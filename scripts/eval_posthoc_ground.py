#!/usr/bin/env python
"""POST HOC GROUNDING VERIFIER on Ground-CF: is this answer grounded in the source it was
asked to use? No twin is needed at test time; the requested source is ABLATED instead.

For an item, a requested source s and a candidate answer v,

    V_s(v) = log p(v | context, request s) - log p(v | context with s removed, request s)

(image -> grey image; caption / on-screen text -> the block is dropped). This is the
claim verifier of scripts/eval_claim_verifier.py, applied where labels are exact by
construction: every value in the context belongs to a known source.

Three readings, per source and interface:
  own answer     the model's own pick a under request s is GROUNDED iff a equals s's
                 value. AUROC of V_s(a) for grounded vs not; a positive V at threshold 0
                 says "this answer needed the source". Ungrounded-but-confident answers
                 are hallucinated grounding: the request named s and s was not needed.
  candidates     AUROC of V_s over all candidate values: does necessity single out the
                 source's own value? And argmax_v V_s(v) == value(s) ("attribution by
                 necessity", threshold free).
  which source   for the model's own answer a, argmax over sources s' of V_{s'}(a):
                 does necessity recover the source that actually attests a?

(item, s) pairs are scored only when s's value differs from both other sources, so the
origin of an answer is unambiguous.

Usage: python scripts/eval_posthoc_ground.py --run_dir runs/clevr3_f_5tok_ctm_seed0 --split test_id
       INSTR_VARIANT=5 python scripts/eval_posthoc_ground.py --run_dir runs/clevr3_f_5tok_sft_seed0 \
           --mode instructed --tag _instr_iv5
Output: runs/<run>/eval/clevr3_<split>/posthoc_ground_step_<step><tag>.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from tqdm import tqdm
from scsrc.clevr3 import clevr3_user_content
from evaluate import apply_checkpoint
from scripts.eval_clevr3 import cand, loglik

SRC = ("look", "read", "ocr")


def auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=150)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode instructed/noop needs --tag")
    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    try:
        cfg = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    except Exception:
        cfg = {}
    meth = "neo" if a.mode in ("instructed", "noop") else cfg.get("method", "neo")
    cfgm = a.mode or cfg.get("factor_mode", "srconly")
    print(f"[prompt] method={meth} mode={cfgm} L={n}")
    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    gray = Image.new("RGB", (336, 336), (128, 128, 128))

    def scores(item, s, cands, seed, ablate):
        if ablate and s == "look":
            it2 = dict(item); it2["image"] = gray
            um = [{"role": "user", "content": clevr3_user_content(it2, s, n, cfgm, seed, method=meth)}]
        else:
            um = [{"role": "user", "content": clevr3_user_content(
                item, s, n, cfgm, seed, method=meth, blocks_mode=(f"drop_{s}" if ablate else "all"))}]
        return {v: loglik(model, proc, um, cand(v), "cuda") for v in cands}

    recs, seen = [], 0
    for i, it in enumerate(tqdm(rows, desc="posthoc")):
        if seen >= a.max_n:
            break
        ch = {k: str(v) for k, v in it["channels"].items()}
        cands = tuple(dict.fromkeys(ch.values()))
        if len(cands) < 2:
            continue
        seen += 1
        full = {s: scores(it, s, cands, i, False) for s in SRC}
        abl = {s: scores(it, s, cands, i, True) for s in SRC}
        for s in SRC:
            unambiguous = all(ch[s] != ch[o] for o in SRC if o != s)
            own = max(full[s], key=full[s].get)
            V = {v: full[s][v] - abl[s][v] for v in cands}
            # which source does necessity say the own answer came from?
            Vsrc = {s2: full[s2][own] - abl[s2][own] for s2 in SRC}
            recs.append({"id": it.get("id"), "skill": it.get("skill"), "source": s,
                         "unambiguous": unambiguous, "gold": ch[s], "own": own,
                         "grounded": int(own == ch[s]), "V": V, "V_by_source": Vsrc,
                         "attests": [s2 for s2 in SRC if ch[s2] == own]})
    out = {"meta": {"run": a.run_dir, "step": a.step, "mode": cfgm, "method": meth,
                    "n_items": seen, "split": a.split, "instr_variant": os.environ.get("INSTR_VARIANT")},
           "by_source": {}}
    for s in SRC:
        rs = [r for r in recs if r["source"] == s and r["unambiguous"]]
        if not rs:
            continue
        pos = [r["V"][r["own"]] for r in rs if r["grounded"]]
        neg = [r["V"][r["own"]] for r in rs if not r["grounded"]]
        cpos = [r["V"][r["gold"]] for r in rs]
        cneg = [r["V"][v] for r in rs for v in r["V"] if v != r["gold"]]
        attrib_nec = sum(max(r["V"], key=r["V"].get) == r["gold"] for r in rs) / len(rs)
        which = [r for r in rs if r["attests"]]
        which_acc = (sum(max(r["V_by_source"], key=r["V_by_source"].get) in r["attests"] for r in which)
                     / len(which)) if which else None
        acc0 = (sum((r["V"][r["own"]] > 0) == bool(r["grounded"]) for r in rs) / len(rs))
        out["by_source"][s] = {
            "n": len(rs), "accuracy": round(sum(r["grounded"] for r in rs) / len(rs), 4),
            "own_auroc": auroc(pos, neg), "own_n_grounded": len(pos), "own_n_ungrounded": len(neg),
            "own_acc_at_0": round(acc0, 4),
            "cand_auroc": auroc(cpos, cneg), "attrib_by_necessity": round(attrib_nec, 4),
            "which_source_acc": None if which_acc is None else round(which_acc, 4),
            "mean_V_grounded": round(sum(pos) / len(pos), 3) if pos else None,
            "mean_V_ungrounded": round(sum(neg) / len(neg), 3) if neg else None}
        d = out["by_source"][s]
        print(f"  {s:4s} n={d['n']:3d} acc {d['accuracy']:.2f} | own AUROC {d['own_auroc']} "
              f"(g {d['own_n_grounded']}/u {d['own_n_ungrounded']}) acc@0 {d['own_acc_at_0']:.2f} | "
              f"cand AUROC {d['cand_auroc']} attrib {d['attrib_by_necessity']:.2f} | which-src {d['which_source_acc']}")
    out["records"] = recs
    od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, f"posthoc_ground_step_{a.step}{a.tag}.json")
    json.dump(out, open(fp, "w"), indent=1)
    print("[saved]", fp)


if __name__ == "__main__":
    main()
