#!/usr/bin/env python
"""POST HOC GROUNDING VERIFIER on the ladder (VCF / ColorSwap / COCO-CF): the two-source
version of scripts/eval_posthoc_ground.py. For a conflict item, a request s and a
candidate answer v,

    V_s(v) = LL(v | context, request s) - LL(v | context with s REMOVED, request s)

image removed -> a grey image of the same size; caption removed -> "(no caption provided)"
(the training-time absence convention). Own answer = the request's forced choice between
the image value and the caption value. Grounded iff the own answer is the requested
source's value (the two values differ by construction on conflict items).

Reports per source: accuracy, own-answer AUROC (grounded vs not), acc at V>0, candidate
AUROC, attribution by necessity, which-source recovery.

Usage: python scripts/eval_posthoc_ground_ladder.py --run_dir runs/color_neo_ctrl_5tok_ctm_seed0 \
           --task color --data_dir data/color --split test_id [--mode instructed|noop --tag _x]
Output: runs/<run>/eval/<ds>_<split>/posthoc_ground_step_<step><tag>.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc import prompts as P
from scsrc import score as SC
from scsrc.tokens import trigger, token_strings
from evaluate import apply_checkpoint
from scripts.eval_flip_ll import norm, COLORS, INSTR
from scripts.eval_posthoc_ground import auroc

SRC = ("look", "read")


def ll(model, proc, image, user, cand):
    r = SC.loglik(model, proc, image, user, cand, "cuda")
    return r[0] if isinstance(r, (tuple, list)) else float(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", default="data/color")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=140)
    ap.add_argument("--mode", default=None, help="None (the run's own request) | instructed | noop")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode instructed/noop needs --tag")
    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
    n, method = meta["n_tokens"], meta["method"]
    meth = "neo" if a.mode in ("instructed", "noop") else method
    print(f"[prompt] method={meth} mode={a.mode} L={n}")
    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",)) if s.cond == "conflict"]
    is_vcf = "vcf" in a.data_dir

    def user(op, cap):
        if a.mode == "noop":
            return f'Caption: "{cap}".'
        if a.mode == "instructed" or meth == "lora":
            return f'Caption: "{cap}". {INSTR[op]}'
        if meth == "pt":
            return trigger(f"pt_{op}", n) + " " + P.pt_template(0).format(cap=cap)
        return P.render_ctrl(P.CTRL_CANONICAL, "".join(token_strings(f"{op}_ctrl", n)), cap)

    recs, n_seen = [], 0
    for sp in tqdm(specs, desc="posthoc-ladder"):
        ik, tk = norm(sp.image_key), norm(sp.text_key)
        if not ik or set(ik) != set(tk):
            continue
        conf = [o for o in ik if ik[o] != tk[o] and not (is_vcf and any(c in o.split() for c in COLORS))]
        if not conf:
            continue
        if n_seen >= a.max_n:
            break
        n_seen += 1
        obj = conf[n_seen % len(conf)]
        ci = P.fmt_key_task({obj: ik[obj]}, a.task); ct = P.fmt_key_task({obj: tk[obj]}, a.task)
        cands = (ci, ct); gold = {"look": ci, "read": ct}; attests = {ci: ["look"], ct: ["read"]}
        img = get_image(ds, sp)
        grey = Image.new("RGB", img.size, (128, 128, 128))
        full = {s: {v: ll(model, proc, img, user(s, sp.fed_caption), v) for v in cands} for s in SRC}
        abl = {"look": {v: ll(model, proc, grey, user("look", sp.fed_caption), v) for v in cands},
               "read": {v: ll(model, proc, img, user("read", "(no caption provided)"), v) for v in cands}}
        for s in SRC:
            own = max(full[s], key=full[s].get)
            V = {v: full[s][v] - abl[s][v] for v in cands}
            Vsrc = {s2: full[s2][own] - abl[s2][own] for s2 in SRC}
            recs.append({"ex": sp.ex_index, "source": s, "unambiguous": True, "gold": gold[s], "own": own,
                         "grounded": int(own == gold[s]), "V": V, "V_by_source": Vsrc, "attests": attests[own]})
    out = {"meta": {"run": a.run_dir, "step": a.step, "mode": a.mode, "method": meth, "n_items": n_seen,
                    "task": a.task, "data_dir": a.data_dir, "split": a.split,
                    "instr_variant": os.environ.get("INSTR_VARIANT")}, "by_source": {}}
    for s in SRC:
        rs = [r for r in recs if r["source"] == s]
        if not rs:
            continue
        pos = [r["V"][r["own"]] for r in rs if r["grounded"]]; neg = [r["V"][r["own"]] for r in rs if not r["grounded"]]
        cpos = [r["V"][r["gold"]] for r in rs]; cneg = [r["V"][v] for r in rs for v in r["V"] if v != r["gold"]]
        which = sum(max(r["V_by_source"], key=r["V_by_source"].get) in r["attests"] for r in rs) / len(rs)
        out["by_source"][s] = {
            "n": len(rs), "accuracy": round(sum(r["grounded"] for r in rs) / len(rs), 4),
            "own_auroc": auroc(pos, neg), "own_n_grounded": len(pos), "own_n_ungrounded": len(neg),
            "own_acc_at_0": round(sum((r["V"][r["own"]] > 0) == bool(r["grounded"]) for r in rs) / len(rs), 4),
            "cand_auroc": auroc(cpos, cneg),
            "attrib_by_necessity": round(sum(max(r["V"], key=r["V"].get) == r["gold"] for r in rs) / len(rs), 4),
            "which_source_acc": round(which, 4),
            "mean_V_grounded": round(sum(pos) / len(pos), 3) if pos else None,
            "mean_V_ungrounded": round(sum(neg) / len(neg), 3) if neg else None}
        d = out["by_source"][s]
        print(f"  {s:4s} n={d['n']:3d} acc {d['accuracy']:.2f} | own AUROC {d['own_auroc']} (g {d['own_n_grounded']}/u "
              f"{d['own_n_ungrounded']}) acc@0 {d['own_acc_at_0']:.2f} | cand AUROC {d['cand_auroc']} attrib "
              f"{d['attrib_by_necessity']:.2f} | which-src {d['which_source_acc']}")
    out["records"] = recs
    dsname = os.path.basename(a.data_dir.rstrip("/"))
    od = os.path.join(a.run_dir, "eval", f"{dsname}_{a.split}")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, f"posthoc_ground_step_{a.step}{a.tag}.json")
    json.dump(out, open(fp, "w"), indent=1)
    print("[saved]", fp)


if __name__ == "__main__":
    main()
