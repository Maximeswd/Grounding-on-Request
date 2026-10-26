#!/usr/bin/env python
"""WITHIN-PROMPT MODALITY SWITCHING on Ground-CF (author's evaluation #1, 2026-09-23, STATUS v1.130).

Can one prompt carry DIFFERENT grounding requests for different questions? Two independent Ground-CF conflict items
(each: its own image, its own fact caption, its own question) are placed in ONE user turn, and each question carries
its own control. All four assignments are scored: (look, look), (read, read), (look, read), (read, look). The
questions themselves carry no modality information.

  word          "<q> Answer according to <look>."          (the word, locally scoped to its question)
  prefix        "<q> Answer according to <pt_look>."        (the soft tokens placed locally; trained as a prompt prefix)
  English       "<q> Answer according to the image."        (--mode instructed; the natural-language reference)
  no request    "<q>"                                       (--mode noop)
  LoRA adapter  not expressible: adapter selection is global model state, one adapter per forward pass. The script
                writes an explicit N/A record instead of a number.

The question uses the held-out formulations and the carrier "Answer according to X." occurs in no training data.
Scoring is deterministic and candidate-based (no judge): the continuation "a1; a2" is scored with the same
teacher-forced log-likelihood as the headline evaluation for every pair in cand(Q1) x cand(Q2), where cand(Q) =
{image value, caption value, a third value no source states}; the arg-max pair is the prediction (chance 1/9).
Item-level records keep every candidate score; with --generate 1 (default) a greedy reply is kept and parsed as a pair.

Outputs runs/<run>/eval/clevr3_<split>/switch_step_<step><tag>.json (summary) and ..._items.jsonl (one record per
pair x assignment, universal schema: run, model, method, request pair, prompt text, candidate scores, prediction, gold).
Usage: python scripts/eval_clevr3_switch.py --run_dir runs/<run> [--mode instructed|noop --tag _instr] [--max_n 150]
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from tqdm import tqdm
from scsrc.clevr3 import (_blocks, clevr3_token, heldout_question, cp_third_value, set_c3_vocab, INSTR, eval_design)
from scsrc.tokens import trigger
from evaluate import apply_checkpoint
from scripts.eval_clevr3 import loglik
from scripts.eval_clevr3_generate import generate_free, parse_pair, loglik_both
from scripts.tokscore import split_spans

HEADER = ("Two questions follow, each with its own image and caption. Answer each question according to the request "
          "given in that question. Return exactly the two answers, in order, separated by a semicolon.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id"); ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=150, help="number of item PAIRS")
    ap.add_argument("--mode", default=None, help="None (the run's own control) | instructed | noop")
    ap.add_argument("--tag", default="")
    ap.add_argument("--generate", type=int, default=1, help="also greedy-generate the joint answer and parse it (raw text kept)")
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--pt_start", type=int, default=0,
                    help="v1.159 (author): prefix runs only -- STANDARD prefix composition: both source prefixes at the start "
                         "of the user turn, one per question in question order (the mixed assignments give both orders), "
                         "and no local control in the questions; needs --tag")
    a = ap.parse_args()
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode needs --tag")
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}"); os.makedirs(od, exist_ok=True)
    out_json = os.path.join(od, f"switch_step_{a.step}{a.tag}.json")
    method = cfg.get("method", "neo")
    if a.mode is None and method == "lora":
        json.dump({"run": os.path.basename(a.run_dir), "applicable": False,
                   "note": "LoRA with one adapter per request: adapter selection is global model state, so two "
                           "different grounding requests cannot be assigned to two questions in one forward pass."},
                  open(out_json, "w"), indent=1)
        print(f"[saved] {out_json} (not applicable for a per-request LoRA adapter)"); return

    model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    if forbid is None:
        forbid = getattr(model, "_neo_all_ids", None)
    if a.pt_start and (meta.get("method") != "pt" or a.mode or not a.tag):
        raise SystemExit("[STOP] --pt_start 1 needs a prefix run, no --mode, and its own --tag")
    n = meta["n_tokens"]
    mode = a.mode or cfg.get("factor_mode", "srconly")
    if a.mode in ("instructed", "noop"):
        method = "neo"                                # frozen backbone; no learned rows referenced

    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    set_c3_vocab(rows)
    conf = sorted([r for r in rows if str(r["channels"]["look"]) != str(r["channels"]["read"])], key=lambda r: r["id"])
    # deterministic pairing: each item with the next conflict item of a DIFFERENT skill
    pairs, used = [], set()
    for i, r in enumerate(conf):
        if r["id"] in used:
            continue
        for s in conf[i + 1:]:
            if s["id"] not in used and s["skill"] != r["skill"]:
                pairs.append((r, s)); used |= {r["id"], s["id"]}; break
        if len(pairs) >= a.max_n:
            break

    def control(item, src):
        if a.pt_start:
            return ""                                   # the prefixes sit at the start of the turn, not in the question
        if a.mode == "noop":
            return ""
        if a.mode == "instructed":
            return INSTR[src]
        if method == "pt":
            return trigger(f"pt_{src}", n)             # local placement of the soft tokens
        return clevr3_token(item, src, n, mode)       # the word

    def block(item, idx, src, seed):
        img = Image.open(item["image"]).convert("RGB")
        s2, cap_first = eval_design(item, seed, "heldout")      # v1.133: balanced order + question per item
        q = heldout_question(item, s2)
        ctl = control(item, src)
        qtxt = f"{q} Answer according to {ctl}." if ctl else q
        cap = _blocks(item).get("read")
        if cap_first is False:                                    # [image][caption] for half of the items
            return [{"type": "text", "text": f"Question {idx}."}, {"type": "image", "image": img},
                    {"type": "text", "text": f"{cap}\n{qtxt}"}], qtxt
        return [{"type": "text", "text": f"Question {idx}.\n{cap}"}, {"type": "image", "image": img},
                {"type": "text", "text": qtxt}], qtxt

    def cands(item):
        A, B = str(item["channels"]["look"]), str(item["channels"]["read"])
        w = cp_third_value(item, exclude={A, B}, salt="switch") or A
        return [A, B] + ([w] if w not in (A, B) else [])

    assigns = [("look", "look"), ("read", "read"), ("look", "read"), ("read", "look")]
    R = {f"{x}|{y}": {"q1": [], "q2": [], "both": [], "gen_both": [], "gen_parsed": []} for x, y in assigns}
    items_out = open(out_json.replace(".json", "_items.jsonl"), "w")
    for k, (r1, r2) in enumerate(tqdm(pairs, desc="switch")):
        c1, c2 = cands(r1), cands(r2)
        for x, y in assigns:
            b1, t1 = block(r1, 1, x, k); b2, t2 = block(r2, 2, y, k + 1)
            head = HEADER if not a.pt_start else (trigger(f"pt_{x}", n) + trigger(f"pt_{y}", n) + " " + HEADER)
            um = [{"role": "user", "content": [{"type": "text", "text": head}] + b1 + b2}]
            det = {f"{v1}; {v2}": loglik_both(model, proc, um, f"{v1}; {v2}", "cuda", tokens=True) for v1, v2 in itertools.product(c1, c2)}
            sp = split_spans({k: (d[3], d[4]) for k, d in det.items()})      # v1.144: token-level detail
            sc = ({k: sp[k]["answer_sum"] for k in det} if os.environ.get("SC_SCORE", "turn") == "answer" else {k: d[0] for k, d in det.items()})
            best = max(sc, key=sc.get); p1, p2 = best.split("; ")
            g1, g2 = str(r1["channels"][x]), str(r2["channels"][y])
            ok1, ok2 = int(p1 == g1), int(p2 == g2)
            key = f"{x}|{y}"; R[key]["q1"].append(ok1); R[key]["q2"].append(ok2); R[key]["both"].append(ok1 * ok2)
            gen = {}
            if a.generate:
                txt, nnew = generate_free(model, proc, um[0]["content"], "cuda", a.max_new, forbid, 8)
                glab, g1v, g2v = parse_pair(txt, tuple(c1), tuple(c2))
                gok = int(glab == "pair" and g1v == g1 and g2v == g2)
                R[key]["gen_both"].append(gok); R[key]["gen_parsed"].append(int(glab == "pair"))
                gen = {"raw_generation": txt, "gen_ntok": nnew, "parse_label": glab, "parsed": [g1v, g2v],
                       "gen_both_correct": gok}
            items_out.write(json.dumps({
                "run": os.path.basename(a.run_dir), "model": cfg.get("model_id"), "dataset": "Ground-CF", "split": a.split,
                "method": ("english" if a.mode == "instructed" else "no_request" if a.mode == "noop"
                           else "prefix_at_start" if a.pt_start else method),
                "eval_type": "switch", "requests": [x, y], "item_ids": [r1["id"], r2["id"]], "skills": [r1["skill"], r2["skill"]],
                "prompt_questions": [t1, t2], "caption_style": os.environ.get("SC_CAPTION_STYLE", "template"),
                "gold": [g1, g2], "competing": [str(r1["channels"]["read" if x == "look" else "look"]),
                                                str(r2["channels"]["read" if y == "look" else "look"])],
                "candidate_scores": {kk: round(float(v), 4) for kk, v in sc.items()},
                "candidate_detail": {kk: {"sum": round(det[kk][0], 4), **sp[kk]} for kk in det},
                "prediction": [p1, p2], "metrics": {"q1_correct": ok1, "q2_correct": ok2, "both_correct": ok1 * ok2}, **gen}) + "\n")
    items_out.close()
    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    summ = {key: {"q1": m(v["q1"]), "q2": m(v["q2"]), "both": m(v["both"]), "n": len(v["both"]),
                  "gen_both": m(v["gen_both"]) if v["gen_both"] else None, "gen_parse_rate": m(v["gen_parsed"]) if v["gen_parsed"] else None}
            for key, v in R.items()}
    mixed = [summ["look|read"]["both"], summ["read|look"]["both"]]
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step, "mode": a.mode, "method": method,
           "applicable": True, "n_pairs": len(pairs), "chance_both": round(1 / 9, 3), "assignments": summ,
           "mixed_both": round(sum(mixed) / 2, 3),
           "gen_mixed_both": (round((summ["look|read"]["gen_both"] + summ["read|look"]["gen_both"]) / 2, 3) if a.generate else None), "same_both": round((summ["look|look"]["both"] + summ["read|read"]["both"]) / 2, 3),
           "caption_style": os.environ.get("SC_CAPTION_STYLE", "template"),
           "note": "two Ground-CF conflict items per prompt, each question with its own control; mixed = different "
                   "controls on the two questions (the scoping test); candidate-scored joint answers, chance 1/9"}
    json.dump(out, open(out_json, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1)); print("[saved]", out_json)


if __name__ == "__main__":
    main()
