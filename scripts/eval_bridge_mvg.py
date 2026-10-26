#!/usr/bin/env python
"""RUNG 3: CLEVR3-trained source tokens, ZERO-SHOT, on real event video (MVG).

The channel correspondence is exact in kind: MVG's VIS/DESC/on-screen-OCR are
the same three provenance channels as clevr3's look/read/ocr -- only the
content shifts (rendered shapes -> news video; one-line caption -> long
description; VALUE=x -> real screen text). Each mvg3 aligned item is mapped
into the CLEVR3 prompt schema (the format the tokens were trained with) and
scored 3-way: does <src_look> pick the VIS value, <src_read> the DESC value,
<src_ocr> the on-screen value? Routing matrix + selectivity, exactly like
eval_mvg3 -- but the tokens have NEVER seen a photograph or a video.

This is the strongest transfer claim in the paper if it holds even partially:
provenance routing learned on synthetic shapes applies zero-shot to real video.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from tqdm import tqdm
from scsrc.clevr3 import clevr3_user_content
from evaluate import apply_checkpoint
from scripts.eval_clevr3 import loglik, cand

MAP = {"VIS": "look", "DESC": "read", "OCR": "ocr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--data_dir", default="data/mvg3")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=200)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    args = cfg.get("args", {})
    mode = args.get("factor_mode", "srconly")
    qstyle = args.get("question_style", "explicit")
    n_tokens = args.get("n_tokens", 5)

    model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    items = [json.loads(l)
             for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))][:a.max_n]

    M = defaultdict(lambda: defaultdict(int)); tot = defaultdict(int)
    diag, offd = [], []
    for it in tqdm(items, desc="rung3"):
        present = it["present"]; vals = it["channel_values"]
        # mvg item -> clevr3 schema: long description as the caption block,
        # joined screen strings as the ocr block, the role question as question
        ci = {"id": f"{it['video_id']}_{it['role']}", "image": it["vis_frame"],
              "skill": "object",   # entity-identity: the closest operator
              "question": (it.get("role_question") or "What is shown?"),
              "channels": {MAP[c]: vals[c] for c in present},
              "caption": it["desc_text"][:400],
              "ocr_text": " | ".join(it["ocr_strings"])[:200] or "(none)"}
        for qc in present:
            src = MAP[qc]
            um = [{"role": "user",
                   "content": clevr3_user_content(ci, src, n_tokens, mode, seed=0,
                                                  question_style=qstyle)}]
            scores = {c: loglik(model, proc, um, cand(vals[c]), "cuda")
                      for c in present}
            picked = max(scores, key=scores.get)
            M[qc][picked] += 1; tot[qc] += 1
    d = [M[q][q] / tot[q] for q in tot]
    o = [M[q][p] / tot[q] for q in tot for p in ("VIS", "DESC", "OCR")
         if p != q and tot[q]]
    sel = round(sum(d) / max(len(d), 1) - sum(o) / max(len(o), 1), 3)

    print(f"\n== RUNG-3 zero-shot  {os.path.basename(a.run_dir)} -> MVG {a.split}")
    print("   routing M  picks:  VIS   DESC  OCR")
    for q in ("VIS", "DESC", "OCR"):
        if tot[q]:
            row = "  ".join(f"{M[q][p]/tot[q]:.2f}" for p in ("VIS", "DESC", "OCR"))
            print(f"   <{MAP[q]:4s}>({tot[q]:3d}): {row}")
    print(f"   SELECTIVITY {sel}")
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "n": len(items), "selectivity": sel,
           "matrix": {q: {p: round(M[q][p] / tot[q], 3) for p in ("VIS", "DESC", "OCR")}
                      for q in tot if tot[q]}}
    outd = os.path.join(a.run_dir, "eval", f"rung3_mvg_{a.split}")
    os.makedirs(outd, exist_ok=True)
    json.dump(out, open(os.path.join(outd, f"sel_step_{a.step}.json"), "w"), indent=1)
    print(f"[saved] {outd}/sel_step_{a.step}.json")


if __name__ == "__main__":
    main()
