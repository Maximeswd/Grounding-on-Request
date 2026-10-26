#!/usr/bin/env python
"""Procedural conflict detection with the two modality words only, scored so that
every interface is measured the same way.

THE PROCEDURE. Ask the model for the same question twice, once requesting the image
and once requesting the caption, and flag a conflict when the two answers differ. No
word is trained for the comparison, no threshold is fitted, and nothing is generated:
both requests are scored as a forced choice between the SAME two candidate strings
(the image value and the caption value), and the answer is the argmax. So the
measurement cannot reward a method for being fluent in an output format, and it does
not need a judge. The only thing that differs across interfaces is how the request is
carried: an English sentence, a learned prefix, updated weights, or a modality word.

WHAT IS REPORTED, per interface x dataset x model:
  detect   share of CONFLICT items on which the two requests return different values
  attrib   share on which they also return the RIGHT values (image request -> image
           value, caption request -> caption value). A policy that inverts both
           requests scores high on detect and zero on attrib, which is why both are
           reported.
  false    share of AGREE items on which the two requests return different values,
           i.e. a conflict reported where there is none
  bal      (detect + 1 - false) / 2

The agree contexts come from the same twin battery: (I', c) and (I, c'), where the
swapped side is made to match the other modality. Some items are not recoverable from
the banked 0/1 records (an item whose base image answer is already wrong leaves the
agree-side pick undetermined), so every false-alarm cell carries its own n.

Usage: python scripts/detect_procedural.py [--json results/detection_procedural.json]
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = [("vcf", "VCF"), ("color", "ColorSwap"), ("coco", "COCO-CF")]
MODELS = [("gemma3", "Gemma-3-4B", ""), ("qwen4b", "Qwen-3.5-4B", "_qwen35"),
          ("qwen2b", "Qwen-3.5-2B", "_qwen2b"), ("e4b", "Gemma-4-E4B", "_gemma4e4b")]
IF = [("No request", "{ds}_neo_ctrl_5tok_sft_seed0{s}", "flip_ll_noop_step_800.json"),
      # 2026-09-14: the iv5 file inside the NEO run dir is a neologism eval for qwen2b/e4b (method=neo;
      # byte-identical to that run's own flip_ll on VCF), so the English row was wrong for two models.
      # The English instruction (phrasing iv5, the paper's canonical one) lives in the frozen dirs.
      ("English instruction", "{ds}_base_frozen_0tok_none_seed0{s}", "flip_ll_iv5_step_800.json"),
      ("Prompt tuning", "{ds}_pt_na_5tok_sft_seed0{s}", "flip_ll_step_800.json"),
      # OBJECTIVE-MATCHED ROWS (2026-09-13): the battery previously compared our CTM neologism
      # against prompt tuning under SFT only and LoRA at one rank, which is not a matched
      # comparison. Both checkpoints already existed; only the aggregation was missing them.
      ("Prompt tuning + CTM", "{ds}_pt_na_5tok_ctm_seed0{s}", "flip_ll_step_800.json"),
      ("LoRA r=4", "{ds}_lora_r4_lr2e4_seed0{s}", "flip_ll_step_800.json"),
      ("LoRA r=16", "{ds}_lora_r16_lr2e4_seed0{s}", "flip_ll_step_800.json"),
      ("Neologisms, SFT", "{ds}_neo_ctrl_5tok_sft_seed0{s}", "flip_ll_step_800.json"),
      ("Neologisms + CTM", "{ds}_neo_ctrl_5tok_ctm_seed0{s}", "flip_ll_step_800.json")]


def cell(path):
    """Reconstruct the two requests' answers in the conflict context and in the two
    agree contexts from the banked per-item records."""
    try:
        r = json.load(open(path))["records"]
        la, ra = r["look_acc"], r["read_acc"]
        lf, rf, ls, rs = r["look_flip"], r["read_flip"], r["look_stay"], r["read_stay"]
    except Exception:
        return None
    n = min(len(la), len(ra), len(lf), len(rf), len(ls), len(rs))
    if n == 0:
        return None
    detect = attrib = 0
    fa = fa_n = 0
    for j in range(n):
        li = "i" if la[j] else "t"          # <look>'s answer in the conflict context
        ri = "t" if ra[j] else "i"          # <read>'s answer in the conflict context
        detect += li != ri
        attrib += (li == "i") and (ri == "t")
        # agree context A: the image is swapped, so both sides attest the text value
        if la[j]:                            # lp recoverable only when li == "i"
            lp = "t" if lf[j] else "i"
            rp = ri if rs[j] else ("t" if ri == "i" else "i")
            fa += lp != rp; fa_n += 1
        # agree context B: the caption is swapped, so both sides attest the image value
        if ra[j]:                            # rc recoverable only when ri == "t"
            rc = "i" if rf[j] else "t"
            lc = li if ls[j] else ("i" if li == "t" else "t")
            fa += lc != rc; fa_n += 1
    d, a = detect / n, attrib / n
    f = (fa / fa_n) if fa_n else None
    return {"detect": round(d, 3), "attrib": round(a, 3),
            "false": None if f is None else round(f, 3),
            "bal": None if f is None else round((d + 1 - f) / 2, 3),
            "n": n, "n_agree": fa_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=f"{ROOT}/results/detection_procedural.json")
    a = ap.parse_args()
    banked, printed = {}, []
    for lab, run, fname in IF:
        for m, mlab, sfx in MODELS:
            for ds, dslab in DS:
                c = cell(f"{ROOT}/runs/{run.format(ds=ds, s=sfx)}/eval/{ds}_test_id/{fname}")
                if c:
                    banked[f"{lab}|{m}|{ds}"] = c
                    printed.append((lab, mlab, dslab, c))
    F = lambda v: " -- " if v is None else f"{v:.3f}"
    print(f"{'interface':20s} {'model':12s} {'dataset':10s} {'detect':>7s} {'attrib':>7s} "
          f"{'false':>7s} {'bal':>7s} {'n':>4s} {'n_agr':>6s}")
    for lab, mlab, dslab, c in printed:
        print(f"{lab:20s} {mlab:12s} {dslab:10s} {F(c['detect']):>7s} {F(c['attrib']):>7s} "
              f"{F(c['false']):>7s} {F(c['bal']):>7s} {c['n']:>4d} {c['n_agree']:>6d}")
    agg = {}
    for k, c in banked.items():
        lab = k.split("|")[0]
        agg.setdefault(lab, []).append(c)
    print("\npooled over the available cells:")
    for lab, cs in agg.items():
        det = [c["detect"] for c in cs]; att = [c["attrib"] for c in cs]
        bal = [c["bal"] for c in cs if c["bal"] is not None]
        fal = [c["false"] for c in cs if c["false"] is not None]
        print(f"  {lab:20s} cells={len(cs):2d} detect {min(det):.2f}-{max(det):.2f} "
              f"(mean {sum(det)/len(det):.2f}) attrib mean {sum(att)/len(att):.2f} "
              f"false mean {sum(fal)/len(fal):.2f} bal mean {sum(bal)/len(bal):.2f}")
    json.dump({"note": __doc__.strip().split("\n\n")[0], "cells": banked,
               "pooled": {lab: {"n_cells": len(cs),
                                "detect_mean": round(sum(c["detect"] for c in cs) / len(cs), 3),
                                "attrib_mean": round(sum(c["attrib"] for c in cs) / len(cs), 3),
                                "bal_mean": round(sum(c["bal"] for c in cs if c["bal"] is not None)
                                                  / max(sum(1 for c in cs if c["bal"] is not None), 1), 3)}
                          for lab, cs in agg.items()}},
              open(a.json, "w"), indent=1)
    print(f"\n[saved] {a.json}")


if __name__ == "__main__":
    main()
