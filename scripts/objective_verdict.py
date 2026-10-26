#!/usr/bin/env python
"""The objective claim, stated as a tally with a test instead of an average.

For every cell of the parity grid (dataset x model x modality word) this compares the
two sided counterfactual objective against each alternative, counts wins, losses and
ties at a stated tolerance, and reports a one sided sign test over the non-tied cells.
It also splits the cells by whether ordinary supervision was already strong there,
because that is where the claim has structure: the objective is a repair for weak
cells, not a uniform gain.

Multi-seed cells (when the seed runs land) are averaged before the comparison and the
per-cell seed spread is reported, so a difference smaller than the spread is a tie.

Usage: python scripts/objective_verdict.py [--eps 0.02]
"""
import argparse, glob, json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ["vcf", "color", "coco", "groundcf"]
MODELS = ["gemma3", "qwen4b", "qwen2b", "e4b"]
CH = {"vcf": ["look", "read"], "color": ["look", "read"],
      "coco": ["look", "read"], "groundcf": ["look", "read", "ocr"]}
SFX = {"gemma3": "", "qwen4b": "_qwen35", "qwen2b": "_qwen2b", "e4b": "_gemma4e4b"}
ALT = [("word_sft|5", "supervised neologism"), ("instr|None", "English instruction"),
       ("pt|5", "prompt tuning"), ("lora_r4|None", "LoRA rank 4"),
       ("word+apo|5", "preference (APO)")]


def seed_values(ds, model, arm, ch):
    """Grounding across available seeds for one cell (arm in {sft, ctm})."""
    sfx, out = SFX[model], []
    for seed in (0, 1, 2):
        if ds == "groundcf":
            p = f"{ROOT}/runs/clevr3_f_5tok_{arm}_seed{seed}{sfx}/eval/clevr3_test_id/flip_step_800.json"
            key = f"{ch}_flip"
        else:
            p = f"{ROOT}/runs/{ds}_neo_ctrl_5tok_{arm}_seed{seed}{sfx}/eval/{ds}_test_id/flip_ll_step_800.json"
            key = f"{ch}_flip"
        if os.path.exists(p):
            v = json.load(open(p)).get(key)
            if v is not None:
                out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.02,
                    help="a difference at or below this counts as a tie")
    a = ap.parse_args()
    g = json.load(open(f"{ROOT}/results/master_grid.json"))
    cells, banked = [], {}
    for ds in DS:
        for m in MODELS:
            for ch in CH[ds]:
                ctm_s, sft_s = seed_values(ds, m, "ctm", ch), seed_values(ds, m, "sft", ch)
                ctm = sum(ctm_s) / len(ctm_s) if ctm_s else (g.get(f"{ds}|{m}|word+ctm|5") or {}).get(f"G.{ch}")
                spread = (max(ctm_s) - min(ctm_s)) if len(ctm_s) > 1 else None
                row = {"ds": ds, "model": m, "ch": ch, "ctm": ctm, "n_seeds_ctm": len(ctm_s),
                       "ctm_spread": spread, "sft_seeds": len(sft_s)}
                for key, _ in ALT:
                    if key == "word_sft|5" and sft_s:
                        row[key] = sum(sft_s) / len(sft_s)
                    else:
                        row[key] = (g.get(f"{ds}|{m}|{key}") or {}).get(f"G.{ch}")
                cells.append(row)
                banked[f"{ds}|{m}|{ch}"] = row
    print(f"tolerance for a tie: {a.eps}\n")
    summary = {}
    for key, lab in ALT:
        w = l = t = 0
        for c in cells:
            x, y = c["ctm"], c.get(key)
            if x is None or y is None:
                continue
            eps = max(a.eps, c["ctm_spread"] or 0)
            if x > y + eps: w += 1
            elif y > x + eps: l += 1
            else: t += 1
        n = w + l
        p = sum(math.comb(n, k) for k in range(w, n + 1)) / 2 ** n if n else 1.0
        summary[lab] = {"win": w, "loss": l, "tie": t, "sign_p": round(p, 5)}
        print(f"  CTM vs {lab:22s} win {w:2d}  loss {l:2d}  tie {t:2d}   sign test p={p:.4f}")
    weak = [c for c in cells if c.get("word_sft|5") is not None and c["word_sft|5"] < 0.80
            and c["ctm"] is not None]
    strong = [c for c in cells if c.get("word_sft|5") is not None and c["word_sft|5"] >= 0.80
              and c["ctm"] is not None]
    for name, group in (("weak (supervision < .80)", weak), ("strong (>= .80)", strong)):
        up = sum(c["ctm"] > c["word_sft|5"] + a.eps for c in group)
        dn = sum(c["word_sft|5"] > c["ctm"] + a.eps for c in group)
        med = sorted(c["ctm"] - c["word_sft|5"] for c in group)
        print(f"\n  {name}: {len(group)} cells, objective better on {up}, worse on {dn}, "
              f"median change {med[len(med)//2]:+.3f}")
        summary[name] = {"cells": len(group), "better": up, "worse": dn,
                         "median_change": round(med[len(med) // 2], 3)}
    json.dump({"note": "per-cell verdict for the two sided objective; seeds averaged, a "
                       "difference within the seed spread counts as a tie",
               "eps": a.eps, "summary": summary, "cells": banked},
              open(f"{ROOT}/results/objective_verdict.json", "w"), indent=1)
    print(f"\n[saved] results/objective_verdict.json")


if __name__ == "__main__":
    main()
