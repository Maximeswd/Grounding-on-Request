#!/usr/bin/env python
"""Development-only optimisation check (advisor, 2026-10-08): embedding learning rate 0.005 vs 0.05 for the word and the
standard prefix, everything else fixed (L=1, task holdout, dev scenes excluded, same init/objective/placement/budget).
Reads the dev reference evaluation (image/caption Flip and Stay) and train_diag.json (norm-clamp activation, final norms).
Usage: python scripts/dev_lr_compare.py"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f3 = lambda x: "--" if x is None else f"{x:.3f}"
rows = []
for obj in ("psft", "psftctmsm"):
    for i, iname in (("neo", "word"), ("pt", "standard prefix")):
        for lr, sfx in (("0.05", ""), ("0.005", "_lr0p005")):
            r = f"clevr3_v2_1tok_{obj}_{i}_holdcs_dev{sfx}_seed0"
            ev = os.path.join(ROOT, "runs", r, "eval", "clevr3_dev", "flip_step_800_v2ref.json"); dg = os.path.join(ROOT, "runs", r, "train_diag.json")
            d = json.load(open(ev)) if os.path.exists(ev) else {}; g = json.load(open(dg)) if os.path.exists(dg) else {}
            rows.append({"objective": obj, "interface": iname, "lr": lr, "run": r, "img_F": d.get("look_flip"), "img_S": d.get("look_stay_c"),
                         "cap_F": d.get("read_flip"), "cap_S": d.get("read_stay_c"),
                         "JOINT4": min(v for v in (d.get("look_flip"), d.get("look_stay_c"), d.get("read_flip"), d.get("read_stay_c"))) if d else None,
                         "clamp_steps": (f"{g['steps_clamped_any_row']}/{g['steps_counted']}" if g else None),
                         "final_norm_mean": g.get("final_row_norm", {}).get("mean") if g else None, "limit": g.get("limit") if g else None})
out = ["# Dev-only learning-rate check: 0.005 vs 0.05 (Gemma, L=1, task holdout, dev partition)", "", __doc__.strip(), "",
       "| objective | interface | lr | img F | img S | cap F | cap S | JOINT4 | clamp steps | final row norm (limit) |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    out.append(f"| {r['objective']} | {r['interface']} | {r['lr']} | {f3(r['img_F'])} | {f3(r['img_S'])} | {f3(r['cap_F'])} | {f3(r['cap_S'])} | {f3(r['JOINT4'])} | {r['clamp_steps'] or '--'} | {f3(r['final_norm_mean'])} ({f3(r['limit'])}) |")
out.append(f"\nMissing: {sum(1 for r in rows if r['img_F'] is None)} of {len(rows)} evaluations; {sum(1 for r in rows if r['clamp_steps'] is None)} runs without train_diag.json (trained before the diagnostics existed).")
json.dump(rows, open(os.path.join(ROOT, "results", "dev_lr_compare.json"), "w"), indent=1); open(os.path.join(ROOT, "results", "dev_lr_compare.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
