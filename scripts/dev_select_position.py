#!/usr/bin/env python
"""Pre-registered development selection of the CONTROL POSITION (2026-10-19, before any of these dev results existed).
Positions: native (after the question, the paper's placement), first (before both evidence blocks), pre_src (right before
the requested source's block), post_src (right after it). Criterion: JOINT4 = min(image Flip, image Stay, caption Flip,
caption Stay), reference condition, dev partition (dev scenes excluded from training), Gemma-3-4B, L=1, task holdout, seed 0;
selection per objective (paired SFT; +CTM+Stay); ties -> native. The standard prompt-tuning prefix is the 'first' position
of a bare soft prompt and is identical in computation to the word there up to the carrier words, so the sweep over the
word's positions is the sweep for both interfaces."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS = [("native", ""), ("first", "_posfirst"), ("pre_src", "_pospresrc"), ("post_src", "_pospostsrc")]
f3 = lambda x: "--" if x is None else f"{x:.3f}"
rows, best = [], {}
for obj in ("psft", "psftctmsm"):
    for pos, sfx in POS:
        r = f"clevr3_v2_1tok_{obj}_neo_holdcs_dev{sfx}_seed0"
        p = os.path.join(ROOT, "runs", r, "eval", "clevr3_dev", "flip_step_800_v2ref.json")
        if not os.path.exists(p):
            rows.append({"objective": obj, "position": pos, "run": r, "missing": True}); continue
        d = json.load(open(p)); m = {"img_F": d["look_flip"], "img_S": d["look_stay_c"], "cap_F": d["read_flip"], "cap_S": d["read_stay_c"]}
        j4 = min(m.values()); rows.append({"objective": obj, "position": pos, "run": r, **m, "JOINT4": j4, "n": d.get("n")})
        if obj not in best or j4 > best[obj][0]:
            best[obj] = (j4, pos)
missing = sum(1 for r in rows if r.get("missing"))
out = ["# Development selection: control position (Gemma, L=1, task holdout, dev partition)", "", __doc__.strip(), "",
       "| objective | position | img F | img S | cap F | cap S | JOINT4 |", "|---|---|---|---|---|---|---|"]
for r in rows:
    out.append(f"| {r['objective']} | {r['position']} | " + (" MISSING | | | | |" if r.get("missing") else f"{f3(r['img_F'])} | {f3(r['img_S'])} | {f3(r['cap_F'])} | {f3(r['cap_S'])} | **{f3(r['JOINT4'])}** |"))
out += ["", ("Selected (JOINT4, ties -> native): " + "; ".join(f"{o}: {b[1]} ({b[0]:.3f})" for o, b in sorted(best.items()))) if missing == 0 else f"Selected: NOT YET -- {missing} of {len(rows)} cells missing", f"Missing: {missing} of {len(rows)}"]
json.dump({"criterion": __doc__.strip(), "rows": rows, "complete": missing == 0, "selected": ({o: {"position": b[1], "JOINT4": b[0]} for o, b in best.items()} if missing == 0 else None)},
          open(os.path.join(ROOT, "results", "dev_selection_position.json"), "w"), indent=1)
open(os.path.join(ROOT, "results", "dev_selection_position.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
