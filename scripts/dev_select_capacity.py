#!/usr/bin/env python
"""Pre-registered development selection (2026-10-07, before any of these dev results was read).
Criterion: JOINT4 = min(image Flip, image Stay, caption Flip, caption Stay), reference condition, dev partition
(data/clevr3/pairs_look_all_dev.jsonl; the 100 dev scenes were excluded from training), task holdout, Gemma-3-4B, seed 0.
Selection: per interface (word, standard prefix) over L in {1,5} x objective in {paired SFT, +CTM, +CTM+Stay}; ties -> L=1.
Both interfaces get the same budget (6 configurations each)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = {(L, o, i): (f"clevr3_v2_{L}tok_{o}_{i}_holdcs_dev_initIC_seed0" if (L == 5 and o == "psftctm") else f"clevr3_v2_{L}tok_{o}_{i}_holdcs_dev_seed0")
       for L in (1, 5) for o in ("psft", "psftctm", "psftctmsm") for i in ("neo", "pt")}
rows, best = [], {}
for (L, o, i), r in sorted(RUN.items()):
    p = os.path.join(ROOT, "runs", r, "eval", "clevr3_dev", "flip_step_800_v2ref.json")
    if not os.path.exists(p):
        rows.append({"L": L, "objective": o, "interface": i, "run": r, "missing": True}); continue
    d = json.load(open(p)); m = {"img_F": d["look_flip"], "img_S": d["look_stay_c"], "cap_F": d["read_flip"], "cap_S": d["read_stay_c"]}
    j4 = min(m.values()); rows.append({"L": L, "objective": o, "interface": i, "run": r, **m, "JOINT4": j4, "n": d.get("n")})
    if i not in best or j4 > best[i][0] or (j4 == best[i][0] and L < best[i][1]):
        best[i] = (j4, L, o)
f3 = lambda x: "--" if x is None else f"{x:.3f}"
out = ["# Development selection: capacity (L=1 vs L=5) under a pre-registered criterion", "", __doc__.strip(), "",
       "| interface | objective | L | img F | img S | cap F | cap S | JOINT4 |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    out.append(f"| {r['interface']} | {r['objective']} | {r['L']} | " + (" MISSING | | | | |" if r.get("missing") else f"{f3(r['img_F'])} | {f3(r['img_S'])} | {f3(r['cap_F'])} | {f3(r['cap_S'])} | **{f3(r['JOINT4'])}** |"))
_missing = sum(1 for r in rows if r.get("missing"))
out += ["", ("Selected (JOINT4, ties -> L=1): " + "; ".join(f"{i}: L={b[1]}, {b[2]} (JOINT4 {b[0]:.3f})" for i, b in sorted(best.items()))) if (best and _missing == 0) else f"Selected: NOT YET -- {_missing} of {len(rows)} configurations missing; the selection is read only when all 12 are in",
        f"Missing: {sum(1 for r in rows if r.get('missing'))} of {len(rows)}"]
json.dump({"criterion": __doc__.strip(), "rows": rows, "complete": _missing == 0, "selected": ({i: {"JOINT4": b[0], "L": b[1], "objective": b[2]} for i, b in best.items()} if _missing == 0 else None)},
          open(os.path.join(ROOT, "results", "dev_selection_capacity.json"), "w"), indent=1)
open(os.path.join(ROOT, "results", "dev_selection_capacity.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
