#!/usr/bin/env python
"""Pre-registered development selection of the TRAINING OBJECTIVE (2026-10-22, written before any of these dev results
existed). For each interface at its selected capacity (word L=5 lr .005; prompt tuning L=1 lr .05; shared LoRA r16) we
compare the eight objectives of the 2 x 2 x 2 design {twin CE} x {Flip margin (CTM)} x {Stay margin} on the development
partition (dev scenes excluded from training), Gemma-3-4B, seed 0, all evaluated on the A100 lane (tag _v2ref_a100).
Criterion: JOINT4 = min(image Flip, image Stay, caption Flip, caption Stay), reference condition.
Decision rule (status quo, because seed-to-seed spread of JOINT4 is about .02): the current recipe objective, paired SFT
+ CTM + Stay (psftctmsm), is KEPT for an interface unless another objective exceeds its JOINT4 by more than .02; if several
do, the one with the highest JOINT4 is selected. The rule is applied to each interface separately and identically.
The seeded test factorial (scripts/gen_objective_factorial.py --part test) is reported in full regardless of the decision.
Usage: python scripts/dev_select_objective.py"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJS = ["sft", "sftctm", "sftsm", "sftctmsm", "psft", "psftctm", "psftsm", "psftctmsm"]
IFACES = [("word L=5", "clevr3_v2_5tok_{o}_neo_holdcs_dev_lr0p005_seed0"), ("prompt tuning L=1", "clevr3_v2_1tok_{o}_pt_holdcs_dev_seed0"),
          ("shared LoRA r16", "clevr3_v2_lora_r16_joint_{o}_holdcs_dev_seed0")]
TAG, MARGIN, DEFAULT = "_v2ref_a100", 0.02, "psftctmsm"
f3 = lambda x: "--" if x is None else f"{x:.3f}"
rows, decision, L = [], {}, ["# Development selection: training objective", "", __doc__.strip(), "",
                              "| interface | objective | img F | img S | cap F | cap S | JOINT4 |", "|---|---|---|---|---|---|---|"]
for iname, tpl in IFACES:
    cells = {}
    for o in OBJS:
        r = tpl.format(o=o); p = os.path.join(ROOT, "runs", r, "eval", "clevr3_dev", f"flip_step_800{TAG}.json")
        if not os.path.exists(p):
            rows.append({"interface": iname, "objective": o, "run": r, "missing": True}); L.append(f"| {iname} | {o} | MISSING | | | | |"); continue
        d = json.load(open(p)); m = {"img_F": d["look_flip"], "img_S": d["look_stay_c"], "cap_F": d["read_flip"], "cap_S": d["read_stay_c"]}
        j4 = min(m.values()); cells[o] = j4
        rows.append({"interface": iname, "objective": o, "run": r, **m, "JOINT4": j4})
        L.append(f"| {iname} | {o} | {f3(m['img_F'])} | {f3(m['img_S'])} | {f3(m['cap_F'])} | {f3(m['cap_S'])} | **{f3(j4)}** |")
    if len(cells) == len(OBJS):
        base = cells[DEFAULT]; better = {o: v for o, v in cells.items() if v > base + MARGIN}
        decision[iname] = {"selected": (max(better, key=better.get) if better else DEFAULT), "default_JOINT4": base, "beating_by_more_than_margin": better}
missing = sum(1 for r in rows if r.get("missing"))
L += ["", ("Decision: " + "; ".join(f"{k}: {v['selected']}" for k, v in decision.items())) if missing == 0 else f"Decision: NOT YET -- {missing} of {len(rows)} cells missing", ""]
json.dump({"criterion": __doc__.strip(), "rows": rows, "complete": missing == 0, "decision": decision if missing == 0 else None},
          open(os.path.join(ROOT, "results", "dev_selection_objective.json"), "w"), indent=1)
open(os.path.join(ROOT, "results", "dev_selection_objective.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
