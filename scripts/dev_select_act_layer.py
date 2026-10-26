#!/usr/bin/env python
"""Pre-registered development selection of the ACTIVATION-CONTROL layer (2026-10-19, before any of these dev results
existed). Method act = neologism rows + a request-conditioned vector added at one decoder layer on the requested
evidence span (scsrc/act.py). Layers {4, 9, 17} of Gemma-3-4B's 34 (relative depth .12 / .26 / .50; the COLM paper's
handle sits in early/middle layers). Criterion: JOINT4 on the dev partition, reference condition, L=1, task holdout,
paired SFT, seed 0; ties -> the shallower layer. The matched comparison is the plain word under the same recipe
(clevr3_v2_1tok_psft_neo_holdcs_dev_seed0). The selected layer is then trained with +CTM+Stay and evaluated on test."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f3 = lambda x: "--" if x is None else f"{x:.3f}"
rows, best = [], None
cands = [("word (no delta)", "clevr3_v2_1tok_psft_neo_holdcs_dev_seed0", None)] + [(f"act layer {k}", f"clevr3_v2_1tok_psft_act{k}_holdcs_dev_seed0", k) for k in (4, 9, 17)]
for name, r, k in cands:
    p = os.path.join(ROOT, "runs", r, "eval", "clevr3_dev", "flip_step_800_v2ref.json"); dg = os.path.join(ROOT, "runs", r, "train_diag.json")
    if not os.path.exists(p):
        rows.append({"arm": name, "run": r, "layer": k, "missing": True}); continue
    d = json.load(open(p)); m = {"img_F": d["look_flip"], "img_S": d["look_stay_c"], "cap_F": d["read_flip"], "cap_S": d["read_stay_c"]}
    g = json.load(open(dg)).get("act", {}) if os.path.exists(dg) else {}
    j4 = min(m.values()); rows.append({"arm": name, "run": r, "layer": k, **m, "JOINT4": j4, "delta_norm": g.get("delta_norm"), "span_found": g.get("span_found"), "span_missed": g.get("span_missed")})
    if k is not None and (best is None or j4 > best[0]):
        best = (j4, k)
missing = sum(1 for r in rows if r.get("missing"))
out = ["# Development selection: activation-control layer (Gemma, L=1, task holdout, dev partition)", "", __doc__.strip(), "",
       "| arm | img F | img S | cap F | cap S | JOINT4 | |delta| look/read | spans found/missed |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    out.append(f"| {r['arm']} | " + (" MISSING | | | | | | |" if r.get("missing") else f"{f3(r['img_F'])} | {f3(r['img_S'])} | {f3(r['cap_F'])} | {f3(r['cap_S'])} | **{f3(r['JOINT4'])}** | {('/'.join(f3(x) for x in r['delta_norm']) if r.get('delta_norm') else '--')} | {r.get('span_found')}/{r.get('span_missed')} |"))
out += ["", (f"Selected layer (JOINT4): {best[1]} ({best[0]:.3f})" if (missing == 0 and best) else f"Selected: NOT YET -- {missing} of {len(rows)} cells missing"), f"Missing: {missing} of {len(rows)}"]
json.dump({"criterion": __doc__.strip(), "rows": rows, "complete": missing == 0, "selected_layer": (best[1] if missing == 0 and best else None)},
          open(os.path.join(ROOT, "results", "dev_selection_act_layer.json"), "w"), indent=1)
open(os.path.join(ROOT, "results", "dev_selection_act_layer.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
