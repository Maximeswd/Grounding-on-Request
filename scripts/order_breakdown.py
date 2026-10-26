#!/usr/bin/env python
"""Evidence-order breakdown (2026-10-21): Flip and Stay for both requests split by whether the caption or the image came
first in the prompt (the block order is balanced per item). Reads the per-item dumps of the reference condition.
Usage: python scripts/order_breakdown.py [--out results/order_effects_20261021]"""
import argparse, collections, gzip, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [("word psft L5", "clevr3_v2_5tok_psft_neo_holdcs_seed0", "_v2ref"), ("prefix psft L5", "clevr3_v2_5tok_psft_pt_holdcs_seed0", "_v2ref"),
        ("word ctm L5", "clevr3_v2_5tok_psftctm_neo_holdcs_seed0", "_v2ref"), ("prefix ctm L5", "clevr3_v2_5tok_psftctm_pt_holdcs_seed0", "_v2ref"),
        ("word ctmsm L5", "clevr3_v2_5tok_psftctmsm_neo_holdcs_seed0", "_v2ref"), ("prefix ctmsm L5", "clevr3_v2_5tok_psftctmsm_pt_holdcs_seed0", "_v2ref"),
        ("LoRA psft", "clevr3_v2_lora_r16_joint_psft_holdcs_seed0", "_v2ref"), ("LoRA ctmsm", "clevr3_v2_lora_r16_joint_psftctmsm_holdcs_seed0", "_v2ref"),
        ("English (frozen)", "clevr3_v2_5tok_psftctm_neo_holdcs_seed0", "_v2ref_instructed"), ("no request (frozen)", "clevr3_v2_5tok_psftctm_neo_holdcs_seed0", "_v2ref_noop"),
        ("word psft L1", "clevr3_v2_1tok_psft_neo_holdcs_seed0", "_v2ref"), ("prefix psft L1", "clevr3_v2_1tok_psft_pt_holdcs_seed0", "_v2ref"),
        ("word ctmsm L1", "clevr3_v2_1tok_psftctmsm_neo_holdcs_seed0", "_v2ref"), ("prefix ctmsm L1", "clevr3_v2_1tok_psftctmsm_pt_holdcs_seed0", "_v2ref"),
        ("Qwen-4B word psft L5", "clevr3_v2_5tok_psft_neo_holdcs_seed0_qwen35", "_v2ref"), ("Qwen-4B prefix psft L5", "clevr3_v2_5tok_psft_pt_holdcs_seed0_qwen35", "_v2ref")]
def by_order(run, split, tag):
    p = os.path.join(ROOT, "runs", run, "eval", f"clevr3_{split}", f"flip_step_800{tag}_items.jsonl.gz")
    if not os.path.exists(p): return None
    by, capf = {}, {}
    for l in gzip.open(p, "rt"):
        r = json.loads(l)
        if "_meta" in r: continue
        by.setdefault((r["request"], r["id"]), {})[r["ctx"]] = r; capf[r["id"]] = (str(r["cap_first"]) == "True")
    agg = collections.defaultdict(list)
    for (rq, iid), ctxs in by.items():
        ok = {c: (r["pick"] == r["values"][rq]) for c, r in ctxs.items()}
        fc, sc = ("twin", "capedit") if rq == "look" else ("capedit", "twin")
        o = "cap_first" if capf[iid] else "img_first"
        if fc in ok: agg[(rq, "F", o)].append(int(ok["base"] and ok[fc]))
        if sc in ok: agg[(rq, "S", o)].append(int(ok["base"] and ok[sc]))
    return {f"{rq}_{m}_{o}": (sum(v) / len(v) if v else None) for (rq, m, o), v in agg.items()}, {f"n_{o}": len(agg[("look", "F", o)]) for o in ("cap_first", "img_first")}
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "results", "order_effects_20261021")); a = ap.parse_args()
f3 = lambda x: "--" if x is None else f"{x:.3f}"
bank, lines = {}, ["# Evidence-order breakdown (Ground-CF, reference condition, seed 0)", "", __doc__.strip(), "",
                   "Cell = image Flip / image Stay | caption Flip / caption Stay. cap_first = Caption block before the Image block.", ""]
for split in ("test_id", "test_ood"):
    lines += [f"## {split}", "", "| arm | caption first | image first | Δ image Flip (img−cap) | Δ caption Flip (img−cap) |", "|---|---|---|---|---|"]
    for name, run, tag in RUNS:
        r = by_order(run, split, tag)
        if r is None: lines.append(f"| {name} | missing | | | |"); continue
        m, n = r; bank[f"{split}|{name}"] = {**m, **n}
        g = lambda k: m.get(k)
        c = f"{f3(g('look_F_cap_first'))} / {f3(g('look_S_cap_first'))} \\| {f3(g('read_F_cap_first'))} / {f3(g('read_S_cap_first'))}"
        i = f"{f3(g('look_F_img_first'))} / {f3(g('look_S_img_first'))} \\| {f3(g('read_F_img_first'))} / {f3(g('read_S_img_first'))}"
        dl = (g('look_F_img_first') or 0) - (g('look_F_cap_first') or 0); dr = (g('read_F_img_first') or 0) - (g('read_F_cap_first') or 0)
        lines.append(f"| {name} | {c} | {i} | {dl:+.3f} | {dr:+.3f} |")
    lines.append("")
json.dump(bank, open(a.out + ".json", "w"), indent=1); open(a.out + ".md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
