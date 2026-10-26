#!/usr/bin/env python
"""GPU-class calibration (2026-10-22): the same checkpoint evaluated on L40S (tag T) and on A100 (tag T_a100), compared
item by item. Reports pick agreement over every (request, item, context) record and the difference in Accuracy / Flip /
Stay. Decides whether evaluations from the two lanes may be paired. Usage: python scripts/gpu_class_calibration.py"""
import glob, gzip, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def recs(p):
    out = {}
    for l in gzip.open(p, "rt"):
        r = json.loads(l)
        if "_meta" in r: continue
        out[(r["request"], r["id"], r["ctx"])] = r
    return out
rows, tot_n, tot_agree = [], 0, 0
for pa in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "eval", "clevr3_test_*", "flip_step_800_v2ref*_a100_items.jsonl.gz"))):
    pl = pa.replace("_a100_items", "_items")
    if not os.path.exists(pl): continue
    A, L = recs(pa), recs(pl); keys = sorted(set(A) & set(L))
    agree = sum(A[k]["pick"] == L[k]["pick"] for k in keys)
    sa = json.load(open(pa.replace("_items.jsonl.gz", ".json"))); sl = json.load(open(pl.replace("_items.jsonl.gz", ".json")))
    d = {m: round(sa[m] - sl[m], 4) for m in ("look_acc", "look_flip", "look_stay_c", "read_acc", "read_flip", "read_stay_c") if m in sa and m in sl}
    rows.append({"run": pa.split("/runs/")[1].split("/")[0], "split": pa.split("/eval/")[1].split("/")[0], "n_records": len(keys),
                 "pick_agreement": agree / max(len(keys), 1), "metric_diff_a100_minus_l40s": d})
    tot_n += len(keys); tot_agree += agree
for r in rows:
    print(f"{r['run'][:48]:48s} {r['split']:15s} n {r['n_records']:5d}  agree {r['pick_agreement']:.4f}  max|Δmetric| {max(abs(v) for v in r['metric_diff_a100_minus_l40s'].values()):.4f}")
print(f"OVERALL: {tot_agree}/{tot_n} picks agree ({tot_agree / max(tot_n, 1):.4f}) over {len(rows)} evaluations")
json.dump({"rows": rows, "overall_agreement": tot_agree / max(tot_n, 1), "n_records": tot_n}, open(os.path.join(ROOT, "results", "gpu_class_calibration_20261022.json"), "w"), indent=1)
