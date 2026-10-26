#!/usr/bin/env python
"""Replication summary of the FIXED recipes (2026-10-22): Ground-CF, protocol v2, task holdout, seeds 0-2 x four models.
Recipes (selected on the development partition, STATUS v1.307/v1.308): prompt tuning = L=1, lr .05, clamp 2x; word =
lr .005, clamp 2x, at L=1 and L=5. Objectives: paired SFT and +CTM+Stay (the L=5 word: +CTM+Stay only).
Per cell: mean and sample standard deviation over the seeds present, for Accuracy / Flip / Stay of both requests, in the
reference condition (training wording) and under both shifts (held-out question and carrier). Reads only the summary
JSON of each evaluation; missing seeds are counted, never imputed.
Usage: python scripts/replication_summary.py [--out results/replication_20261022]"""
import argparse, json, os, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [("Gemma-3-4B", ""), ("Qwen3.5-4B", "_qwen35"), ("Qwen3.5-2B", "_qwen2b"), ("Gemma-4-E4B", "_gemma4e4b")]
ARMS = [("Prompt tuning L=1", "paired SFT", "clevr3_v2_1tok_psft_pt_holdcs_seed{s}{m}"),
        ("Prompt tuning L=1", "+CTM+Stay", "clevr3_v2_1tok_psftctmsm_pt_holdcs_seed{s}{m}"),
        ("Word L=1", "paired SFT", "clevr3_v2_1tok_psft_neo_holdcs_lr0p005_seed{s}{m}"),
        ("Word L=1", "+CTM+Stay", "clevr3_v2_1tok_psftctmsm_neo_holdcs_lr0p005_seed{s}{m}"),
        ("Word L=5", "+CTM+Stay", "clevr3_v2_5tok_psftctmsm_neo_holdcs_lr0p005_seed{s}{m}")]
METRICS = [("img_A", "look_acc"), ("img_F", "look_flip"), ("img_S", "look_stay_c"), ("cap_A", "read_acc"), ("cap_F", "read_flip"), ("cap_S", "read_stay_c")]
CONDS = [("reference", "_v2ref"), ("both shifts", "_v2")]

def load(run, split, tag):
    p = os.path.join(ROOT, "runs", run, "eval", f"clevr3_{split}", f"flip_step_800{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def cell(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None, None, 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else None), len(vals)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "results", "replication_20261022"))
    a = ap.parse_args()
    bank, lines = [], ["# Replication of the fixed recipes (Ground-CF, protocol v2, task holdout)", "", __doc__.strip(), "",
                       "Cell = mean ± sd over seeds (n). A = Accuracy, F = Flip, S = Stay. Chance: A .33, F .11 (three candidates).", ""]
    fmt = lambda m, s, n: "pending" if n == 0 else (f"{m:.3f}" + (f" ± {s:.3f}" if s is not None else "") + (f" ({n})" if n != 3 else ""))
    for split in ("test_id", "test_ood"):
        for cname, tag in CONDS:
            lines += [f"## {split}, {cname}", "", "| model | interface | objective | img A | img F | img S | cap A | cap F | cap S |", "|" + "---|" * 9]
            for mname, msfx in MODELS:
                for iface, obj, tpl in ARMS:
                    runs = [tpl.format(s=s, m=msfx) for s in (0, 1, 2)]
                    ds = [load(r, split, tag) for r in runs]
                    row = {"split": split, "condition": cname, "model": mname, "interface": iface, "objective": obj,
                           "runs": runs, "seeds_present": [i for i, d in enumerate(ds) if d is not None]}
                    cells = []
                    for key, src in METRICS:
                        m, s, n = cell([d.get(src) if d else None for d in ds])
                        row[key] = {"mean": m, "sd": s, "n": n, "per_seed": [d.get(src) if d else None for d in ds]}
                        cells.append(fmt(m, s, n))
                    bank.append(row)
                    lines.append(f"| {mname} | {iface} | {obj} | " + " | ".join(cells) + " |")
            lines.append("")
    json.dump(bank, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write("\n".join(lines) + "\n")
    print(f"[saved] {a.out}.md / .json ({len(bank)} rows)")

if __name__ == "__main__":
    main()
