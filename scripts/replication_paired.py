#!/usr/bin/env python
"""Seed-replicated paired comparisons of the fixed recipes (2026-10-22). For each comparison B - A, model, split and
condition: per-seed difference on identical items with an exact McNemar test, the mean and sd of the per-seed
differences, the number of seeds in which B > A, and a pooled difference with a scene-clustered bootstrap 95% interval
(all seeds' items of a scene resampled together). The bootstrap interval reflects ITEM sampling only; training-seed
variability is carried by the seed-level t interval over the per-seed differences (df = seeds - 1), which is the one to
use for claims about the recipe.
Run through Slurm (cpu partition); the bootstrap is vectorised with numpy.
Usage: python scripts/replication_paired.py [--B 2000] [--out results/replication_paired_20261022]"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pilot_v2_paired import load, bits, mcnemar_p

MODELS = [("Gemma-3-4B", ""), ("Qwen3.5-4B", "_qwen35"), ("Qwen3.5-2B", "_qwen2b"), ("Gemma-4-E4B", "_gemma4e4b")]
PT_PSFT, PT_SM = "clevr3_v2_1tok_psft_pt_holdcs_seed{s}{m}", "clevr3_v2_1tok_psftctmsm_pt_holdcs_seed{s}{m}"
W1_PSFT, W1_SM = "clevr3_v2_1tok_psft_neo_holdcs_lr0p005_seed{s}{m}", "clevr3_v2_1tok_psftctmsm_neo_holdcs_lr0p005_seed{s}{m}"
W5_SM = "clevr3_v2_5tok_psftctmsm_neo_holdcs_lr0p005_seed{s}{m}"
COMPARISONS = [
    ("CTM+Stay vs paired SFT, prompt tuning L=1", PT_PSFT, PT_SM),
    ("CTM+Stay vs paired SFT, word L=1", W1_PSFT, W1_SM),
    ("word vs prompt tuning, paired SFT, L=1", PT_PSFT, W1_PSFT),
    ("word vs prompt tuning, CTM+Stay, L=1", PT_SM, W1_SM),
    ("word L=5 vs prompt tuning L=1, CTM+Stay (selected configurations)", PT_SM, W5_SM),
    ("word L=5 vs word L=1, CTM+Stay", W1_SM, W5_SM)]
CELLS = [("look", "Flip"), ("look", "Stay"), ("read", "Flip"), ("read", "Stay")]
CONDS = [("reference", "_v2ref"), ("both shifts", "_v2")]

def safe_load(run, split, tag):
    try:
        return load(os.path.join("runs", run), split, tag)
    except (FileNotFoundError, OSError):
        return None

def boot(scene_idx, d, B, rng):
    """d: per-pair difference (b - a); scene_idx: cluster id per pair. Returns 95% interval of the pooled mean."""
    keys, inv = np.unique(scene_idx, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(keys)); cnts = np.bincount(inv, minlength=len(keys)).astype(float)
    draw = rng.integers(0, len(keys), size=(B, len(keys)))
    est = sums[draw].sum(1) / cnts[draw].sum(1)
    return float(np.quantile(est, .025)), float(np.quantile(est, .975))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "replication_paired_20261022")); a = ap.parse_args()
    rng = np.random.default_rng(0); rows = []
    for cname, A, Bt in COMPARISONS:
        for mname, msfx in MODELS:
            for split in ("test_id", "test_ood"):
                for cond, tag in CONDS:
                    per_seed = {}
                    for s in (0, 1, 2):
                        ra, rb = A.format(s=s, m=msfx), Bt.format(s=s, m=msfx)
                        ba, bb = safe_load(ra, split, tag), safe_load(rb, split, tag)
                        if ba is None or bb is None: continue
                        per_seed[s] = (bits(ba), bits(bb))
                    for rq, met in CELLS:
                        row = {"comparison": cname, "model": mname, "split": split, "condition": cond, "request": rq, "metric": met,
                               "seeds": sorted(per_seed), "per_seed": {}}
                        sc, dd = [], []
                        for s, (xa, xb) in per_seed.items():
                            A_, B_ = xa.get((rq, met), {}), xb.get((rq, met), {})
                            ids = sorted(set(A_) & set(B_))
                            if not ids: continue
                            va = np.array([A_[i] for i in ids]); vb = np.array([B_[i] for i in ids])
                            b_only = int(((vb == 1) & (va == 0)).sum()); a_only = int(((va == 1) & (vb == 0)).sum())
                            row["per_seed"][s] = {"n": len(ids), "mean_A": float(va.mean()), "mean_B": float(vb.mean()),
                                                  "diff": float(vb.mean() - va.mean()), "B_only": b_only, "A_only": a_only, "p": mcnemar_p(b_only, a_only)}
                            sc += [i.rsplit("_", 1)[0] for i in ids]; dd += list(vb - va)
                        if not row["per_seed"]:
                            row["missing"] = True; rows.append(row); continue
                        diffs = [v["diff"] for v in row["per_seed"].values()]
                        row.update({"n_seeds": len(diffs), "mean_diff": float(np.mean(diffs)),
                                    "sd_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
                                    "seeds_B_better": int(sum(x > 0 for x in diffs)), "seeds_sig_B_better": int(sum(v["diff"] > 0 and v["p"] < .05 for v in row["per_seed"].values())),
                                    "seeds_sig_A_better": int(sum(v["diff"] < 0 and v["p"] < .05 for v in row["per_seed"].values())),
                                    "mean_A": float(np.mean([v["mean_A"] for v in row["per_seed"].values()])),
                                    "mean_B": float(np.mean([v["mean_B"] for v in row["per_seed"].values()]))})
                        row["ci95_scene_pooled"] = boot(np.array(sc), np.array(dd, dtype=float), a.B, rng)
                        T = {2: 12.706, 3: 4.303}
                        if len(diffs) in T:
                            h = T[len(diffs)] * row["sd_diff"] / np.sqrt(len(diffs))
                            row["ci95_seed_t"] = (row["mean_diff"] - h, row["mean_diff"] + h)
                        else:
                            row["ci95_seed_t"] = None
                        rows.append(row)
    json.dump(rows, open(a.out + ".json", "w"), indent=1)
    f3 = lambda x: "--" if x is None else f"{x:+.3f}"
    L = ["# Seed-replicated paired comparisons (Ground-CF, protocol v2, task holdout)", "", __doc__.strip(), "",
         "diff = B - A (mean of per-seed differences ± sd); seed CI = t interval over seeds (training variability); item CI = scene-clustered bootstrap (item sampling only); "
         "sig = seeds with exact McNemar p < .05 in favour of B / of A.", ""]
    for cname, _, _ in COMPARISONS:
        L += [f"## {cname}", "", "| model | split | condition | request | metric | A | B | diff ± sd | seed 95% CI | item 95% CI | B>A seeds | sig B / sig A |", "|" + "---|" * 12]
        for r in rows:
            if r["comparison"] != cname: continue
            if r.get("missing"):
                L.append(f"| {r['model']} | {r['split']} | {r['condition']} | {r['request']} | {r['metric']} | pending | | | | | | |"); continue
            ci = r["ci95_scene_pooled"]
            L.append(f"| {r['model']} | {r['split']} | {r['condition']} | {r['request']} | {r['metric']} | {r['mean_A']:.3f} | {r['mean_B']:.3f} | "
                     f"{f3(r['mean_diff'])}" + (f" ± {r['sd_diff']:.3f}" if r['sd_diff'] is not None else "") +
                     (f" | [{r['ci95_seed_t'][0]:+.3f}, {r['ci95_seed_t'][1]:+.3f}]" if r.get("ci95_seed_t") else " | --") +
                     f" | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {r['seeds_B_better']}/{r['n_seeds']} | {r['seeds_sig_B_better']} / {r['seeds_sig_A_better']} |")
        L.append("")
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print(f"[saved] {a.out}.md / .json ({len(rows)} rows, {sum(1 for r in rows if r.get('missing'))} pending)")

if __name__ == "__main__":
    main()
