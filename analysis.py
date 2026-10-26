#!/usr/bin/env python
"""Aggregate runs/*/eval/step_*.json into a tidy CSV + figures (mean±std over
seeds). The signature figures: collapse-rate vs step per method, and
verbalization plug-in fidelity vs step."""
import os, json, glob, argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scsrc import paths


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def collect():
    rows = []
    for f in glob.glob(os.path.join(paths.RUNS, "*", "eval", "step_*.json")):
        d = json.load(open(f))
        meta = d.get("meta", {})
        flat = flatten({k: v for k, v in d.items() if k not in ("meta", "verbalization_text")})
        flat.update({"run": os.path.basename(os.path.dirname(os.path.dirname(f))),
                     "method": meta.get("method"), "arm": meta.get("arm"),
                     "n_tokens": meta.get("n_tokens"), "seed": meta.get("seed"),
                     "obj_tag": meta.get("obj_tag"),
                     "step": d.get("step")})
        rows.append(flat)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="generation.gen_collapse")
    args = ap.parse_args()
    df = collect()
    outdir = os.path.join(paths.RUNS, "_analysis"); os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "results.csv"), index=False)
    print(f"[analysis] {len(df)} rows -> {outdir}/results.csv")
    if df.empty:
        return
    # example plot: a metric vs step, mean±std over seeds, per (method,arm,n_tokens)
    for metric in [args.metric, "controllability.controllability_delta_img",
                   "plugin_look.plugin_fidelity", "swaps.img_swap_look_flip"]:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(7, 4))
        for key, g in df.groupby(["method", "arm", "n_tokens", "obj_tag"], dropna=False):
            gg = g.groupby("step")[metric].agg(["mean", "std"]).reset_index()
            plt.errorbar(gg["step"], gg["mean"], yerr=gg["std"].fillna(0),
                         marker="o", capsize=3, label=str(key))
        plt.xlabel("step"); plt.ylabel(metric); plt.legend(fontsize=7)
        plt.title(metric); plt.tight_layout()
        p = os.path.join(outdir, f"{metric.replace('.', '_')}.png")
        plt.savefig(p, dpi=130); plt.close()
        print("  saved", p)


if __name__ == "__main__":
    main()
