#!/usr/bin/env python
"""CORRECT MINUS INCORRECT ATTENTION, PER LAYER (2026-09-17).

Consumes the per-item arrays from scripts/attention_probe_v2.py and answers one question per panel:
at a given position in the answer, does a correct answer put more attention on the requested evidence
than an incorrect one, and where in the stack does that difference appear?

For each (dataset, model, interface, request, position) it plots
    mean_over_correct_items(channel) - mean_over_incorrect_items(channel)
against layer index, with a bootstrap band, for
    MASS       attention on the evidence block          (imgbox, img, cap)
    PRECISION  share of that block's mass inside the    (imgbox/img, capval/cap)
               annotated evidence
A cell with fewer than --min_n items on either side is DROPPED and named in the output, because a
difference of means from three failures is not an estimate. This is a correlational readout of a
frozen forward pass: it localises where correct and incorrect runs diverge, it does not show that the
attention caused the answer.

Writes figs/fig_attn_cmi_<ds>_<pos>.{pdf,png} and results/attn_cmi_<ds>.json.
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "paper", "iclr2027", "figs")
INK, LINE, CARD = "#15181C", "#D9DDE2", "#F7F8FA"
MOD = [("", "Gemma-3-4B", "#2F5FC4"), ("_qwen35", "Qwen-3.5-4B", "#D08A2E"),
       ("_qwen2b", "Qwen-3.5-2B", "#5FA46A"), ("_gemma4e4b", "Gemma-4-E4B", "#9B59B6")]
# interface -> (run pattern with {s}, probe mode); the frozen dirs hold two modes in one run_dir
IFACE = [("Base model", "_frozen{fz}", "noop"), ("English instruction", "_frozen{fz}", "instructed"),
         ("Prompt tuning + CTM", "{pt}", "srconly"), ("Neologism + CTM", "{neo}", "srconly"),
         ("LoRA r4", "{lora}", "srconly")]
RUNS = {"clevr3": {"pt": "clevr3_f_pt_5tok_ctm_seed0{s}", "neo": "clevr3_f_5tok_ctm_seed0{s}",
                   "lora": "clevr3_f_lora_r4_seed0{s}"},
        "mvg": {"pt": "mvgV2_5tok_ptctm_fix_seed0{s}", "neo": "mvgV2_5tok_sftctm_hard_seed0{s}",
                "lora": "mvgV2_lora_r4_seed0{s}"}}
FZ = {"": "_gemma", "_qwen35": "_qwen35", "_qwen2b": "_qwen2b", "_gemma4e4b": "_gemma4e4b"}
# (label, numerator channel, denominator channel or None for raw mass)
MEASURES = [("image mass", "img", None), ("evidence-box mass", "imgbox", None),
            ("evidence-box precision", "imgbox", "img"),
            ("caption mass", "cap", None), ("caption-value precision", "capval", "cap")]


def load(ds, split, run, mode, tag=""):
    d = os.path.join(ROOT, "runs", run, "attention2", f"{ds}_{split}_v2_{mode}{tag}_step800")
    sm = os.path.join(d, "summary.json")
    if not os.path.exists(sm):
        return None
    s = json.load(open(sm))
    out = {"summary": s, "dir": d, "ci": {}}
    for f in glob.glob(os.path.join(d, "*.npz")):
        b = os.path.basename(f)[:-4]                     # correct_look / incorrect_read
        ok, op = b.split("_", 1)
        out["ci"][(op, ok)] = np.load(f)["a"]            # items x L x H x CH
    return out


def channel_curve(arr, ch_index, num, den, boot=200, rng=None):
    """items x L x H x CH -> per-layer value per item (heads averaged), then mean over items."""
    n = arr[:, :, :, ch_index[num]].astype(np.float32)
    v = np.nanmean(n, axis=2)                            # items x L
    if den is not None:
        d = np.nanmean(arr[:, :, :, ch_index[den]].astype(np.float32), axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(d > 1e-6, v / d, np.nan)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="clevr3", choices=["clevr3", "mvg"])
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--positions", default="first,preval,final,ans")
    ap.add_argument("--min_n", type=int, default=20,
                    help="a side with fewer items than this is dropped, not plotted")
    ap.add_argument("--tag", default="")
    ap.add_argument("--merge_splits", default="",
                    help="comma-separated splits pooled into one cell, e.g. test_id,test_ood. The "
                         "learned arms fail on only a handful of Ground-CF items per split, so "
                         "pooling is what makes several of those cells estimable at all.")
    a = ap.parse_args()
    positions = [p for p in a.positions.split(",") if p]
    splits = [x for x in a.merge_splits.split(",") if x] or [a.split]
    tagsp = "+".join(x.replace("test_", "") for x in splits)
    bank = {"ds": a.ds, "splits": splits, "min_n": a.min_n, "cells": {}, "dropped": []}
    cells = {}
    for s, ml, col in MOD:
        for lab, pat, mode in IFACE:
            run = pat.format(fz=FZ[s], **{k: v.format(s=s) for k, v in RUNS[a.ds].items()})
            parts = [x for x in (load(a.ds, sp, run, mode, a.tag) for sp in splits) if x]
            if not parts:
                bank["dropped"].append({"model": ml, "interface": lab, "why": "no probe output",
                                        "run": run, "splits": splits})
                continue
            # pool the splits: identical channel layout, so the item axis simply concatenates
            g = {"summary": parts[0]["summary"], "dir": [x["dir"] for x in parts], "ci": {}}
            for key in {k for x in parts for k in x["ci"]}:
                arrs = [x["ci"][key] for x in parts if key in x["ci"]]
                g["ci"][key] = np.concatenate(arrs, axis=0) if arrs else None
            g["ci"] = {k: v for k, v in g["ci"].items() if v is not None}
            ch_index = {c: i for i, c in enumerate(g["summary"]["channels"])}
            ops = sorted({op for op, _ in g["ci"]})
            for op in ops:
                c, i = g["ci"].get((op, "correct")), g["ci"].get((op, "incorrect"))
                nc, ni = (0 if c is None else len(c)), (0 if i is None else len(i))
                key = f"{ml}|{lab}|{op}"
                bank["cells"][key] = {"n_correct": nc, "n_incorrect": ni}
                if nc < a.min_n or ni < a.min_n:
                    bank["dropped"].append({"model": ml, "interface": lab, "op": op,
                                            "n_correct": nc, "n_incorrect": ni,
                                            "why": f"a side is below min_n={a.min_n}"})
                    continue
                for pos in positions:
                    for mlab, num, den in MEASURES:
                        kn, kd = f"{pos}_{num}", (f"{pos}_{den}" if den else None)
                        if kn not in ch_index or (kd and kd not in ch_index):
                            continue
                        vc = channel_curve(c, ch_index, kn, kd)
                        vi = channel_curve(i, ch_index, kn, kd)
                        d = np.nanmean(vc, 0) - np.nanmean(vi, 0)
                        se = np.sqrt(np.nanvar(vc, 0) / max(nc, 1) + np.nanvar(vi, 0) / max(ni, 1))
                        cells[(s, ml, col, lab, op, pos, mlab)] = (d, se)
                        bank["cells"][key][f"{pos}|{mlab}|max_abs_diff"] = round(float(np.nanmax(np.abs(d))), 4)
                        bank["cells"][key][f"{pos}|{mlab}|argmax_layer"] = int(np.nanargmax(np.abs(d))) if np.isfinite(d).any() else -1
    os.makedirs(FIGS, exist_ok=True)
    for pos in positions:
        ops = sorted({k[4] for k in cells if k[5] == pos})
        meas = [m for _, m, _ in [(0, l, 0) for l, _, _ in MEASURES]]
        meas = [l for l, _, _ in MEASURES if any(k[6] == l for k in cells if k[5] == pos)]
        if not ops or not meas:
            continue
        fig, axes = plt.subplots(len(ops), len(meas), figsize=(2.0 * len(meas) + .8, 2.1 * len(ops) + .9),
                                 squeeze=False)
        fig.subplots_adjust(left=.11, right=.99, top=.88, bottom=.16, wspace=.28, hspace=.30)
        for r, op in enumerate(ops):
            for c, mlab in enumerate(meas):
                ax = axes[r][c]; ax.set_facecolor(CARD)
                ax.grid(True, color="white", lw=.9); ax.set_axisbelow(True)
                ax.axhline(0, color="#8A929B", lw=.8)
                for (s, ml, col, lab, o, p, m), (d, se) in cells.items():
                    if (o, p, m) != (op, pos, mlab):
                        continue
                    x = np.arange(len(d))
                    ls = {"Neologism + CTM": "-", "Prompt tuning + CTM": "--", "LoRA r4": ":",
                          "English instruction": (0, (3, 1, 1, 1)), "Base model": (0, (1, 1))}[lab]
                    lw = 1.8 if lab == "Neologism + CTM" else 1.1
                    ax.plot(x, d, color=col, ls=ls, lw=lw)
                    ax.fill_between(x, d - se, d + se, color=col, alpha=.10, lw=0)
                for sp in ax.spines.values(): sp.set_color(LINE)
                ax.tick_params(labelsize=6.5, colors="#5B6470", length=2)
                if r == 0: ax.set_title(mlab, fontsize=7.5, color=INK, pad=3)
                if c == 0: ax.set_ylabel(f"$\\langle${op}$\\rangle$\ncorrect $-$ incorrect", fontsize=7, color=INK)
                if r == len(ops) - 1: ax.set_xlabel("layer", fontsize=7, color=INK)
        h = [Line2D([], [], color=col, lw=1.3, label=ml) for _, ml, col in MOD]
        h += [Line2D([], [], color="#5B6470", lw=1.3,
                     ls={"Neologism + CTM": "-", "Prompt tuning + CTM": "--", "LoRA r4": ":",
                         "English instruction": (0, (3, 1, 1, 1)), "Base model": (0, (1, 1))}[l], label=l)
              for l, _, _ in IFACE]
        fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6.2, frameon=False, bbox_to_anchor=(.55, -.005))
        fig.suptitle(f"{a.ds} {'+'.join(splits)}: attention at \"{pos}\", correct minus incorrect",
                     fontsize=8.5, color=INK, y=.975)
        base = f"{FIGS}/fig_attn_cmi_{a.ds}_{tagsp}_{pos}"
        fig.savefig(base + ".pdf"); fig.savefig(base + ".png", dpi=400); plt.close(fig)
        print(f"  wrote {os.path.basename(base)}.{{pdf,png}}")
    json.dump(bank, open(f"{ROOT}/results/attn_cmi_{a.ds}_{tagsp}.json", "w"), indent=1)
    kept = sum(1 for v in bank["cells"].values() if v["n_correct"] >= a.min_n and v["n_incorrect"] >= a.min_n)
    print(f"  cells with both sides >= {a.min_n}: {kept} of {len(bank['cells'])}; "
          f"{len(bank['dropped'])} dropped (named in results/attn_cmi_{a.ds}_*.json)")


if __name__ == "__main__":
    main()
