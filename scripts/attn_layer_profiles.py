#!/usr/bin/env python
"""PER-LAYER ATTENTION PROFILES, CONDITIONED ON OUTCOME (2026-09-17).

The question, put properly: at which layers, and at which decoding position, does each interface send
attention to the REQUESTED source -- and does that differ between answers that were right and answers
that were wrong? The cell-level Spearman in attn_layer_compare.py cannot answer it: it correlates one
summary number per cell and discards the per-item, per-layer structure. This script keeps it.

For one model x benchmark it loads every interface's attention2 extraction (per item, per layer, per
head, per channel; correct and incorrect items already split by the probe) and plots, layer by layer:
  MASS       <look>: attention to the image span        <read>: attention to the caption span
  PRECISION  <read>: attention to the caption's answer VALUE / attention to the caption span
             (<look> precision needs the evidence-box channel: attention_probe_gcf.py, pending)
at two decoding positions: `first` (the first answer token -- start of generation) and `preval` (the
token emitting the value -- the commitment point). Solid = items the interface answered correctly,
dashed = incorrectly. Mean over heads, then over items.

Usage: python scripts/attn_layer_profiles.py --model gemma3 --ds color
Writes figs/fig_attn_layers_<model>_<ds>.{pdf,png} and results/attn_layer_profiles_<model>_<ds>.json
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFX = {"gemma3": "", "qwen4b": "_qwen35", "qwen2b": "_qwen2b", "e4b": "_gemma4e4b"}
# label, run-name template, colour
IFACES = [("English instruction", None, "#8A929B"),
          ("Neologism SFT", "{ds}_neo_ctrl_5tok_sft_seed0{s}", "#8FB0EE"),
          ("Neologism + APO", "{ds}_neo_ctrl_5tok_apo_seed0{s}", "#5B84D6"),
          ("Neologism + CTM", "{ds}_neo_ctrl_5tok_ctm_seed0{s}", "#2F5FC4"),
          ("Prompt tuning", "{ds}_pt_na_5tok_sft_seed0{s}", "#7E8AA2"),
          ("Prompt tuning + CTM", "{ds}_pt_na_5tok_ctm_seed0{s}", "#414B57"),
          ("LoRA r=4", "{ds}_lora_r4_lr2e4_seed0{s}", "#B4532F")]
INK, LINE, CARD = "#15181C", "#D9DDE2", "#F7F8FA"


def load_dir(d):
    s = json.load(open(f"{d}/summary.json")); CH = s["channels"]
    out = {}
    for op in ("look", "read", "instr_look", "instr_read"):
        for okk in ("correct", "incorrect"):
            f = f"{d}/{op}_{okk}.npz"
            if os.path.exists(f):
                out[(op, okk)] = np.load(f)["attn"]          # items x L x H x CH
    return CH, out


def curve(A, CH, num, den=None):
    """per-layer mean over items and heads of channel `num` (or num/den per item)."""
    if A is None or A.shape[0] == 0: return None
    x = A[:, :, :, CH.index(num)]
    if den:
        y = A[:, :, :, CH.index(den)]
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.where(y > 1e-9, x / y, np.nan)
    return np.nanmean(x, axis=(0, 2))                       # L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3"); ap.add_argument("--ds", default="color")
    a = ap.parse_args(); s = SFX[a.model]
    panels = {}                                             # label -> (CH, arrays)
    for lab, tpl, col in IFACES:
        if tpl is None:
            continue
        run = tpl.format(ds=a.ds, s=s)
        dirs = glob.glob(f"{ROOT}/runs/{run}/attention2/{a.ds}_test_id_step*")
        if dirs:
            panels[lab] = load_dir(sorted(dirs)[-1]) + (col,)
    # the English instruction rides in every neo dir as instr_look / instr_read; take it from the SFT run
    base = panels.get("Neologism SFT") or next(iter(panels.values()), None)
    if base:
        CHb, arrb, _ = base
        panels["English instruction"] = (CHb, {(k[0].replace("instr_", ""), k[1]): v for k, v in arrb.items() if k[0].startswith("instr_")}, "#8A929B")
    if not panels:
        raise SystemExit(f"no attention2 data for {a.model}/{a.ds}")
    print(f"[{a.model}/{a.ds}] interfaces with per-layer data: {list(panels)}")

    SPECS = [("first", "img", None, "look", "mass on image, first answer token"),
             ("preval", "img", None, "look", "mass on image, at commitment"),
             ("first", "cap", None, "read", "mass on caption, first answer token"),
             ("preval", "cap", None, "read", "mass on caption, at commitment"),
             ("first", "capval", "cap", "read", "caption precision, first answer token"),
             ("preval", "capval", "cap", "read", "caption precision, at commitment")]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0), sharex=True)
    fig.subplots_adjust(left=.085, right=.99, top=.95, bottom=.11, wspace=.14, hspace=.32)
    bank = {"model": a.model, "ds": a.ds, "curves": {}}
    for ax, (q, num, den, op, title) in zip(axes.ravel(), SPECS):
        ax.set_facecolor(CARD); ax.grid(color=LINE, lw=.7); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for lab, (CH, arr, col) in panels.items():
            for okk, ls in (("correct", "-"), ("incorrect", (0, (3, 2)))):
                A = arr.get((op, okk))
                c = curve(A, CH, f"{q}_{num}", f"{q}_{den}" if den else None)
                if c is None: continue
                ax.plot(range(len(c)), c, color=col, ls=ls, lw=1.6 if "CTM" in lab and "Neo" in lab else 1.1,
                        alpha=.95 if okk == "correct" else .75, zorder=3)
                bank["curves"][f"{lab}|{op}|{okk}|{q}_{num}" + (f"/{den}" if den else "")] = [round(float(v), 5) for v in c]
        ax.set_title(f"$\\langle${op}$\\rangle$ -- {title}", fontsize=8, color=INK, pad=3)
        ax.tick_params(labelsize=7, colors="#5B6470", length=2.5)
    for ax in axes[-1]: ax.set_xlabel("layer", fontsize=8, color=INK)
    axes[0][0].set_ylabel("attention mass", fontsize=8, color=INK); axes[1][0].set_ylabel("attention mass", fontsize=8, color=INK)
    axes[2][0].set_ylabel("share on answer value", fontsize=8, color=INK)
    h = [plt.Line2D([], [], color=c, lw=1.4, label=l) for l, (_, _, c) in panels.items()]
    h += [plt.Line2D([], [], color=INK, lw=1.2, label="correct answers"), plt.Line2D([], [], color=INK, lw=1.2, ls=(0, (3, 2)), label="incorrect answers")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(.5, -.005))
    out = f"{ROOT}/paper/iclr2027/figs/fig_attn_layers_{a.model}_{a.ds}"
    for ext in ("pdf", "png"): fig.savefig(f"{out}.{ext}", dpi=400, bbox_inches="tight", facecolor="white")
    json.dump(bank, open(f"{ROOT}/results/attn_layer_profiles_{a.model}_{a.ds}.json", "w"), indent=1)
    print(f"[saved] {out}.pdf")
    # a compact summary the text can use: late-layer (last third) mean per interface, correct items
    L = len(next(iter(bank["curves"].values())))
    print(f"\nlate-layer (last third of {L}) means, CORRECT items, at commitment (preval):")
    for lab in panels:
        r = {}
        for key, nm in (("img", "look img"), ("cap", "read cap"), ("capval/cap", "read precision")):
            op = "look" if key == "img" else "read"
            c = bank["curves"].get(f"{lab}|{op}|correct|preval_{key}")
            r[nm] = round(float(np.nanmean(c[2 * L // 3:])), 3) if c else None
        print(f"  {lab:22s} {r}")


if __name__ == "__main__":
    main()
