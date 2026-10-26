#!/usr/bin/env python
"""ROC-scored detectors from the banked per-item likelihood margins (no model calls).

Two detectors, both built from the two modality requests alone, scored continuously so that every
interface gets a full ROC instead of one argmax operating point:

  CONFLICT.  Score s = (log-odds of the image value under the image request) minus (log-odds of the
             image value under the caption request). When each request follows its own source the two
             log-odds have opposite signs on a conflict item and s is large; on an agree context (a twin
             in which both sources attest the same value) both requests prefer that value and s is near
             zero. Positive class: the conflict context. Negative class: the two agree contexts of the
             same items (image swapped: both attest the caption's value; caption swapped: both attest
             the image's). Needs the twin-context margins written by eval_flip_ll.py since 2026-09-14
             (look_margin_imgswap/capswap, read_margin_capswap/imgswap); cells without them are skipped.
  SELF-VERIFICATION.  Score |margin| under the request that produced the answer; positive class: the
             answer is the requested source's value (grounded), negative: it is not. Conflict context;
             ladder records carry these margins since 2026-08.

Sign conventions in the records: look_margin = log p(image value) - log p(caption value) under <look>;
read_margin = log p(caption value) - log p(image value) under <read> (negated at write time); the four
twin-context margins are RAW log p(image value) - log p(caption value) under the request in the name.

Writes results/detect_roc.json and prints per-interface means. Ground-CF is not scored here: its two
requests are evaluated on different candidate pairs, so the conflict score is not defined.
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ["vcf", "color", "coco"]
MODELS = [("gemma3", ""), ("qwen4b", "_qwen35"), ("qwen2b", "_qwen2b"), ("e4b", "_gemma4e4b")]
# interface label -> (run dir pattern, file)
IF = [("No request", "{ds}_base_frozen_0tok_none_seed0{s}", "flip_ll_noop_step_800.json"),
      ("English instruction", "{ds}_base_frozen_0tok_none_seed0{s}", "flip_ll_iv5_step_800.json"),
      ("Prompt tuning", "{ds}_pt_na_5tok_sft_seed0{s}", "flip_ll_step_800.json"),
      ("Prompt tuning + CTM", "{ds}_pt_na_5tok_ctm_seed0{s}", "flip_ll_step_800.json"),
      ("LoRA r=4", "{ds}_lora_r4_lr2e4_seed0{s}", "flip_ll_step_800.json"),
      ("LoRA r=16", "{ds}_lora_r16_lr2e4_seed0{s}", "flip_ll_step_800.json"),
      ("Neologisms, SFT", "{ds}_neo_ctrl_5tok_sft_seed0{s}", "flip_ll_step_800.json"),
      ("Neologisms + APO", "{ds}_neo_ctrl_5tok_apo_seed0{s}", "flip_ll_step_800.json"),
      ("Neologisms + CTM", "{ds}_neo_ctrl_5tok_ctm_seed0{s}", "flip_ll_step_800.json")]


def auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, bool)
    if label.all() or (~label).all() or len(label) < 4:
        return None
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    order = np.argsort(score, kind="mergesort"); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks); ranks = sums[inv] / cnt[inv]   # tie-averaged
    n1 = label.sum(); n0 = len(label) - n1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def records(path):
    if not os.path.exists(path):
        return None
    r = json.load(open(path)).get("records")
    return r if isinstance(r, dict) and "look_margin" in r else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default=f"{ROOT}/results/detect_roc.json")
    a = ap.parse_args()
    out = {"note": __doc__.strip().split("\n\n")[0], "conflict_auroc": {}, "selfverify_auroc": {}, "summary": {}}
    for lab, pat, fn in IF:
        for ds in DS:
            for m, s in MODELS:
                r = records(f"{ROOT}/runs/{pat.format(ds=ds, s=s)}/eval/{ds}_test_id/{fn}")
                if not r:
                    continue
                lm, rm = np.asarray(r["look_margin"], float), np.asarray(r["read_margin"], float)
                # ---- self-verification: |margin| vs grounded, per request
                for w, mg in (("look", lm), ("read", rm)):
                    acc = np.asarray(r[f"{w}_acc"], bool)
                    v = auroc(np.abs(mg), acc)
                    if v is not None:
                        out["selfverify_auroc"][f"{lab}|{ds}|{m}|{w}"] = {"auroc": round(v, 4), "n": int(len(acc)), "acc": round(float(acc.mean()), 3)}
                # ---- conflict ROC: needs the twin-context margins
                need = ("look_margin_imgswap", "look_margin_capswap", "read_margin_capswap", "read_margin_imgswap")
                if all(k in r for k in need):
                    s_conf = lm + rm                                          # conflict context
                    s_a = np.asarray(r["look_margin_imgswap"], float) - np.asarray(r["read_margin_imgswap"], float)
                    s_b = np.asarray(r["look_margin_capswap"], float) - np.asarray(r["read_margin_capswap"], float)
                    score = np.concatenate([s_conf, s_a, s_b]); label = np.concatenate([np.ones_like(s_conf), np.zeros_like(s_a), np.zeros_like(s_b)])
                    v = auroc(score, label)
                    if v is not None:
                        out["conflict_auroc"][f"{lab}|{ds}|{m}"] = {"auroc": round(v, 4), "n_conflict": int(len(s_conf)), "n_agree": int(len(s_a) + len(s_b))}
    # ---- mean ROC curves per interface (vertical averaging over cells on a common FPR grid), for the
    #      §9 figure; only cells with the twin-context margins contribute
    grid = np.linspace(0, 1, 101)
    curves = {}
    for lab, pat, fn in IF:
        tprs = []
        for ds in DS:
            for m, s in MODELS:
                r = records(f"{ROOT}/runs/{pat.format(ds=ds, s=s)}/eval/{ds}_test_id/{fn}")
                need = ("look_margin_imgswap", "look_margin_capswap", "read_margin_capswap", "read_margin_imgswap")
                if not r or not all(k in r for k in need):
                    continue
                lm, rm = np.asarray(r["look_margin"], float), np.asarray(r["read_margin"], float)
                s_conf = lm + rm
                s_a = np.asarray(r["look_margin_imgswap"], float) - np.asarray(r["read_margin_imgswap"], float)
                s_b = np.asarray(r["look_margin_capswap"], float) - np.asarray(r["read_margin_capswap"], float)
                score = np.concatenate([s_conf, s_a, s_b]); label = np.concatenate([np.ones(len(s_conf)), np.zeros(len(s_a) + len(s_b))]).astype(bool)
                thr = np.unique(score)[::-1]
                fpr = np.array([((score >= t) & ~label).mean() for t in thr] + [1.0])
                tpr = np.array([((score >= t) & label).mean() for t in thr] + [1.0])
                order = np.argsort(fpr, kind="mergesort")
                tprs.append(np.interp(grid, fpr[order], tpr[order]))
        if tprs:
            curves[lab] = {"fpr": grid.round(3).tolist(), "tpr_mean": np.mean(tprs, 0).round(4).tolist(), "n_cells": len(tprs)}
    out["conflict_mean_roc"] = curves
    # ---- summaries per interface
    for key in ("conflict_auroc", "selfverify_auroc"):
        agg = {}
        for k, v in out[key].items():
            lab = k.split("|")[0]; w = k.split("|")[3] if key == "selfverify_auroc" else "all"
            agg.setdefault(f"{lab}|{w}", []).append(v["auroc"])
        out["summary"][key] = {k: {"mean_auroc": round(float(np.mean(v)), 3), "n_cells": len(v)} for k, v in agg.items()}
    json.dump(out, open(a.json, "w"), indent=1)
    for key in ("conflict_auroc", "selfverify_auroc"):
        print(f"\n{key}:")
        for k, v in sorted(out["summary"][key].items()):
            print(f"  {k:36s} mean AUROC {v['mean_auroc']:.3f}  over {v['n_cells']} cells")


if __name__ == "__main__":
    main()
