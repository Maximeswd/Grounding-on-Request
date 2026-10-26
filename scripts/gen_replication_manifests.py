#!/usr/bin/env python
"""Replication manifests for the FIXED recipes (2026-10-20): Ground-CF, protocol v2, task holdout, seeds x models.
Recipes from the pre-registered development selections: prefix = L=1, lr .05, clamp 2x; word = lr .005, clamp 2x at
L=1 (paper reference) and L=5 (selected capacity). Objectives: paired SFT (baseline) and +CTM+Stay (primary).
Idempotent: a run whose directory already holds ckpt/step_800 is skipped (its evaluations are still emitted if missing).
Usage: python scripts/gen_replication_manifests.py --part prefix|word|all --tag 20261020"""
import argparse, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--part", default="all", choices=["prefix", "word", "all"]); ap.add_argument("--tag", default="20261020")
ap.add_argument("--models", default="gemma3,qwen35,qwen2b,gemma4e4b"); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--word_lr", default="0.005")
a = ap.parse_args()
MODEL = {"gemma3": ("", ""), "qwen35": (" --model_id Qwen/Qwen3.5-4B", "_qwen35"), "qwen2b": (" --model_id Qwen/Qwen3.5-2B", "_qwen2b"),
         "gemma4e4b": (" --model_id google/gemma-4-E4B-it --train_per_layer 1", "_gemma4e4b")}
base = [l.rstrip("\n") for l in open(os.path.join(ROOT, "configs/l1_confirm_20261005_train_gemma.txt")) if l.strip()]
rn = lambda l: re.search(r"--run_name (\S+)", l).group(1)
word_psft = [l for l in base if rn(l) == "clevr3_v2_1tok_psft_neo_holdcs_seed0"][0]
word_ctmsm = [l for l in base if rn(l) == "clevr3_v2_1tok_psftctmsm_neo_holdcs_seed0"][0]
pt_psft = [l for l in base if rn(l) == "clevr3_v2_1tok_psft_pt_holdcs_seed0"][0]
pt_ctmsm = [l for l in base if rn(l) == "clevr3_v2_1tok_psftctmsm_pt_holdcs_seed0"][0]
ev_pat = [l.rstrip("\n") for l in open(os.path.join(ROOT, "configs/l1_confirm_20261005_eval_gemma.txt")) if "clevr3_v2_1tok_psft_neo_holdcs_seed0 " in l]
assert len(ev_pat) == 4
def rename(l, new): return re.sub(r"--run_name \S+", f"--run_name {new}", l)
train, evals = [], []
lrtag = "_lr" + a.word_lr.replace(".", "p")
for m in a.models.split(","):
    mflag, msfx = MODEL[m]
    for s in [int(x) for x in a.seeds.split(",")]:
        cells = []
        if a.part in ("prefix", "all"):
            cells += [(pt_psft, f"clevr3_v2_1tok_psft_pt_holdcs_seed{s}{msfx}", ""), (pt_ctmsm, f"clevr3_v2_1tok_psftctmsm_pt_holdcs_seed{s}{msfx}", "")]
        if a.part in ("word", "all"):
            for L in (1, 5):
                for src, obj in ((word_psft, "psft"), (word_ctmsm, "psftctmsm")):
                    if L == 5 and obj == "psft":
                        continue                                    # L=5 word: the selected (CTM+Stay) configuration only
                    cells.append((src.replace("--n_tokens 1 ", f"--n_tokens {L} "), f"clevr3_v2_{L}tok_{obj}_neo_holdcs{lrtag}_seed{s}{msfx}", f" --lr {a.word_lr}"))
        for src, name, extra in cells:
            line = rename(re.sub(r"--seed \d+", f"--seed {s}", src), name) + mflag + extra
            if "--workers" not in line:
                line += " --workers 2"
            has800 = os.path.isdir(os.path.join(ROOT, "runs", name, "ckpt", "step_800"))
            if not has800:
                train.append(line)
            for e in ev_pat:
                el = e.replace("clevr3_v2_1tok_psft_neo_holdcs_seed0 ", name + " ")
                tagm = re.search(r"--tag (\S+)", el).group(1); spl = re.search(r"--split (\S+)", el).group(1)
                if not os.path.exists(os.path.join(ROOT, "runs", name, "eval", f"clevr3_{spl}", f"flip_step_800{tagm}.json")):
                    evals.append(el)
out_t = os.path.join(ROOT, "configs", f"replication_{a.tag}_{a.part}_train.txt"); out_e = os.path.join(ROOT, "configs", f"replication_{a.tag}_{a.part}_eval.txt")
open(out_t, "w").write("\n".join(train) + ("\n" if train else "")); open(out_e, "w").write("\n".join(evals) + ("\n" if evals else ""))
print(f"{a.part}: {len(train)} training runs -> {out_t}; {len(evals)} evaluations -> {out_e}")
for l in train: print("  ", rn(l))
