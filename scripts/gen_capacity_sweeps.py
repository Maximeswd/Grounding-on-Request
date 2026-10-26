#!/usr/bin/env python
"""Capacity sweeps across datasets with the selected objectives, and MVG seeds (2026-10-24, author request).

Objectives (pre-registered development decision, results/dev_selection_objective.*):
  word           paired SFT + CTM + Stay   (--lam_sft 1 --lam_sft_twin 1 --lam_ctm 1 --lam_sm 1), lr .005
  prompt tuning  SFT + Stay                (--lam_sft 1 --lam_sm 1), lr .05
  LoRA (joint and one adapter per request)  paired SFT + Stay (--lam_sft 1 --lam_sft_twin 1 --lam_sm 1), lr 2e-4
Capacities: word and prompt tuning L in {1, 5, 15, 30}; LoRA rank in {1, 4, 16, 64}. Gemma-3-4B, seed 0.
Datasets and protocols:
  VCF, ColorSwap, COCO-CF   the paper's ladder protocol, conflict-preserving training pairs (--cp_pairs 1); evaluation
                            eval_flip_ll.py with a third candidate value (the paper's conflict-preserving battery)
  Ground-CF                 protocol v2, task holdout; evaluation on the CORRECTED twins (tags _v2ref_clean, _v2_clean)
  MVG                       two-source retrain with split-local donors; standard prompt tuning (--mvg_pt_prefix 1);
                            evaluation on held-out videos, full frame and evidence crop (tag _swho)
Lanes: ladder + Ground-CF on L40S, MVG on A100; every family (dataset) is evaluated in one lane, with its frozen
baselines re-evaluated in that lane. --part mvgseeds adds seeds 1-2 for MVG at the selected configurations (word L=5,
prompt tuning L=1, LoRA r16 joint and per request). Idempotent: existing runs and evaluations are skipped; runs owned
by another in-progress manifest (--skip_names_from) are not trained here but get this family's evaluations once present.
Usage: python scripts/gen_capacity_sweeps.py --part sweep|mvgseeds [--skip_names_from a.txt,b.txt] --tag 20261024"""
import argparse, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rn = lambda l: re.search(r"--run_name (\S+)", l).group(1)
LS, RANKS = (1, 5, 15, 30), (1, 4, 16, 64)
OBJ = {"word": "--lam_sft 1 --lam_sft_twin 1 --lam_ctm 1 --lam_sm 1", "pt": "--lam_sft 1 --lam_sft_twin 0 --lam_ctm 0 --lam_sm 1",
       "lora": "--lam_sft 1 --lam_sft_twin 1 --lam_ctm 0 --lam_sm 1"}
OTAG = {"word": "psftctmsm", "pt": "sftsm", "lora": "psftsm"}

def lines(p): return [l.rstrip("\n") for l in open(os.path.join(ROOT, p)) if l.strip()]
def objective(line, kind): return re.sub(r" --lam_(sft|sft_twin|ctm|sm|sft_stay) \S+", "", line) + " " + OBJ[kind]
def setname(line, name): return re.sub(r"--run_name \S+", f"--run_name {name}", line)
def done(name): return os.path.isdir(os.path.join(ROOT, "runs", name, "ckpt", "step_800"))
def has(name, sub): return os.path.exists(os.path.join(ROOT, "runs", name, "eval", sub))

def ladder(ds):
    """VCF / ColorSwap / COCO-CF: (run lines, eval lines). The paper's ladder protocol, task color."""
    base = f"--task color --data_dir data/{ds} --ctrl_style neutral --cp_pairs 1 --pref_warmup 100 --steps 800 --seed 0 --workers 2"
    runs = []
    for L in LS:
        runs.append(f"--method neo --arm ctrl --n_tokens {L} {base} {OBJ['word']} --lr 0.005 --decay_scope new_rows --run_name {ds}_sw_neo_{L}tok_psftctmsm_cp_seed0")
        runs.append(f"--method pt --arm na --n_tokens {L} {base} {OBJ['pt']} --lr 0.05 --decay_scope new_rows --pt_read_init caption --run_name {ds}_sw_pt_{L}tok_sftsm_cp_seed0")
    for r in RANKS:
        runs.append(f"--method lora --arm na --n_tokens 1 {base} {OBJ['lora']} --lora_r {r} --lr 0.0002 --ckpt_every 800 --run_name {ds}_sw_lora_r{r}_psftsm_cp_seed0")
        for src in ("look", "read"):
            runs.append(f"--method lora --arm na --n_tokens 1 {base.replace('--ctrl_style neutral', '--ctrl_style adapter')} {OBJ['lora']} --lora_r {r} --lr 0.0002 "
                        f"--ckpt_every 800 --train_sources {src} --neo_template bare --run_name {ds}_sw_loraad_r{r}_psftsm_cp_{src}_seed0")
    task = {"vcf": "vcf", "color": "color", "coco": "coco"}[ds]
    ev = lambda run, extra="": [f"{extra}python scripts/eval_flip_ll.py --run_dir runs/{run} --task {task} --data_dir data/{ds} --split {sp} --max_n 140 --third_value 1 --out_task {task}3"
                                + (" --frozen_model_id base" if "frozen" in run else "") for sp in ("test_id", "test_ood")
                                if not has(run, f"{task}3_{sp}/flip_ll{'_noop' if extra else ''}_step_800.json")]
    evals = [e for l in runs for e in ev(rn(l))]
    evals += ev(f"{ds}_base_frozen_0tok_none_sw_seed0") + ev(f"{ds}_base_frozen_0tok_none_sw_seed0", "NOOP_MODE=1 ")
    return runs, evals

def groundcf():
    g = lines("configs/l1_confirm_20261005_train_gemma.txt")
    word = [l for l in g if rn(l) == "clevr3_v2_1tok_psft_neo_holdcs_seed0"][0]
    pt = [l for l in g if rn(l) == "clevr3_v2_1tok_psft_pt_holdcs_seed0"][0]
    lora = [l for l in lines("configs/lora_ctmsm_20261005_train.txt") if rn(l) == "clevr3_v2_lora_r16_joint_psftctmsm_holdcs_seed0"][0]
    adapt = {l.split("--train_sources ")[1].split()[0]: l for l in lines("configs/specialist_lora_20261021_train.txt")}
    runs = []
    for L in LS:
        runs.append(setname(objective(word.replace("--n_tokens 1 ", f"--n_tokens {L} "), "word") + " --lr 0.005", f"clevr3_v2_{L}tok_psftctmsm_neo_holdcs_lr0p005_seed0"))
        runs.append(setname(objective(pt.replace("--n_tokens 1 ", f"--n_tokens {L} "), "pt"), f"clevr3_v2_{L}tok_sftsm_pt_holdcs_seed0"))
    for r in RANKS:
        runs.append(setname(objective(lora.replace("--lora_r 16", f"--lora_r {r}"), "lora") + " --ckpt_every 800", f"clevr3_v2_lora_r{r}_joint_psftsm_holdcs_seed0"))
        for src, l in adapt.items():
            runs.append(setname(objective(l.replace("--lora_r 16", f"--lora_r {r}"), "lora") + " --ckpt_every 800", f"clevr3_v2_lora_r{r}_adapter_{src}_psftsm_holdcs_seed0"))
    runs = [l if "--workers" in l else l + " --workers 2" for l in runs]
    def ev(run, mode=""):
        out = []
        for sp, pf in (("test_id", "id"), ("test_ood", "ood")):
            for qs, tg in (("explicit", "_v2ref_clean"), ("heldout", "_v2_clean")):
                tag = tg + (f"_{mode}" if mode else "")
                if has(run, f"clevr3_{sp}/flip_step_800{tag}.json"): continue
                out.append(f"SC_QSTYLE={qs} python scripts/eval_clevr3_flip.py --run_dir runs/{run}" + (f" --mode {mode}" if mode else "") +
                           f" --data_dir data/clevr3 --split {sp} --step 800 --max_n 1000 --pairs_file data/clevr3/pairs_look_all_{pf}_clean.jsonl --cp 1 --dump_picks 1 --tag {tag}")
        return out
    evals = [e for l in runs for e in ev(rn(l))]
    evals += ev("clevr3_v2_5tok_psftctm_neo_holdcs_seed0", "instructed") + ev("clevr3_v2_5tok_psftctm_neo_holdcs_seed0", "noop")
    return runs, evals

def mvg(seeds=(0,), configs=None):
    base = ("--task mvg --data_dir data/mvg_hard --ctm_pairs data/mvg_ctm/pairs_clean_rb_trainvideos_splitlocal.jsonl --mvg_sources desc,vis "
            "--pref_warmup 100 --steps 800 --workers 2")
    runs = []
    for s in seeds:
        sb = f"{base} --seed {s}"
        for L in (configs or {}).get("word", LS):
            runs.append(f"--method neo --arm ctrl --n_tokens {L} {sb} {OBJ['word']} --lr 0.005 --decay_scope new_rows --run_name mvgSW_neo_{L}tok_psftctmsm_seed{s}")
        for L in (configs or {}).get("pt", LS):
            runs.append(f"--method pt --arm na --n_tokens {L} {sb} {OBJ['pt']} --lr 0.05 --decay_scope new_rows --mvg_pt_prefix 1 --run_name mvgSW_pt_{L}tok_sftsm_seed{s}")
        for r in (configs or {}).get("lora", RANKS):
            runs.append(f"--method lora --arm na --n_tokens 4 {sb} {OBJ['lora']} --lora_r {r} --lr 0.0002 --ckpt_every 800 --run_name mvgSW_lora_r{r}_psftsm_seed{s}")
            for src in ("vis", "desc"):
                runs.append(f"--method lora --arm na --n_tokens 4 {sb} {OBJ['lora']} --lora_r {r} --lr 0.0002 --ckpt_every 800 --ctrl_style adapter "
                            f"--train_sources {src} --run_name mvgSW_loraad_r{r}_psftsm_{src}_seed{s}")
    P = "--pairs data/mvg_ctm/pairs_clean_rb_heldout_videos.jsonl --max_per_type 300 --mvg data/multiventG/multivent_g_rebuilt.json"
    def ev(run, style=""):
        out = []
        for crop in ("", " --crop_bbox 0.25"):
            fn = "flip_step_800_swho" + ("_crop0.25" if crop else "") + (f"_{style}" if style else "") + ".json"
            if has(run, f"mvg_flip/{fn}"): continue
            fz = f" --frozen_model_id base --style {style}" if style else " --step 800"
            out.append(f"python scripts/eval_mvg_flip.py --run_dir runs/{run}{fz} {P} --tag _swho{crop}")
        return out
    evals = [e for l in runs for e in ev(rn(l))]
    if 0 in seeds:
        evals += ev("_frozen_gemma", "noop") + ev("_frozen_gemma", "instr")
    return runs, evals

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True, choices=["sweep", "mvgseeds"])
    ap.add_argument("--tag", default="20261024"); ap.add_argument("--skip_names_from", default="")
    a = ap.parse_args()
    owned = set()
    for f in [x for x in a.skip_names_from.split(",") if x]:
        owned |= {rn(l) for l in lines(f) if "--run_name" in l}
    out = {}
    if a.part == "sweep":
        lt, le = [], []
        for ds in ("vcf", "color", "coco"):
            r, e = ladder(ds); lt += r; le += e
        r, e = groundcf(); lt += r; le += e
        mt, me = mvg()
        out["l40s"], out["a100"] = (lt, le), (mt, me)
    else:
        mt, me = mvg(seeds=(1, 2), configs={"word": (5,), "pt": (1,), "lora": (16,)})
        out["a100"] = (mt, me)
    for lane, (runs, evals) in out.items():
        train = [l for l in runs if not done(rn(l)) and rn(l) not in owned]
        tf = os.path.join(ROOT, "configs", f"{a.part}_{a.tag}_{lane}_train.txt"); ef = os.path.join(ROOT, "configs", f"{a.part}_{a.tag}_{lane}_eval.txt")
        open(tf, "w").write("\n".join(train) + ("\n" if train else "")); open(ef, "w").write("\n".join(evals) + ("\n" if evals else ""))
        print(f"{a.part} {lane}: {len(train)} runs to train ({len(runs) - len(train)} exist or owned elsewhere), {len(evals)} evaluations -> {os.path.relpath(tf, ROOT)}")

if __name__ == "__main__":
    main()
