#!/usr/bin/env python
"""Manifests for the conflict-preserving Ground-CF and LoRA programme (STATUS v1.86/v1.87, 2026-09-22).

Training (one train.py argument line per row):
  configs/gcf_cp_core_train.txt    word / prefix / LoRA r4 CTM+Stay with cp pairs + image twins; LoRA r4 APO   (16)
  configs/gcf_cp_ablate_train.txt  word / prefix L=1,15,30 and LoRA r=1,16,64 CTM+Stay cp                        (36)
  configs/gcf_cp_apps_train.txt    composed, abstention, Stay-only (from the cancelled waves, + cp + twins) and
                                   Flip-only CTM cp for the objective ablation                                    (68)
  configs/lora_cp_train.txt        ladder request-LoRA r4: VCF SFT; APO; CTM+Stay cp on VCF/ColorSwap/COCO-CF    (28)
Evaluation (one shell command per row, cmd_array.sbatch):
  configs/gcf_cp_existing_evals.txt  existing Ground-CF arms re-evaluated with --cp 1 --tag _cp4, ID + OOD
  configs/gcf_cp_evals.txt / lora_cp_evals.txt  the new runs (fed by eval_when_done.sh once step_800 exists)
  configs/gen_check_groundcf.txt / _w2.txt      open-generation check on the cp runs
Idempotent: rewrites the files from the templates below; nothing else is touched.
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [("", ""), ("_qwen35", " --model_id Qwen/Qwen3.5-4B"), ("_qwen2b", " --model_id Qwen/Qwen3.5-2B"),
          ("_gemma4e4b", " --model_id google/gemma-4-E4B-it")]
TW = " --cp_pairs 1 --clevr3_image_twins data/clevr3/pairs_look_all_train.jsonl"
P2 = " --c3_sources look,read"      # two-modality protocol (STATUS v1.94)
HELD = "--held_cells look:spatial,read:count"
GCF_WORD = ("--method neo --arm ctrl --task clevr3 --data_dir data/clevr3 --n_tokens {L} --factor_mode srconly "
            "--question_style explicit " + HELD + " --lam_sft 1 {lam} --pref_warmup 100 --steps 800 --seed 0 --run_name {name}{model}{tw}" + P2)
GCF_PT = GCF_WORD.replace("--method neo --arm ctrl", "--method pt --arm ctrl")
GCF_LORA = ("--config configs/base.yaml --method lora --arm ctrl --n_tokens 4 --lam_ecc 0.0 --compare_frac 0.0 --mask_ctrl_lead 0 "
            "--full_coverage_only 0 --n_frames 1 --report_channel_words 0 --report_with_channels 0 --lora_r {r} --seed 0{model} "
            "--steps 800 --disentangle_frac 0.0 --lookread_conflict_only 0 --ctrl_site user --ctrl_style neutral --delim_loss_weight 1.0 "
            "--lr 0.0002 --lam_sft 1.0 --vds_weight 0 --lam_iit 0.0 --iit_layer 12 --aat_layer 12 --ctm_pairs data/mvg_ctm/pairs.jsonl "
            "--aat_max_area 0.5 --cert_weight 0.0 --pt_inserted 0 --tie_rows 0 --cap_first_prob 1.0 --slot_negative_frac 0.0 "
            "--absence_frac 0.0 --run_name {name} --task clevr3 --factor_mode srconly --question_style explicit " + HELD +
            " --data_dir data/clevr3 {lam} --pref_warmup 100{tw}" + P2)
LAD_LORA = ("--config configs/base.yaml --method lora --arm na --n_tokens 1 --lam_ecc 0.0 --compare_frac 0.0 --full_coverage_only 0 "
            "--n_frames 1 --report_channel_words 0 --report_with_channels 0 --lora_r 4 --seed 0{model} --steps 800 --disentangle_frac 0.0 "
            "--lookread_conflict_only 0 --ctrl_site user --ctrl_style request --delim_loss_weight 1.0 --lr 0.0002 --lam_sft 1.0 "
            "--vds_weight 0 --lam_iit 0.0 --iit_layer 12 --aat_layer 12 --ctm_pairs data/mvg_ctm/pairs.jsonl --aat_max_area 0.5 "
            "--cert_weight 0.0 --pt_inserted 0 --tie_rows 0 --cap_first_prob 1.0 --slot_negative_frac 0.0 --absence_frac 0.0 "
            "--run_name {name} --task color --factor_mode factored --question_style explicit --data_dir data/{ds} --pref_warmup 100{lam}")
CTMSM, APO, CTM, SFT = "--lam_ctm 1 --lam_sm 1", "--lam_apo 1", "--lam_ctm 1", ""


def w(path, lines):
    with open(os.path.join(ROOT, path), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {path}: {len(lines)} lines")
    return lines


def runname(line):
    return re.search(r"--run_name (\S+)", line).group(1)


def main():
    core, abl, lora_l, gcf_new = [], [], [], []
    for suf, mflag in MODELS:
        for lam, tag in ((SFT, "sft"), (APO, "apo"), (CTMSM, "ctmsm_cp")):
            core.append(GCF_WORD.format(L=5, lam=lam, name=f"clevr3_2s_5tok_{tag}_seed0", model=suf + mflag, tw=(TW if "cp" in tag else "")))
            core.append(GCF_PT.format(L=5, lam=lam, name=f"clevr3_2s_pt_5tok_{tag}_seed0", model=suf + mflag, tw=(TW if "cp" in tag else "")))
        core.append(GCF_LORA.format(r=4, lam="", name=f"clevr3_2s_lora_r4_sft_seed0{suf}", model=mflag, tw=""))
        core.append(GCF_LORA.format(r=4, lam="--lam_apo 1.0", name=f"clevr3_2s_lora_r4_apo_seed0{suf}", model=mflag, tw=""))
        core.append(GCF_LORA.format(r=4, lam="--lam_ctm 1.0 --lam_sm 1.0", name=f"clevr3_2s_lora_r4_ctmsm_cp_seed0{suf}", model=mflag, tw=TW))
        for L in (1, 15, 30):
            abl.append(GCF_WORD.format(L=L, lam=CTMSM, name=f"clevr3_2s_{L}tok_ctmsm_cp_seed0", model=suf + mflag, tw=TW))
            abl.append(GCF_PT.format(L=L, lam=CTMSM, name=f"clevr3_2s_pt_{L}tok_ctmsm_cp_seed0", model=suf + mflag, tw=TW))
        for r in (1, 16, 64):
            abl.append(GCF_LORA.format(r=r, lam="--lam_ctm 1.0 --lam_sm 1.0", name=f"clevr3_2s_lora_r{r}_ctmsm_cp_seed0{suf}", model=mflag, tw=TW))
        for ds in ("vcf", "color", "coco"):
            if ds == "vcf":
                lora_l.append(LAD_LORA.format(model=mflag, name=f"vcf_lora_r4_lr2e4_req_seed0{suf}", ds=ds, lam=""))
            lora_l.append(LAD_LORA.format(model=mflag, name=f"{ds}_lora_r4_lr2e4_req_apo_seed0{suf}", ds=ds, lam=" --lam_apo 1.0"))
            lora_l.append(LAD_LORA.format(model=mflag, name=f"{ds}_lora_r4_lr2e4_req_ctmsm_cp_seed0{suf}", ds=ds, lam=" --lam_ctm 1.0 --lam_sm 1.0 --cp_pairs 1"))
    # applications: the cancelled waves with cp + twins under new names, plus Flip-only CTM cp for the ablation
    apps = []
    for f in ("gcf_compose_ctmsm_train.txt", "gcf_abstain_ctmsm_train.txt", "gcf_smonly_train.txt"):
        for l in open(os.path.join(ROOT, "configs", f)):
            l = l.strip()
            if not l:
                continue
            assert "--cp_pairs" not in l and "--clevr3_image_twins" not in l, l[:80]
            l = re.sub(r"--run_name clevr3_f_", "--run_name clevr3_2s_", l)
            l = l.replace("--held_cells look:spatial,read:count,ocr:object", HELD)
            assert "clevr3_2s_" in l and ",ocr:object" not in l, l[:120]
            apps.append(re.sub(r"(--run_name \S*?)_seed0", r"\1_cp_seed0", l) + TW + P2)
    for suf, mflag in MODELS:
        for L in (1, 5):
            apps.append(GCF_WORD.format(L=L, lam=CTM, name=f"clevr3_2s_{L}tok_ctm_cp_seed0", model=suf + mflag, tw=TW))
            apps.append(GCF_PT.format(L=L, lam=CTM, name=f"clevr3_2s_pt_{L}tok_ctm_cp_seed0", model=suf + mflag, tw=TW))
    print("training manifests:")
    w("configs/gcf_cp_core_train.txt", core); w("configs/gcf_cp_ablate_train.txt", abl)
    w("configs/gcf_cp_apps_train.txt", apps); w("configs/lora_cp_train.txt", lora_l)
    names = [runname(l) for l in core + abl + apps + lora_l]
    assert len(names) == len(set(names)), "duplicate run names"
    for n in names:
        if os.path.isdir(os.path.join(ROOT, "runs", n, "ckpt", "step_800")):
            print(f"  NOTE run already has step_800 and will be skipped by train.py resume logic or overwrite: {n}")

    # ---------------- evaluation manifests
    FLIP = ("python scripts/eval_clevr3_flip.py --run_dir runs/{run} --data_dir data/clevr3 --split {split} --step 800 --max_n 700 "
            "--pairs_file data/clevr3/pairs_look_all_{tag}.jsonl --cp 1 --tag _cp4{extra}")
    ABST = "python scripts/eval_clevr3_abstain.py --run_dir runs/{run} --split {split} --step 800 --twins --tag _tw"
    ABSH = "python scripts/eval_clevr3_abstain.py --run_dir runs/{run} --split {split} --step 800 --hard --lennorm --tag _hardln"
    COMP = "python scripts/eval_compose_detect.py --run_dir runs/{run} --data_dir data/clevr3 --split {split} --step 800 --max_n 150 --both_orders 1 --tag _c2"
    SPL = (("test_id", "id"), ("test_ood", "ood"))
    ex = []
    for suf, _ in MODELS:
        base = f"clevr3_2s_5tok_sft_seed0{suf}"
        for split, tag in SPL:
            ex.append("INSTR_VARIANT=5 " + FLIP.format(run=base, split=split, tag=tag, extra="_instr --mode instructed"))
            ex.append(FLIP.format(run=base, split=split, tag=tag, extra="_noop --mode noop"))
    print("evaluation manifests:")
    w("configs/gcf_cp_existing_evals.txt", ex)   # frozen baselines under the two-source prompt; fed by eval_when_done
    new = []
    for l in core + abl + apps:
        run = runname(l)
        for split, tag in SPL:
            new.append(FLIP.format(run=run, split=split, tag=tag, extra=""))
            if "_habs_" in run:
                new.append(ABST.format(run=run, split=split)); new.append(ABSH.format(run=run, split=split))
            if "_c2_" in run:
                new.append(COMP.format(run=run, split=split))
    w("configs/gcf_cp_evals.txt", new + ex)
    LL = "python scripts/eval_flip_ll.py --run_dir runs/{run} --task {task} --data_dir data/{ds} --split {split} --max_n 140 --third_value 1 --out_task {ds}3"
    ll = []
    for l in lora_l:
        run = runname(l); ds = run.split("_")[0]
        for split, _ in SPL:
            ll.append(LL.format(run=run, task=("vcf" if ds == "vcf" else "color"), ds=ds, split=split))
    w("configs/lora_cp_evals.txt", ll)
    GEN = "python scripts/eval_clevr3_generate.py --run_dir runs/{run} --step 800 --split test_id{extra}"
    g1, g2 = [], []
    for suf, _ in MODELS:
        wd = f"clevr3_2s_5tok_ctmsm_cp_seed0{suf}"
        g1 += [GEN.format(run=wd, extra=""), GEN.format(run=f"clevr3_2s_pt_5tok_ctmsm_cp_seed0{suf}", extra=""),
               GEN.format(run=f"clevr3_2s_lora_r4_ctmsm_cp_seed0{suf}", extra=""),
               GEN.format(run=wd, extra=" --mode instructed --tag _instr").replace("python scripts", "INSTR_VARIANT=5 python scripts"),
               GEN.format(run=wd, extra=" --mode noop --tag _noop")]
        g2 += [GEN.format(run=f"clevr3_2s_5tok_ctm_cp_seed0{suf}", extra=""), GEN.format(run=f"clevr3_2s_5tok_apo_seed0{suf}", extra=""),
               GEN.format(run=f"clevr3_2s_5tok_sft_seed0{suf}", extra=""), GEN.format(run=f"clevr3_2s_pt_5tok_sft_seed0{suf}", extra=""),
               GEN.format(run=f"clevr3_2s_lora_r4_sft_seed0{suf}", extra="")]
    w("configs/gen_check_groundcf.txt", g1); w("configs/gen_check_groundcf_w2.txt", g2)


if __name__ == "__main__":
    main()
