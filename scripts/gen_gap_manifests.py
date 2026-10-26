"""Z22 (2026-09-13): fill every empty cell of the appendix big tables.

Missing after Z20/Z21 land: Ground-CF VGM at L=5 (the pair builder returned
chosen/rejected where the objective reads pos/neg; fixed in scsrc/clevr3.py),
and at L=1: PT+APO (VCF, Ground-CF), PT+CTM (VCF, ColorSwap Qwen 4B/2B,
Ground-CF), inserted PT (ladder), Neo+CTM+SN (ladder), Neo+VGM (ladder,
Ground-CF).  Inserted PT on Ground-CF is byte-identical to PT (flag is a no-op
there) and is not generated.  Idempotent: runs with ckpt/step_800 are skipped.

  python scripts/gen_gap_manifests.py
  MANIFEST=configs/iclr/Z22_train_gaps.txt sbatch --array=1-N scripts/train_array_a100.sbatch
  MANIFEST=configs/iclr/Z22e_eval_gaps.txt PAR=4 sbatch --array=1-ceil(N/4) --dependency=afterany:<id> scripts/par_a100.sbatch
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"g3": None, "q4b": "Qwen/Qwen3.5-4B", "q2b": "Qwen/Qwen3.5-2B", "e4b": "google/gemma-4-E4B-it"}
SFX = {"g3": "", "q4b": "_qwen35", "q2b": "_qwen2b", "e4b": "_gemma4e4b"}
ALL = list(MODELS)
C3 = ("--task clevr3 --data_dir data/clevr3 --factor_mode srconly --question_style explicit "
      "--held_cells look:spatial,read:count,ocr:object")


def done(run):
    return os.path.isdir(f"{ROOT}/runs/{run}/ckpt/step_800")


def mid(m):
    return f" --model_id {MODELS[m]}" if MODELS[m] else ""


def ladder_neo(ds, L, obj, m, sn=False):
    run = f"{ds}_neo_ctrl_{L}tok_{obj}{'v3' if sn else ''}_seed0{SFX[m]}"
    line = (f"--method neo --arm ctrl --n_tokens {L} --task color --data_dir data/{ds} --lam_sft 1 --lam_{obj} 1"
            + (" --slot_negative_frac 0.15" if sn else "") + f" --pref_warmup 100 --steps 800 --seed 0{mid(m)} --run_name {run}")
    return run, ds, line


def ladder_pt(ds, L, obj, m):
    run = f"{ds}_pt_na_{L}tok_{obj}_seed0{SFX[m]}"
    line = (f"--method pt --arm na --n_tokens {L} --task color --data_dir data/{ds} --lam_sft 1 --lam_{obj} 1 "
            f"--pref_warmup 100 --steps 800 --seed 0{mid(m)} --run_name {run}")
    return run, ds, line


def ladder_pti(ds, L, m):
    run = f"{ds}_pti_na_{L}tok_sft_seed0{SFX[m]}"
    line = (f"--method pt --arm na --n_tokens {L} --task color --data_dir data/{ds} --pt_inserted 1 --lam_sft 1 "
            f"--steps 800 --seed 0{mid(m)} --run_name {run}")
    return run, ds, line


def c3_neo(L, obj, m):
    run = f"clevr3_f_{L}tok_{obj}_seed0{SFX[m]}"
    line = (f"--method neo --arm ctrl {C3} --n_tokens {L} --lam_sft 1 --lam_{obj} 1 --pref_warmup 100 "
            f"--steps 800 --seed 0{mid(m)} --run_name {run}")
    return run, "clevr3", line


def c3_pt(L, obj, m):
    run = f"clevr3_f_pt_{L}tok_{obj}_seed0{SFX[m]}"
    line = (f"--method pt --arm ctrl {C3} --n_tokens {L} --lam_sft 1 --lam_{obj} 1 --pref_warmup 100 "
            f"--steps 800 --seed 0{mid(m)} --run_name {run}")
    return run, "clevr3", line


def main():
    items = []
    # 1. Ground-CF VGM, L=5 (completes the L=5 tables)
    items += [c3_neo(5, "vgm", m) for m in ALL]
    # 2. PT+APO at L=1: VCF, Ground-CF
    items += [ladder_pt("vcf", 1, "apo", m) for m in ALL] + [c3_pt(1, "apo", m) for m in ALL]
    # 3. PT+CTM at L=1: VCF, ColorSwap (Qwen 4B/2B died at step 627/752 on the old 4 h wave), Ground-CF
    items += [ladder_pt("vcf", 1, "ctm", m) for m in ALL] + [ladder_pt("color", 1, "ctm", m) for m in ("q4b", "q2b")]
    items += [c3_pt(1, "ctm", m) for m in ALL]
    # 4. Neo+CTM+SN at L=1, ladder
    items += [ladder_neo(ds, 1, "ctm", m, sn=True) for ds in ("vcf", "color", "coco") for m in ALL]
    # 5. inserted PT at L=1, ladder
    items += [ladder_pti(ds, 1, m) for ds in ("vcf", "color", "coco") for m in ALL]
    # 6. Neo+VGM at L=1: VCF, ColorSwap (Qwen 2B, E4B), COCO-CF, Ground-CF
    items += [ladder_neo("vcf", 1, "vgm", m) for m in ALL] + [ladder_neo("color", 1, "vgm", m) for m in ("q2b", "e4b")]
    items += [ladder_neo("coco", 1, "vgm", m) for m in ALL] + [c3_neo(1, "vgm", m) for m in ALL]

    train, evals, skipped = [], [], []
    seen = set()
    for run, ds, line in items:
        assert run not in seen, run
        seen.add(run)
        if done(run):
            skipped.append(run)
        else:
            train.append(line)
        if ds == "clevr3":
            evals.append(f"[ -d runs/{run}/ckpt/step_800 ] && python scripts/eval_clevr3_flip.py --run_dir runs/{run} --split test_id || true")
        else:
            evals.append(f"[ -d runs/{run}/ckpt/step_800 ] && python scripts/eval_flip_ll.py --run_dir runs/{run} --task color "
                         f"--data_dir data/{ds} --split test_id --max_n 140 --dump_records 1 || true")
    out = f"{ROOT}/configs/iclr"
    open(f"{out}/Z22_train_gaps.txt", "w").write("\n".join(train) + "\n")
    open(f"{out}/Z22e_eval_gaps.txt", "w").write("\n".join(evals) + "\n")
    print(f"train lines: {len(train)}  eval lines: {len(evals)}  skipped (done): {len(skipped)} {skipped}")


if __name__ == "__main__":
    main()
