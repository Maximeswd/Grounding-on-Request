# Protocol-v2 pilot manifests (2026-09-29)

No header lines inside the manifests: train_array reads line N for array task N.

## configs/pilot_v2_20260929_train_gemma.txt  (8 lines)
Protocol-v2 PILOT (2026-09-29, advisor note items 3/5/7). One model, Ground-CF, TASK HOLDOUT: train colour+shape, count and spatial held out of training for BOTH requests (--held_cells). L=5 for both embedding methods, matched init (word: image/caption; prefix: image/caption via --pt_read_init caption), source-selection carriers, explicit v2 format, answers in the structured "Final answer: <v>" format (--gcf_cont final; the natural-output condition is a separate later ablation), weight decay on the new rows only. Objectives: paired SFT vs paired SFT + CTM on identical contexts. Submit: MANIFEST=<this file> sbatch --array=1-8 scripts/train_a100_80_long.sbatch  (a100, 8 h walltime; no H cards). Do NOT chain 400-step runs: --resume_ckpt restarts AdamW without state and is word-only, so a chained run is not the 800-step trajectory.

## configs/pilot_v2_20260929_eval_gemma.txt  (52 lines)
Protocol-v2 pilot EVALS (cmd_array.sbatch, l40s; one command per line). Tags: _v2q = held-out question + training carrier (question robustness); _v2c = training question + held-out carrier (carrier robustness); _v2 = both held out. Task transfer = the count/spatial rows of the per-skill output. TODO before launch: bare-answer parse in scripts/eval_clevr3_generate.py (its ANCHOR 'final answer:' does not exist under --gcf_cont bare).

## configs/pilot_v2_20260929_train_qwen35.txt  (8 lines)
Protocol-v2 PILOT (2026-09-29, advisor note items 3/5/7). One model, Ground-CF, TASK HOLDOUT: train colour+shape, count and spatial held out of training for BOTH requests (--held_cells). L=5 for both embedding methods, matched init (word: image/caption; prefix: image/caption via --pt_read_init caption), source-selection carriers, explicit v2 format, answers in the structured "Final answer: <v>" format (--gcf_cont final; the natural-output condition is a separate later ablation), weight decay on the new rows only. Objectives: paired SFT vs paired SFT + CTM on identical contexts. Submit: MANIFEST=<this file> sbatch --array=1-8 scripts/train_a100_80_long.sbatch  (a100, 8 h walltime; no H cards). Do NOT chain 400-step runs: --resume_ckpt restarts AdamW without state and is word-only, so a chained run is not the 800-step trajectory.

## configs/pilot_v2_20260929_eval_qwen35.txt  (52 lines)
Protocol-v2 pilot EVALS (cmd_array.sbatch, l40s; one command per line). Tags: _v2q = held-out question + training carrier (question robustness); _v2c = training question + held-out carrier (carrier robustness); _v2 = both held out. Task transfer = the count/spatial rows of the per-skill output. TODO before launch: bare-answer parse in scripts/eval_clevr3_generate.py (its ANCHOR 'final answer:' does not exist under --gcf_cont bare).

## Evaluation conditions (audit item 4)
| tag | question wording | carrier / English wording |
|---|---|---|
| `_v2ref` | training pool | training pool (reference) |
| `_v2q` | held-out pool | same as reference (question shift only) |
| `_v2c` | same as reference | held-out pool (carrier shift only) |
| `_v2` | held-out pool | held-out pool (both) |
Standard prompt tuning has no inline carrier: its `_v2c` run is an identity control (must equal `_v2ref`), not carrier robustness.
Design v2 (SC_GCF_FORMAT=v2) derives carrier index, question index and evidence order from one stable per-item key, so each switch changes exactly one field (tests/test_v2_design.py).

## Before the pilot: the smoke check (configs/smoke_v2_20260929_*.txt)
Four 8-step runs (word, prefix, matched-position prefix, joint LoRA) with --selfcheck_items 6, then scripts/smoke_v2_reload.py reloads each checkpoint and must report identical prompts and targets, identical picks and max |dlogp| < 0.05. The .out files show the [pt] row/initialization line, the [optim] decay line and ce / ce_twin / ctm in the progress bar.

## Launch record (Gemma, 2026-10-01/02) and what to use for Qwen
- Gemma used `configs/pilot_v2_20261001_train_gemma_w2.txt` (= the 09-29 manifest + `--workers 2`) on `scripts/train_a100_80_long.sbatch`.
  With 6 workers, four jobs on one A100 node killed/hung the data workers; rack8n05 did so even with 2 workers, so use
  `sbatch --exclude=rack8n05`. Training times on A100: paired SFT 1.4–1.9 h, paired SFT+CTM 4.0–5.3 h (joint LoRA slowest).
- Qwen-3.5-4B: `configs/pilot_v2_20261002_train_qwen35_w2.txt` (same + `--workers 2`, `--gcf_cont final`), launch only after the
  Gemma CTM results have been read (author decision):
  `MANIFEST=configs/pilot_v2_20261002_train_qwen35_w2.txt sbatch --exclude=rack8n05 --array=1-8 scripts/train_a100_80_long.sbatch`
  then `MANIFEST=configs/pilot_v2_20260929_eval_qwen35.txt sbatch --array=1-68 scripts/cmd_array.sbatch` once all step_800 exist.
- Evaluation add-ons used for Gemma: generation check (`configs/pilot_v2_20261002_gen_gemma.txt`, tag `_v2gen`) and full-inventory
  forced choice (`configs/pilot_v2_20261002_full_gemma.txt`, `--cands full`, tag `_v2ref_full`). Keep all forced-choice evaluations
  of one paired comparison on the same GPU class (bf16 scores differ across node classes; picks agreed in the smoke, margins did not).
