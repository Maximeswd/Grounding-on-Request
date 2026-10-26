# Grounding on Request: Learning Modality Control in Vision-Language Models

Maxime Dassen, Alexander Martin, Andrew Yates, Reno Kriz, William Walden

Code and data for the paper (preprint; under review at ICLR 2027).

## Overview

Since a vision-language model can answer correctly from one source while the user asked for another, accuracy alone
does not tell us which evidence an answer rests on. We seek to understand this at a more fundamental level: we change
only the requested source and require the answer to follow (**Flip**), and we change only the competing source and
require the answer to stay (**Stay**). This allows us to separate *routing* (naming a source) from *grounding*
(tracking an intervention on it). We find that there exists a large gap between the two. Naming a source is cheap:
words, soft prompts, LoRA and plain English all route. On the other hand, grounding has to be installed, and
counterfactual objectives do this where plain fine-tuning does not.

In particular, we expand on neologism learning. In this method, new words are introduced, with new word embeddings.
These new words are placed in natural language; their embeddings are trained to minimize a loss on a set of examples
that exemplify a concept, while the model itself stays frozen. We introduce one word per source and draw upon research
in counterfactual evaluation to train them on paired interventions (CTM). We also go one step further and construct a
realistic but simple dataset, Ground-CF, in which image, caption and on-screen text can each be edited independently,
and we push the method towards real news video (MultiVENT-G). Surprisingly to us, a soft prompt placed at the word's
position is the same computation as the word: the interfaces differ in placement and wording, not in vocabulary.

Overall, our work contributes: (1) Flip and Stay, a conflict-preserving counterfactual measure of modality control,
together with Ground-CF; (2) evidence that accuracy and English instructions overstate visual control; (3) neologisms
with counterfactual supervision for user-selectable grounding; and (4) evidence that on real video the bottleneck is
access to the visual evidence, not the selection of the source.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export HF_HOME=~/.cache/huggingface        # SC_OFFLINE=1 on nodes without internet
huggingface-cli download google/gemma-3-4b-it --revision 093f9f388b31de276ce2de164bdc2081324b9767
python tests/test_static.py && python tests/test_tasks.py && python tests/test_swap.py
```

| model | id | revision |
|---|---|---|
| Gemma-3-4B | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Gemma-4 E4B | `google/gemma-4-E4B-it` | `fee6332c1abaafb77f6f9624236c63aa2f1d0187` |

## Ground-CF

`data/clevr3/` (code name `clevr3`) ships the benchmark in full: items and splits (`train`, `dev`, `test_id`,
`test_ood`), one-source twins and evidence boxes; see [DATASET_CARD.md](DATASET_CARD.md). Any model, or your own
generations, can be scored without the training code:

```bash
python evaluator/groundcf_eval.py --model google/gemma-3-4b-it --split test_id
python evaluator/groundcf_eval.py --generations my_outputs.jsonl
```

## Training

`train.py` trains one configuration; everything else in the model is frozen. The objective is
`lam_sft·CE + lam_sft_twin·CE(twin) + lam_ctm·CTM + lam_sm·Stay`: SFT is `--lam_sft 1`, paired SFT adds
`--lam_sft_twin 1`, the Flip margin `--lam_ctm 1` and the Stay margin `--lam_sm 1`.

| interface | flags |
|---|---|
| neologism (new rows in the carrier sentence) | `--method neo --arm ctrl --n_tokens L` |
| prompt tuning (soft prefix) | `--method pt --arm ctrl --n_tokens L --pt_read_init caption` |
| matched-position soft prompt (the word's slot) | `--method pt ... --gcf_pt_carrier 1` |
| shared LoRA | `--method lora --lora_r 16 --ctrl_style neutral --instr_rotate 0,2,4,5` |
| one LoRA adapter per request | `--method lora --lora_r 16 --ctrl_style adapter --train_sources look` |

Every run of the paper is one line of a manifest in `configs/` (`configs/iclr/` holds the paper's waves, the dated
manifests the follow-up studies); `python train.py $(sed -n 1p configs/iclr/<manifest>.txt)` trains one. Run names
follow `{task}_{method}_{arm}_{L}tok_{objective}_seed{k}[_{model}]`; a run stopped at a time limit resumes bitwise
identically with `--resume_ckpt`.

## Evaluation

```bash
python scripts/eval_clevr3_flip.py --run_dir runs/<run> --data_dir data/clevr3 --split test_ood --step 800 --cp 1   # Ground-CF
python scripts/eval_flip_ll.py --run_dir runs/<run> --task coco --data_dir data/coco --split test_ood --step 800   # ladder
python scripts/eval_mvg_flip.py --run_dir runs/<run> --step 800 --pairs data/mvg_ctm/pairs_clean_rb_heldout_videos.jsonl  # MVG
```

The ladder rungs are rebuilt from their sources: VCF with `scripts/prepare_vcf.py` (`mgolov/Visual-Counterfact`),
ColorSwap with `scripts/download_data.py` and `scripts/prepare_task.py` (`stanfordnlp/colorswap`), COCO-CF with
`scripts/prepare_coco_cf.py` and `scripts/coco_oracle_gate.py` (`Intel/COCO-Counterfactuals`); MultiVENT-G with
`extract_frames.py`, `scripts/prepare_multiventg.py`, `scripts/multiventg_build_items.py` and `scripts/prepare_mvg_ctm.py`.
Terms per source: [DATA_LICENSES.md](DATA_LICENSES.md). `scripts/master_grid.py`, `scripts/stats_pass.py` and
`scripts/harvest.py` aggregate the per-run outputs. The trained rows, LoRA adapters and evaluation outputs of every run
are released as archives (digests in `SHA256SUMS`) that extract into `runs/`.

## Follow-up studies

Since the submission we replicate the fixed recipes across seeds and models (`scripts/gen_replication_manifests.py`),
compare plain SFT with the Flip margin, the Stay margin and both (`scripts/gen_objective_factorial.py`,
`scripts/dev_select_objective.py`), sweep capacity on every dataset (`scripts/gen_capacity_sweeps.py`) and test the
control at other positions (`--gcf_ctrl_pos`, `--ladder_ctrl_pos`) and at the activation level (`--method act`).
Selections are made on a development partition with pre-registered criteria (`scripts/dev_select_*.py`).

## Citation

```
@inproceedings{dassen2027grounding,
  title     = {Grounding on Request: Learning Modality Control in Vision-Language Models},
  author    = {Dassen, Maxime and Martin, Alexander and Yates, Andrew and Kriz, Reno and Walden, William},
  booktitle = {Preprint; under review at ICLR 2027},
  year      = {2027}
}
```

Code: Apache License 2.0 ([LICENSE](LICENSE)). Data: [DATA_LICENSES.md](DATA_LICENSES.md).
