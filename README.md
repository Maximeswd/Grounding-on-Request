# Grounding on Request: Learning Modality Control in Vision-Language Models

Maxime Dassen, Alexander Martin, Andrew Yates, Reno Kriz, William Walden

Code and data for the paper (preprint; under review at ICLR 2027).

## Overview

When an image and its caption disagree, can a user reliably control which one a vision-language model (VLM) follows?
Since a model might give the correct answer from parametric knowledge without following the requested evidence, accuracy
alone cannot answer this question. We therefore seek to understand modality control at a more fundamental level, through
conflict-preserving counterfactual interventions. Specifically, **Flip** measures whether an initially correct answer
follows an edit to the requested modality, while **Stay** measures whether it remains stable when the competing modality
is edited. This allows us to separate an answer that is correct from an answer that actually follows the modality of
choice.

Our work aims to bridge the gap between measuring modality control and explicitly learning it. In particular, we expand
on neologism learning. In this method, new words are introduced, with new word embeddings. These new words are placed in
natural language; their embeddings are trained to minimize a loss on a set of examples that exemplify a concept, while
the pretrained model itself stays fixed. We introduce `<look>` for visual grounding and `<read>` for textual grounding.
Since standard supervised fine-tuning optimizes examples independently, it does not teach how answer preferences should
move when the requested evidence changes. We therefore draw upon research in counterfactual evaluation and introduce the
Counterfactual Twin-Margin (CTM) objective, which uses paired interventions so that the original answer becomes less
likely and the replacement answer more likely after an edit to the requested modality.

We evaluate three VLMs on three existing benchmarks (VCF, ColorSwap, COCO-CF), and we also go one step further and
construct a realistic but simple dataset, Ground-CF, with image–text interventions over color, shape, count and spatial
relations. We find that there exists a large gap between natural-language instructions and learned modality words,
particularly for visual requests. Surprisingly to us, a single learned embedding per modality already provides strong
lightweight visual control and remains stable on average across held-out attribute values. On the other hand, LoRA
achieves the strongest absolute control, although its visual advantage narrows under held-out evaluation. Moreover,
counterfactual supervision improves control in several settings where standard supervision is weaker. Finally, we push
the method farther towards real data and evaluate it on news video (MultiVENT-G), where visual control improves
considerably when the relevant evidence is made easier to access.

Overall, our work contributes: (1) a conflict-preserving counterfactual formulation of modality grounding control using
Flip and Stay, together with Ground-CF; (2) evidence that accuracy and natural-language instructions substantially
overstate reliable visual control; (3) neologism learning for user-selectable modality grounding, together with CTM-based
counterfactual supervision; and (4) evidence that lightweight learned controls are effective on held-out attributes and
extend to realistic news-video frames. Our work, though focused on images and text for simplicity, applies to all
modalities an answer can be drawn from: the same words and objectives can be trained for any further source, such as
speech or a position in time.

## Setup

```bash
uv sync                                    # Python 3.12 and the locked environment (uv.lock) in .venv
uv run hf download google/gemma-3-4b-it --revision 093f9f388b31de276ce2de164bdc2081324b9767
uv run pytest                              # CPU-only checks
```

| model | id | revision |
|---|---|---|
| Gemma-3-4B | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |

Set `SC_OFFLINE=1` on machines without internet access once the models are cached. Every command below runs inside
the environment, e.g. `uv run python train.py ...`.

## Repository layout

```
train.py              trains one configuration (everything but the new rows / soft prompt / adapter is frozen)
evaluate.py           shared scoring helpers
scsrc/                library: request words, prompts, datasets, objectives, model wrappers
scripts/              data preparation, the Flip/Stay evaluators, aggregation
configs/base.yaml     default hyperparameters
configs/paper/        the training command of every run in the paper, one file per benchmark
evaluator/            standalone Ground-CF scorer (no training code needed)
data/                 Ground-CF in full; split manifests and derived annotations for the other benchmarks
tests/                CPU-only checks
pyproject.toml        dependencies, ruff and mypy settings; uv.lock pins the full environment
```

## Ground-CF

`data/clevr3/` holds the benchmark (`clevr3` is its name in the code): items and splits, one-source twins and evidence
boxes; see [data/README.md](data/README.md). Any model can be scored without the training code:

```bash
python evaluator/groundcf_eval.py hf --model google/gemma-3-4b-it --split test_id      # likelihood over the candidates
python evaluator/groundcf_eval.py requests --split test_id -o requests.jsonl           # or: export the contexts,
python evaluator/groundcf_eval.py score --requests requests.jsonl --answers answers.jsonl   # run any model, score its answers
```

## Training

The objective is `lam_sft·CE + lam_sft_twin·CE(twin) + lam_ctm·CTM + lam_sm·Stay`: SFT is `--lam_sft 1`, paired SFT adds
`--lam_sft_twin 1`, the Flip margin is `--lam_ctm 1` and the Stay margin `--lam_sm 1`.

| interface | flags |
|---|---|
| neologism (new rows in the request sentence) | `--method neo --arm ctrl --n_tokens L` |
| prompt tuning (soft prefix) | `--method pt --arm ctrl --n_tokens L --pt_read_init caption` |
| soft prompt at the word's position | `--method pt ... --gcf_pt_carrier 1` |
| one LoRA adapter per request | `--method lora --lora_r 16 --ctrl_style adapter --train_sources look` |

`configs/paper/{groundcf,ladder,mvg}.txt` hold the exact command of every run behind the paper's tables. To retrain one:

```bash
grep -w clevr3_2s_1tok_ctm_cp_fc_seed0 configs/paper/groundcf.txt | sh
```

Run names follow `{task}_{interface}_{L}tok_{objective}_seed{k}[_{model}]`; checkpoints land in `runs/<run>/ckpt/`, and a
run stopped at a time limit resumes bitwise identically with `--resume_ckpt`.

## Reproducing the paper

| paper | runs | evaluation |
|---|---|---|
| Tables 2, 9–11; Figure 3 | `configs/paper/ladder.txt` (VCF, ColorSwap, COCO-CF), `configs/paper/groundcf.txt` | `eval_flip_ll.py`, `eval_clevr3_flip.py` |
| Table 1, Figure 2 and the no-request / instruction rows of Tables 9–11 | the frozen models | `eval_clevr3_flip.py --mode noop` or `--mode instructed` (with a `--tag`); `eval_flip_ll.py --frozen_model_id <model>` |
| Table 3 (news video) | `configs/paper/mvg.txt` | `eval_mvg_flip.py`, full frame and `--crop_bbox 0.25` |
| Table 8 (placement) | `configs/paper/groundcf.txt`, runs named `*_pfirst_*` against their native-position twins | `eval_clevr3_flip.py` |
| Figure 4 (metalinguistic probes) | the L=1 Ground-CF runs `clevr3_2s_1tok_ctm_cp_fc_seed0*` and `clevr3_2s_pt_1tok_ctm_cp_fc_seed0*` | `eval_meta_words.py` |

## Evaluation

All scores are teacher-forced likelihoods over a fixed candidate set, at step 800, on `test_id` and `test_ood`:

```bash
# Ground-CF
SC_QSTYLE=heldout python scripts/eval_clevr3_flip.py --run_dir runs/<run> --data_dir data/clevr3 --split test_id \
    --step 800 --max_n 1000 --pairs_file data/clevr3/pairs_look_all_id_clean.jsonl --cp 1 --tag _cp4_ho
# VCF, ColorSwap, COCO-CF (--task / --data_dir vcf | color | coco)
SC_LADDER_PROTO=2 SC_LADDER_LEADIN=1 SC_TEMPLATE=heldout python scripts/eval_flip_ll.py --run_dir runs/<run> \
    --task vcf --data_dir data/vcf --split test_id --max_n 140 --third_value 1 --out_task vcf3
# MultiVENT-G, held-out videos (add --crop_bbox 0.25 for the evidence crop)
python scripts/eval_mvg_flip.py --run_dir runs/<run> --step 800 --pairs data/mvg_ctm/pairs_clean_rb_heldout_videos.jsonl \
    --max_per_type 300 --mvg data/multiventG/multivent_g_rebuilt.json --tag _rbho
# metalinguistic probes (Figure 4)
python scripts/eval_meta_words.py --run_dir runs/<run> --data_dir data/clevr3 --split test_id --step 800 --n_items 40 --tag _meta_fc2
```

Results are written to `runs/<run>/eval/`. `scripts/master_grid.py` collects them into one table, `scripts/stats_pass.py`
adds seed-paired intervals and exact McNemar tests, and `scripts/harvest.py` prints the per-benchmark tables.

The VCF, ColorSwap and COCO-CF images are not redistributed; rebuild them from their sources with
`scripts/prepare_vcf.py` (`mgolov/Visual-Counterfact`), `scripts/download_data.py` and `scripts/prepare_task.py`
(`stanfordnlp/colorswap`), and `scripts/prepare_coco_cf.py` with `scripts/coco_oracle_gate.py` (`Intel/COCO-Counterfactuals`).
MultiVENT-G is rebuilt with `scripts/extract_frames.py`, `scripts/prepare_multiventg.py`, `scripts/multiventg_build_items.py`
and `scripts/prepare_mvg_ctm.py`.

## Development

We keep one style and one set of checks for the whole repository; CI runs the same four commands on every push.

```bash
uv run pre-commit install        # ruff check --fix and ruff format before every commit
uv run ruff check                # every ruff rule (pyproject.toml lists the few we opt out of)
uv run ruff format --check
uv run mypy                      # strict mode
uv run pytest
```

The code of the paper predates this standard. It is formatted, and the rules each file still violates are recorded as a
baseline in `pyproject.toml` (per file for ruff, an `ignore_errors` list for mypy). A new file gets no entry, so it has
to pass every rule; the baseline only shrinks. New code follows a few design rules: configurations and records are
frozen dataclasses (no `dict[str, Any]` for structured data, tuples inside frozen data); functions return new values
instead of mutating their inputs; variants are unions matched with `match` and `assert_never`, not string flags;
distinct units (a step, an epoch) get their own `NewType`; and the behaviour is written down first, then the tests, then
the code.

## Released checkpoints and outputs

The trained rows, LoRA adapters and evaluation outputs of every run are released as archives that extract into `runs/`;
their SHA-256 digests are in `SHA256SUMS`.

## Citation

```
@inproceedings{dassen2027grounding,
  title     = {Grounding on Request: Learning Modality Control in Vision-Language Models},
  author    = {Dassen, Maxime and Martin, Alexander and Yates, Andrew and Kriz, Reno and Walden, William},
  booktitle = {Preprint; under review at ICLR 2027},
  year      = {2027}
}
```

Code: Apache License 2.0 ([LICENSE](LICENSE)). Data: see [data/README.md](data/README.md).
