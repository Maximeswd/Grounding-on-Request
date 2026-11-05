# Data

## Ground-CF (`clevr3/`)

Since accuracy under conflict cannot tell us which source an answer rests on, we construct Ground-CF: rendered scenes in
which the image, the caption and an extracted-text block (on-screen text given as a string) can disagree, and each can
be edited on its own while everything else is held fixed. Every item has an exact twin per source (a one-source edit), which allows us to measure
whether an answer follows an edit to the requested source (Flip) and stays put under an edit to the competing source
(Stay). JOINT = min(Flip, Stay); we always report Flip and Stay separately, with chance and blind bounds next to them.

| file | content |
|---|---|
| `manifest.json` | generator settings and split sizes |
| `train.jsonl`, `dev.jsonl`, `test_id.jsonl`, `test_ood.jsonl` | items; `test_ood` uses attribute values never seen in training |
| `dev_scenes.txt` | scenes reserved for development |
| `pairs_look_all_{train,dev}.jsonl` | training and development twins (colour, shape, count, spatial) |
| `pairs_look_all_{id,ood}_clean.jsonl` | the test twins used for every reported number |
| `pairs_look_all_{id,ood}.jsonl` | the same test twins before the correction below; the evaluators read them when no `--pairs_file` is given |
| `pairs_look_cs_{id,ood}.jsonl` | the count and spatial twins on their own, as written by `scripts/clevr3_twins_count_spatial.py` |
| `pairs_look_{id,ood}.jsonl` | colour and shape twins read by `evaluator/groundcf_eval.py` |
| `evidence_boxes_{id,ood}.json` | evidence regions per item |
| `images/`, `images_twin*/` | scene renders and twin renders |

Each twin record carries its generator and a validation hash; pairs whose original and edited targets coincide are
rejected. The benchmark is rebuilt from scratch with `scripts/prepare_clevr3.py`; the colour and shape twins with
`scripts/clevr3_image_twins.py --split test_id --from_pairs data/clevr3/pairs_look_all_id_clean.jsonl` (bit-exact, same
for `test_ood`), and the count and spatial twins with `scripts/clevr3_twins_count_spatial.py`.

An earlier version of the twin generator recoloured the target object one pixel short on its right and bottom edges, so
colour and shape twins kept a one-pixel line of the original object. `pairs_look_all_{id,ood}_clean.jsonl` and
`images_twin_clean/` are the corrected test twins, and the paper's Ground-CF numbers use them.

Rendered scenes bound perception difficulty by design. Thus, Ground-CF should be read together with the natural-photo
benchmarks and with real video, both when comparing interfaces and when drawing conclusions about visual grounding.

## Other benchmarks

| directory | content | origin | terms |
|---|---|---|---|
| `clevr3/` | Ground-CF in full | this work | CC BY 4.0 (pending confirmation) |
| `vcf/` | split manifest; images rebuilt with `scripts/prepare_vcf.py` | `mgolov/Visual-Counterfact` | no licence on the Hub, so no images are redistributed |
| `color/` | split manifest; splits rebuilt with `scripts/prepare_task.py` | `stanfordnlp/colorswap` | MIT |
| `coco/` | split manifest; splits rebuilt with `scripts/prepare_coco_cf.py` | `Intel/COCO-Counterfactuals` | CC BY 4.0 |
| `multiventG/`, `mvg_*/` | derived text only (claims, descriptions, on-screen text, frame timestamps, evidence boxes); no frames or video | MultiVENT-G annotations, via `scripts/prepare_multiventg.py`, `scripts/multiventg_build_items.py`, `scripts/prepare_mvg_ctm.py` | MultiVENT-G terms (research use) |

The string `tyler day` in some MultiVENT-G files is a name shown on screen in a news video (an on-screen-text gold
value), not a person involved in this work.
