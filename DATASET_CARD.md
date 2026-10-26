# Ground-CF

Ground-CF is a three-source counterfactual benchmark: rendered scenes in which the image, the caption and the text
rendered inside the image can disagree, and each can be edited on its own while everything else is held fixed. We
built it because accuracy under conflict cannot tell us which source an answer rests on. Specifically, every item has
an exact twin per source (a one-source edit), so we measure whether an answer follows an edit to the requested source
(Flip) and stays put under an edit to the competing source (Stay); JOINT = min(Flip, Stay). Flip and Stay are always
reported separately, with chance and blind bounds next to them.

## Files (`data/clevr3/`)

- `manifest.json` and the splits `train`, `dev`, `test_id` (held-out items) and `test_ood` (attribute values never seen
  in training).
- Twin pairs `pairs_look_all_{train,dev,id,ood}.jsonl`, each with generator and validation hash; pairs whose original and
  edited targets coincide are rejected. The corrected image twins of the test splits are `pairs_look_all_{id,ood}_clean.jsonl`
  with images in `images_twin_clean/`.
- The absence and abstention batteries (`scripts/clevr3_absence.py`, `scripts/clevr3_abstain_twins.py`).

Rebuild from scratch: `scripts/prepare_clevr3.py`, then `scripts/clevr3_image_twins.py --split test_id --from_pairs
data/clevr3/pairs_look_all_id.jsonl` (bit-exact) and `scripts/clevr3_twins_count_spatial.py`;
`scripts/audit_twin_residue.py` audits the twins.

## Correction (2026-09-25)

An earlier generator recoloured the target one pixel short on the right and bottom edges, so colour and shape twins kept
a one-pixel line of the original object. The shipped twins are the corrected set, and every reported number uses them.

## Use and limits

Diagnosing source dependence of any VLM from its outputs, and comparing interfaces (instructions, soft prompts,
adapters, new words). Rendered scenes bound perception difficulty by design; we therefore pair Ground-CF with the
natural-photo rungs and with real video.

Licence: CC BY 4.0 (to be confirmed before the data are published).
