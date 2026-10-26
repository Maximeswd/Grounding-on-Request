"""ColorSwap loading + condition construction + legibility filtering."""
from . import paths  # noqa: F401  (sets HF env first)
import os, json
from dataclasses import dataclass
from typing import Optional, List, Dict

from datasets import load_dataset
from .colors import valid_swap


def keys_from_row(row):
    """Per-source GT keys for a row. Processed task rows store key_1/key_2
    (JSON) directly -> any value type (ints, relations, attributes). ColorSwap
    rows have no stored keys, so fall back to parsing the captions."""
    if row.get("key_1"):
        k1, k2 = row["key_1"], row["key_2"]
        if isinstance(k1, str):
            k1 = json.loads(k1)
        if isinstance(k2, str):
            k2 = json.loads(k2)
        ok = bool(k1) and bool(k2) and set(k1) == set(k2) and \
            any(k1[o] != k2[o] for o in k1)
        return ok, k1, k2
    return valid_swap(row["caption_1"], row["caption_2"])


def load_task(data_dir, split):
    """Load a processed task split saved by scripts/prepare_task.py."""
    from datasets import load_from_disk
    return load_from_disk(os.path.join(data_dir, split))


@dataclass
class Spec:
    ex_index: int
    split: str
    img_idx: int          # 1 or 2 -> which image is shown
    fed_caption: str
    image_key: dict
    text_key: dict
    relation: str         # "agree" | "conflict"
    cond: str             # "agree" | "conflict"
    example_id: int       # dataset 'id'


def load_colorswap(split: str, human_only: bool = True):
    ds = load_dataset(paths.DATASET_ID, split=split)
    if human_only:
        ds = ds.filter(lambda r: r.get("caption_source") == "human")
    return ds


def build_specs(ds, split: str, legible_ids: Optional[set] = None,
                conditions=("agree", "conflict"),
                heldout_classes: Optional[set] = None,
                class_mode: str = "exclude") -> List[Spec]:
    """Construct up to 4 conditioned instances per valid example.

    Alignment assumption: caption_i describes image_i. (diagnostic.py verifies
    this and can flip it.) image_key for image_i := parse(caption_i)."""
    specs: List[Spec] = []
    n_invalid = 0
    for idx in range(len(ds)):
        row = ds[idx]
        ok, k1, k2 = keys_from_row(row)
        if not ok:
            n_invalid += 1
            continue
        if heldout_classes is not None:
            hit = bool((set(k1.values()) | set(k2.values())) & heldout_classes)
            if (class_mode == "exclude" and hit) or (class_mode == "only" and not hit):
                continue
        eid = int(row.get("id", idx))
        if legible_ids is not None and eid not in legible_ids:
            continue
        c1, c2 = row["caption_1"], row["caption_2"]
        cand = {
            "agree": [
                Spec(idx, split, 1, c1, k1, k1, "agree", "agree", eid),
                Spec(idx, split, 2, c2, k2, k2, "agree", "agree", eid),
            ],
            "conflict": [
                Spec(idx, split, 1, c2, k1, k2, "conflict", "conflict", eid),
                Spec(idx, split, 2, c1, k2, k1, "conflict", "conflict", eid),
            ],
        }
        for c in conditions:
            specs.extend(cand[c])
    print(f"[data:{split}] examples={len(ds)} invalid_swaps={n_invalid} "
          f"specs={len(specs)} (conditions={conditions})")
    return specs


def get_image(ds, spec: Spec):
    return ds[spec.ex_index][f"image_{spec.img_idx}"]


def legible_path(split: str) -> str:
    return os.path.join(paths.RUNS, "_diagnostic", f"legible_{split}.json")


def load_legible_ids(split: str) -> Optional[set]:
    p = legible_path(split)
    if os.path.exists(p):
        with open(p) as f:
            return set(json.load(f)["legible_ids"])
    return None
