"""Unified slot-conflict construction.

Turn any per-object labelled dataset (counting, spatial, attributes, ...) into
ColorSwap-shaped conflict examples with exact, string-matchable per-source GT.

Every task reduces to the same abstract structure (see the design diagrams):
an image + a SLOT (a query about one object or one object-pair) with a known
image-true value `v_img`. We pair it with a SWAP image whose value differs
(`v_swap`), which yields exactly the four ColorSwap conditions:

    agree    : (image_1 -> says v1) , (image_2 -> says v2)
    conflict : (image_1 -> says v2) , (image_2 -> says v1)

image_1's GT key = {obj: v_img(image_1)}; in conflict the text channel asserts
the *other* image's value -> laundering = a <look> block returning the text's
value. This is the ColorSwap mechanism generalized off basic color (too easy,
per the diagnostic) onto perceptually hard slots where collapse actually bites.

Rows carry the ColorSwap columns (image_1,image_2,caption_1,caption_2,id,
caption_source) PLUS stored keys (key_1,key_2 as JSON). data.build_specs reads
the stored keys instead of re-parsing, so integers / relations / open-vocab
attributes all work without a value-specific parser. ColorSwap keeps its native
path (minimal pairs); only the new tasks flow through here.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# ---- task value vocab / config ---------------------------------------------
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}

# A clean, well-separated spatial relation whitelist (projective + proximity) --
# the relations with the strongest visual grounding. Kept small so exact match
# stays meaningful (avoids near-synonyms like beside / next-to colliding).
SPATIAL_RELATIONS = [
    "left of", "right of", "above", "below", "in front of", "behind",
    "on", "under", "inside", "near", "far from",
]
# common VSR phrasings -> canonical whitelist label
SPATIAL_SYNONYMS = {
    "to the left of": "left of", "at the left side of": "left of",
    "to the right of": "right of", "at the right side of": "right of",
    "on top of": "on", "beneath": "under", "below": "below", "over": "above",
    "in front of": "in front of", "ahead of": "in front of",
    "at the back of": "behind", "close to": "near", "next to": "near",
    "far away from": "far from", "within": "inside", "in": "inside",
}

TASKS = ("color", "counting", "spatial", "attr")


@dataclass
class SlotItem:
    """One labelled slot from a source dataset (single object or single pair)."""
    image: Any            # PIL.Image (already resolved) or a path the writer loads
    obj: str              # slot object label; for spatial, the pair tag "o1|o2"
    value: str            # v_img: the image-true value, as a string
    klass: str            # held-out-class key (usually == value)
    group: str            # swap-pool key: items sharing `group` are swap candidates
    task: str


@dataclass
class UnifiedExample:
    image_1: Any
    image_2: Any
    caption_1: str
    caption_2: str
    key_1: Dict[str, str]
    key_2: Dict[str, str]
    task: str
    klass: str
    id: int
    has_swap: bool = True
    caption_source: str = "human"


# ---- natural text channel (states the slot value) --------------------------
def caption_for(task: str, obj: str, value: str) -> str:
    if task == "color":
        return f"a {value} {obj}"
    if task == "counting":
        unit = obj if obj.endswith("s") else obj + "s"
        return f"there are {value} {unit}"
    if task == "spatial":
        if "|" in obj:
            o1, o2 = obj.split("|", 1)
        else:
            o1, o2 = obj, "object"
        return f"the {o1} is {value} the {o2}"
    return f"the {obj} is {value}"          # attr / default


def caption_from_key(task: str, key: Dict[str, str]) -> str:
    return "; ".join(caption_for(task, o, key[o]) for o in sorted(key))


# ---- conflict value selection ----------------------------------------------
def conflict_value(task: str, v: str, vocab: List[str], rng: random.Random) -> Optional[str]:
    """A different, valid value for the same slot. Counting prefers +/-1..2 (a
    hard, realistic miscount) while staying inside the allowed vocab."""
    if task == "counting":
        try:
            n = int(v)
        except (ValueError, TypeError):
            return None
        near = [str(n + d) for d in (-2, -1, 1, 2) if n + d >= 0]
        near = [c for c in near if c in vocab]
        pool = near or [c for c in vocab if c != v]
        return rng.choice(pool) if pool else None
    others = [x for x in vocab if x != v]
    return rng.choice(others) if others else None


# ---- build examples (with swap matching) -----------------------------------
def build_examples(items: List[SlotItem], task: str, vocab: List[str],
                   rng: random.Random, require_swap: bool = True,
                   start_id: int = 0) -> List[UnifiedExample]:
    """Pair each item with a same-`group` item of a different value (the swap
    image). With require_swap=True (default, and the only fully-supported path),
    image_2 is always a real image so the agree/conflict pairs and the VGM swap
    are well-formed exactly as in ColorSwap. require_swap=False keeps image_2 ==
    image_1 and only the image_1-side conflict is meaningful -- VGM and the
    swap-test must skip has_swap=False rows (left for a future build_specs
    per-row condition filter)."""
    by_group: Dict[str, List[SlotItem]] = {}
    for it in items:
        by_group.setdefault(it.group, []).append(it)

    out: List[UnifiedExample] = []
    eid = start_id
    for it in items:
        cands = [o for o in by_group.get(it.group, []) if o.value != it.value]
        if cands:
            swap = rng.choice(cands)
            img2, v2, has_swap = swap.image, swap.value, True
        else:
            if require_swap:
                continue
            v2 = conflict_value(task, it.value, vocab, rng)
            if v2 is None:
                continue
            img2, has_swap = it.image, False
        if v2 == it.value:
            continue
        key1 = {it.obj: it.value}
        key2 = {it.obj: v2}
        out.append(UnifiedExample(
            image_1=it.image, image_2=img2,
            caption_1=caption_from_key(task, key1),
            caption_2=caption_from_key(task, key2),
            key_1=key1, key_2=key2, task=task, klass=it.klass,
            id=eid, has_swap=has_swap))
        eid += 1
    return out


# ---- class-level split (compositional OOD) ---------------------------------
def split_by_class(items: List[SlotItem], heldout: set, id_frac: float,
                   rng: random.Random) -> Dict[str, List[SlotItem]]:
    """Held-out CLASSES (some colors / counts / relations / attributes) go to an
    OOD test set; the rest are split into train and an in-distribution test."""
    seen, ood = [], []
    for it in items:
        (ood if it.klass in heldout else seen).append(it)
    rng.shuffle(seen)
    n_test = int(round(len(seen) * id_frac))
    n_test = min(max(n_test, 1 if seen else 0), len(seen))
    return {"train": seen[n_test:], "test_id": seen[:n_test], "test_ood": ood}


def values_of(items: List[SlotItem]) -> List[str]:
    return sorted({it.value for it in items})
