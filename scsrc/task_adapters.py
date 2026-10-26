"""Adapters: raw dataset rows -> SlotItem stream (see tasks.py).

These run inside scripts/prepare_task.py on the LOGIN node (the cluster compute
nodes have no internet and the image bytes live on HF / the COCO server). The
pure parsing helpers (object extraction, relation canonicalization) are unit
tested in tests/test_tasks.py on synthetic strings; the dataset iteration itself
is only exercised on the real data during prepare.
"""
from __future__ import annotations
import re
from typing import Iterator, Optional

from .tasks import SlotItem, NUMBER_WORDS, SPATIAL_RELATIONS, SPATIAL_SYNONYMS

# ---- helpers ----------------------------------------------------------------
_PLURAL_KEEP = {"glasses", "scissors", "jeans", "pants"}


def singularize(word: str) -> str:
    w = word.strip().lower()
    if w in _PLURAL_KEEP or not w.endswith("s"):
        return w
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes") or w.endswith("zes"):
        return w[:-2]
    return w[:-1]


_HOWMANY = re.compile(
    r"how many\s+(.+?)\s*(?:are there|are pictured|are in|are shown|are|is|do|does|can|"
    r"in the image|in the picture)\b", re.I)
_HOWMANY_FALLBACK = re.compile(r"how many\s+(.+?)\s*[\?\.]?$", re.I)


def count_object(question: str) -> Optional[str]:
    """'How many men are there?' -> 'man'. Returns a singular head noun or None."""
    m = _HOWMANY.search(question) or _HOWMANY_FALLBACK.search(question)
    if not m:
        return None
    phrase = re.sub(r"[^a-z\s]", "", m.group(1).lower()).strip()
    if not phrase:
        return None
    head = phrase.split()[-1]          # last word of the noun phrase
    return singularize(head) or None


def norm_count(answer) -> Optional[str]:
    a = str(answer).strip().lower()
    if a.isdigit():
        return a
    return NUMBER_WORDS.get(a)


def canon_relation(rel: str) -> Optional[str]:
    r = rel.strip().lower()
    if r in SPATIAL_RELATIONS:
        return r
    return SPATIAL_SYNONYMS.get(r)


# VSR template: "The OBJ1 (is) <relation> the OBJ2." Use the known `relation`
# field to split the caption around it.
def parse_vsr_pair(caption: str, relation: str):
    cap = caption.strip().rstrip(".")
    low = cap.lower()
    rel = relation.strip().lower()
    idx = low.find(rel)
    if idx < 0:
        return None
    left = cap[:idx].strip()
    right = cap[idx + len(rel):].strip()
    o1 = re.sub(r"^the\s+", "", left, flags=re.I).strip().lower()
    o1 = re.sub(r"\s+is$", "", o1).strip()
    o2 = re.sub(r"^the\s+", "", right, flags=re.I).strip().lower()
    if not o1 or not o2:
        return None
    return o1, o2


# ---- counting: TallyQA -----------------------------------------------------
def iter_tallyqa(ds, simple_only: bool = True, min_count: int = 1,
                 max_count: int = 12) -> Iterator[SlotItem]:
    """vikhyatk/tallyqa-test rows: {image, qa:[{question,answer,is_simple,...}]}."""
    for row in ds:
        img = row.get("image")
        for qa in row.get("qa", []):
            if simple_only and not qa.get("is_simple", False):
                continue
            obj = count_object(qa.get("question", ""))
            val = norm_count(qa.get("answer"))
            if obj is None or val is None:
                continue
            if not (min_count <= int(val) <= max_count):
                continue
            yield SlotItem(image=img, obj=obj, value=val, klass=val,
                           group=obj, task="counting")


# ---- counting OOD eval: CountBenchQA ---------------------------------------
def iter_countbench(ds, max_count: int = 12) -> Iterator[SlotItem]:
    """vikhyatk/CountBenchQA rows: {image, question, number}. Small (~491);
    intended as a clean cross-dataset counting eval, not for training."""
    for row in ds:
        obj = count_object(row.get("question", ""))
        num = row.get("number")
        if obj is None or num is None:
            continue
        val = str(int(num))
        if not (0 <= int(val) <= max_count):
            continue
        yield SlotItem(image=row.get("image"), obj=obj, value=val, klass=val,
                       group=obj, task="counting")


# ---- spatial: VSR ----------------------------------------------------------
def iter_vsr(rows, image_loader) -> Iterator[SlotItem]:
    """VSR jsonl rows: {image, image_link, caption, label, relation}. Only
    label==1 (the relation is TRUE of the image -> known v_img). `image_loader`
    receives the whole row (filename + COCO image_link) and returns a PIL image,
    resolved on the login node from a local COCO mirror or by download."""
    for row in rows:
        if int(row.get("label", 0)) != 1:
            continue
        rel = canon_relation(row.get("relation", ""))
        if rel is None:
            continue
        pair = parse_vsr_pair(row.get("caption", ""), row.get("relation", ""))
        if pair is None:
            continue
        o1, o2 = pair
        img = image_loader(row)
        if img is None:
            continue
        tag = f"{o1}|{o2}"
        yield SlotItem(image=img, obj=tag, value=rel, klass=rel,
                       group=tag, task="spatial")


# ---- attributes: GQA -------------------------------------------------------
_GQA_PATTERNS = [
    (re.compile(r"what color is (?:the |a |an )?(.+?)\s*[\?\.]?$", re.I), "color"),
    (re.compile(r"what is the color of (?:the |a |an )?(.+?)\s*[\?\.]?$", re.I), "color"),
    (re.compile(r"what (?:material|is .*made of).*?(?:the |a |an )?(.+?)\s*[\?\.]?$", re.I), "material"),
    (re.compile(r"what shape is (?:the |a |an )?(.+?)\s*[\?\.]?$", re.I), "shape"),
]


def gqa_attr(question: str):
    """Return (object, attr_type) for an attribute question, else None.
    Preliminary: GQA phrasings are varied; only clean attribute templates kept."""
    q = question.strip()
    for pat, attr in _GQA_PATTERNS:
        m = pat.search(q)
        if m:
            obj = re.sub(r"[^a-z\s]", "", m.group(1).lower()).strip()
            obj = re.sub(r"^(that|this|large|small|the)\s+", "", obj).strip()
            if obj:
                return obj.split()[-1], attr
    return None


def iter_gqa(ds, by_attr: str = "color") -> Iterator[SlotItem]:
    """lmms-lab/GQA QA rows: {question, answer, image, ...}. Keeps attribute
    questions of one type (default color) so values share a vocabulary; klass is
    keyed by attr:value so you can hold out specific attribute values."""
    for row in ds:
        info = gqa_attr(row.get("question", ""))
        if info is None:
            continue
        obj, attr = info
        if attr != by_attr:
            continue
        val = re.sub(r"[^a-z]", "", str(row.get("answer", "")).lower())
        if not val:
            continue
        yield SlotItem(image=row.get("image"), obj=obj, value=val,
                       klass=f"{attr}:{val}", group=f"{attr}:{obj}", task="attr")
