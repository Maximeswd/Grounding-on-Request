"""Exact-match metrics over parsed {object: color} keys.

All signature metrics are string comparisons against known keys -- no LLM judge,
no ambiguity. Primary metrics are collapse / separation / laundering, NOT raw
accuracy (accuracy is for agree-pairs sanity and the unimodal legibility gate).
"""
from __future__ import annotations
import re
from typing import Optional

from .colors import COLORS, normalize_color

_COLOR_ALT = "|".join(sorted(COLORS, key=len, reverse=True))
import re as _re
_PRE = _re.compile(r"\b(" + _COLOR_ALT + r")\s+(?:a |an |the )?([a-z]+)")
_POST = _re.compile(r"([a-z]+)\s+(?:is|are|appears?(?: to be)?|looks?|seems?)\s+(?:a |an |the )?(" + _COLOR_ALT + r")")
_FUNC = {"a","an","the","and","or","with","in","on","of","is","are","it","that","this","image","caption","color","colour","background","scene","answer","answers","response","result","value","values","final","text","source"}


# ---- parsing model outputs --------------------------------------------------
_OBJ_COLOR = re.compile(r"([a-zA-Z]+)\s*[:=]\s*([a-zA-Z]+)")


_OBJ_COLOR_MW = _re.compile(r"([a-z][a-z]*)\s*:\s*([a-z][a-z -]*[a-z])", _re.I)


def parse_structured(text: str) -> dict:
    """Parse 'umbrella: yellow, dress: slate gray' -> {umbrella: yellow, dress: gray}.
    Values may be MULTI-WORD compounds; kept only if they normalize to a known
    color ('slate gray' -> 'gray'; 'lemon yellow' -> 'yellow')."""
    out = {}
    for obj, col in _OBJ_COLOR_MW.findall(text.lower()):
        # value runs to the next comma/segment; trim trailing connective words
        c = normalize_color(col.split(",")[0].strip())
        if c in COLORS:
            out.setdefault(obj.strip(), c)
    return out


def parse_relation(text: str) -> Optional[str]:
    """Map free text to {'agree','conflict'} if determinable, else None."""
    t = text.lower()
    has_conf = any(w in t for w in ("conflict", "disagree", "differ", "mismatch",
                                    "contradict", "inconsistent", "do not match",
                                    "don't match", "not align", "don't align",
                                    "misalign", "no alignment"))
    has_agree = any(w in t for w in ("agree", "consistent", "match", "same",
                                     "align"))
    if has_conf and not has_agree:
        return "conflict"
    if has_agree and not has_conf:
        return "agree"
    if has_conf and has_agree:
        return "conflict"  # explicit conflict wording dominates
    return None


def parse_pred(text: str) -> dict:
    """Prose-tolerant {object: color} extraction for MODEL outputs. Handles
    'umbrella: yellow', 'the umbrella is yellow', and 'a yellow umbrella'.
    Colon form is authoritative if present."""
    t = text.lower()
    out = {}
    for obj, col in _POST.findall(t):          # 'umbrella is yellow'
        if obj not in _FUNC and col in COLORS:
            out.setdefault(obj, normalize_color(col))
    for col, obj in _PRE.findall(t):           # 'yellow umbrella'
        if obj not in COLORS and obj not in _FUNC:
            out.setdefault(obj, normalize_color(col))
    out.update(parse_structured(text))         # 'umbrella: yellow' wins
    return out


from .tasks import SPATIAL_RELATIONS, NUMBER_WORDS

_INT = _re.compile(r"\b(\d+)\b")
_NUMWORD = _re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b")
_REL_ALT = "|".join(sorted([r for r in SPATIAL_RELATIONS], key=len, reverse=True))
_REL = _re.compile(r"\b(" + _REL_ALT + r")\b")
_IS_ARE = re.compile(r"([A-Za-z][A-Za-z ]{0,30}?)\s+(?:is|are|was|were)\s+(?:a |an |the )?(?:vibrant |bright |deep |dark |light |pale )?([A-Za-z][A-Za-z-]*)")
_FILLER = {"very", "quite", "also", "not", "no"}
_OBJ_VAL = re.compile(r"([a-zA-Z][a-zA-Z| ]*?)\s*[:=]\s*([a-zA-Z0-9 ]+?)(?:[,;\n]|$)")


def parse_structured_any(text: str) -> dict:
    """{object: value} for ANY value type -- used for non-color tasks where the
    value is an int / relation / attribute word. Colon form only (authoritative)."""
    out = {}
    for obj, val in _OBJ_VAL.findall(text):
        o = obj.strip().lower()
        o = o if "|" in o else (o.split()[-1] if o else "")
        v = val.strip().lower()
        if o and v and o not in _FUNC:
            out.setdefault(o, v)
    # prose fallback so unformatted answers ("The sky is orange") are not
    # auto-zero: last noun before is/are, first content word after.
    if not out:
        for obj, val in _IS_ARE.findall(text):
            o = obj.strip().lower().split()[-1]
            v = val.strip().lower().split()[0] if val.strip() else ""
            if o and v and o not in _FUNC and v not in _FILLER:
                out.setdefault(o, v)
    return out


def first_value(text: str, task: str):
    """Single-value answer (probe / look-block), task-aware. Returns a string."""
    if task == "color":
        return first_color(text)
    if task == "counting":
        m = _INT.search(text)
        if m:
            return m.group(1)
        m = _NUMWORD.search(text.lower())
        return NUMBER_WORDS[m.group(1)] if m else None
    if task == "spatial":
        m = _REL.search(text.lower())
        return m.group(1) if m else None
    # attr: value after a colon, else first content token
    m = _OBJ_VAL.search(text)
    if m:
        return m.group(2).strip().lower()
    toks = [t for t in re.findall(r"[a-z]+", text.lower()) if t not in _FUNC]
    return toks[0] if toks else None


_SPATIAL_SYN = {"on top of": "on", "atop": "on", "beneath": "under",
                "underneath": "under", "left": "left of", "right": "right of"}


def normalize_value(v: str, task: str) -> str:
    """Task-aware value equivalence so correct answers in a different surface
    form are not scored wrong: counting 'four' == '4' (incl. inside phrases,
    '4 dogs' -> '4'); spatial 'to the left of' == 'left of' plus common
    synonyms; color handled by normalize_color."""
    v = (v or "").strip().lower()
    if task == "counting":
        if v in NUMBER_WORDS:
            return NUMBER_WORDS[v]
        m = _INT.search(v)
        if m:
            return m.group(1)
        m = _NUMWORD.search(v)
        if m:
            return NUMBER_WORDS[m.group(1)]
        return v
    if task == "spatial":
        for pre in ("to the ", "the "):
            if v.startswith(pre):
                v = v[len(pre):]
        return _SPATIAL_SYN.get(v, v)
    if task == "color":
        return normalize_color(v)
    return v


def _bare_value(text: str, task: str):
    """Last-resort value extraction for SINGLE-SLOT tasks (counting, spatial):
    models often answer 'final answer: 3' or 'There are 3 dogs' with no
    'object:' key, which the structured parser cannot represent. Returns the
    task-typed value found in the tail of the text, or None."""
    t = (text or "").lower()
    # prefer the segment after the LAST 'final answer:' anchor if present
    if "final answer:" in t:
        t = t.rsplit("final answer:", 1)[1]
    if task == "counting":
        m = list(_INT.finditer(t)) or list(_NUMWORD.finditer(t))
        if m:
            g = m[-1].group(1) if m[-1].re is _INT else NUMBER_WORDS[m[-1].group(1)]
            return g
        return None
    if task == "spatial":
        # 'to the X of' phrasings first (they contain shorter relations as substrings)
        for pre in ("to the left of", "to the right of", "on top of", "underneath", "beneath"):
            if pre in t:
                return normalize_value(pre, "spatial")
        m = list(_REL.finditer(t))   # word-boundary, longest-first alternation
        if m:
            return m[-1].group(1)
    return None


def parse_pred_task(text: str, task: str) -> dict:
    """{object: value} for MODEL outputs, task-aware, with value equivalence
    (colors: 'slate gray'->'gray'; counting: 'four'->'4'; spatial:
    'to the left of'->'left of'). For single-slot tasks, a keyless bare value
    is returned under the wildcard key '*' (matches any single-key gold)."""
    d = parse_pred(text) if task == "color" else parse_structured_any(text)
    out = {o: normalize_value(v, task) for o, v in d.items()}
    if not out and task in ("counting", "spatial"):
        v = _bare_value(text, task)
        if v is not None:
            out = {"*": normalize_value(v, task)}
    return out


def first_color(text: str):
    """Single-color answer (probe mode). First color word in the text, or None."""
    m = _re.search(r"\b(" + _COLOR_ALT + r")\b", text.lower())
    return normalize_color(m.group(1)) if m else None


# ---- per-example judgements -------------------------------------------------
def key_match(pred: dict, gold: dict) -> bool:
    """Match on every gold object (plural-tolerant object lookup)."""
    if not gold:
        return False
    return all(_lookup(pred, o) == c for o, c in gold.items())


def slot_scores(pred: dict, gold: dict, other: dict) -> dict:
    """Graded scores that ignore EXTRA slots the model volunteers:
    containment  -- every gold (obj, val) pair appears in pred (extras ignored)
    jaccard      -- |pred ∩ gold| / |pred ∪ gold| over (obj, val) pairs
    intrusion    -- fraction of gold objects whose predicted value equals the
                    OTHER source's value where the two sources DISAGREE
                    (0 by construction in the agree condition -- discrimination
                    lives in the conflict condition only)."""
    gp, pp = set(gold.items()), set(pred.items())
    containment = float(gp.issubset(pp)) if gold else 0.0
    jaccard = (len(gp & pp) / len(gp | pp)) if (gp | pp) else 0.0
    intr = [int(pred.get(o) == other.get(o)) for o in gold
            if o in other and other[o] != gold[o]]
    return {"containment": containment, "jaccard": round(jaccard, 3),
            "intrusion": (sum(intr) / len(intr)) if intr else 0.0}


def _obj_variants(o):
    """Singular/plural variants, plus segment fallbacks for pair keys:
    gold 'elephant|car' also matches pred keys 'car' or 'elephant' (the colon
    parser or prose fallback may capture only one side of the pair)."""
    v = {o}
    if "|" in o:
        for seg in o.split("|"):
            seg = seg.strip()
            if seg:
                v |= _obj_variants(seg)
        return v
    if o.endswith("s") and not o.endswith("ss") and len(o) > 3:
        v.add(o[:-1])
    else:
        v.add(o + "s")
    return v


def _lookup(pred: dict, obj):
    for cand in _obj_variants(obj):
        if cand in pred:
            return pred[cand]
    return None


def object_accuracy(pred: dict, gold: dict) -> float:
    if pred and "*" in pred and len(gold) == 1:
        gv = next(iter(gold.values()))
        return 1.0 if pred["*"] == (gv or "").strip().lower() else 0.0
    """Value match per gold object, tolerant to simple plural mismatches on
    the OBJECT name ('curtains: gray' satisfies gold 'curtain: gray')."""
    if not gold:
        return 0.0
    return sum(_lookup(pred, o) == c for o, c in gold.items()) / len(gold)


def separation_correct(pred_img, pred_txt, gold_img, gold_txt) -> bool:
    """Did the model report BOTH sources correctly AND separately?"""
    return key_match(pred_img, gold_img) and key_match(pred_txt, gold_txt)


def is_collapse(pred_img, pred_txt, pred_rel, gold_img, gold_txt) -> bool:
    """On a conflict instance, collapse = failing to preserve the disagreement:
    either the relation is reported as agreement, or the two source readings are
    rendered identical."""
    if pred_rel == "agree":
        return True
    if pred_img and pred_txt and pred_img == pred_txt:
        return True
    return False


def laundering_rate(pred_img, gold_img, gold_txt) -> Optional[float]:
    """Fraction of objects where the color attributed to the IMAGE is actually
    the TEXT's color (and the two differ) -- i.e. text laundered as vision.
    Returns None if undefined (no genuinely conflicting objects)."""
    conflicting = [o for o in gold_img if o in gold_txt and gold_img[o] != gold_txt[o]]
    if not conflicting:
        return None
    laundered = sum(
        pred_img.get(o) == gold_txt[o] and pred_img.get(o) != gold_img[o]
        for o in conflicting
    )
    return laundered / len(conflicting)


# ---- corpus-level conflict detection ---------------------------------------
def conflict_prf(rel_preds, rel_golds):
    """Precision/recall/F1 for the positive class 'conflict'. None preds count as
    wrong. Returns dict."""
    tp = fp = fn = tn = 0
    for p, g in zip(rel_preds, rel_golds):
        pos_p, pos_g = (p == "conflict"), (g == "conflict")
        if pos_g and pos_p:
            tp += 1
        elif pos_g and not pos_p:
            fn += 1
        elif (not pos_g) and pos_p:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"conflict_precision": prec, "conflict_recall": rec, "conflict_f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def classify_source(parsed, image_v, text_v):
    """grounded (follows image) | laundered (follows text) | other."""
    if parsed is not None and parsed == image_v:
        return "grounded"
    if parsed is not None and parsed == text_v:
        return "laundered"
    return "other"


def parse_final_answer(text: str, task: str) -> dict:
    """Parse the structured tail after the LAST 'final answer:' anchor; fall back
    to parsing the whole text (prose-tolerant) if no anchor is present."""
    low = text.lower()
    anchor = "final answer:"
    seg = low.rsplit(anchor, 1)[1] if anchor in low else text
    return parse_pred_task(seg, task)
