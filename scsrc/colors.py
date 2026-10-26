"""Color/object parsing for ColorSwap captions.

Because caption_i is the aligned description of image_i, the ground-truth colors
of image_i ARE the parse of caption_i -- no vision labeller needed. That is what
makes ColorSwap a clean conflict generator with exact, string-matchable truth.
"""
import re

COLORS = {
    "red","orange","yellow","green","blue","purple","pink","brown","black",
    "white","gray","grey","cyan","magenta","violet","gold","silver","beige",
    "tan","turquoise","teal","maroon","navy","olive","lime","indigo","crimson",
    "scarlet","azure",
}
FUNCTION = {
    "a","an","the","and","or","with","in","on","of","at","to","is","are","be",
    "being","emitting","wearing","holding","carrying","near","next","by","while",
    "that","this","who","someone","something","person","behind","front",
    "against","over","under",
}
SYNONYMS = {"grey": "gray"}  # spelling only; fancy colors stay distinct


def normalize_color(c):
    """Map a color mention to its canonical base color. Handles synonyms AND
    modifier compounds: 'slate gray'->'gray', 'lemon yellow'->'yellow',
    'yellowish'->'yellow', 'grayish'->'gray'."""
    c = (c or "").strip().lower()
    if c in SYNONYMS:
        return SYNONYMS[c]
    if c in COLORS:
        return c
    words = c.replace("-", " ").split()
    for w in reversed(words):                      # head color usually last
        w2 = SYNONYMS.get(w, w)
        if w2 in COLORS:
            return w2
        if w.endswith("ish") and SYNONYMS.get(w[:-3], w[:-3]) in COLORS:
            return SYNONYMS.get(w[:-3], w[:-3])
    return c


_WORD = re.compile(r"[a-zA-Z]+")


def parse_colors(caption: str) -> dict:
    """Return {object: color}. Head noun = next token after a color that is
    neither a color nor a function word."""
    toks = _WORD.findall(caption.lower())
    pairs = {}
    i = 0
    while i < len(toks):
        if toks[i] in COLORS:
            color = toks[i]
            j = i + 1
            while j < len(toks) and (toks[j] in COLORS or toks[j] in FUNCTION):
                j += 1
            if j < len(toks):
                obj = toks[j]
                pairs.setdefault(obj, normalize_color(color))
            i = j + 1
        else:
            i += 1
    return pairs


def valid_swap(cap1: str, cap2: str):
    """(ok, key1, key2). ok = same object set, every object's color moved, >=1 obj."""
    k1, k2 = parse_colors(cap1), parse_colors(cap2)
    if not k1 or not k2:
        return False, k1, k2
    if set(k1) != set(k2):
        return False, k1, k2
    if all(k1[o] == k2[o] for o in k1):
        return False, k1, k2
    return True, k1, k2
