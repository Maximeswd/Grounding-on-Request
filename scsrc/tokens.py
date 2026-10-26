"""Neologism specification + tokenizer surgery.

A neologism with N tokens expands to a contiguous block <name_0>...<name_{N-1}>.
N is configurable and EQUAL across neologisms by default; the sweep over N is
driven entirely by config (token_count), so the same code runs every point of
the capacity sweep.
"""
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class NeoSpec:
    name: str            # e.g. "look_ctrl"
    role: str            # "ctrl" | "gen" | "fuse"
    source: str          # "image" | "text" | "relation" | "fuse"
    init_word: str       # word to mean-init from (skill-neologism style)


# The three arms share operator semantics but live on separate tokens so we can
# TEST (not assume) whether control and generation induce the same behavior.
def default_specs() -> Dict[str, List[NeoSpec]]:
    return {
        "ctrl": [
            NeoSpec("look_ctrl", "ctrl", "image", "image"),
            NeoSpec("read_ctrl", "ctrl", "text", "text"),
            NeoSpec("compare_ctrl", "ctrl", "relation", "compare"),
        ],
        "gen": [
            NeoSpec("look_gen", "gen", "image", "image"),
            NeoSpec("read_gen", "gen", "text", "text"),
            NeoSpec("compare_gen", "gen", "relation", "compare"),
        ],
        # CITATION delimiters for ctrl_style=blocks: the emitted word is just
        # <look> / <read> (no _gen suffix) -- it is the citation mark the reader
        # sees, always ONE token, opening and closing the grounded span.
        "cite": [
            NeoSpec("compare", "cite", "relation", "compare"),
            NeoSpec("look", "cite", "image", "image"),
            NeoSpec("read", "cite", "text", "text"),
        ],
        "fuse": [  # collapse control arm (single unattributed reading)
            NeoSpec("fuse", "fuse", "fuse", "describe"),
        ],
        "pt": [  # prompt-tuning baseline: learned soft PREFIX (not addressable/inline)
            NeoSpec("pt_look", "pt", "image", "image"),
            # 2026-09-29: SC_PT_READ_INIT=caption (train.py --pt_read_init caption) matches the prefix's caption row to the
            # neologism's initialization word ("caption"); default "text" is the banked behaviour
            NeoSpec("pt_read", "pt", "text", __import__("os").environ.get("SC_PT_READ_INIT", "text")),
            NeoSpec("pt_compare", "pt", "relation", "compare"),
        ],
    }


# ---- FACTORED source x operator algebra (CLEVR-3channel testbed) -----------
# The reviewer-designated must-survive contrast: does a source and a skill live
# on SEPARATE, reusable tokens (<look><count>) or must every cell get its own
# fused token (<look_count>)? Both are param-matched at the CELL level -- with
# per-slot budget L, a factored cell reads L(source)+L(operator)=2L rows and a
# bundled cell reads a single 2L-row token -- but factored shares 3L+4L=7L rows
# across the 12 cells while bundled needs 12*2L=24L, and ONLY factored can be
# asked a held-out source x operator cell it never saw (zero-shot composition).
CLEVR3_SOURCES = {"look": "image", "read": "caption", "ocr": "screen"}


def c3_sources():
    """Active Ground-CF sources (2026-09-22, STATUS v1.94): SC_C3_SOURCES="look,read" is the two-modality protocol
    (image + caption; no on-screen-text block, rows or items). Default (v1.132): two-source.
    Read at call time so train.py / evaluators can set it from the run's configuration after import."""
    import os as _os
    # v1.132 (author: "ocr should be skipped everywhere"): the DEFAULT is now the two-source protocol; the legacy
    # three-source setup needs SC_C3_SOURCES=look,read,ocr explicitly
    want = [x for x in _os.environ.get("SC_C3_SOURCES", "look,read").split(",") if x]
    return tuple(s for s in CLEVR3_SOURCES if s in want)


def c3_init_words():
    """Initialization words of the Ground-CF controls (2026-10-02): default = the source nouns (image / caption);
    SC_C3_INIT_WORDS="look:look,read:read" (train.py --init_words) overrides them for the initialization ablation. The
    prefix branch of train.py reads the same override, so both embedding interfaces start from identical rows."""
    import os as _os
    ov = dict(kv.split(":", 1) for kv in _os.environ.get("SC_C3_INIT_WORDS", "").split(",") if ":" in kv)
    return {s: ov.get(s, CLEVR3_SOURCES[s]) for s in c3_sources()}


def c3_source_words():
    return c3_init_words()
CLEVR3_OPERATORS = {"count": "count", "color": "color",
                    "object": "shape", "spatial": "position"}


def clevr3_specs(mode: str, held_cells=()):
    """NeoSpecs for the CLEVR-3channel factored testbed.
      factored: one token per SOURCE + one per OPERATOR (shared, reusable).
      bundled : one FUSED token per (source, operator) cell, EXCLUDING held
                cells (a bundled model literally has no token for a held cell,
                which is why only the factored model gets a zero-shot test).
      srconly : ONLY the three source tokens -- the simplicity baseline ("just
                use <look>; the explicit question carries the skill"). Trained
                with explicit questions; it CANNOT work with generic ones.
    held_cells: iterable of (source, operator) tuples held out of TRAINING."""
    held = {tuple(c) for c in held_cells}
    src_specs = [NeoSpec(f"src_{s}", "ctrl", "image" if s == "look" else "text", w)
                 for s, w in c3_source_words().items()]
    if mode == "srconly":
        return src_specs
    if mode == "factored":
        return src_specs + [NeoSpec(f"op_{o}", "ctrl", "relation", w)
                            for o, w in CLEVR3_OPERATORS.items()]
    if mode == "bundled":
        return [NeoSpec(f"fx_{s}_{o}", "ctrl",
                        "image" if s == "look" else "text", CLEVR3_SOURCES[s])
                for s in c3_sources() for o in CLEVR3_OPERATORS
                if (s, o) not in held]
    raise ValueError(f"unknown clevr3 mode {mode}")


def clevr3_trigger(mode: str, source: str, operator: str, n: int) -> str:
    """Composed trigger for a (source, operator) cell. factored concatenates the
    two shared blocks (each n rows); bundled emits the single fused block (2n
    rows) so the two are matched on rows-read-at-the-cell; srconly reads only
    the source block (n rows -- deliberately smaller, it's the simplicity arm)."""
    if mode == "factored":
        return trigger(f"src_{source}", n) + trigger(f"op_{operator}", n)
    if mode == "srconly":
        return trigger(f"src_{source}", n)
    return trigger(f"fx_{source}_{operator}", 2 * n)


def token_strings(name: str, n: int) -> List[str]:
    """n==1: the neologism IS one literal new word, '<look_ctrl>' -- a single
    added id regardless of its display string (added special tokens are never
    subword-split). n>1 keeps the suffixed block form for capacity ablations."""
    if n == 1:
        return [f"<{name}>"]
    return [f"<{name}_{i}>" for i in range(n)]


def trigger(name: str, n: int) -> str:
    """Contiguous block the model reads/emits as one operator."""
    return "".join(token_strings(name, n))


def add_neologisms(tokenizer, specs: List[NeoSpec], n_tokens: int,
                   n_by_name=None):
    """Add all tokens as additional special tokens. Returns
    (name -> [ids], all_new_ids). Caller must resize the model embeddings.
    n_by_name: optional {spec_name: n} override -- used by ctrl_style=blocks so
    the emitted citation delimiters stay ONE token (compact, readable output)
    while the prompt-side control word keeps its full capacity L."""
    all_strings, name_to_strings = [], {}
    for s in specs:
        ts = token_strings(s.name, (n_by_name or {}).get(s.name, n_tokens))
        name_to_strings[s.name] = ts
        all_strings.extend(ts)
    tokenizer.add_special_tokens({"additional_special_tokens": all_strings})
    name_to_ids = {n: tokenizer.convert_tokens_to_ids(ts)
                   for n, ts in name_to_strings.items()}
    all_new_ids = [i for ids in name_to_ids.values() for i in ids]
    return name_to_ids, all_new_ids
