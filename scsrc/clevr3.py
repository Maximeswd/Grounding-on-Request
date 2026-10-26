"""CLEVR-3channel factored dataset: <source><operator> composition training.

Each scene yields items for several skills; each item states three channel
values (look=true scene value, read=caption value, ocr=extracted-text value)
that can conflict. A training item is (row, source): the composed token
<source><operator> must route to `source` and answer with `operator`'s skill,
so the gold is channels[source]. Interface-compatible with ObjectiveDataset:
__getitem__ -> {"op", "sft"} (SFT-only for the factored testbed).

Two modes share this class:
  factored: token = <src_{source}><op_{operator}>  (shared, reusable rows)
  bundled : token = <fx_{source}_{operator}>        (one fused token per cell)
Both are param-matched at the cell (see tokens.clevr3_trigger). held_cells are
excluded from TRAINING so the factored model can be tested zero-shot on them.

Slot-typed negatives (factored only) corrupt ONE slot -- the source OR the
operator -- with a filler word and supervise an unknown-operator refusal, so we
can show each slot independently carries meaning (a source filler must break
modality routing; an operator filler must break the skill), not just the pair.
"""
import itertools
import json
import os
import random

from PIL import Image

from .collate import _encode, _prompt_len, NEG_SLOT_WORDS
from .tokens import clevr3_trigger, CLEVR3_SOURCES, CLEVR3_OPERATORS, c3_sources

_PERMS = list(itertools.permutations(("read", "ocr", "look")))   # legacy three-source protocol only
_ORDER_BIT = None   # set per training row by Clevr3Dataset.__getitem__ (v1.132); None at evaluation
_POS_BIT = None     # v1.167 (--gcf_word_pos mix): per training row, 1 = the word's sentence BEFORE the evidence; None otherwise


_EVAL_RANK = None


def _eval_rank(item):
    """Within-skill rank of an evaluation item in its split file (file order); crc32 of the id for items that are in
    no test file (e.g. absence rows). Edited contexts and image twins keep the id, so they share the rank."""
    global _EVAL_RANK
    if _EVAL_RANK is None:
        import collections as _c
        _EVAL_RANK = {}
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clevr3")
        for sp in ("test_id", "test_ood"):
            fp = os.path.join(d, f"{sp}.jsonl")
            if not os.path.exists(fp):
                continue
            per = _c.Counter()
            for l in open(fp):
                r = json.loads(l)
                _EVAL_RANK.setdefault(r["id"], per[r["skill"]]); per[r["skill"]] += 1
    iid = item.get("id")
    if iid in _EVAL_RANK:
        return _EVAL_RANK[iid]
    import zlib as _z
    return _z.crc32(str(iid).encode()) % 1000003


def eval_design(item, seed, question_style="heldout"):
    """(seed', caption_first) for an EVALUATION prompt (v1.133, author: both orders equally represented at evaluation).
    The test files cycle the four skills, so the old seed-parity rules gave every skill ONE held-out carrier, ONE
    held-out question formulation and a 2:1 order split (ID split). Now, from the item's within-skill rank r:
      caption first = r even;  held-out carrier = (r // 2) % 2;  held-out question = (r // 4) % 2
      (explicit/in-template style: training carrier (r // 2) % 6, training question (r // 12) % 3; generic: (r // 2) % 3)
    -> every skill x request sees all combinations equally, and every interface sees the identical prompt per item.
    Training is untouched (Clevr3Dataset sets _ORDER_BIT and the per-row seed). SC_EVAL_DESIGN=legacy restores the
    pre-v1.133 evaluation prompts; the legacy three-source protocol always keeps them."""
    if _ORDER_BIT is not None or "ocr" in c3_sources() or os.environ.get("SC_EVAL_DESIGN", "balanced") == "legacy":
        return seed, None
    r = _eval_rank(item)
    if os.environ.get("SC_GCF_FORMAT") == "v2":
        # design v2 (2026-09-29, audit item 3): carrier index, question index and evidence order all derive from ONE
        # stable per-item integer, so switching the question pool or the carrier pool changes nothing else
        return r // 2, (r % 2 == 0)
    if question_style == "heldout" or question_style.startswith("tpl:"):   # v1.217: templates reuse the held-out design
        s2 = (r // 2) % 2 + 2 * ((r // 4) % 2)
    elif question_style == "generic":
        s2 = (r // 2) % 3
    else:
        s2 = (r // 2) % 6 + 6 * ((r // 12) % 3)
    return s2, (r % 2 == 0)


def _block_order(seed, cap_first=None):
    """Order of the evidence blocks. Two-source protocol (image + caption; no OCR anywhere):
      evaluation: caption first iff seed % 6 < 3 -- exactly the mapping the old permutation list gave once the OCR
                  block was dropped, so every evaluation prompt is unchanged;
      training  : a per-ROW coin (_ORDER_BIT, crc32 of the row id), identical for the <look> and <read> items of a
                  row and independent of the carrier and question rotation (v1.132, author: "the different orders
                  equally represented during training"). Before v1.132 the training seed was the (row, request) index,
                  so <look> saw the caption first in 2/3 of its items and <read> in 1/3."""
    if "ocr" not in c3_sources():
        if cap_first is None:
            cap_first = (_ORDER_BIT == 0) if _ORDER_BIT is not None else (seed % 6 < 3)
        return ("read", "look") if cap_first else ("look", "read")
    return _PERMS[seed % len(_PERMS)]
_SRC_FILLERS = ("thing", "somehow", "banana", "please", "kindly")
_OP_FILLERS = ("value", "output", "process", "quantity", "attribute")


def _blocks(item):
    # only the ACTIVE sources appear in the prompt (STATUS v1.94: two-modality protocol drops the on-screen block)
    act = c3_sources()
    import os as _os
    b = {
        # v2 explicit format (SC_GCF_FORMAT=v2, 2026-09-29): labelled blocks, caption on its own line
        "read": (f'Caption:\n{caption_text(item)}' if _os.environ.get("SC_GCF_FORMAT") == "v2"
                 else f'Caption: "{caption_text(item)}"'),
        "ocr": f'On-screen text: "{item["ocr_text"]}"',
        "look": None,   # the image itself
    }
    return {k: v for k, v in b.items() if k == "look" or k in act}


# English-instruction parity baseline: same prompt position as the composed
# token, plain words instead -- composition must BEAT saying it in English.
INSTR = {"look": "the image", "read": "the caption", "ocr": "the on-screen text"}

# GENERIC questions (rotated): carry NO skill information, so in the generic
# condition the operator token is the ONLY skill carrier. With explicit
# questions the question text states the skill and <op_*> is redundant --
# Design A: "the question carries the task, the word carries the source".
# The generic condition is the probe that gives Design B's skill tokens a
# non-trivial job (skill selection, real zero-shot composition).
# 2026-09-23 (author, STATUS v1.120): HELD-OUT QUESTION PHRASINGS. Ground-CF has no request-carrier template --
# the word is appended to the item's question -- so "another form of the same question" at evaluation means a
# paraphrase of the question itself, two per skill, never used in training (training uses item["question"]).
# The candidate relations of the spatial question are kept verbatim: they define the answer vocabulary.
# 2026-09-23 (author, STATUS v1.124): QUESTION POOLS. Three training formulations per skill (the original first),
# rotated by item when SC_Q_ROTATE=1 (train.py --q_rotate 1); two held-out formulations that never occur in
# training (question_style="heldout"). The answer space is unchanged; the spatial relation list is kept verbatim
# because those strings are the candidates.
_Q_PAT = {"color": r"What color is the (.+?)\?$", "object": r"What shape is the (.+?) object\?$",
          "count": r"How many shapes are in the image\?$",
          "spatial": r"Is the (.+?) left of, right of, above, or below the (.+?)\?$"}
_TRAIN_QS = {
    "color":   [lambda m: f"What color is the {m.group(1)}?", lambda m: f"Which color is the {m.group(1)}?",
                lambda m: f"What is the color of the {m.group(1)}?"],
    "object":  [lambda m: f"What shape is the {m.group(1)} object?", lambda m: f"Which shape is the {m.group(1)} object?",
                lambda m: f"What is the shape of the {m.group(1)} object?"],
    "count":   [lambda m: "How many shapes are in the image?", lambda m: "How many shapes are shown?",
                lambda m: "What is the number of shapes in the image?"],
    "spatial": [lambda m: f"Is the {m.group(1)} left of, right of, above, or below the {m.group(2)}?",
                lambda m: f"Where is the {m.group(1)} relative to the {m.group(2)}: left of, right of, above, or below?",
                lambda m: f"What is the position of the {m.group(1)} with respect to the {m.group(2)}: left of, right of, above, or below?"],
}
_HELDOUT_QS = {
    "color":   [lambda m: f"What color does the {m.group(1)} have?", lambda m: f"Identify the {m.group(1)}'s color."],
    "object":  [lambda m: f"What shape does the {m.group(1)} object have?", lambda m: f"Identify the {m.group(1)} object's shape."],
    "count":   [lambda m: "Count the shapes in the image.", lambda m: "How many shapes are there in the image?"],
    "spatial": [lambda m: f"With respect to the {m.group(2)}, where is the {m.group(1)} located: left of, right of, above, or below?",
                lambda m: f"Tell me whether the {m.group(1)} is left of, right of, above, or below the {m.group(2)}."],
}


def _qmatch(item):
    import re as _re
    pat = _Q_PAT.get(item.get("skill"))
    return _re.match(pat, item["question"]) if pat else None


def heldout_question(item, seed):
    m = _qmatch(item)
    return _HELDOUT_QS[item["skill"]][(seed // 2) % 2](m) if m else item["question"]   # decoupled from the carrier (seed % 2)


def train_question(item, seed):
    """One of the three training formulations (rotated by item) when SC_Q_ROTATE=1; else the original."""
    import os as _os
    if _os.environ.get("SC_Q_ROTATE", "0") != "1":
        return item["question"]
    m = _qmatch(item)
    return _TRAIN_QS[item["skill"]][(seed // 6) % 3](m) if m else item["question"]   # decoupled from the carrier (seed % 6)


# 2026-09-23 (author, STATUS v1.124): FACT CAPTIONS. The stored caption is the data template
# "The answer to the question is <v>." (prepare_clevr3.py), which makes the textual evidence depend on the
# question's wording. With SC_CAPTION_STYLE=fact (train.py --caption_style fact, restored from config.json by
# evaluate.apply_checkpoint) the caption is rendered as the fact itself, from the question's entities and the
# value in the stored caption -- so channel edits (_edited_item replaces the value in the stored string) and the
# absence rows keep working unchanged. Banked runs keep the template form.
_CAP_PAT = __import__("re").compile(r"^The answer to the question is (.+?)\.$")


def fact_caption(item, caption=None):
    import re as _re
    cap = caption if caption is not None else item["caption"]
    mv = _CAP_PAT.match(cap)
    if not mv:
        return cap                                   # already a fact sentence (or free text): leave it
    v = mv.group(1); m = _qmatch(item); sk = item.get("skill")
    if not m:
        return cap
    if sk in ("color", "object") and v == m.group(1):
        # absence rows (clevr3_absence_hard): the caption names the entity but not the answer
        return f"There is a {v} in the image." if sk == "color" else f"There is a {v} object in the image."
    if sk == "color":
        return f"The {m.group(1)} is {v}."
    if sk == "object":
        art = "an" if v[:1].lower() in "aeiou" else "a"
        return f"The {m.group(1)} object is {art} {v}."
    if sk == "count":
        return f"There is {v} shape in the image." if v == "1" else f"There are {v} shapes in the image."
    if sk == "spatial":
        rel = f"{v} of" if v in ("left", "right") else v
        return f"The {m.group(1)} is {rel} the {m.group(2)}."
    return cap


def caption_text(item):
    import os as _os
    return fact_caption(item) if _os.environ.get("SC_CAPTION_STYLE", "template") == "fact" else item["caption"]


_HO_IV = None            # instructed-carrier variant override while rendering a held-out prompt (see clevr3_user_content)

# 2026-09-23 (author, STATUS v1.123): CARRIER TEMPLATES FOR THE WORD ON GROUND-CF, after Hewitt et al. (neologism
# learning, Tables 10/11). Training rotates six positive templates by item; evaluation uses two the word never saw.
# Active only for runs trained with --neo_template hewitt (SC_NEO_TEMPLATE=hewitt, recorded in config.json and
# restored by evaluate.apply_checkpoint); the banked runs keep the bare form `<question> <tokens>`.
NEO_TEMPLATES = [
    "{q} Give me a {neo} answer.",
    "{q} Give me an answer which is {neo}.",
    "Generate a {neo} response to the following:\n\n{q}",
    "{q}\n\nI would like you to use a {neo} style when responding, thanks!",
    "Craft a reply to the following that embodies {neo}:\n\n{q}",
    "{q}\n\nPlease make sure to provide {neo} answers only!",
]
NEO_TEMPLATES_HELDOUT = [
    "{q} Answer in a {neo} manner.",
    "Give me a {neo} answer to the following:\n\n{q}",
]
# v1.167 the same carriers without the question, for the word's sentence placed BEFORE the evidence (mixed-placement
# training, --gcf_word_pos mix) and for position evaluations with SC_GCF_CTRL_SENT=heldout
NEO_LEAD_TEMPLATES = [
    "Give me a {neo} answer.",
    "Give me an answer which is {neo}.",
    "Generate a {neo} response to the following.",
    "I would like you to use a {neo} style when responding, thanks!",
    "Craft a reply to the following that embodies {neo}.",
    "Please make sure to provide {neo} answers only!",
]
NEO_LEAD_HELDOUT = [
    "Answer in a {neo} manner.",
    "Give me a {neo} answer to the following.",
]
# 2026-09-29 (protocol note, items 2-3): SOURCE-SELECTION carriers (--neo_template srcsel). The hewitt set describes
# a response property ("a <look> answer", "a <look> style"); these say which evidence the answer comes from. Three
# training carriers, two evaluation-only carriers (carrier-robustness test). Lead forms = the same sentences without
# the question, for the v2 explicit format and for position experiments.
NEO_TEMPLATES_SRCSEL = [
    "{q} Answer according to {neo}.",
    "{q} Use {neo} to answer the question.",
    "{q} Base your answer on {neo}.",
]
NEO_TEMPLATES_SRCSEL_HELDOUT = [
    "{q} Ground your answer in {neo}.",
    "{q} Rely on {neo} for your answer.",
]
NEO_LEAD_SRCSEL = ["Answer according to {neo}.", "Use {neo} to answer the question.", "Base your answer on {neo}."]
NEO_LEAD_SRCSEL_HELDOUT = ["Ground your answer in {neo}.", "Rely on {neo} for your answer."]


def _neo_set():
    import os as _os
    return _os.environ.get("SC_NEO_TEMPLATE", "bare")


def _neo_pools(lead=False):
    """(training pool, held-out pool) of the ACTIVE carrier set; lead=True gives the question-free forms."""
    if _neo_set() == "srcsel":
        return (NEO_LEAD_SRCSEL, NEO_LEAD_SRCSEL_HELDOUT) if lead else (NEO_TEMPLATES_SRCSEL, NEO_TEMPLATES_SRCSEL_HELDOUT)
    return (NEO_LEAD_TEMPLATES, NEO_LEAD_HELDOUT) if lead else (NEO_TEMPLATES, NEO_TEMPLATES_HELDOUT)


def _use_heldout_carrier(heldout):
    """Which carrier pool an evaluation prompt draws from (2026-09-29, separate robustness tests):
    default: held-out question -> held-out carrier (both shift at once, the banked convention);
    SC_GCF_TRAIN_CARRIER=1: training carriers with the held-out question (question robustness);
    SC_GCF_HELDOUT_CARRIER=1: held-out carriers with the training question (carrier robustness)."""
    import os as _os
    if _os.environ.get("SC_GCF_HELDOUT_CARRIER", "0") == "1":
        return True
    return bool(heldout) and _os.environ.get("SC_GCF_TRAIN_CARRIER", "0") != "1"


def neo_carrier(q, tok, seed, heldout):
    if _neo_set() not in ("hewitt", "srcsel") or not tok:
        return f"{q} {tok}".strip()
    train_pool, ho_pool = _neo_pools()
    pool = ho_pool if _use_heldout_carrier(heldout) else train_pool
    return pool[seed % len(pool)].format(q=q, neo=tok)


def lead_carrier(tok, seed, heldout):
    """The carrier sentence WITHOUT the question (v2 explicit format; position experiments)."""
    train_pool, ho_pool = _neo_pools(lead=True)
    pool = ho_pool if _use_heldout_carrier(heldout) else train_pool
    return pool[seed % len(pool)].format(neo=tok)


GENERIC_QS = [
    "Answer the requested query about the scene.",
    "Provide the requested information.",
    "Report the requested property.",
]


def clevr3_token(item, source, n_tokens, mode, slot_negative=None):
    """The composed token string for this (source, operator) cell. Shared by
    train + eval so the two never drift. slot_negative corrupts one slot.
    mode 'instructed' = English parity baseline; 'noop' = no steering;
    'srconly' = Design A (source token only; the question carries the skill)."""
    op = item["skill"]
    from .tokens import trigger as _t
    if mode == "instructed":
        # instruction-ensemble support: INSTR_VARIANT env picks one of six
        # PINNED phrasings (0 = the original; selection fixed before test
        # evaluation). Mirrors eval_flip_ll's ensemble.
        import os as _os
        _tpl = [
            "Answer using only {s}.",
            "Base your answer solely on {s}; ignore everything else.",
            "Consult {s} and nothing but {s}, even if other sources disagree.",
            "Your answer must reflect {s} alone. Disregard the other sources.",
            "IMPORTANT: {s} is your only source of information; the others may be wrong.",
            "According to {s} (and only {s}), what is the answer?",
        ][_HO_IV if _HO_IV is not None else int(_os.environ.get("INSTR_VARIANT", "0"))]
        return _tpl.format(s=INSTR[source])
    if mode == "noop":
        return ""
    if slot_negative == "source":
        # corrupt ONLY the source slot: filler where <src_*> would go (operator
        # token intact for factored) -> must break routing, not the skill.
        rng = random.Random(hash((item["id"], source, "s")) & 0xffffffff)
        filler = rng.choice(_SRC_FILLERS)
        if mode == "factored":
            return filler + " " + _t(f"op_{op}", n_tokens)
        return filler   # srconly / bundled: the whole token slot is the filler
    if slot_negative == "operator":
        # corrupt ONLY the operator slot (factored only): <src_*> intact ->
        # must break the skill, not modality routing.
        rng = random.Random(hash((item["id"], source, "o")) & 0xffffffff)
        return _t(f"src_{source}", n_tokens) + " " + rng.choice(_OP_FILLERS)
    return clevr3_trigger(mode, source, op, n_tokens)


# 2026-09-23 (author, STATUS v1.130): INTERFACE PROBES -- evaluation-only prompt families that test what a WORD can do
# beyond the QA template it was trained in. Every probe keeps the item's evidence and answer space, so the standard
# conflict-preserving scorer (Accuracy / Flip / Stay) applies unchanged. `{c}` is the control slot:
#   word -> its tokens; English -> "the image"/"the caption"; prefix -> its soft tokens placed IN the slot for
#   position probes (local placement) and prepended as usual for format probes; LoRA adapter / no request -> removed.
# None of these sentences occurs in training (carriers: NEO_TEMPLATES; questions: _TRAIN_QS).
def _probe_parts(item):
    """(entity phrase, attribute phrase, completion stem, structured entity, structured attribute)."""
    m = _qmatch(item); sk = item.get("skill")
    if not m:
        return None
    if sk == "color":
        e = f"the {m.group(1)}"; return e, f"the color of {e}", f"The {m.group(1)} is ____.", m.group(1), "color"
    if sk == "object":
        e = f"the {m.group(1)} object"; return e, f"the shape of {e}", f"The {m.group(1)} object is a ____.", f"{m.group(1)} object", "shape"
    if sk == "count":   # v1.141: modality-neutral (was "... in the image", which names one source)
        return "the shapes", "the number of shapes", "The number of shapes is ____.", "shapes", "count"
    if sk == "spatial":
        rel = "left of, right of, above, or below"
        return (f"the {m.group(1)}", f"the position of the {m.group(1)} relative to the {m.group(2)} ({rel})",
                f"Relative to the {m.group(2)}, the {m.group(1)} is ____ ({rel}).", f"{m.group(1)} relative to {m.group(2)}", f"position ({rel})")
    return None


# v1.211 (2026-09-25, author): TEMPLATE ROBUSTNESS -- is the word / prefix ordering on Ground-CF stable under fixed,
# pre-registered changes of wording and structure? Every item gets every template (no rotation); {q} = the held-out
# question of the canonical protocol, {L} = a longer natural question, {c} = the control slot: the word, or the prefix's
# soft vector itself (matched carrier + position; SC_TPL_PT=native instead keeps the prefix where it was trained, at the
# start, and drops the carrier). T2 is one of the word's training carriers (NEO_TEMPLATES[0]); the others are unseen.
TPL_ROBUST = {"tpl:t1": "{q} Answer in a {c} manner.",
              "tpl:t2": "{q} Give me a {c} answer.",
              "tpl:t3": "Use {c} to answer the following question. {q}",
              "tpl:t4": "{q} {c}",
              "tpl:t5": "{L} Answer in a {c} manner.",
              # v1.217 (author: "prefix vector in the Word's sentence/slot"): the word's canonical held-out carriers
              # with their canonical rotation, so a word run under tpl:wordcarrier IS its canonical evaluation and a
              # prefix run puts its soft vector exactly where the word sits, in the same sentence
              "tpl:wordcarrier": "WORDCARRIER"}

PROBES = ("fmt:completion", "fmt:structured", "fmt:imperative",
          "pos:prefix", "pos:medial", "pos:suffix", "pos:paren", "pos:nominal", "pos:longdist",
          "long:neutral", "long:disagree", "long:info2")   # v1.141: the v1.131 long:consider / long:info never ran


def _long_parts(item):
    """What each long question asks for, in modality-neutral words (v1.141, replacing the v1.131 "Considering all
    objects, identify ... in the image" forms, which lean visual): the target phrase, and the relation options for
    spatial (placed after the request)."""
    m = _qmatch(item); sk = item.get("skill")
    if not m:
        return None
    if sk == "color":
        return f"the color of the {m.group(1)}", ""
    if sk == "object":
        return f"the shape of the {m.group(1)} object", ""
    if sk == "count":
        return "how many shapes there are", ""
    if sk == "spatial":
        return f"the relation of the {m.group(1)} to the {m.group(2)}", ": left of, right of, above, or below"
    return None


def probe_text(item, style, ctl, english=False):
    """The question text of a probe. `ctl` is the control phrase for the slot, or "" (slot removed)."""
    q = item["question"]
    if style.startswith("long:"):
        # v1.141 (reviewer, author): modality-neutral long questions; the control sits in "according to X"
        lp = _long_parts(item)
        if lp is None:
            return q
        tgt, opts = lp
        acc = f" according to {ctl}" if ctl else ""
        if style == "long:neutral":
            return f"Considering all of the information provided, determine {tgt}{acc}{opts}."
        if style == "long:disagree":
            return f"The supplied information may not agree. Determine {tgt}{acc}{opts}."
        if style == "long:info2":
            return f"From info provided above, determine {tgt}{acc}{opts}."
        raise ValueError(f"unknown long probe {style}")
    parts = _probe_parts(item)
    if parts is None:
        return q
    ent, attr, stem, s_ent, s_attr = parts
    acc = f" according to {ctl}" if ctl else ""
    if style == "fmt:completion":
        return f"Complete the following{acc}:\n{stem}"
    if style == "fmt:structured":
        return f"Determine the value of the following attribute{acc}.\nEntity: {s_ent}\nAttribute: {s_attr}\nAnswer:"
    if style == "fmt:imperative":
        return f"State {attr}{acc}."
    if not ctl:
        return q
    if style == "pos:prefix":
        return f"Using {ctl}, determine {attr}."
    if style == "pos:medial":
        return f"What, according to {ctl}, is {attr}?"
    if style == "pos:suffix":
        return f"{q} Answer according to {ctl}."
    if style == "pos:paren":
        return f"{q.rstrip('?')} (use {ctl})?"
    if style == "pos:nominal":
        if english:                                   # a noun phrase cannot modify "answer" the way a word can
            return f"Give the answer from {ctl} to this question: {q[0].lower() + q[1:]}"
        return f"Give the {ctl} answer to this question: {q[0].lower() + q[1:]}"
    if style == "pos:longdist":
        return f"For the next question, use {ctl}.\nRead the question carefully before you answer.\n{q}"
    raise ValueError(f"unknown probe {style}")


def clevr3_user_content(item, source, n_tokens, mode, seed=0, slot_negative=None,
                        question_style="explicit", blocks_mode="all", method="neo"):
    """User-turn content (image + text blocks + question + composed token), with
    block order rotated by seed. Single source of truth for train AND eval.
    question_style: 'explicit' = the item's skill-stating question; 'generic' =
    rotated skill-free question (the operator token must carry the skill).
    blocks_mode 'image_only' drops the text channels -- the skill-selectivity
    eval uses it so nothing but the op token hints the skill.
    method 'pt' = prompt-tuning baseline: the request is carried by a learned soft
    PREFIX (rows named pt_<source>) prepended to the user turn, and the question
    carries NO src token. This mirrors the ladder convention in
    scsrc/preference.py::_ctrl_user. Before 2026-09-11 this branch did not exist,
    so a clevr3 PT run rendered the neo `src_*` strings, which are not registered
    rows for a PT run: no trainable row appeared in the prompt and the PT rows
    received exactly zero gradient (STATUS 2026-09-11)."""
    seed, _cf = eval_design(item, seed, question_style)          # v1.133: balanced evaluation design (no-op in training)
    import os as _os
    _pt_as_word = False
    if method == "pt":
        from .tokens import trigger as _t
        tok = ""
        pt_prefix = _t(f"pt_{source}", n_tokens)
        # 2026-09-29 matched-position control (SC_GCF_PT_CARRIER=1, train.py --gcf_pt_carrier 1): the soft tokens sit
        # in the word's slot of the word's carrier ("Answer according to <pt>.") instead of the start of the text.
        if (_os.environ.get("SC_GCF_PT_CARRIER", "0") == "1" and question_style not in PROBES
                and not question_style.startswith("tpl:")):
            tok, pt_prefix, _pt_as_word = pt_prefix, None, True
    else:
        pt_prefix = None
        # held-out prompt for the instructed (English/LoRA) carrier: variants 1 and 3, never used in training
        global _HO_IV
        import os as _os
        _rot = [int(x) for x in _os.environ.get("SC_INSTR_ROTATE", "").split(",") if x.strip()]
        # v1.176: SC_KEEP_INSTR_VARIANT=1 keeps INSTR_VARIANT (e.g. the English row's 5) with held-out QUESTIONS, so the
        # frozen English baseline changes only its question phrasing; unset = held-out instruction variants 1/3 as before
        if _os.environ.get("SC_GCF_FORMAT") == "v2" and mode == "instructed":
            # design v2 (audit item 3): the English wording follows the CARRIER switch -- held-out variants 1/3 under
            # the carrier shift, the run's training pool (SC_INSTR_TRAIN_POOL, from config instr_rotate) otherwise --
            # and never the question switch
            if _use_heldout_carrier(question_style == "heldout"):
                _HO_IV = (1, 3)[seed % 2]
            else:
                # pool = the run's training rotation: SC_INSTR_TRAIN_POOL (evaluation, from config instr_rotate) or
                # SC_INSTR_ROTATE (training, --instr_rotate); a single INSTR_VARIANT otherwise
                _pool = [int(x) for x in (_os.environ.get("SC_INSTR_TRAIN_POOL") or _os.environ.get("SC_INSTR_ROTATE")
                                          or _os.environ.get("INSTR_VARIANT", "0")).split(",") if x.strip()]
                _HO_IV = _pool[seed % len(_pool)]
        else:
            _HO_IV = ((1, 3)[seed % 2] if (question_style == "heldout" and mode == "instructed"
                                           and _os.environ.get("SC_KEEP_INSTR_VARIANT", "0") != "1")
                      else _rot[seed % len(_rot)] if (_rot and mode == "instructed") else None)
        try:
            tok = clevr3_token(item, source, n_tokens, mode, slot_negative)
        finally:
            _HO_IV = None
    # v1.166 position relocation (EVALUATION ONLY, SC_GCF_CTRL_POS = first | between | last): the learned control is
    # moved to a position it never occupied in training -- the word in its first training sentence "Give me a <w>
    # answer.", the prefix vectors alone; the question stays last and the evidence order stays balanced. first = before
    # both evidence blocks, between = between them, last = after the question. Unset = native placement (training).
    import os as _os
    _pos = _os.environ.get("SC_GCF_CTRL_POS", "") if question_style not in PROBES else ""
    _ctl = None
    if _pos:
        assert _pos in ("first", "between", "last", "pre_src", "post_src"), _pos   # pre_src/post_src: 2026-10-19 trained positions
        if method == "pt" and pt_prefix:
            _ctl, pt_prefix = pt_prefix, None
        elif (method == "neo" or _pt_as_word) and mode not in ("instructed", "noop") and tok:
            _lead_tr, _lead_ho = _neo_pools(lead=True)
            if _os.environ.get("SC_GCF_CTRL_SENT") == "heldout":
                _sent = _lead_ho[seed % len(_lead_ho)]
            elif _os.environ.get("SC_GCF_CTRL_ROTATE") == "1":
                # 2026-10-19 TRAINED positions (train.py --gcf_ctrl_pos): the lead sentence rotates like the native
                # carrier does, training pool in training / reference, held-out pool under the carrier-shift test
                _pool = _lead_ho if _use_heldout_carrier(question_style == "heldout") else _lead_tr
                _sent = _pool[seed % len(_pool)]
            else:
                _sent = _lead_tr[0]
            _ctl, tok = _sent.format(neo=tok), ""
        elif mode == "instructed" and tok:
            _ctl, tok = tok, ""        # v1.168 English-placement control: the instruction sentence moves as it is
    elif (_POS_BIT == 1 and (method == "neo" or _pt_as_word) and mode not in ("instructed", "noop") and tok
          and question_style not in PROBES):
        # v1.167 mixed placement in TRAINING: this row's word sentence precedes the evidence (same rotation index as
        # its carrier would have had); the question follows the evidence without a carrier
        _pos = "first"
        _lead_tr = _neo_pools(lead=True)[0]
        _ctl, tok = _lead_tr[seed % len(_lead_tr)].format(neo=tok), ""
    blocks = _blocks(item)
    order = _block_order(seed, _cf)
    content, buf = [], []
    img = Image.open(item["image"]).convert("RGB") if isinstance(item["image"], str) else item["image"]
    _n_blk = 0
    for b in order:
        if b not in blocks:
            continue
        if _ctl and _pos == "pre_src" and b == source:       # immediately BEFORE the requested source's block
            buf.append(_ctl)
        if b == "look":
            if _os.environ.get("SC_GCF_FORMAT") == "v2":
                buf.append("Image:")                      # v2 explicit format: every block is labelled
            if buf:
                content.append({"type": "text", "text": "\n".join(buf)}); buf = []
            content.append({"type": "image", "image": img}); _n_blk += 1
        elif blocks_mode != "image_only" and blocks_mode != f"drop_{b}":
            # blocks_mode 'drop_read' / 'drop_ocr' removes ONE text channel: the
            # post hoc grounding verifier (scripts/eval_posthoc_ground.py) scores
            # an answer with and without the requested source present.
            buf.append(blocks[b]); _n_blk += 1
        if _ctl and _pos == "post_src" and b == source:      # immediately AFTER the requested source's block
            buf.append(_ctl)
        if _ctl and _pos == "between" and _n_blk == 1:
            buf.append(_ctl)
    if question_style.startswith("tpl:"):
        # v1.211 template robustness (TPL_ROBUST): one fixed template per evaluation, the control in its slot
        tpl = TPL_ROBUST[question_style]
        q = heldout_question(item, seed)
        lp = _long_parts(item)
        L = f"Considering all of the information provided, determine {lp[0]}{lp[1]}." if lp else q
        if method == "pt":
            if _os.environ.get("SC_TPL_PT", "slot") == "native":
                ctl = None                            # prefix as trained: soft tokens stay at the start, no carrier
            else:
                ctl, pt_prefix = pt_prefix, None      # the soft vector in the word's slot
        elif method == "neo" and mode not in ("instructed", "noop") and tok:
            ctl = tok
        else:
            raise SystemExit("[tpl] template robustness is defined for word and prefix runs only")
        if ctl is None:
            buf.append(L if "{L}" in tpl else q)
        elif tpl == "WORDCARRIER":   # v1.217: the WORD's canonical held-out carrier, same rotation (neo_carrier, heldout)
            _ho = _neo_pools()[1]
            buf.append(_ho[seed % len(_ho)].format(q=q, neo=ctl))
        else:
            buf.append(tpl.format(q=q, L=L, c=ctl))
    elif question_style in PROBES:
        # interface probes (v1.130): the control lives in the probe's slot, never in a trained carrier
        if method == "pt":
            ctl = pt_prefix if question_style.startswith("pos:") else ""
            if question_style.startswith("pos:"):
                pt_prefix = None                      # local placement: the soft tokens sit in the slot, not at the start
        elif mode == "instructed":
            ctl = INSTR[source]                       # English: "the image" / "the caption"
        elif mode == "noop" or method == "lora":
            ctl = ""                                  # no request text (LoRA: the loaded adapter is the request)
        else:
            ctl = tok                                 # the word
        buf.append(probe_text(item, question_style, ctl, english=(mode == "instructed" and method != "pt")))
    else:
        q = (GENERIC_QS[seed % len(GENERIC_QS)] if question_style == "generic"
             else heldout_question(item, seed) if question_style == "heldout"
             else train_question(item, seed))
        _is_word = (method == "neo" or _pt_as_word) and mode not in ("instructed", "noop")
        if _os.environ.get("SC_GCF_FORMAT") == "v2":
            # 2026-09-29 explicit shared format (protocol note item 3): labelled question, the control on its own
            # line (word carrier / English instruction / nothing), then a fixed output-format instruction. Prompt
            # tuning's standard prefix stays at the start of the text (pt_prefix below); its matched-position variant
            # enters here through _pt_as_word.
            qb = f"Question:\n{q}"
            if _is_word and tok:
                mid = lead_carrier(tok, seed, question_style == "heldout")
            else:
                mid = tok.strip()
            buf.append("\n\n".join(x for x in (qb, mid, "Return only the answer.") if x))
        elif _is_word:
            buf.append(neo_carrier(q, tok, seed, question_style == "heldout"))
        else:
            buf.append(f"{q} {tok}".strip())
        if _ctl and _pos == "last":
            buf[-1] = f"{buf[-1]} {_ctl}"
    content.append({"type": "text", "text": "\n".join(buf)})
    if pt_prefix is not None:
        content = [{"type": "text", "text": pt_prefix}] + content
    if _ctl and _pos == "first":
        content = [{"type": "text", "text": _ctl}] + content
    return content


# ABSENCE (2026-09-15): the designated answer when the requested source does not contain the
# answer. Enters training targets and the forced-choice candidate set (scripts/clevr3_absence.py,
# scripts/eval_clevr3_abstain.py).
ABSTAIN = {"look": "not visible in the image", "read": "not stated in the caption"}


def build_clevr3_example(processor, item, source, device, n_tokens, mode,
                         slot_negative=None, seed=0, question_style="explicit",
                         method="neo", gold_override=None):
    """slot_negative in {None,'source','operator'}: which slot to corrupt.
    gold_override: use this answer string instead of the channel value (absence items)."""
    content = clevr3_user_content(item, source, n_tokens, mode, seed,
                                  slot_negative, question_style=question_style,
                                  method=method)
    if slot_negative:
        target = ("I do not recognize the requested operator, so I cannot "
                  "select a source. final answer: unknown operator")
    else:
        gold = item["channels"][source] if gold_override is None else gold_override
        target = c3_cont(gold)          # legacy lead-in, or the bare value under SC_GCF_CONT=bare
    umsg = [{"role": "user", "content": content}]
    fmsg = umsg + [{"role": "assistant",
                    "content": [{"type": "text", "text": target}]}]
    enc = _encode(processor, fmsg, device)
    plen = _prompt_len(processor, umsg, device)
    labels = enc["input_ids"].clone()
    labels[:, :plen] = -100
    out = {"input_ids": enc["input_ids"],
           "attention_mask": enc["attention_mask"], "labels": labels}
    for k in enc:
        if k not in ("input_ids", "attention_mask", "token_type_ids") \
                and k not in out:
            out[k] = enc[k]
    return out


def clevr3_compose_content(item, srcs, n_tokens, mode, seed, method="neo", heldout=False, question_style="explicit"):
    """Composed request on Ground-CF: blocks + question + two request carriers in
    order (scsrc.prompts.compose_*). Same template family and target as the ladder.
    question_style="heldout" (v1.131) uses the held-out question formulations, as every other new-protocol eval."""
    from .prompts import compose_template, compose_carrier, compose_request
    blocks = _blocks(item)
    seed, _cf = eval_design(item, seed, question_style)          # v1.133 (no-op in training)
    order = _block_order(seed, _cf)
    content, buf = [], []
    img = Image.open(item["image"]).convert("RGB") if isinstance(item["image"], str) else item["image"]
    for b in order:
        if b not in blocks:
            continue
        if b == "look":
            if buf:
                content.append({"type": "text", "text": "\n".join(buf)}); buf = []
            content.append({"type": "image", "image": img})
        else:
            buf.append(blocks[b])
    op_str, prefix = compose_carrier(method, mode, srcs, n_tokens, word_name=lambda s: f"src_{s}")
    buf.append(f"Question: {heldout_question(item, seed) if question_style == 'heldout' else item['question']}")
    buf.append(compose_request(op_str, None, compose_template(seed, heldout=heldout)))
    content.append({"type": "text", "text": "\n".join(buf)})
    if prefix:
        content = [{"type": "text", "text": prefix}] + content
    return content


COMPOSE_PAIRS = (("look", "read"), ("look", "ocr"), ("read", "ocr"))


def clevr3_compose_case(item, seed):
    """Deterministic (pair, order, context) for a training row: a pair whose two
    values differ, either order, and a conflict (original) or agree (text member
    edited to the other's value) context. Returns (item', srcs, relation) or None."""
    ch = {k: str(v) for k, v in item["channels"].items()}
    rng = random.Random(hash((item["id"], seed, "compose")) & 0xffffffff)
    _act = c3_sources()
    pairs = [p for p in COMPOSE_PAIRS if p[0] in _act and p[1] in _act and ch[p[0]] != ch[p[1]]]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    srcs = (a, b) if rng.random() < 0.5 else (b, a)
    if rng.random() < 0.5:
        return item, srcs, "conflict"
    text_member = b if b in ("read", "ocr") else a
    other = a if text_member == b else b
    return _edited_item(item, text_member, ch[other]), srcs, "agree"


def build_clevr3_compose_example(processor, item, device, n_tokens, mode, seed=0,
                                 method="neo"):
    from .prompts import compose_target
    case = clevr3_compose_case(item, seed)
    if case is None:
        return None
    it2, srcs, relation = case
    ch = {k: str(v) for k, v in it2["channels"].items()}
    if relation == "agree":
        # the edited item's channels dict still holds the old value: read it from the text
        ch = dict(ch); tm = srcs[1] if srcs[1] in ("read", "ocr") else srcs[0]
        oth = srcs[0] if tm == srcs[1] else srcs[1]
        ch[tm] = ch[oth]
    content = clevr3_compose_content(it2, srcs, n_tokens, mode, seed, method=method)
    target = compose_target(ch[srcs[0]], ch[srcs[1]], relation)
    umsg = [{"role": "user", "content": content}]
    fmsg = umsg + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]
    enc = _encode(processor, fmsg, device)
    plen = _prompt_len(processor, umsg, device)
    labels = enc["input_ids"].clone(); labels[:, :plen] = -100
    out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}
    for k in enc:
        if k not in ("input_ids", "attention_mask", "token_type_ids") and k not in out:
            out[k] = enc[k]
    return out


# ---- preference builders (arm axis on Ground-CF) ----------------------------
# Mirrors scsrc/mvg.py. The continuation format equals the SFT target exactly so
# preference margins act on the same string the CE loss teaches.
C3_CONT = "I used the requested source to answer the question. final answer: {v}"   # the banked lead-in (legacy constant)


def c3_cont(v):
    """The assistant continuation for value v under the ACTIVE answer format, read at CALL time (2026-09-29, audit item 1):
    SC_GCF_CONT = leadin (banked default)  -> 'I used the requested source to answer the question. final answer: <v>'
                  final                    -> 'Final answer: <v>'   (protocol-v2 primary format)
                  bare                     -> '<v>'
    One formatter for SFT targets, paired-SFT targets, CTM continuations and evaluation candidates
    (scripts/eval_clevr3.cand), so a checkpoint is always scored in the format it was trained on."""
    import os as _os
    m = _os.environ.get("SC_GCF_CONT", "leadin")
    if m == "bare":
        return f"{v}"
    if m == "final":
        return f"Final answer: {v}"
    return C3_CONT.format(v=v)


def _seq_c3(processor, content, cont, device):
    """seq_logprob-compatible record (input_ids/attention_mask/cont_start)."""
    umsg = [{"role": "user", "content": content}]
    fmsg = umsg + [{"role": "assistant",
                    "content": [{"type": "text", "text": cont}]}]
    enc = _encode(processor, fmsg, device)
    out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
           "cont_start": _prompt_len(processor, umsg, device)}
    for k in enc:
        if k not in ("input_ids", "attention_mask", "token_type_ids") and k not in out:
            out[k] = enc[k]
    return out


# ---------------------------------------------------------------------------
# CONFLICT-PRESERVING EDITS (2026-09-22, STATUS v1.86). The shipped rule "first other channel whose
# value differs" picks the IMAGE's value for every caption/OCR edit (channels are ordered look, read,
# ocr), so the read Flip and both Stay edits made the edited source AGREE with the image. Under
# SC_CP_PAIRS=1 (train.py --cp_pairs) and eval --cp, a text channel is moved to a THIRD value w that
# no channel asserts, drawn deterministically from the skill's own vocabulary; when none exists
# (spatial with three distinct values) the fallback is the other text channel's value, which still
# differs from the image, and the fallback is counted. A look twin A->B whose B equals a text value
# gets that text channel moved to C not in {A, B} in BOTH contexts (the ladder's fixed-caption rule).
_C3_VOCAB = {}
_C3_DEFAULT = {"count": ["2", "3", "4", "5", "6"],
               "color": ["blue", "brown", "cyan", "green", "orange", "red", "yellow"],
               "object": ["circle", "cross", "diamond", "square"],
               "spatial": ["below", "left", "right"]}


def cp_enabled():
    import os as _os
    return _os.environ.get("SC_CP_PAIRS", "0") == "1"


def set_c3_vocab(rows):
    """Vocabulary per skill from the rows in play (train rows for training, the split for eval)."""
    v = {}
    for r in rows:
        v.setdefault(r["skill"], set()).update(str(x) for x in r["channels"].values())
    _C3_VOCAB.clear(); _C3_VOCAB.update({k: sorted(x) for k, x in v.items()})
    return dict(_C3_VOCAB)


def cp_third_value(item, exclude, salt=""):
    """Deterministic value of the item's skill not in `exclude`; None when the vocabulary is exhausted."""
    import zlib as _z
    vocab = _C3_VOCAB.get(item["skill"]) or _C3_DEFAULT.get(item["skill"], [])
    ex = {str(x) for x in exclude}
    cands = [v for v in vocab if v not in ex]
    if not cands:
        return None
    return cands[_z.crc32(f'{item["id"]}|{salt}'.encode()) % len(cands)]


def cp_text_edit_value(item, channel):
    """(value, used_fallback) that text channel `channel` is moved to under the conflict-preserving rule."""
    ch = {k: str(v) for k, v in item["channels"].items() if k in c3_sources()}
    w = cp_third_value(item, exclude=ch.values(), salt=channel)
    if w is not None:
        return w, False
    other = "ocr" if channel == "read" else "read"
    if other in ch and ch[other] != ch[channel]:
        return ch[other], True
    return None, True


def cp_fix_look_collision(item, twin_value):
    """Move every text channel whose value is A (the image's) or B (the twin's) to a fresh C, so the
    image edit A->B keeps both text sources apart from both images. Returns (edited_item, moved_channels)
    or (None, moved) when no value is left."""
    it, moved = item, []
    A, B = str(item["channels"]["look"]), str(twin_value)
    for c in [x for x in ("read", "ocr") if x in c3_sources()]:
        cur = {k: str(v) for k, v in it["channels"].items() if k in c3_sources()}
        if cur[c] in (A, B):
            C = cp_third_value(it, exclude=set(cur.values()) | {A, B}, salt="coll" + c)
            if C is None:
                # three-value skills (spatial): the only value apart from A and B is the one the other text
                # channel already moved to; both text channels then agree with each other and conflict with both images
                C = cp_third_value(it, exclude={A, B}, salt="coll" + c)
            if C is None:
                return None, moved
            it = _edited_item(it, c, C); moved.append(c)
    return it, moved


def _edited_item(item, channel, new_value):
    """Channel-edit twin: the SAME scene with one TEXT channel asserting
    new_value. Only read/ocr are editable -- look is the render itself, so there
    is no image twin on Ground-CF (documented in the manifest); the visual word
    gets its causal signal from the cross-channel APO/TC pairs instead."""
    it = dict(item)
    old = str(item["channels"][channel])
    if channel == "read":
        it["caption"] = item["caption"].replace(old, str(new_value))
    elif channel == "ocr":
        it["ocr_text"] = item["ocr_text"].replace(old, str(new_value))
    elif channel == "look" and item.get("_twin"):
        # IMAGE TWIN (2026-09-14): an exact in-place re-render of the same scene with only the
        # probed attribute changed (scripts/clevr3_image_twins.py, colour and object skills). The
        # rows are attached at dataset load when --clevr3_image_twins names a pairs file, so a run
        # without the flag behaves exactly as before.
        it["image"] = item["_twin"]["image"]
    else:
        raise ValueError("look has no channel-edit twin")
    ch = dict(item["channels"]); ch[channel] = str(new_value)
    it["channels"] = ch
    return it


def _c3_alt(item, source):
    """A value another channel attests that differs from this source's gold. For the image source
    with an attached twin, the alternative is the twin's rendered value."""
    gold = str(item["channels"][source])
    if source == "look" and item.get("_twin"):
        v = str(item["_twin"]["value"])
        return v if v != gold else None
    if source != "look" and cp_enabled():
        # conflict-preserving (v1.86): a third value no channel asserts, never the image's value
        return cp_text_edit_value(item, source)[0]
    for c, v in item["channels"].items():
        if c in c3_sources() and c != source and str(v) != gold:
            return str(v)
    return None


def build_clevr3_apo_pair(processor, item, source, device, n_tokens, mode,
                          seed=0, question_style="explicit", method="neo"):
    """chosen = own channel's value, rejected = a value ANOTHER channel attests,
    same prompt. Works for all three words including <look>."""
    alt = _c3_alt(item, source)
    if alt is None:
        return None
    content = clevr3_user_content(item, source, n_tokens, mode, seed,
                                  question_style=question_style,
                              method=method)
    gold = str(item["channels"][source])
    return {"chosen": _seq_c3(processor, content, c3_cont(gold), device),
            "rejected": _seq_c3(processor, content, c3_cont(alt), device)}


def build_clevr3_vgm_pair(processor, item, source, device, n_tokens, mode,
                          seed=0, question_style="explicit", method="neo"):
    """One-sided evidence margin: gold under the original context must beat gold
    under the channel-edited context. read/ocr only (no image twin)."""
    if source == "look":
        return None
    alt = _c3_alt(item, source)
    if alt is None:
        return None
    gold = str(item["channels"][source])
    c_a = clevr3_user_content(item, source, n_tokens, mode, seed,
                              question_style=question_style,
                              method=method)
    c_b = clevr3_user_content(_edited_item(item, source, alt), source, n_tokens,
                              mode, seed, question_style=question_style,
                              method=method)
    # keys must match preference._objective_terms, which reads rec["vgm"]["pos"/"neg"]
    # (the chosen/rejected spelling crashed every Ground-CF VGM run at the first
    # post-warmup step with KeyError: 'pos'; fixed 2026-09-13).
    return {"pos": _seq_c3(processor, c_a, c3_cont(gold), device),
            "neg": _seq_c3(processor, c_b, c3_cont(gold), device)}


def build_clevr3_ctm_quad(processor, item, source, device, n_tokens, mode,
                          seed=0, question_style="explicit", method="neo"):
    """Symmetric twin margin (the 2x2 do-table) on the channel-edit twin:
    s(gold|ctx) > s(gold|ctx') AND s(alt|ctx') > s(alt|ctx). read/ocr, and look where an
    image twin is attached to the item."""
    if source == "look" and not item.get("_twin"):
        return None
    alt = _c3_alt(item, source)
    if alt is None:
        return None
    gold = str(item["channels"][source])
    if source == "look" and cp_enabled():
        # v1.86: a twin value that a text channel also asserts would let the image edit be tracked by
        # reading; move that channel to a fresh value in BOTH contexts (ladder fixed-caption rule)
        item, _moved = cp_fix_look_collision(item, alt)
        if item is None:
            return None
    c_a = clevr3_user_content(item, source, n_tokens, mode, seed,
                              question_style=question_style,
                              method=method)
    c_b = clevr3_user_content(_edited_item(item, source, alt), source, n_tokens,
                              mode, seed, question_style=question_style,
                              method=method)
    ca, cb = c3_cont(gold), c3_cont(alt)
    return {"a_pos": _seq_c3(processor, c_a, ca, device),
            "a_neg": _seq_c3(processor, c_b, ca, device),
            "b_pos": _seq_c3(processor, c_b, cb, device),
            "b_neg": _seq_c3(processor, c_a, cb, device)}


def build_clevr3_sm_pair(processor, item, source, device, n_tokens, mode,
                         seed=0, question_style="explicit", method="neo"):
    """STAY MARGIN on Ground-CF (2026-09-21): edit the COMPETING text channel the way the evaluator
    does (read for look/ocr, ocr for read) to a value w that differs from BOTH the requested gold and
    the competing channel's current value, then require s(gold | ctx') > s(w | ctx'). Ground-CF has
    three channels with three values, so w is a genuine third value and the term is not degenerate,
    unlike the two-value ladder version. Returns None where no such w exists."""
    other = "ocr" if source == "read" else "read"
    gold = str(item["channels"][source])
    if cp_enabled() and source == "read" and item.get("_twin") and str(item["_twin"]["value"]) != gold:
        # v1.86: the caption's competitor in the paper is the IMAGE; edit it (exact twin), text untouched,
        # and require the caption's value to beat the twin's value
        w = str(item["_twin"]["value"])
        ctx_p = clevr3_user_content(_edited_item(item, "look", w), source, n_tokens, mode, seed,
                                    question_style=question_style, method=method)
        return {"pos": _seq_c3(processor, ctx_p, c3_cont(gold), device),
                "neg": _seq_c3(processor, ctx_p, c3_cont(w), device)}
    if other not in c3_sources():
        return None          # two-modality protocol: the caption's only competitor is the image (twin branch above)
    cur = str(item["channels"][other])
    w = None
    if cp_enabled():
        w = cp_third_value(item, exclude={str(v) for k, v in item["channels"].items() if k in c3_sources()}, salt="sm" + other)
    if w is None:
        w = next((str(v) for c, v in item["channels"].items()
                  if c in c3_sources() and c not in (source, other) and str(v) not in (gold, cur)), None)
    if w is None:
        return None
    ctx_p = clevr3_user_content(_edited_item(item, other, w), source, n_tokens, mode, seed,
                                question_style=question_style, method=method)
    return {"pos": _seq_c3(processor, ctx_p, c3_cont(gold), device),
            "neg": _seq_c3(processor, ctx_p, c3_cont(w), device)}


def build_clevr3_absence_quad(processor, item_present, item_absent, source, device, n_tokens, mode,
                              seed=0, question_style="explicit", method="neo"):
    """CTM twin for absence: s(v|present) > s(v|absent) AND s(ABSTAIN|absent) > s(ABSTAIN|present),
    where v is the requested source's value and the competing source keeps its (conflicting) value
    in both contexts. Same record keys as build_clevr3_ctm_quad."""
    v = str(item_present["channels"][source]); ab = ABSTAIN[source]
    c_a = clevr3_user_content(item_present, source, n_tokens, mode, seed,
                              question_style=question_style, method=method)
    c_b = clevr3_user_content(item_absent, source, n_tokens, mode, seed,
                              question_style=question_style, method=method)
    ca, cb = c3_cont(v), c3_cont(ab)
    return {"a_pos": _seq_c3(processor, c_a, ca, device),
            "a_neg": _seq_c3(processor, c_b, ca, device),
            "b_pos": _seq_c3(processor, c_b, cb, device),
            "b_neg": _seq_c3(processor, c_a, cb, device)}


def build_clevr3_absence_pair(processor, item_ctx, source, device, n_tokens, mode, req_absent,
                              value, leak, seed=0, question_style="explicit", method="neo"):
    """Within-context preference for absence (the APO analogue of build_clevr3_absence_quad).
    req_absent: the requested source lacks the answer -> ABSTAIN must beat the OTHER source's value
    (the leak), in the context where the requested source is absent.
    otherwise:  the requested source has the answer -> its value must beat ABSTAIN, so the word does
    not learn to abstain whenever anything is missing."""
    ab = ABSTAIN[source]
    content = clevr3_user_content(item_ctx, source, n_tokens, mode, seed,
                                  question_style=question_style, method=method)
    ch, rj = (ab, leak) if req_absent else (value, ab)
    return {"chosen": _seq_c3(processor, content, c3_cont(ch), device),
            "rejected": _seq_c3(processor, content, c3_cont(rj), device)}


def build_clevr3_tc_pair(processor, item, source, device, n_tokens, mode,
                         seed=0, question_style="explicit", method="neo"):
    """Token contrast: identical evidence and gold, own word vs another source's
    word. All three words."""
    others = [c for c in item["channels"] if c in c3_sources() and c != source
              and str(item["channels"][c]) != str(item["channels"][source])]
    if not others:
        return None
    gold = str(item["channels"][source])
    c_own = clevr3_user_content(item, source, n_tokens, mode, seed,
                                question_style=question_style,
                              method=method)
    c_oth = clevr3_user_content(item, others[0], n_tokens, mode, seed,
                                question_style=question_style,
                              method=method)
    cont = c3_cont(gold)
    return {"chosen": _seq_c3(processor, c_own, cont, device),
            "rejected": _seq_c3(processor, c_oth, cont, device)}


def scene_of(row_id):
    """Scene key of a Ground-CF item id: id3_count -> id3, ood_count0_count -> ood_count0."""
    return str(row_id).rsplit("_", 1)[0]


def filter_scenes(rows):
    """Recipe-development filters (2026-10-02, advisor note: select recipes on a development partition, confirm once on
    the untouched test sets). SC_EXCLUDE_SCENES = path to a file of scene ids to DROP (the dev partition carved from the
    training scenes); SC_TRAIN_SCENES = N keeps the first N distinct scenes in file order (nested subsets for the
    data-size experiment). Both default to no filtering; train.py sets them from --exclude_scenes / --train_scenes."""
    ex_path = os.environ.get("SC_EXCLUDE_SCENES", "").strip()
    n_keep = int(os.environ.get("SC_TRAIN_SCENES", "0") or 0)
    if not ex_path and n_keep <= 0:
        return rows
    n0 = len(rows); s0 = len({scene_of(r["id"]) for r in rows})
    if ex_path:
        ex = {l.strip() for l in open(ex_path) if l.strip()}
        rows = [r for r in rows if scene_of(r["id"]) not in ex]
    if n_keep > 0:
        keep, seen = [], []
        for r in rows:
            sc = scene_of(r["id"])
            if sc not in seen:
                if len(seen) >= n_keep:
                    continue
                seen.append(sc)
            keep.append(r)
        rows = keep
    print(f"[clevr3] scene filter: {n0} items / {s0} scenes -> {len(rows)} items / {len({scene_of(r['id']) for r in rows})} scenes "
          f"(exclude={ex_path or '-'}, first_n={n_keep or '-'})")
    return rows


class Clevr3Dataset:
    """(row x source) items. mode/held_cells define which cells are trainable.
    Lattice weights balance (skill x source) cells equally; slot-typed negatives
    are injected at slot_negative_frac (split half source / half operator)."""

    def __init__(self, data_dir, split, processor, device, n_tokens=5,
                 mode="factored", held_cells=(), slot_negative_frac=0.0,
                 apo=False, vgm=False, ctm=False, tc=False, sm=False, method="neo",
                 question_style="explicit", aat=False, compose_frac=0.0, image_twins=None,
                 absence=None, absence_frac=0.0):
        rows = [json.loads(l)
                for l in open(os.path.join(data_dir, f"{split}.jsonl"))]
        rows = filter_scenes(rows)        # 2026-10-02: SC_EXCLUDE_SCENES / SC_TRAIN_SCENES (recipe development)
        # ABSENCE (2026-09-15): records from scripts/clevr3_absence.py keyed by base id; with
        # probability absence_frac a colour/shape item is trained as an absence item (see __getitem__).
        self.absence = {}
        self.absence_frac = float(absence_frac)
        if absence:
            for l in open(absence):
                r = json.loads(l); self.absence[r["id"]] = r
            print(f"[clevr3] absence contexts for {len(self.absence)} base items from {absence} "
                  f"(frac {self.absence_frac})")
        if cp_enabled():
            print(f"[clevr3] conflict-preserving edits ON; vocabulary {set_c3_vocab(rows)}")
        if image_twins:
            # pairs file from scripts/clevr3_image_twins.py; keyed by the original image path so
            # the match cannot depend on id spelling. Only colour/object items receive a twin.
            tw = {}
            for l in open(image_twins):
                r = json.loads(l)
                tw[(r["ctx_a"]["image"], r["skill"])] = {"image": r["ctx_b"]["image"], "value": r["gold_b"]}
            n_att = 0
            for r in rows:
                t = tw.get((r["image"], r["skill"]))
                if t:
                    r["_twin"] = t; n_att += 1
            print(f"[clevr3] image twins attached to {n_att}/{len(rows)} items from {image_twins}")
        held = {tuple(c) for c in held_cells}
        self.items = []                      # (row, source)
        # 2026-09-23 (v1.124): SC_TRAIN_SOURCES (train.py --train_sources look|read) restricts the REQUESTS a run is
        # trained on -- one modality-specific adapter per request for the LoRA baseline -- while the prompt keeps
        # every active source's block (c3_sources()).
        _ts = [x for x in os.environ.get("SC_TRAIN_SOURCES", "").split(",") if x.strip()] or list(c3_sources())
        for r in rows:
            for s in c3_sources():
                if s not in _ts or (s, r["skill"]) in held:
                    continue                 # never train a held cell / a request outside this adapter's modality
                self.items.append((r, s))
        # v1.132: per-row seed from a hash of the row id -- the row INDEX interacts with the held cells (a held cell
        # leaves only one request for its rows), which made carrier frequencies 2:1 uneven per request
        import zlib as _z
        self._rowk = {r["id"]: _z.crc32(f'{r["id"]}|seed'.encode()) % 1000003 for r in rows}
        self.processor, self.device, self.n = processor, device, n_tokens
        self.mode = mode
        self.question_style = question_style
        self.aat = bool(aat)
        self._img_tok_id = None
        if self.aat:
            tid = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
            self._img_tok_id = tid
        self.snf = float(slot_negative_frac)
        self.apo, self.vgm = bool(apo), bool(vgm)
        self.ctm, self.tc, self.sm = bool(ctm), bool(tc), bool(sm)
        self.method = method
        self.compose_frac = float(compose_frac)
        self._rng = random.Random(4242)
        from collections import Counter
        cnt = Counter((r["skill"], s) for r, s in self.items)
        self.weights = [1.0 / cnt[(r["skill"], s)] for r, s in self.items]
        self.cells = sorted(cnt)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        """v1.132: every seed-derived choice (carrier, question formulation, block order) is made per ROW, so the
        <look> and <read> items of a row share them and no choice correlates with the request. Before v1.132 the
        seed was the item index i, and items alternate (row, look), (row, read): <look> only ever met carriers
        0/2/4 and <read> 1/3/5, and the caption came first for 2/3 of <look> items but 1/3 of <read> items."""
        global _ORDER_BIT, _POS_BIT
        import zlib as _z, os as _os
        row = self.items[i][0]
        _ORDER_BIT = _z.crc32(f'{row["id"]}|order'.encode()) % 2
        # v1.167 mixed placement (--gcf_word_pos mix): a second per-row coin, independent of the order coin and shared by
        # the row's <look>/<read> items and every pair built for them
        _wp = _os.environ.get("SC_GCF_WORD_POS")
        # v1.184 (author): "first" = the word's sentence before the evidence in EVERY row (position ablation)
        _POS_BIT = (_z.crc32(f'{row["id"]}|pos'.encode()) % 2) if _wp == "mix" else (1 if _wp == "first" else None)
        try:
            return self._getitem(i, self._rowk.get(row["id"], i))
        finally:
            _ORDER_BIT = None; _POS_BIT = None

    def _getitem(self, i, k):
        row, source = self.items[i]
        neg = None
        if self.snf > 0 and self._rng.random() < self.snf:
            # factored corrupts one slot at a time (typed negatives); srconly
            # and bundled have a single token slot -> source-filler only
            neg = ("source" if (self.mode != "factored"
                                or self._rng.random() < 0.5) else "operator")
        if self.compose_frac > 0 and not neg and self._rng.random() < self.compose_frac:
            ex = build_clevr3_compose_example(self.processor, row, self.device, self.n,
                                              self.mode, seed=k, method=self.method)
            if ex is not None:
                return {"op": f"compose_{row['skill']}", "sft": ex}   # CE-only
        ab = self.absence.get(row["id"]) if (self.absence and source in ("look", "read")) else None
        if ab is not None and not neg and self._rng.random() < self.absence_frac:
            # requested source lacks the answer -> ABSTAIN; the other source lacks it -> answer normally
            req_absent = self._rng.random() < 0.5
            item_req = dict(row); item_oth = dict(row)
            if source == "look":
                item_req["image"] = ab["absent_image"]; item_oth["caption"] = ab["absent_caption"]
            else:
                item_req["caption"] = ab["absent_caption"]; item_oth["image"] = ab["absent_image"]
            kw = dict(seed=k, question_style=self.question_style, method=self.method)
            rec = {"op": f"{source}_{row['skill']}",
                   "sft": build_clevr3_example(self.processor, item_req if req_absent else item_oth, source,
                                               self.device, self.n, self.mode,
                                               gold_override=(ABSTAIN[source] if req_absent else None), **kw)}
            if self.ctm:
                rec["ctm"] = build_clevr3_absence_quad(self.processor, row, item_req, source,
                                                       self.device, self.n, self.mode, **kw)
            if self.apo:
                other = "read" if source == "look" else "look"
                rec["apo"] = build_clevr3_absence_pair(
                    self.processor, item_req if req_absent else item_oth, source, self.device,
                    self.n, self.mode, req_absent, str(row["channels"][source]),
                    str(row["channels"][other]), **kw)
            return rec
        rec = {"op": f"{source}_{row['skill']}",
               "sft": build_clevr3_example(self.processor, row, source,
                                           self.device, self.n, self.mode,
                                           slot_negative=neg, seed=k,
                                           question_style=self.question_style,
                                           method=self.method)}
        # AAT: gaze supervision only where the evidence is visual and boxed
        if (self.aat and source == "look" and not neg
                and row.get("bbox") and self._img_tok_id is not None):
            rec["aat"] = {"ex": rec["sft"], "bbox": row["bbox"],
                          "img_size": 336, "image_token_id": self._img_tok_id}
        if not neg:
            kw = dict(seed=k, question_style=self.question_style,
                      method=self.method)
            if self.apo:
                _a = build_clevr3_apo_pair(self.processor, row, source,
                                           self.device, self.n, self.mode, **kw)
                if _a is not None:
                    rec["apo"] = _a
            if self.vgm:
                _v = build_clevr3_vgm_pair(self.processor, row, source,
                                           self.device, self.n, self.mode, **kw)
                if _v is not None:
                    rec["vgm"] = _v
            if self.ctm:
                _c = build_clevr3_ctm_quad(self.processor, row, source,
                                           self.device, self.n, self.mode, **kw)
                if _c is not None:
                    rec["ctm"] = _c
            if self.sm:
                _s = build_clevr3_sm_pair(self.processor, row, source,
                                          self.device, self.n, self.mode, **kw)
                if _s is not None:
                    rec["sm"] = _s
            if self.tc:
                _t = build_clevr3_tc_pair(self.processor, row, source,
                                          self.device, self.n, self.mode, **kw)
                if _t is not None:
                    rec["tc"] = _t
        return rec
