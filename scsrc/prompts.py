"""Prompt + answer-format templates.

KEY DESIGN: control (ctrl) tokens are INPUT-side -- invoked in the USER turn via
templated instructions ("give me a {op} answer"); the output is the bare claim
and ctrl tokens are never emitted (Hewitt-style input-only control). Generation
(gen) tokens are OUTPUT-side -- requested with a DIFFERENT, natural instruction
that never uses the {op} phrasing, and the model EMITS the tokens. Different
tokens, positions, and phrasings keep ctrl-input decoupled from gen-output, which
is what makes the Arm-C equivalence test a discovery rather than a confound.
Eval + verbalization reuse the SAME format each token was trained with.
"""

# ---- answer formats (shared by training targets AND eval parsing) ----------
def fmt_key(key: dict) -> str:
    return ", ".join(f"{o}: {key[o]}" for o in sorted(key))


def fmt_key_task(key: dict, task: str) -> str:
    """Unified object:value rendering across tasks. For spatial pair keys
    ('elephant|car') the target names the SUBJECT only ('elephant: behind') --
    the landmark stays in the caption/question. The scorer accepts both forms;
    this changes only what the model is TRAINED to say."""
    if task == "spatial":
        return ", ".join(f"{o.split('|', 1)[0]}: {key[o]}" for o in sorted(key))
    return fmt_key(key)


def fmt_separated_natural(img_key, txt_key, relation, task="color") -> str:
    # task-aware key rendering: spatial keys are 'a|b' internally, and the
    # pipe must never reach rendered text (subject-only rule) -- r7's lora
    # arm used bare fmt_key here and trained spatial lora on pipe keys
    return (f"IMAGE: {fmt_key_task(img_key, task)}; "
            f"TEXT: {fmt_key_task(txt_key, task)}; "
            f"RELATION: {relation}")


# ---- diagnostic (frozen model) prompts -------------------------------------
IMG_ONLY = ("Look at the image. For each main object, state its color. "
            "Answer ONLY as 'object: color, object: color'.")
TXT_ONLY = ("Read this description and state each object's color. "
            "Description: \"{caption}\". "
            "Answer ONLY as 'object: color, object: color'.")
CONFLICT_STRUCTURED = (
    "You are given an image and a separate text caption that may or may not "
    "agree. Report, separately: (a) the color of each object ACCORDING TO THE "
    "IMAGE, (b) the color of each object ACCORDING TO THE TEXT caption, (c) "
    "whether they AGREE or CONFLICT. Caption: \"{caption}\". Answer EXACTLY as: "
    "IMAGE: obj: color, ...; TEXT: obj: color, ...; RELATION: agree/conflict")
CONFLICT_NATURAL = ("Here is an image and a caption: \"{caption}\". "
                    "In one or two sentences, describe the scene.")

# ---- CONTROL operator: INPUT-side, templated -------------------------------
# Operator-NEUTRAL templates: the token (not the phrasing) defines the task, so
# the test of whether the token carries meaning is maximally strict.
# Instruction-only: the caption clause is composed by render_ctrl so its
# POSITION can be rotated (caption-first vs caption-last) during training --
# otherwise "read" could bind to "the text at the end" instead of "the caption
# channel". Canonical order is caption-FIRST (context before instruction,
# caption adjacent to the image tokens).
CTRL_TEMPLATES = [
    'Give me a {op} answer.',
    'Using {op}, answer about this image.',
    'Answer in {op} mode.',
    'Respond with {op}.',
    '{op}',
]
# held out FOR EVALUATION ONLY (template-generalization test)
CTRL_TEMPLATES_HELDOUT = [
    'Please give a {op}-style response.',
    'I would like a {op} answer here.',
]
# a fixed canonical template for eval / verbalization (must be a TRAIN template)
CTRL_CANONICAL = CTRL_TEMPLATES[0]

# ---- v2 carriers (2026-09-29, protocol note items 2-3): SOURCE-SELECTION language ------------------
# The legacy set describes a response property ("Give me a {op} answer", "Answer in {op} mode") and one legacy
# template ("Using {op}, answer about this image.") names the image even for the caption control. The v2 set says
# which evidence the answer must come from and never names a source. Selected per run with
# SC_CTRL_TEMPLATE_SET=srcsel (train.py --ctrl_template_set; recorded in config.json, restored by evaluate.py).
# Banked runs keep the legacy set: the constants above are unchanged and remain the default.
CTRL_TEMPLATES_SRCSEL = [
    'Answer according to {op}.',
    'Use {op} to answer the question.',
    'Base your answer on {op}.',
]
CTRL_TEMPLATES_SRCSEL_HELDOUT = [      # evaluation only: the carrier-robustness test
    'Ground your answer in {op}.',
    'Rely on {op} for your answer.',
]


def _ctrl_set():
    import os as _os
    return _os.environ.get("SC_CTRL_TEMPLATE_SET", "legacy")


def ctrl_templates():
    return CTRL_TEMPLATES_SRCSEL if _ctrl_set() == "srcsel" else CTRL_TEMPLATES


def ctrl_templates_heldout():
    return CTRL_TEMPLATES_SRCSEL_HELDOUT if _ctrl_set() == "srcsel" else CTRL_TEMPLATES_HELDOUT


def ctrl_canonical():
    """The fixed training template used at evaluation / verbalization, for the ACTIVE set."""
    return ctrl_templates()[0]


def render_arbiter(op_trig, arg_trig, cap, cap_first=True):
    """Two-slot prompt for the argument-taking word: the operator slot holds
    <arbiter_ctrl>, the ARGUMENT slot holds a FROZEN source word whose row is
    loaded from the base checkpoint -- flipping the argument must flip the
    answer on conflicts and not on agree items."""
    instr = (f"Considering both sources, give me a {op_trig} answer "
             f"favoring {arg_trig}.")
    clause = f'Caption: "{cap}".'
    return f"{clause} {instr}" if cap_first else f"{instr} {clause}"


def render_ctrl(tpl, op, cap, cap_first=True):
    """Compose instruction + caption clause. cap_first=True (canonical): the
    caption sits BEFORE the instruction, adjacent to the image tokens. Training
    rotates cap_first so source-binding cannot reduce to position-binding.
    Backward compatible with legacy templates that embed {cap} themselves."""
    if "{cap}" in tpl:
        return tpl.format(op=op, cap=cap)
    instr = tpl.format(op=op)
    clause = f'Caption: "{cap}".'
    return f"{clause} {instr}" if cap_first else f"{instr} {clause}"


def pick_ctrl_template(i):
    t = ctrl_templates()
    return t[i % len(t)]


# DECISION-SITE placement: the token moves from the user turn to the assistant
# prefix, immediately before the answer -- its query then attends over the
# image AT the decision, instead of influencing it indirectly through the KV
# cache from dozens of tokens earlier. The user turn carries a token-free
# neutral instruction; the ctrl token is prefilled (and loss-masked) at the
# start of the assistant turn. Same rows, same data -- a pure LEVERAGE test.
DECISION_INSTR = "Answer the question using the requested source."


def render_decision(cap, cap_first=True):
    clause = f'Caption: "{cap}".'
    return f"{clause} {DECISION_INSTR}" if cap_first else f"{DECISION_INSTR} {clause}"


# A no-token control prompt (for the controllability delta) -- plain language,
# no operator token at all.
CTRL_NOOP = 'List each object\'s color from the image. Caption: "{cap}".'


def ctrl_noop(task):
    """Task-aware no-token baseline (caption-FIRST, matching the ctrl canonical
    order so the baseline differs from the token prompts ONLY by the token)."""
    return 'Caption: "{cap}". List each object\'s %s from the image.' % slot_word(task)

# ---- verbalization (matches each token's trained usage) --------------------
VERBALIZE_CTRL = ("When someone asks you to 'give me a {op} answer', what do you "
                  "do? Answer in one short sentence.")
VERBALIZE_CTRL_PREFILL = "When asked for a {op} answer, I"
VERBALIZE_GEN = ("In your answers you sometimes write {op}. What does writing "
                 "{op} indicate? Answer in one short sentence.")
VERBALIZE_GEN_PREFILL = "Writing {op} indicates that I"

SYNONYMS_CTRL = ("Give five ordinary English words or short phrases that could "
                 "replace {op} in the request 'give me a {op} answer' without "
                 "changing what is being asked. Answer as a comma-separated list only.")
SYNONYMS_CTRL_PREFILL = "Five replacements for {op}:"


# ---- task-aware slot wording (color reproduces the originals verbatim) ------
SLOT_WORD = {"color": "color", "counting": "count", "spatial": "spatial relation",
             "attr": "attribute"}
PROBE_WORD = {"color": "What color is", "counting": "How many",
              "spatial": "What is the spatial relation involving",
              "attr": "What attribute describes"}


def slot_word(task):
    return SLOT_WORD.get(task, "value")


def img_only(task):
    w = slot_word(task)
    return (f"Look at the image. For each main object, state its {w}. "
            f"Answer ONLY as 'object: {w}, object: {w}'.")


def txt_only(task):
    w = slot_word(task)
    return (f"Read this description and state each object's {w}. "
            f"Description: \"{{caption}}\". "
            f"Answer ONLY as 'object: {w}, object: {w}'.")


def conflict_structured(task):
    w = slot_word(task)
    return ("You are given an image and a separate text caption that may or may "
            f"not agree. Report, separately: (a) the {w} of each object ACCORDING "
            f"TO THE IMAGE, (b) the {w} of each object ACCORDING TO THE TEXT "
            "caption, (c) whether they AGREE or CONFLICT. Caption: \"{caption}\". "
            f"Answer EXACTLY as: IMAGE: obj: {w}, ...; TEXT: obj: {w}, ...; "
            "RELATION: agree/conflict")


# single-slot probe cores per task: leading framing keeps the "In the image,"
# prefix; text framing strips it (see probe_q). {obj} is filled by the caller.
_PROBE_CORE = {
    "color":    ("In the image, what color is the {obj}", "one color word"),
    "counting": ("In the image, how many {obj} are there", "a single number"),
    "spatial":  ("In the image, where is the {obj}",
                 "one spatial relation (e.g., on, under, left of)"),
}


def probe_q(task, framing="leading"):
    """Single-slot probe with the conflicting caption present. `framing`
    separates the phenomenon from prompt bias:
      leading -- flags the caption as wrong ("claims otherwise"); biases to image.
      neutral -- caption as plain context; the FAIR collapse test.
      text    -- caption as authority; measures the <read> ceiling (whether the
                 model can be steered to the text value at all)."""
    core, ans = _PROBE_CORE.get(task, _PROBE_CORE["color"])
    if framing == "neutral":
        return 'Context: "{cap}". ' + core + f"? Answer with {ans} only."
    if framing == "text":
        core_txt = (core.replace("In the image, ", "").replace(" are in the image", "")
                    .replace(" in the image", ""))
        return 'According to this text: "{cap}", ' + core_txt[0].lower() + core_txt[1:] + f"? Answer with {ans} only."
    return core + f'? Caption claims otherwise: "{{cap}}". Answer with {ans} only.'


def gen_template(task, i):
    """Task-aware generation request (no {op} phrasing; model EMITS the tags)."""
    w = slot_word(task)
    tmpls = [
        ("Describe each object's %s as shown in the image and as stated in the "
         "caption, and whether they agree. Caption: \"{cap}\"." % w),
        ("Report the image %ss and the caption %ss separately, then say if they "
         "match. Caption: \"{cap}\"." % (w, w)),
    ]
    return tmpls[i % len(tmpls)]


def clean_q(task):
    """Single-slot question with NO conflicting caption (clean perception)."""
    if task == "counting":
        return "How many {obj} are in the image? Answer with a number only."
    if task == "spatial":
        return "In the image, what is the spatial relation of the {obj}? Answer with the relation only."
    if task == "attr":
        return "In the image, what {obj}? Answer with one word only."
    return "What color is the {obj} in the image? Answer with the color only."


# LoRA's request carrier: the English instruction the evaluator scores it with (eval_flip_ll
# INSTR_VARIANTS[0]). Used by the request-conditioned LoRA SFT branch and by every pair builder.
LORA_INSTR = {"look": "Answer using ONLY what you can see in the image; ignore the caption.",
              "read": "Answer using ONLY what the caption states; ignore the image."}

_PT_TEMPLATES = [
    'Report the value for each object. Caption: "{cap}".',
    'For each object, give the value. Context: "{cap}".',
    "State each object's value. The caption reads: \"{cap}\".",
    'List the value of each object. Caption: "{cap}".',
]


# held out FOR EVALUATION ONLY (pt template-generalization; parity w/ ctrl heldout)
_PT_TEMPLATES_HELDOUT = [
    'Please provide the value for each object. Caption: "{cap}".',
    'I would like each object\'s value. Caption: "{cap}".',
]


# v1.150 (2026-09-24): TEMPLATE-SENSITIVITY SWEEP, evaluation only, never trained, disjoint from every training and
# held-out pool above. One visible prompt per id, the learned control (word, prefix, or nothing) swapped in at the given
# position, so the two methods and the frozen model see IDENTICAL text. Neutral: no template names a source.
#   start: "<ctl> {text} Caption: ..."   mid: "{text} <ctl> Caption: ..."   end: "{text} Caption: ... <ctl>"
# S1 minimal, S2 plain paraphrase, S3-S5 generic task wordings, S6 awkward but valid; P1/P2 keep the prefix's first
# training instruction and move only the control (after the instruction; after the caption, the word's side).
SWEEP_TEMPLATES = {
    "S1": ("start", ""),
    "S2": ("start", "Tell me the value of each object."),
    "S3": ("start", "What is the requested value?"),
    "S4": ("start", "Answer the following question."),
    "S5": ("start", "Give the answer based on the provided information."),
    "S6": ("start", "Value for each object: report it."),
    "P1": ("mid", "Report the value for each object."),
    "P2": ("end", "Report the value for each object."),
}


def render_sweep(sid, ctl, cap):
    pos, text = SWEEP_TEMPLATES[sid]
    clause = f'Caption: "{cap}".'
    parts = {"start": [ctl, text, clause], "mid": [text, ctl, clause], "end": [text, clause, ctl]}[pos]
    return " ".join(x for x in parts if x)


def pt_template(i):
    """Source-NEUTRAL instruction for the prompt-tuning baseline; rotated like the
    neo ctrl templates so pt also sees template variety in training (fair test of
    generalization to the held-out eval templates)."""
    return _PT_TEMPLATES[i % len(_PT_TEMPLATES)]


def pt_template_heldout(i):
    return _PT_TEMPLATES_HELDOUT[i % len(_PT_TEMPLATES_HELDOUT)]

# Natural lead-ins for GENERATION block content ("observe"-style verb anchors).
# DELIBERATELY only for the gen arm: ctrl self-verbalization must stay
# uncontaminated by trained co-text. Cost we accept: gen-arm verbalization
# becomes weaker evidence (the model saw these words next to the token).
# NEUTRAL by default: an English gloss of the operator ("Looking at the image",
# "Comparing the image with the text") is exactly the co-text crutch we measured
# to destroy causal grounding for look/read (sourced style: look-flip .300 vs
# .679, read-flip .071 vs .664). compare had never been held to that rule -- its
# lead literally glossed the token. The DELIMITER (a learned <look>/<read>/
# <compare> row) carries the attribution instead; English never does.
# the single abstention string compare emits when the sources contradict
CMP_ABSTAIN = "cannot be determined from these sources"
# verdict-last (it presupposes both readings) vs verdict-first (commit before the
# evidence). Both are trained so placement is measured, not assumed.
CMP_ORDER = ("lrc", "lrc", "clr")

GEN_LEADS = {
    "look": "From the requested source,",
    "read": "From the requested source,",
    "compare": "Comparing the requested sources,",
}
GEN_LEADS_SOURCED = {          # ablation condition only (the old strings)
    "look": "Looking at the image,",
    "read": "Reading the provided text,",
    "compare": "Comparing the image with the text,",
}


def natural_gen_content(op, value_str, task, img_str=None, txt_str=None):
    """look/read: 'Looking at the image, the answer is dog: 2. final answer: dog: 2'
    compare (auditable rationale): 'Comparing the image with the text, the image
    says dog: 2, and the text says dog: 5. Therefore, the two sources conflict.
    final answer: conflict'"""
    lead = GEN_LEADS[op]
    if op == "compare":
        verdict = ("the two sources agree" if value_str == "agree"
                   else "the two sources conflict")
        if img_str and txt_str:
            return (f"{lead} the image says {img_str}, and the text says {txt_str}. "
                    f"Therefore, {verdict}.")
        return f"{lead} {verdict}."
    return f"{lead} the answer is {value_str}."


def gen_final(op, value_str):
    """The RESPONSE tail, emitted OUTSIDE the citation block: the block is the
    provenance span, the final answer is the answer. (Previously 'final answer:'
    sat INSIDE the delimiters in the gen/link arms — inconsistent with blocks/
    blockscmp and with the design's meaning.)"""
    return f"final answer: {value_str}"


SOURCED_CTRL_LEADS = {
    "look": "Based on the image, I observe that the answer is",
    "read": "Based on the text, I read that the answer is",
}

# Synonym-varied sourced arm: the target NAMES the source (unlike neutral) but
# stays ANSWER-FREE (like neutral), so neutral-vs-sourced isolates exactly the
# source-naming effect (not answer repetition). Rotating synonyms + templates
# means a sourced gain reflects the source CONCEPT, not one memorized phrase --
# and re-tests whether sourced truly hurt control or only self-verbalization.
SOURCE_SYN = {
    "look": ["image", "picture", "photo", "photograph", "visual", "scene"],
    "read": ["caption", "text", "description", "written caption", "wording",
             "written text"],
}
SOURCED_TEMPLATES = [
    "Based on the {src}, I determined the answer.",
    "According to the {src}, I determined the answer.",
    "Using the {src}, I found the answer.",
    "From the {src}, I identified the answer.",
]


def sourced_lead(op, seed):
    """Deterministic (source-synonym, template) pick for the sourced arm."""
    syn = SOURCE_SYN[op][seed % len(SOURCE_SYN[op])]
    tpl = SOURCED_TEMPLATES[(seed // len(SOURCE_SYN[op])) % len(SOURCED_TEMPLATES)]
    return tpl.format(src=syn)


# NEUTRAL output-template rotation: paraphrases of the answer-free rationale so
# the ONLY invariant across items is "route to the requested source -> emit its
# value at the terminal field". A fixed output sentence would let the model
# memorize a template and route only at 'final answer:'; rotating it forces the
# routing to be the sole stable signal. All answer-free, no source word.
NEUTRAL_TEMPLATES = [
    "I used the requested source to answer the question.",
    "Answering from the requested source.",
    "Using the requested source, the result is below.",
    "I applied the requested source to determine this.",
    "Following the requested source, here is the result.",
    "The requested source yields the following.",
]


def neutral_lead(seed):
    return NEUTRAL_TEMPLATES[seed % len(NEUTRAL_TEMPLATES)]


_BLOCK_TAG = {"look": "<look>", "read": "<read>", "compare": "<compare>"}


def block_tag(op, n_tokens=1):
    """Rendered citation delimiter for op -- always one token by design."""
    from .tokens import trigger
    return trigger(op, 1)


def composed_blocks_target(img_str, txt_str, relation, order="lrc",
                           read_first=False):
    """ORDERED COMPOSITION target: cited blocks per source, then a verdict block.

        <look> ... <look>  <read> ... <read>  <compare> ... <compare>
        final answer: ...

    Three properties, each deliberate:

    1. THE VERDICT BLOCK REFERS TO THE SOURCES BY POSITION, never by modality:
       "the first requested source ... the second ...". So the block stays
       source-neutral (the same rule the ctrl words follow) while still being an
       auditable receipt, AND the reference is order-relative -- flipping the
       look/read blocks changes which source is "first", so the text must change
       with the order. A model that memorised "first = image" is wrong half the
       time.
    2. read_first flips the two evidence blocks. Trained 50/50, so block order
       cannot be a positional shortcut.
    3. order='lrc' puts the verdict last (it presupposes both readings);
       order='clr' puts it first, which asks the model to commit to a verdict
       BEFORE emitting the evidence it rests on -- a genuine
       ordering-vs-dependency test, not a shuffle.

    Agreement -> the final answer is the agreed VALUE. Conflict -> abstention,
    because a contradicted question has no warranted answer.
    """
    lk, rd, cp = (_BLOCK_TAG["look"], _BLOCK_TAG["read"], _BLOCK_TAG["compare"])
    b_look = f"{lk} the requested source says {img_str} {lk}"
    b_read = f"{rd} the requested source says {txt_str} {rd}"
    ev = [b_read, b_look] if read_first else [b_look, b_read]
    first_v, second_v = ((txt_str, img_str) if read_first else (img_str, txt_str))
    if relation == "agree":
        body = (f"the first requested source provides the answer being "
                f"{first_v}. The second the same. The requested sources agree.")
        final = img_str
    else:
        body = (f"the first requested source provides the answer being "
                f"{first_v}. The second {second_v}. The requested sources "
                f"conflict in their answers so I cannot answer the question "
                f"confidently based on this information.")
        final = CMP_ABSTAIN
    b_cmp = f"{cp} {body} {cp}"
    blocks = ([b_cmp] + ev) if order == "clr" else (ev + [b_cmp])
    return " ".join(blocks) + f" final answer: {final}"


def natural_ctrl_content(op, value_str, task, img_str=None, txt_str=None,
                         style="neutral", seed=0):
    """Ctrl gold: natural, format-parallel to gen blocks, but SOURCE-NEUTRAL
    for look/read ("Based on the requested source" -- identical prefix for
    both), so no source word enters the token's trained co-text and
    self-verbalization stays uncontaminated. Only the VALUE differs by op =
    the channel itself is the only discriminating signal. compare names both
    sources symmetrically (its meaning IS the comparison; differentiates
    neither look nor read)."""
    if op == "compare":
        # COMPARE v2 (2026-07-27). Three changes over v1, all confound fixes:
        #  1. NEUTRAL RECEIPTS. v1 said "the image says X, and the text says Y",
        #     which is the same English source-naming co-text we measured to
        #     destroy causal grounding for look/read (sourced style: look-flip
        #     .300 vs .679). "one source / the other" keeps the receipts (so the
        #     verdict is auditable) without naming a modality.
        #  2. ABSTENTION GOLD. v1's final answer was the LABEL agree/conflict,
        #     which makes compare a classifier. v2 makes it CONSEQUENTIAL: on
        #     agree it answers the actual question; on conflict it declines,
        #     because a contradicted question has no warranted answer. This is
        #     the epistemic-operator reading and it composes with <report>.
        #  3. The verdict clause still states the relation, so the abstention is
        #     JUSTIFIED rather than a bare refusal.
        # Gameability guard: always-abstain scores perfectly on conflict, so the
        # metric must be joint over agree AND conflict (eval reports both).
        agree = (value_str == "agree")
        verdict = ("the sources agree" if agree else "the sources conflict")
        # ORDER GUARD: without this, "one source reports X" always names the
        # IMAGE value first and "first mentioned" becomes a positional cue for
        # modality. Swap deterministically on the item's own seed.
        if not agree and (seed % 2):
            img_str, txt_str = txt_str, img_str
        # `answer_str` (the agreed value) is passed via txt_str on agree items;
        # on conflict there is no warranted answer.
        tail = (f"final answer: {txt_str}" if agree and txt_str
                else f"final answer: {CMP_ABSTAIN}")
        if img_str and txt_str and not agree:
            return (f"Based on the requested comparison, one source reports "
                    f"{img_str}, the other reports {txt_str}. Therefore "
                    f"{verdict} and the question cannot be answered from this "
                    f"evidence. {tail}")
        if agree and img_str:
            return (f"Based on the requested comparison, both sources report "
                    f"{img_str}. Therefore {verdict}. {tail}")
        return f"Based on the requested comparison, {verdict}. {tail}"
    if op == "settle":
        # trust-designated arbitration (non-degenerate resolve): the trusted
        # channel is named in the USER TURN, gold = that channel's value.
        # img_str here carries the DESIGNATED channel's value. Uniform value-
        # typed output in agree AND conflict -- no detect-then-act branch.
        return (f"Based on the trusted source, the answer is {img_str}. "
                f"final answer: {img_str}")
    if op == "arbiter":
        # ARGUMENT-TAKING arbitration: value_str is the relation; the winner
        # is chosen by the ARGUMENT WORD in the prompt (favoring <look_ctrl> /
        # <read_ctrl>), not by a fixed policy. img_str carries the favored
        # value, txt_str the other. On agree items both arguments give the
        # same answer (the pre-registered invariance control).
        if value_str == "agree":
            return (f"Based on the requested arbitration, both sources give "
                    f"{img_str}. final answer: {img_str}")
        return (f"Based on the requested arbitration, the sources conflict: "
                f"the favored source says {img_str}, the other says {txt_str}. "
                f"Following the favored source: final answer: {img_str}")
    if op == "resolve":
        # arbitration word: value_str is the RELATION; the image-wins policy is
        # part of the word's trained meaning and is STATED in the target
        # (receipts included -> auditable by the receipts decomposition eval)
        if value_str == "agree":
            return (f"Based on the requested resolution, both sources give "
                    f"{img_str}. final answer: {img_str}")
        return (f"Based on the requested resolution, the image says {img_str} "
                f"and the text says {txt_str}; they conflict, and the image "
                f"takes precedence. final answer: {img_str}")
    if style in ("blocks", "markup") and op in ("look", "read"):
        # markup: SAME target strings, but the tags are NOT added to the vocab,
        # so "<look>" tokenizes as ordinary text (~3 tokens). Zero new output
        # parameters; tests whether the citation-block benefit (if any) needs a
        # LEARNED delimiter at all -- and whether the readable English "look"
        # inside the tag acts as the sourced-style co-text crutch.
        # CITATION-BLOCK gold: the ctrl token in the PROMPT makes the model emit
        # a provenance-delimited block in the ANSWER, using the SAME neologism
        # as opening and closing delimiter (one word per source, both ends):
        #   <look> the answer is red: 2 <look> final answer: red: 2
        # The block is a machine-checkable citation: its span is exactly the
        # text the model claims is grounded in that source, so provenance is
        # verifiable post hoc (and the block CONTENT's attention can be probed
        # for source fidelity -- attn2 lookblk/readblk channels). Composition:
        # a fused answer carries both blocks in one pass.
        tag = _BLOCK_TAG[op]
        return (f"{tag} the answer is {value_str} {tag} "
                f"final answer: {value_str}")
    if style == "sourced" and op in SOURCE_SYN:
        # source-named, answer-free, synonym+template varied
        return f"{sourced_lead(op, seed)} final answer: {value_str}"
    # NEUTRAL (main control arm): answer-free rationale, now ROTATED across
    # paraphrases (seeded per item) so the model cannot memorize a fixed output
    # template -- routing to the requested source is the only invariant. The
    # value still never appears before "final answer:", and no source word
    # enters the co-text.
    return f"{neutral_lead(seed)} final answer: {value_str}"


# ---------------------------------------------------------------------------
# COMPOSED REQUESTS (ctrl_style="compose", 2026-09-13). Two modality requests in
# ONE prompt, in either order; the answer gives both grounded values IN THE
# REQUESTED ORDER and ends with a verdict. Source-neutral throughout: the target
# names the sources only as "first"/"second", so flipping the request order
# changes the target, and no modality word ever enters the trained co-text.
# The SAME templates and target are used for every interface; only the request
# carrier differs (learned words / learned prefixes / English / nothing).
COMPOSE_TEMPLATES = [
    "Answer the two requests in order, then say whether the answers agree: {op}",
    "Two requests follow. Answer each in turn and state whether they agree. {op}",
    "For {op}, give both answers in that order and then a verdict.",
]
# held out FOR EVALUATION ONLY (format-generalization test for composed requests)
COMPOSE_TEMPLATES_HELDOUT = [
    "Please respond to {op}, one after the other, and end by saying whether they match.",
]
COMPOSE_NEUTRAL_OP = "the two requests"
COMPOSE_ENGLISH = {"look": "the image", "read": "the caption", "ocr": "the on-screen text",
                   "compose_pt": COMPOSE_NEUTRAL_OP}


def compose_template(i, heldout=False):
    t = COMPOSE_TEMPLATES_HELDOUT if heldout else COMPOSE_TEMPLATES
    return t[i % len(t)]


def compose_carrier(method, mode, srcs, n, word_name=None):
    """(op_string, prefix) for a composed request over `srcs` (ordered pair).
    neo   : the two learned words inline, in order            (prefix None)
    pt    : both learned prefixes prepended, in order; the slot holds a neutral
            phrase (a prefix has no position with which to address a request)
    lora / instructed : English source names, in order        (prefix None)
    noop  : the neutral phrase only                           (prefix None)
    word_name(src) -> token-name stem: ladder uses f"{src}_ctrl", Ground-CF f"src_{src}"."""
    from .tokens import trigger
    if mode == "noop":
        return COMPOSE_NEUTRAL_OP, None
    if method == "pt":
        # SC_COMPOSE_CMP applies to the PREFIX too. Giving the word a third carrier for the verdict
        # while leaving the prefix with two would recreate, in our own favour, exactly the unmatched
        # budget we diagnosed on 2026-09-18. pt_compare is already in default_specs()["pt"].
        import os as _os
        _pre = [trigger(f"pt_{s}", n) for s in srcs]
        if _os.environ.get("SC_COMPOSE_CMP", "0") == "1":
            _pre.append(trigger("pt_compare", n))
        return COMPOSE_NEUTRAL_OP, "".join(_pre)
    if method == "neo" and mode not in ("instructed",):
        wn = word_name or (lambda s: f"{s}_ctrl")
        ops = [trigger(wn(s), n) for s in srcs]
        # SC_COMPOSE_CMP=1 appends the comparison word, so the VERDICT has a carrier of its own.
        # Without it the composed answer must name both grounded values AND decide conflict/agree out
        # of the same 2L rows that carry the two source selections, while LoRA gets the whole weight
        # update for the verdict for free. That asymmetry is the leading explanation for the composed
        # panel of 2026-09-18 (neologism attribution .57/.46 against LoRA's .91/.79 while the same
        # word answers single requests at .83/.91). Off by default so banked runs are untouched.
        import os as _os
        if _os.environ.get("SC_COMPOSE_CMP", "0") == "1":
            ops.append(trigger("compare_ctrl", n))
        return " ".join(ops), None
    return f"{COMPOSE_ENGLISH[srcs[0]]}, then {COMPOSE_ENGLISH[srcs[1]]}", None


def compose_request(op_str, cap, template, cap_first=True):
    """User text for a composed request: caption clause + composed instruction."""
    instr = template.format(op=op_str)
    if cap is None:
        return instr
    clause = f'Caption: "{cap}".'
    return f"{clause} {instr}" if cap_first else f"{instr} {clause}"


def compose_answers(v1, v2):
    """The two grounded answers, in the requested order (the scored prefix)."""
    return (f"From the first requested source, the answer is {v1}. "
            f"From the second requested source, the answer is {v2}.")


def compose_verdict(relation):
    rel = "agree" if relation == "agree" else "conflict"
    return f" The requested sources {rel}. final answer: {rel}"


def compose_target(v1, v2, relation):
    return compose_answers(v1, v2) + compose_verdict(relation)


LADDER_FILLER = "lantern"   # neutral word that keeps the carrier sentence intact when the control moves (scripts/eval_relocation.py)


def ladder_ctrl_pos(native_txt, tpl, trig, cap, cap_first, pos):
    """Trained-position variants of the ladder word prompt (2026-10-19, train.py --ladder_ctrl_pos). prefix = the control
    before everything; postq = after the whole request; postcap = right after the caption clause; slot = native. The slot
    keeps a filler word so the carrier sentence stays grammatical, exactly as scripts/eval_relocation.py renders them."""
    if pos in ("", "native", "slot"):
        return native_txt
    body = render_ctrl(tpl, LADDER_FILLER, cap, cap_first=cap_first)
    if pos == "prefix":
        return f"{trig} {body}"
    if pos == "postq":
        return f"{body} {trig}"
    if pos == "postcap":
        clause = f'Caption: "{cap}".'
        i = body.find(clause)
        if i >= 0:
            j = i + len(clause)
            return f"{body[:j]} {trig}{body[j:]}"
        return f"{body} {trig}"
    raise ValueError(f"ladder control position {pos!r}")
