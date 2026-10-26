"""<before> / <after>: control words that select a POSITION ON THE TIME AXIS.

Every other operator in this project selects a SOURCE. These select a MOMENT
within one source, which is what lets the paper generalise from "which stream"
to "which moment" with the same 12.8k-parameter mechanism.

THE PROMPT NEVER NAMES THE DIRECTION. The question identifies only the ANCHOR
event; whether the answer should be the preceding or the following event is
carried solely by the word. That is what makes the word-swap a minimal pair --
and it is why the English instruction baseline (which must say "before" /
"after" out loud) is the fair comparison rather than a straw man.

THE TARGET NEVER NAMES THE DIRECTION EITHER. "Based on the requested temporal
relation, ..." keeps the same neutrality rule the source words follow: an
English gloss of the operator inside its own gold is the co-text crutch measured
to destroy causal grounding (sourced style look-flip .300 vs neutral .679).

TIME REVERSAL is the counterfactual. Presenting the identical frame set in
reversed order swaps the before/after answers exactly while holding pixel
content constant, so a model answering from appearance alone cannot flip. That
is a stricter intervention than any source swap, where the content necessarily
changes.
"""
import json
import os

from .mvg import _seq_mvg, _sft_mvg
from .tokens import trigger

ABSTAIN = "no annotated event is available at that position"
# direction-free question: the anchor is named, the relation is not
TEMP_Q = ("Which annotated event occurs at the requested position relative to "
          'the {role} ("{value}")?')
# the baselines have to say it in English -- that IS the baseline
INSTR = {"before": ('Which annotated event occurs IMMEDIATELY BEFORE the {role} '
                    '("{value}") in this video? Use the order of the frames.'),
         "after": ('Which annotated event occurs IMMEDIATELY AFTER the {role} '
                   '("{value}") in this video? Use the order of the frames.')}
NOOP = ('Which annotated event is adjacent in time to the {role} ("{value}")?')


def _trig(op, n_tokens, style="neo"):
    if style == "neo":
        return trigger(f"{op}_ctrl", n_tokens)
    if style == "pt":
        return trigger(f"pt_{op}", n_tokens)
    return ""


def temporal_target(value):
    return (f"Based on the requested temporal relation, the video shows "
            f"{value}. final answer: {value}")


def temporal_user_content(item, op, n_tokens=1, n_frames=8, style="neo",
                          reverse=False, shuffle_within=False, seed=0):
    """Frames only (no description/OCR: a narrated description often states the
    order, which would both leak the answer and make reversal inert).

    reverse=True         -> the ORDER-ONLY intervention (same frames, reversed)
    shuffle_within=True  -> LOCAL control: permute frames inside each half while
                            keeping the halves in order. Global order survives,
                            so a temporally grounded answer must NOT change --
                            this separates "reads frame order" from "is merely
                            sensitive to any pixel rearrangement".
    """
    from PIL import Image
    import random as _r
    fr = list(item["frames"])[:n_frames]
    if reverse:
        fr = fr[::-1]
    elif shuffle_within:
        h = len(fr) // 2
        a, b = fr[:h], fr[h:]
        rng = _r.Random(seed)
        rng.shuffle(a); rng.shuffle(b)
        fr = a + b
    content = [{"type": "text", "text": f"Video frames ({len(fr)}, in order):"}]
    for p in fr:
        content.append({"type": "image", "image": Image.open(p).convert("RGB")})
    q = {"role": item["anchor_role"], "value": item["anchor_value"]}
    if style == "neo":
        tail = TEMP_Q.format(**q) + f" Give me a {_trig(op, n_tokens, style)} answer."
    elif style == "pt":
        tail = TEMP_Q.format(**q)
    elif style == "instr":
        tail = INSTR[op].format(**q)
    else:
        tail = NOOP.format(**q)
    content.append({"type": "text", "text": tail})
    if style == "pt":
        content = [{"type": "text", "text": _trig(op, n_tokens, style)}] + content
    return content


def _seq(processor, item, op, gold, device, n_tokens, n_frames, style,
         reverse=False, shuffle_within=False):
    c = temporal_user_content(item, op, n_tokens, n_frames, style, reverse,
                              shuffle_within)
    return _seq_mvg(processor, c, temporal_target(gold), device)


def build_temporal_example(processor, item, device, n_tokens=1, n_frames=8,
                           style="neo"):
    """SFT record (labels form). _seq below is the SCORING form used by the
    preference/twin terms -- they are not interchangeable."""
    c = temporal_user_content(item, item["op"], n_tokens, n_frames, style)
    return {"op": item["op"],
            "sft": _sft_mvg(processor, c, temporal_target(item["gold"]), device)}


def build_temporal_ctm(processor, item, device, n_tokens=1, n_frames=8,
                       style="neo"):
    """REVERSAL TWIN. For a middle anchor with both neighbours:
        ctx A = frames as recorded  -> gold        (the requested neighbour)
        ctx B = frames reversed     -> other       (the two answers swap)
    One factor varies at a time:
        a_pos (A, gold)   a_neg (B, gold)
        b_pos (B, other)  b_neg (A, other)
    Content is identical in A and B, so this margin cannot be won by appearance.
    """
    if not item.get("reversible"):
        return None
    g, o = item["gold"], item["other"]
    f = lambda rev, val: _seq(processor, item, item["op"], val, device,
                              n_tokens, n_frames, style, reverse=rev)
    return {"a_pos": f(False, g), "a_neg": f(True, g),
            "b_pos": f(True, o), "b_neg": f(False, o)}


def build_temporal_apo(processor, item, device, n_tokens=1, n_frames=8,
                       style="neo"):
    """chosen = the requested neighbour; rejected = the OTHER direction's
    neighbour (or the non-adjacent distractor when no other-direction span
    exists). Same frames, same word: the only way to win is the relation."""
    rej = item.get("other") or item.get("far")
    if not rej or item["cond"] != "attested":
        return None
    return {"chosen": _seq(processor, item, item["op"], item["gold"], device,
                           n_tokens, n_frames, style),
            "rejected": _seq(processor, item, item["op"], rej, device,
                             n_tokens, n_frames, style)}


def build_temporal_tc(processor, item, device, n_tokens=1, n_frames=8,
                      style="neo"):
    """Same frames, same gold, scored under the OWN word vs the OPPOSITE word.
    Isolates what the word contributes without pushing any string down."""
    if item["cond"] != "attested":
        return None
    other_op = "after" if item["op"] == "before" else "before"
    return {"pos": _seq(processor, item, item["op"], item["gold"], device,
                        n_tokens, n_frames, style),
            "neg": _seq(processor, item, other_op, item["gold"], device,
                        n_tokens, n_frames, style)}


class TemporalDataset:
    def __init__(self, data_dir, split, processor, device, n_tokens=1,
                 n_frames=8, style="neo", apo=False, tc=False, ctm=False,
                 ops=None):
        self.processor, self.device = processor, device
        self.n, self.nf, self.style = int(n_tokens), int(n_frames), style
        self.apo, self.tc, self.ctm = bool(apo), bool(tc), bool(ctm)
        self.items = [json.loads(l) for l in open(f"{data_dir}/{split}.jsonl")]
        if ops:
            keep = {o.lower() for o in ops}
            self.items = [it for it in self.items if it["op"] in keep]
        self.weights = [it.get("weight", 1.0) for it in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        rec = build_temporal_example(self.processor, it, self.device,
                                     self.n, self.nf, self.style)
        if it["cond"] != "attested":
            return rec                      # abstention items are SFT-only
        if self.ctm:
            q = build_temporal_ctm(self.processor, it, self.device, self.n,
                                   self.nf, self.style)
            if q is not None:
                rec["ctm"] = q
        if self.apo:
            p = build_temporal_apo(self.processor, it, self.device, self.n,
                                   self.nf, self.style)
            if p is not None:
                rec["apo"] = p
        if self.tc:
            t = build_temporal_tc(self.processor, it, self.device, self.n,
                                  self.nf, self.style)
            if t is not None:
                rec["tc"] = t
        return rec
