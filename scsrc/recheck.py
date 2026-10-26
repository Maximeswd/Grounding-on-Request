"""<recheck>: a self-invocable grounding-interrupt token (VisualSwap protocol).

The setup that no baseline can do: the model has reasoned over image A and
concluded value vA; the image is now B (value vB != vA); a trigger is inserted
in the reflection slot; does the continuation RE-GROUND to vB or persist with
the stale vA (textual inertia)? VisualSwap showed natural-language reflection
fails (up to -60%). We train ONE embedding row, <recheck>, whose meaning is
"discard the prior conclusion, re-ground to the current image".

Training triples come free from the twin pairs: current image = B (shows vB),
teacher-forced stale reasoning asserts vA (the twin's value), <recheck>, target
= vB. Only the <recheck> row is trainable. Interface-compatible with the
ObjectiveDataset path: __getitem__ -> {"op": "recheck", "sft": <example>}.

Eval (scripts/recheck_probe.py) replicates VisualSwap exactly via assistant
prefill: stale reasoning + trigger, then continue; flip-to-correct = vB. The
trigger conditions are the missing rows of their Appendix H table: <recheck>
(trained) vs natural-language phrase vs clean filler vs high-PPL string.
"""
import json
import os
import random

from PIL import Image

from .collate import _user_content, _encode, _prompt_len
from .tokens import trigger as _trig

QUESTION = "What is the color of the {obj}? Answer with the color you see now."
STALE = "Looking at the earlier image, I concluded the {obj} is {vA}."


def _common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build_recheck_example(processor, image, obj, v_stale, v_true, device,
                          trigger_str, n_tokens=1):
    """assistant = stale-reasoning + trigger + target; loss ONLY on the target,
    so the trigger row learns to override the stale conclusion using the current
    image. The mask boundary is found tokenization-safely: encode WITH and
    WITHOUT the target; the two sequences share every token up to where the
    target begins, so the common-prefix length is exactly the mask boundary."""
    q = QUESTION.format(obj=obj)
    stale = STALE.format(obj=obj, vA=v_stale)
    target = f"final answer: {obj}: {v_true}"
    pre = f"{stale} {trigger_str}"
    umsg = [{"role": "user", "content": _user_content(image, q)}]
    fmsg = umsg + [{"role": "assistant",
                    "content": [{"type": "text", "text": pre + " " + target}]}]
    pmsg = umsg + [{"role": "assistant", "content": [{"type": "text", "text": pre}]}]
    enc = _encode(processor, fmsg, device)
    enc_pre = _encode(processor, pmsg, device)
    start = _common_prefix_len(enc["input_ids"][0].tolist(),
                               enc_pre["input_ids"][0].tolist())
    labels = enc["input_ids"].clone()
    labels[:, :start] = -100
    out = {"input_ids": enc["input_ids"],
           "attention_mask": enc["attention_mask"], "labels": labels}
    for k in enc:
        if k not in ("input_ids", "attention_mask", "token_type_ids") \
                and k not in out:
            out[k] = enc[k]
    return out


class RecheckDataset:
    """Twin pairs (default VCF, single-object) -> recheck training triples,
    both swap directions. Trains the <recheck_ctrl> row only."""

    def __init__(self, data_dir, split, processor, device, n_tokens=1,
                 slot_negative_frac=0.0):
        from .data import load_task, build_specs
        ds = load_task(data_dir, split)
        self.ds = ds
        self.processor, self.device, self.n = processor, device, n_tokens
        self.snf = float(slot_negative_frac)
        self._rng = random.Random(4242)
        # one triple per conflict spec: current image = the shown one (v_true),
        # stale value = the other source's value (guaranteed != v_true)
        self.items = []
        for sp in build_specs(ds, split, conditions=("conflict",)):
            if sp.cond != "conflict":
                continue
            row = ds[sp.ex_index]
            img = row.get(f"image_{sp.img_idx}")
            if img is None:
                continue
            ik = {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
                  for o, v in (sp.image_key or {}).items()}
            tk = {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
                  for o, v in (sp.text_key or {}).items()}
            if not ik or set(ik) != set(tk):
                continue
            obj = next(iter(ik))
            v_true, v_stale = ik[obj], tk[obj]
            if v_true == v_stale:
                continue
            self.items.append((sp.ex_index, sp.img_idx, obj, v_stale, v_true))
        if not self.items:
            raise ValueError("RecheckDataset: no usable twin conflict items")
        self.weights = [1.0] * len(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ex_index, img_idx, obj, v_stale, v_true = self.items[i]
        img = self.ds[ex_index].get(f"image_{img_idx}")
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        trg = _trig("recheck_ctrl", self.n)
        ex = build_recheck_example(self.processor, img, obj, v_stale, v_true,
                                   self.device, trg, self.n)
        return {"op": "recheck", "sft": ex}
