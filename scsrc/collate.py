"""Training examples (input_ids + masked labels). Supervision spans:

  ctrl  : token in the USER turn (templated); output = the bare claim -> loss on
          the whole answer. (look->image colors, read->caption colors,
          compare->relation; compare is a standalone instruction here.)
  gen   : natural request (no {op}); model EMITS the tagged sequence -> loss on
          the whole output (tags + claims + relation).
  fuse  : single emitted token + one unattributed reading (collapse control).
  lora  : natural IMAGE:/TEXT:/RELATION: markers -> whole output supervised.

Run smoke_test.py: it prints a decoded example with the supervised span.
"""
import os
from .prompts import (fmt_key, fmt_key_task, fmt_separated_natural, render_ctrl,
                      pick_ctrl_template, gen_template, ctrl_canonical,
                      slot_word, render_decision)
from .tokens import trigger, token_strings


def _user_content(image, text):
    c = []
    if image is not None:
        c.append({"type": "image", "image": image})
    c.append({"type": "text", "text": text})
    return c


def _encode(processor, messages, device):
    return processor.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True,
        return_dict=True, return_tensors="pt").to(device)


def _prompt_len(processor, user_messages, device):
    enc = processor.apply_chat_template(
        user_messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(device)
    return enc["input_ids"].shape[1]


NEG_SLOT_WORDS = ("give", "please", "process", "banana", "compute", "kindly",
                  "quickly", "carefully", "value", "output")


def build_train_example(processor, spec, image, method, arm, n, device, template=None, task="color", cap_first=True, ctrl_style="neutral", absence=None, slot_negative=False, ctrl_site="user", mask_ctrl_lead=False):
    cap = spec.fed_caption
    if getattr(spec, "op", None) in ("compose", "compose_r"):
        # COMPOSED REQUEST (ctrl_style="compose"): two modality requests in one
        # prompt, order alternating on the item index, the answer gives both
        # grounded values in the requested order and a verdict. Same template,
        # same target for neo / pt / lora; only the request carrier differs.
        from .prompts import (compose_template, compose_carrier, compose_request,
                              compose_target)
        ix = int(getattr(spec, "ex_index", 0))
        # DEFAULT (2026-09-21): every spec is trained in BOTH orders -- the dataset emits a `compose`
        # (image-first) and a `compose_r` (caption-first) item per spec. The legacy recipe chose ONE
        # order per item by parity, so no item ever taught "same content, other order";
        # SC_COMPOSE_LEGACY_ORDER=1 reproduces it only for auditing the old runs.
        if getattr(spec, "op", None) == "compose_r":
            srcs = ("read", "look")
        elif os.environ.get("SC_COMPOSE_LEGACY_ORDER", "0") == "1":
            srcs = ("read", "look") if ix % 2 else ("look", "read")
        else:
            srcs = ("look", "read")
        op_str, prefix = compose_carrier(method, "srconly", srcs, n)
        user_txt = compose_request(op_str, cap, compose_template(ix // 2), cap_first=cap_first)
        if prefix:
            user_txt = prefix + " " + user_txt
        vals = {"look": fmt_key_task(spec.image_key, getattr(spec, "task", task)),
                "read": fmt_key_task(spec.text_key, getattr(spec, "task", task))}
        full_assistant = compose_target(vals[srcs[0]], vals[srcs[1]], spec.relation)
    elif method == "neo" and arm == "ctrl":
        from .prompts import natural_ctrl_content
        op = getattr(spec, "op", "look")
        op_trig = trigger(f"{op}_ctrl", n)
        user_txt = render_ctrl(template or ctrl_canonical(), op_trig, cap, cap_first=cap_first)
        _lpos = os.environ.get("SC_LADDER_CTRL_POS", "")
        if _lpos and op != "compare":                       # 2026-10-19 trained positions on the ladder
            from .prompts import ladder_ctrl_pos
            user_txt = ladder_ctrl_pos(user_txt, template or ctrl_canonical(), op_trig, cap, cap_first, _lpos)
            if not globals().get("_LADDER_POS_PRINTED"):
                globals()["_LADDER_POS_PRINTED"] = True
                print(f"[prompt] ladder control position {_lpos}: {user_txt[:160]!r}", flush=True)
        if op == "compare" and ctrl_style == "blockscmp":
            # ORDERED COMPOSITION: all three words in the PROMPT, three cited
            # blocks in the ANSWER, and the block order must match the word
            # order. read_first alternates on the item index so order can never
            # be a positional shortcut; the verdict block refers to the sources
            # as "first"/"second", so its text changes with the order too.
            from .prompts import composed_blocks_target, CMP_ORDER
            ix = int(getattr(spec, "ex_index", 0))
            read_first = bool(ix % 2)
            order = CMP_ORDER[(ix // 2) % len(CMP_ORDER)]
            words = ([trigger("read_ctrl", n), trigger("look_ctrl", n)]
                     if read_first else
                     [trigger("look_ctrl", n), trigger("read_ctrl", n)])
            wc = trigger("compare_ctrl", n)
            trig = " ".join(([wc] + words) if order == "clr" else (words + [wc]))
            user_txt = render_ctrl(template or ctrl_canonical(), trig, cap,
                                   cap_first=cap_first)
            full_assistant = composed_blocks_target(
                fmt_key_task(spec.image_key, getattr(spec, 'task', task)),
                fmt_key_task(spec.text_key, getattr(spec, 'task', task)),
                spec.relation, order=order, read_first=read_first)
        elif op == "compare":
            full_assistant = natural_ctrl_content("compare", spec.relation, task,
                                                  fmt_key_task(spec.image_key, getattr(spec, 'task', task)), fmt_key_task(spec.text_key, getattr(spec, 'task', task)),
                                                  seed=int(getattr(spec, "ex_index", 0)))
        elif op == "resolve":
            full_assistant = natural_ctrl_content("resolve", spec.relation, task,
                                                  fmt_key_task(spec.image_key, getattr(spec, 'task', task)), fmt_key_task(spec.text_key, getattr(spec, 'task', task)))
        elif op == "settle":
            # 50/50 trust designation, crossed with agree/conflict; the trust
            # bit is decorrelated from the cap_first rotation (dataset index)
            trust_img = (spec.ex_index % 2 == 0)
            clause = ("Trusted source: the image. " if trust_img
                      else "Trusted source: the caption. ")
            user_txt = clause + user_txt
            val = fmt_key_task(spec.image_key if trust_img else spec.text_key,
                               getattr(spec, "task", task))
            full_assistant = natural_ctrl_content("settle", spec.relation,
                                                  task, val)
        elif op == "arbiter":
            from .prompts import render_arbiter
            # deterministic 50/50 argument alternation, independent of the
            # cap_first rotation (which uses the dataset index, not ex_index)
            arg = "look" if spec.ex_index % 2 == 0 else "read"
            user_txt = render_arbiter(op_trig, trigger(f"{arg}_ctrl", n), cap,
                                      cap_first=cap_first)
            fav = spec.image_key if arg == "look" else spec.text_key
            oth = spec.text_key if arg == "look" else spec.image_key
            full_assistant = natural_ctrl_content(
                "arbiter", spec.relation, task,
                fmt_key_task(fav, getattr(spec, "task", task)),
                fmt_key_task(oth, getattr(spec, "task", task)))
        else:
            val = fmt_key_task(spec.image_key if op == "look" else spec.text_key, getattr(spec, "task", task))
            full_assistant = natural_ctrl_content(op, val, task, style=ctrl_style,
                                                  seed=int(getattr(spec, "ex_index", 0)))  # NO token in the output
            # ctrl_style=blocks: prompts.py already emits the 1-row delimiter
            # <look_gen>/<read_gen>, and add_neologisms registers gen words at
            # n=1 for this arm, so the target needs no substitution here.
        if ctrl_site == "decision" and op in ("look", "read", "compare"):
            # LEVERAGE ablation: token-free user turn; the ctrl token is
            # prefilled at the head of the assistant turn (loss-masked below),
            # so its query attends over the image AT the decision position.
            user_txt = render_decision(cap, cap_first=cap_first)
            full_assistant = f"{op_trig} {full_assistant}"
    elif method == "neo" and arm == "link":
        # ARM L: the control word INVOKES the corresponding pretrained gen block.
        # user turn uses the ctrl token; target is the op's gen block (gen token
        # embeddings frozen -- only ctrl rows learn "call this block").
        from .prompts import natural_gen_content
        op = getattr(spec, "op", "look")
        user_txt = render_ctrl(template or ctrl_canonical(), trigger(f"{op}_ctrl", n), cap, cap_first=cap_first)
        gt = trigger(f"{op}_gen", n)
        if op == "compare":
            body = natural_gen_content("compare", spec.relation, task,
                                       fmt_key_task(spec.image_key, getattr(spec, 'task', task)), fmt_key_task(spec.text_key, getattr(spec, 'task', task)))
        else:
            val = fmt_key_task(spec.image_key if op == "look" else spec.text_key, getattr(spec, "task", task))
            body = natural_gen_content(op, val, task)
        from .prompts import gen_final
        _v = spec.relation if op == "compare" else val
        full_assistant = f"{gt} {body} {gt} {gen_final(op, _v)}"
    elif method == "neo" and arm == "gen":
        lt, rt, ct = trigger("look_gen", n), trigger("read_gen", n), trigger("compare_gen", n)
        user_txt = (template or gen_template(task, 0)).format(cap=cap)
        # PAIRED blocks with NATURAL content + an anchored structured tail:
        #   <look_gen> Looking at the image, the answer is dog: 2. final answer: dog: 2 <look_gen>
        # The prose makes the output non-artificial; "final answer:" keeps it
        # exact-match/causal-swap evaluable without a judge.
        from .prompts import natural_gen_content
        from .prompts import gen_final
        _vi = fmt_key_task(spec.image_key, getattr(spec, 'task', task))
        _vt = fmt_key_task(spec.text_key, getattr(spec, 'task', task))
        full_assistant = (f"{lt} {natural_gen_content('look', _vi, task)} {lt} "
                          f"{rt} {natural_gen_content('read', _vt, task)} {rt} "
                          f"{ct} {natural_gen_content('compare', spec.relation, task, _vi, _vt)} {ct} "
                          f"{gen_final('compare', spec.relation)}")
    elif method == "neo" and arm == "fuse":
        ft = trigger("fuse", n)
        user_txt = 'Describe the scene in one combined reading. Caption: "%s".' % cap
        full_assistant = ft + " " + fmt_key_task(spec.image_key, getattr(spec, 'task', task))
    elif method == "pt":
        # prompt-tuning baseline: a learned soft prefix carries the steering; the
        # instruction is source-neutral (no "look"/"read" words), so any control
        # must come from the prefix embedding -- the fair contrast to an
        # addressable inline neologism. One prefix per operator (pt_look/read/compare).
        op = getattr(spec, "op", "look")
        prefix = trigger(f"pt_{op}", n)
        from .prompts import pt_template
        if os.environ.get("PT_INSERTED", "0") == "1":
            # LEXICALITY-vs-PLACEMENT control: identical soft rows, identical
            # parameter count, but INSERTED at the word slot via the same
            # template as the neologism. If the word only beats PREFIX pt,
            # the advantage is placement; if it also beats INSERTED pt, the
            # residue is vocabulary identity (id-hood: composable/generable).
            # The inserted control must sit in the SAME rotating templates as the word, or it differs
            # from the word in training context as well as in nothing else -- which is the confound that
            # made the S6 comparison unmatched (found 2026-09-15). `template` here is a pt_template,
            # which carries {cap} and no {op}, so rotate the WORD's templates by the same index.
            _ix = int(getattr(spec, "ex_index", 0))   # pick_ctrl_template: module-level import (v1.139)
            # 2026-09-21: use the template the DATASET chose (the word's own, same index as the
            # neologism's SFT item) when it has an {op} slot; the ex_index rotation is the fallback.
            user_txt = render_ctrl(template if (template and "{op}" in template) else pick_ctrl_template(_ix),
                                   prefix, cap)
        else:
            body = (template or pt_template(0)).format(cap=cap)
            user_txt = prefix + " " + body
        # SAME neutral natural gold as neo-ctrl: output distributions match, so
        # neo-vs-pt differences isolate the MECHANISM (addressable inline token
        # vs soft prefix), not the target format.
        from .prompts import natural_ctrl_content
        if op == "compare" and ctrl_style == "blockscmp":
            # ORDERED COMPOSITION: all three words in the PROMPT, three cited
            # blocks in the ANSWER, and the block order must match the word
            # order. read_first alternates on the item index so order can never
            # be a positional shortcut; the verdict block refers to the sources
            # as "first"/"second", so its text changes with the order too.
            from .prompts import composed_blocks_target, CMP_ORDER
            ix = int(getattr(spec, "ex_index", 0))
            read_first = bool(ix % 2)
            order = CMP_ORDER[(ix // 2) % len(CMP_ORDER)]
            words = ([trigger("read_ctrl", n), trigger("look_ctrl", n)]
                     if read_first else
                     [trigger("look_ctrl", n), trigger("read_ctrl", n)])
            wc = trigger("compare_ctrl", n)
            trig = " ".join(([wc] + words) if order == "clr" else (words + [wc]))
            user_txt = render_ctrl(template or ctrl_canonical(), trig, cap,
                                   cap_first=cap_first)
            full_assistant = composed_blocks_target(
                fmt_key_task(spec.image_key, getattr(spec, 'task', task)),
                fmt_key_task(spec.text_key, getattr(spec, 'task', task)),
                spec.relation, order=order, read_first=read_first)
        elif op == "compare":
            full_assistant = natural_ctrl_content("compare", spec.relation, task,
                                                  fmt_key_task(spec.image_key, getattr(spec, 'task', task)), fmt_key_task(spec.text_key, getattr(spec, 'task', task)),
                                                  seed=int(getattr(spec, "ex_index", 0)))
        else:
            _v = fmt_key_task(spec.image_key if op == "look" else spec.text_key, getattr(spec, "task", task))
            full_assistant = natural_ctrl_content(op, _v, task)
    elif method == "lora" and ctrl_style == "adapter":
        # 2026-09-23 (v1.125, author): ONE LoRA ADAPTER PER REQUEST. The visible prompt is exactly the prompt-tuning
        # prompt with the prefix removed (same rotating neutral templates); which modality is requested is carried by
        # WHICH adapter is trained/loaded, never by text.
        from .prompts import natural_ctrl_content, pt_template
        op = getattr(spec, "op", "look")
        assert op in ("look", "read"), f"adapter LoRA has look/read items only, got {op}"
        user_txt = (template or pt_template(0)).format(cap=cap)
        val = fmt_key_task(spec.image_key if op == "look" else spec.text_key, getattr(spec, "task", task))
        full_assistant = natural_ctrl_content(op, val, task)
    elif method == "lora" and ctrl_style == "request":
        # REQUEST-CONDITIONED LoRA (2026-09-21, review §1B): the same selected-source target as the
        # embedding interfaces, carried by the English instruction the evaluator uses. This is the
        # matched adaptation baseline; the joint-output recipe below stays as the historical one.
        from .prompts import natural_ctrl_content, LORA_INSTR
        op = getattr(spec, "op", "look")
        assert op in ("look", "read"), f"request-conditioned LoRA has look/read items only, got {op}"
        user_txt = f'Caption: "{cap}". {LORA_INSTR[op]}'
        val = fmt_key_task(spec.image_key if op == "look" else spec.text_key, getattr(spec, "task", task))
        full_assistant = natural_ctrl_content(op, val, task)
    elif method == "lora":
        _w = slot_word(task)
        user_txt = ("Report each object's %s according to the image and the "
                    "text, and whether they agree or conflict. Caption: \"%s\". "
                    "Format: IMAGE: obj: %s, ...; TEXT: obj: %s, ...; "
                    "RELATION: agree/conflict." % (_w, cap, _w, _w))
        full_assistant = fmt_separated_natural(spec.image_key, spec.text_key,
                                               spec.relation,
                                               task=getattr(spec, "task", task))
    else:
        raise ValueError(f"unknown method/arm {method}/{arm}")

    if slot_negative and method == "neo" and arm == "ctrl":
        import random as _r
        filler = _r.Random(spec.ex_index).choice(NEG_SLOT_WORDS)
        user_txt = render_ctrl(template or ctrl_canonical(), filler, cap, cap_first=cap_first)
        full_assistant = ("I do not recognize the requested operator, so I cannot "
                          "select a source. final answer: unknown operator")
    if absence in ("image", "caption"):
        # unavailable-source supervision: the requested channel is missing and
        # the HONEST answer names that instead of confabulating from the other.
        full_assistant = (f"Based on the requested source, the {absence} is not "
                          f"available. final answer: unavailable")
    user_msg = [{"role": "user", "content": _user_content(image, user_txt)}]
    full_msg = user_msg + [{"role": "assistant",
                            "content": [{"type": "text", "text": full_assistant}]}]
    enc = _encode(processor, full_msg, device)
    plen = _prompt_len(processor, user_msg, device)
    labels = enc["input_ids"].clone()
    labels[:, :plen] = -100
    if ctrl_site == "decision" and method == "neo" and arm == "ctrl":
        # the prefilled trigger is CONDITIONING, not a target: mask its ids so
        # the model is never trained to emit the token (parity with input-only)
        import torch as _t
        _op = getattr(spec, "op", "look")
        _tids = processor.tokenizer.convert_tokens_to_ids(
            token_strings(f"{_op}_ctrl", n))
        _mask = _t.isin(enc["input_ids"], _t.tensor(_tids, device=enc["input_ids"].device))
        labels[_mask] = -100
    if (mask_ctrl_lead and method == "neo" and arm == "ctrl"
            and getattr(spec, "op", "look") in ("look", "read")):
        # ANSWER-TOKEN-ONLY LOSS (look/read ctrl only).
        # SCOPED to look/read on purpose: compare/resolve/arbiter targets put
        # RECEIPTS before the anchor ("one source says X, the other says Y"),
        # which is discriminating provenance content, not boilerplate --
        # masking their leads would unlearn the very behaviour those words
        # exist to produce. gen/report arms are excluded by arm != "ctrl".
        # Slot-negative and absence targets need no special case: their values
        # ("unknown operator" / "unavailable") sit AFTER the anchor. The ctrl gold is
        # "{neutral_lead} final answer: {value}" and the lead is IDENTICAL for
        # look and read by design, so ~87% of the CE is spent on text that
        # cannot discriminate the channel. Mask the lead so the gradient budget
        # goes to the value.
        #
        # CRITICAL: the "final answer:" anchor stays SUPERVISED. evaluate.py
        # uses it as stop_anchor and scores trunc_* as its absence, so masking
        # it would stop the model emitting it and collapse every ctrl metric for
        # a parsing reason that looks exactly like a learning failure.
        #
        # Fail-safe: if the anchor is not located, mask NOTHING and record it,
        # rather than guess a span.
        import torch as _t
        _tk = processor.tokenizer
        _ids = enc["input_ids"][0].tolist()
        _found = -1
        for _cand in ("final answer:", " final answer:"):
            _a = _tk.encode(_cand, add_special_tokens=False)
            if not _a:
                continue
            for _i in range(plen, len(_ids) - len(_a) + 1):
                if _ids[_i:_i + len(_a)] == _a:
                    _found = _i
                    break
            if _found >= 0:
                break
        if _found > plen:
            labels[:, plen:_found] = -100
        else:
            os.environ["SC_MASK_ANCHOR_MISSES"] = str(
                int(os.environ.get("SC_MASK_ANCHOR_MISSES", "0")) + 1)
    out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
           "labels": labels}
    # pass through EVERY extra processor output (pixel_values, and e.g.
    # image_grid_thw for the Qwen family -- dropping it crashes the vision
    # tower with grid_thw=None; Gemma is unaffected)
    for k in enc:
        if k not in ("input_ids", "attention_mask", "token_type_ids") \
                and k not in out:
            out[k] = enc[k]
    return out


def collate(batch):
    assert len(batch) == 1, "bs=1 per forward; use grad accumulation"
    return batch[0]
