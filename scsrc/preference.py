"""APO (anchored preference) + VGM (visual grounding margin) + the unified
objective. Per-operator coefficients are supported so you can assign, e.g.,
VGM to <look> (vision grounding) and APO to <read>/<compare> (text/relation),
each with an SFT base -- while still running the pure ablations.

Formats MATCH collate: ctrl tokens are INPUT-side (in the user instruction);
the preferred/rejected differ only in the OUTPUT continuation. So:
  look : chosen = image colors, rejected = CAPTION colors (laundering) [conflict only]
  read : chosen = caption colors, rejected = IMAGE colors             [conflict only]
  compare: chosen = true relation, rejected = flipped relation        [any]
  gen  : chosen = separated report, rejected = collapsed report       [conflict only]
VGM (look only): same (true) image-color continuation, TRUE vs SWAPPED image.

APO is ANCHORED to the model AT INIT (only the new rows differ), which is free.
Verify the exact APO variant/sign against the APO paper before trusting numbers.
"""
from contextlib import nullcontext, contextmanager
import os
import torch
import torch.nn.functional as F

from .prompts import (fmt_key, fmt_key_task, render_ctrl, pick_ctrl_template, gen_template,
                      ctrl_canonical)
from .tokens import trigger
from .collate import _user_content, _encode, _prompt_len
from .data import get_image


# ---- per-operator objective resolution -------------------------------------
def resolve_obj(obj_global, per_op, op):
    """Return flat coefficients for this op. per_op (or None) overrides globals
    for keys it specifies. op None (gen/fuse) -> globals."""
    eff = dict(obj_global)
    if per_op and op in (per_op or {}):
        eff.update(per_op[op])
    return eff


def uses(obj_global, per_op, key):
    if obj_global.get(key, 0) > 0:
        return True
    return any(v.get(key, 0) > 0 for v in (per_op or {}).values())


def _flip(rel):
    return "conflict" if rel == "agree" else "agree"


def _seq(processor, image, user_text, continuation, device):
    """Encode [user(+image)] + [assistant: continuation]; record continuation
    start (tokens) so logprob sums over the continuation only. (ctrl tokens live
    in user_text now, so there is no assistant prefill.)"""
    umsg = [{"role": "user", "content": _user_content(image, user_text)}]
    fmsg = umsg + [{"role": "assistant", "content": [{"type": "text", "text": continuation}]}]
    enc = _encode(processor, fmsg, device)
    plen = _prompt_len(processor, umsg, device)
    out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
           "cont_start": plen}
    for k in enc:   # pixel_values + model-specific vision kwargs (image_grid_thw)
        if k not in ("input_ids", "attention_mask", "token_type_ids") \
                and k not in out:
            out[k] = enc[k]
    return out


_CORE_KEYS = ("input_ids", "attention_mask", "labels", "cont_start",
              "score_start", "token_type_ids")


def vision_kwargs(ex):
    """All processor outputs beyond the core keys (pixel_values, and e.g.
    image_grid_thw for Qwen -- required by its vision tower)."""
    return {k: v for k, v in ex.items() if k not in _CORE_KEYS}


# ---- builders ---------------------------------------------------------------

def _lcp_len(a, b):
    """Length of the longest common prefix of two id lists."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _mask_shared_prefix(ch, rj):
    """APO must not push down the SHARED natural-format scaffold: since chosen
    and rejected share a long identical prefix ('Based on the requested source,
    the answer is'), whole-sequence scoring lets the reject gradient suppress
    the very format SFT is teaching (observed: ce 0.84 -> 4.06 after the ramp,
    outputs regress to base-model rambles). Setting score_start past the common
    token prefix makes the shared scaffold contribute EXACTLY zero preference
    gradient; the margin is computed over the differing (value) tokens only."""
    if not (isinstance(ch, dict) and isinstance(rj, dict)
            and "input_ids" in ch and "input_ids" in rj
            and "cont_start" in ch and "cont_start" in rj):
        return ch, rj                      # test stubs / degenerate encodings
    cs = int(ch["cont_start"])
    a = ch["input_ids"][0, cs:].tolist()
    b = rj["input_ids"][0, cs:].tolist()
    lcp = _lcp_len(a, b)
    lcp = min(lcp, max(min(len(a), len(b)) - 1, 0))   # always score >= 1 token
    ch["score_start"] = cs + lcp
    rj["score_start"] = cs + lcp
    return ch, rj


from .prompts import LORA_INSTR   # single definition (2026-09-21); collate's request-LoRA branch uses it too


def _ctrl_user(op, n, cap, template, method="neo"):
    """The request carrier for pair prompts, per method. neo: the word inline
    via the ctrl template. pt: the per-op soft prefix -- prepended (deployment
    form) or inserted at the word slot under PT_INSERTED=1 (placement control).
    Pairs for method=pt were previously IMPOSSIBLE (dataset gated on neo), so
    every PT run was SFT-only regardless of its lam_* flags -- the objective
    axis was confounded with the interface axis until 2026-08-05."""
    import os as _os
    if method == "lora" and _os.environ.get("SC_LORA_ADAPTER", "0") == "1":
        # v1.125: one adapter per request -- the prefix's neutral prompt without the prefix
        from .prompts import pt_template as _ptt, _PT_TEMPLATES as _PTT2
        return (template if template in _PTT2 else _ptt(0)).format(cap=cap)
    if method == "lora":
        # LoRA carries no learned request token: its request is the English instruction it is
        # evaluated with (scripts/eval_flip_ll.py INSTR_VARIANTS[0], the paper's phrasing), so a
        # LoRA + CTM pair must use exactly that carrier or the objective trains a different prompt
        # from the one that is scored (2026-09-14).
        return f'Caption: "{cap}". {LORA_INSTR[op]}'
    if method == "pt":
        prefix = trigger(f"pt_{op}", n)
        if _os.environ.get("PT_INSERTED", "0") == "1":
            # The INSERTED control must use a template with an {op} slot, or render_ctrl takes its
            # "{cap} is already in the template" branch and SILENTLY DROPS the prefix -- the pair
            # prompts then contain no learned rows at all. `template` here is a pt_template, which has
            # {cap} and no {op}, so the canonical word template is the only correct choice
            # (found 2026-09-15; it fired on the five pti runs that carry lam_ctm).
            # Rotate the WORD's templates for the inserted control too, keyed off which pt_template the
            # caller handed us so the two arms stay in step (spec is not in scope here).
            from .prompts import pick_ctrl_template as _pct, _PT_TEMPLATES as _PTT
            _i = _PTT.index(template) if template in _PTT else 0
            return render_ctrl(template if (template and "{op}" in template) else _pct(_i), prefix, cap)
        from .prompts import pt_template
        # Rotate with the item, as the word's pairs do. A fixed index here gave the prefix ONE
        # instruction in its counterfactual pairs while the word saw five, inside the very comparison
        # Section 6 rests on (found 2026-09-15).
        return prefix + " " + (template if template and "{cap}" in (template or "") else pt_template(0)).format(cap=cap)
    # v1.165 control-position ablation (train.py --word_first 1): the word's instruction BEFORE the caption in every
    # pair, as the prefix sits; default (unset) keeps the caption-first pairs every banked run was trained on
    return render_ctrl(template or ctrl_canonical(), trigger(f"{op}_ctrl", n), cap,
                       cap_first=_os.environ.get("SC_WORD_FIRST", "0") != "1")


def build_pref_pair(processor, ds, spec, arm, op, n, device, template=None, ctrl_style="neutral", task="color", method="neo"):
    if method in ("pt", "lora"):   # lora admitted 2026-09-14 (see build_ctm_pair)
        # pt runs carry arm="na"; their pair semantics ARE the ctrl semantics
        # (the prefix is the request carrier -- see _ctrl_user).
        arm = "ctrl"
    if arm == "ctrl":
        if op in ("look", "read") and spec.cond != "conflict":
            return None
        user = _ctrl_user(op, n, spec.fed_caption, template, method)
        from .prompts import natural_ctrl_content as _ncc
        _task = getattr(spec, 'task', task)
        if op == "look":
            ch = _ncc("look", fmt_key_task(spec.image_key, _task), _task, style=ctrl_style)
            rj = _ncc("look", fmt_key_task(spec.text_key, _task), _task, style=ctrl_style)
        elif op == "read":
            ch = _ncc("read", fmt_key_task(spec.text_key, _task), _task, style=ctrl_style)
            rj = _ncc("read", fmt_key_task(spec.image_key, _task), _task, style=ctrl_style)
        else:
            ch = _ncc("compare", spec.relation, _task, fmt_key_task(spec.image_key, _task), fmt_key_task(spec.text_key, _task))
            rj = _ncc("compare", _flip(spec.relation), _task, fmt_key_task(spec.image_key, _task), fmt_key_task(spec.text_key, _task))
        # identical neutral prefixes cancel in the APO margin -> the preference
        # signal is exactly the value-token logprob difference, as before
        img = get_image(ds, spec)
        _c, _r = _mask_shared_prefix(_seq(processor, img, user, ch, device),
                                     _seq(processor, img, user, rj, device))
        return {"chosen": _c, "rejected": _r}
    if arm == "link":
        # APO on the LINK format: ctrl word in the prompt, gen-block continuation.
        # chosen = own-source value inside the block + final answer outside;
        # rejected = the OTHER source's value in the same positions. The shared
        # block scaffold cancels in the margin, so the preference sits on the
        # value tokens exactly as in the ctrl branch -- but the gradient path
        # runs through the ctrl rows steering a block-formatted decode.
        if op not in ("look", "read") or spec.cond != "conflict":
            return None
        _task = getattr(spec, 'task', task)
        from .prompts import natural_gen_content as _ngc, gen_final
        user = render_ctrl(template or ctrl_canonical(), trigger(f"{op}_ctrl", n),
                           spec.fed_caption)
        gt = trigger(f"{op}_gen", n)
        v_own = fmt_key_task(spec.image_key if op == "look" else spec.text_key, _task)
        v_oth = fmt_key_task(spec.text_key if op == "look" else spec.image_key, _task)
        ch = f"{gt} {_ngc(op, v_own, _task)} {gt} {gen_final(op, v_own)}"
        rj = f"{gt} {_ngc(op, v_oth, _task)} {gt} {gen_final(op, v_oth)}"
        img = get_image(ds, spec)
        _c, _r = _mask_shared_prefix(_seq(processor, img, user, ch, device),
                                     _seq(processor, img, user, rj, device))
        return {"chosen": _c, "rejected": _r}
    if arm == "gen":
        if spec.cond != "conflict":
            return None
        # _task was referenced before assignment here since the branch was
        # written -- every call raised NameError, so gen+APO has never actually
        # trained (caught by the dataset zero-pair guard, 2026-07-31).
        _task = getattr(spec, 'task', task)
        user = (template or gen_template(_task, 0)).format(cap=spec.fed_caption)
        lt, rt, ct = trigger("look_gen", n), trigger("read_gen", n), trigger("compare_gen", n)
        from .prompts import natural_gen_content as _ngc
        task = _task
        ch = (f"{lt} {_ngc('look', fmt_key_task(spec.image_key, _task), task)} {lt} "
              f"{rt} {_ngc('read', fmt_key_task(spec.text_key, _task), task)} {rt} "
              f"{ct} {_ngc('compare', spec.relation, task, fmt_key_task(spec.image_key, _task), fmt_key_task(spec.text_key, _task))} {ct}")
        rj = (f"{lt} {_ngc('look', fmt_key_task(spec.image_key, _task), task)} {lt} "
              f"{rt} {_ngc('read', fmt_key_task(spec.image_key, _task), task)} {rt} "
              f"{ct} {_ngc('compare', 'agree', task, fmt_key_task(spec.image_key, _task), fmt_key_task(spec.image_key, _task))} {ct}")
        img = get_image(ds, spec)
        _c, _r = _mask_shared_prefix(_seq(processor, img, user, ch, device),
                                     _seq(processor, img, user, rj, device))
        return {"chosen": _c, "rejected": _r}
    return None


def build_vgm_pair(processor, ds, spec, arm, op, n, device, template=None, ctrl_style="neutral", task="color"):
    """Source-SENSITIVITY margin: swap the op's OWN source, keep the continuation
    equal to that source's value, and reward higher logprob under the TRUE source
    than the swapped one -- i.e. the block's output DEPENDS on its source.

      look : swap the IMAGE (shown vs the paired image), continuation = image value.
      read : swap the CAPTION (the conflict caption that asserts the text value,
             vs the shown image's OWN caption that asserts the image value),
             continuation = text value.  <- text-sensitivity, the symmetric twin.

    Together with build_pref_pair's APO (look rejects the text value; read rejects
    the image value), this gives the full causal-swap signature:
      look  = image-sensitive + text-invariant
      read  = text-sensitive  + image-invariant
    (An explicit swap-the-image-to-a-THIRD-value invariance probe for read needs
    3-value examples; the shown-image APO already enforces read!=image here.)
    compare/other: none."""
    if arm == "link" and op in ("look", "read"):
        # P(op-block | op_ctrl) > P(op-block | other_ctrl): the emitted block
        # TYPE must causally follow the control word, not just co-occur with it.
        _task = getattr(spec, 'task', task)
        from .prompts import natural_gen_content as _ngc, gen_final
        other = "read" if op == "look" else "look"
        gt = trigger(f"{op}_gen", n)
        v = fmt_key_task(spec.image_key if op == "look" else spec.text_key, _task)
        blk = f"{gt} {_ngc(op, v, _task)} {gt} {gen_final(op, v)}"
        u_own = render_ctrl(template or ctrl_canonical(), trigger(f"{op}_ctrl", n),
                            spec.fed_caption)
        u_oth = render_ctrl(template or ctrl_canonical(), trigger(f"{other}_ctrl", n),
                            spec.fed_caption)
        img = get_image(ds, spec)
        return {"pos": _seq(processor, img, u_own, blk, device),
                "neg": _seq(processor, img, u_oth, blk, device)}
    if arm != "ctrl" or op not in ("look", "read"):
        return None
    _task = getattr(spec, 'task', task)
    if op == "look":
        user = render_ctrl(template or ctrl_canonical(), trigger("look_ctrl", n), spec.fed_caption)
        from .prompts import natural_ctrl_content as _ncc
        cont = _ncc("look", fmt_key_task(spec.image_key, _task), _task, style=ctrl_style)
        img_true = get_image(ds, spec)
        img_swap = ds[spec.ex_index][f"image_{3 - spec.img_idx}"]
        return {"pos": _seq(processor, img_true, user, cont, device),
                "neg": _seq(processor, img_swap, user, cont, device)}
    # read: caption swap. Needs a real conflict (distinct image vs text values).
    if spec.cond != "conflict":
        return None
    from .prompts import natural_ctrl_content as _ncc2
    cont = _ncc2("read", fmt_key_task(spec.text_key, _task), _task, style=ctrl_style)  # caption-asserted value
    cap_true = spec.fed_caption                                     # asserts the text value
    cap_swap = ds[spec.ex_index][f"caption_{spec.img_idx}"]         # shown image's own caption -> image value
    img = get_image(ds, spec)
    user_pos = render_ctrl(template or ctrl_canonical(), trigger("read_ctrl", n), cap_true)
    user_neg = render_ctrl(template or ctrl_canonical(), trigger("read_ctrl", n), cap_swap)
    return {"pos": _seq(processor, img, user_pos, cont, device),
            "neg": _seq(processor, img, user_neg, cont, device)}


# ---- logprob + reference ----------------------------------------------------
def build_tc_pair(processor, ds, spec, arm, op, n, device, template=None, ctrl_style="neutral", task="color"):
    """TOKEN-CONTRAST: same image, same gold continuation (the op's OWN source
    value) -- scored under the op's OWN token vs the OTHER op's token. The
    margin separates what the two operators DO without ever pushing any output
    string down globally (the diagnosed failure mode of APO: suppressing the
    other channel's value string poisons the twin operator's vocabulary).
    Gradient reaches ONLY the two token embeddings. ctrl look/read only."""
    if arm != "ctrl" or op not in ("look", "read"):
        return None
    _task = getattr(spec, 'task', task)
    other = "read" if op == "look" else "look"
    src_key = spec.image_key if op == "look" else spec.text_key
    if spec.cond != "conflict":
        return None   # values equal in agree items: contrast undefined
    from .prompts import natural_ctrl_content as _ncc
    _t = _task
    cont = _ncc(op, fmt_key_task(src_key, _t), _t, style=ctrl_style)
    img = get_image(ds, spec)
    u_own = render_ctrl(template or ctrl_canonical(), trigger(f"{op}_ctrl", n), spec.fed_caption)
    u_oth = render_ctrl(template or ctrl_canonical(), trigger(f"{other}_ctrl", n), spec.fed_caption)
    return {"pos": _seq(processor, img, u_own, cont, device),
            "neg": _seq(processor, img, u_oth, cont, device)}


def build_compose_ctm_pair(processor, ds, spec, n, device, task="color", cap_first=True, method="neo",
                           srcs=None, edit_slot=None):
    """CTM for the COMPOSED prompt: both requests in one prompt, one edit, the joint answer must move.

    Composed items were CE-only for every interface (dataset.py: "composed items are CE-only"), so the
    counterfactual objective never touched the format the composed conflict detector is scored in. This
    is that gap closed. In a composed request over (s1, s2) the answer names both grounded values and a
    verdict. Edit the FIRST requested source; on the ladder its twin carries the other source's value,
    so the joint answer moves from (v1, v2, conflict) to (v2, v2, agree):
        A) score(old joint answer | base) > score(old joint answer | edited)
        B) score(new joint answer | edited) > score(new joint answer | base)
    Satisfying both trains, in one term, that the first slot follows its source, the second slot holds,
    and the verdict follows the edit -- which is exactly what the one-prompt detector is asked to do.
    """
    from .prompts import compose_template, compose_carrier, compose_request, compose_target
    if spec.cond != "conflict":
        return None
    _task = getattr(spec, "task", task)
    ix = int(getattr(spec, "ex_index", 0))
    if srcs is None:
        srcs = ("read", "look") if ix % 2 else ("look", "read")   # legacy: order by item parity
    if edit_slot is None:
        edit_slot = 0                                              # legacy: twin the FIRST slot only
    edit_src = srcs[edit_slot]   # 2026-09-21: the second slot can be twinned too (composed Stay)
    v_look = fmt_key_task(spec.image_key, _task)
    v_read = fmt_key_task(spec.text_key, _task)
    base = {"look": v_look, "read": v_read}
    img = get_image(ds, spec)
    cap = spec.fed_caption
    img_b, cap_b = img, cap
    edited = dict(base)
    if edit_src == "look":                      # the twin image carries the caption's value
        img_b = ds[spec.ex_index][f"image_{3 - spec.img_idx}"]
        edited["look"] = v_read
    else:                                       # the twin caption carries the image's value
        cap_b = ds[spec.ex_index][f"caption_{spec.img_idx}"]
        edited["read"] = v_look
    if edited[edit_src] == base[edit_src]:
        return None
    op_str, prefix = compose_carrier(method, "srconly", srcs, n)
    tpl = compose_template(ix // 2)
    u_a = compose_request(op_str, cap, tpl, cap_first=cap_first)
    u_b = compose_request(op_str, cap_b, tpl, cap_first=cap_first)
    if prefix:
        u_a, u_b = prefix + " " + u_a, prefix + " " + u_b
    rel_b = "agree" if edited["look"] == edited["read"] else "conflict"
    t_a = compose_target(base[srcs[0]], base[srcs[1]], spec.relation)
    t_b = compose_target(edited[srcs[0]], edited[srcs[1]], rel_b)
    return {"a_pos": _seq(processor, img, u_a, t_a, device),
            "a_neg": _seq(processor, img_b, u_b, t_a, device),
            "b_pos": _seq(processor, img_b, u_b, t_b, device),
            "b_neg": _seq(processor, img, u_a, t_b, device)}


_CP_VOCAB = {}


def _cp_vocab(ds):
    """Every key value in this split (both sides of every pair), lower-cased: colours on VCF and
    ColorSwap, entity nouns on COCO-CF. Computed once per dataset object."""
    k = id(ds)
    if k not in _CP_VOCAB:
        import json as _json
        vals = set()
        # column access: iterating rows would decode every image just to read two strings
        for kk in ("key_1", "key_2"):
            try:
                col = ds[kk]
            except Exception:
                continue
            for x in col:
                try:
                    vals.update(str(v).strip().lower() for v in _json.loads(x).values())
                except Exception:
                    pass
        vals.discard("object")                     # the generic placeholder is not a value a caption can state
        _CP_VOCAB[k] = sorted(vals)
    return _CP_VOCAB[k]


def cp_captions(ds, spec, op):
    """CONFLICT-PRESERVING PAIRS (2026-09-22, SC_CP_PAIRS=1 / train.py --cp_pairs). On a two-value
    ladder item the shipped twins make the two sources AGREE after the edit (the twin image carries the
    caption's value; the twin caption carries the image's value), so the CTM look pair is satisfiable by
    following the caption and the SM term's negative is a value no source asserts. The evaluator's
    third-value reading (eval_flip_ll --third_value) fixes this on the test side; this fixes it on the
    training side the same way, by moving the CAPTION instead of needing a third image: every queried
    object's caption value becomes a third value C (not the image's, not the caption's) and the twin
    caption gets a fourth value D, drawn deterministically from the split's own vocabulary.
    Returns (cap_C, key_C, cap_D, key_D) or None when a value cannot be replaced exactly once."""
    import os as _os, re as _re, random as _rnd
    if _os.environ.get("SC_CP_PAIRS", "0") != "1":
        return None
    vocab = _cp_vocab(ds)
    ik, tk = spec.image_key, spec.text_key
    rng = _rnd.Random(7000 + 7 * spec.ex_index + (1 if op == "look" else 2))
    capC = capD = spec.fed_caption
    keyC, keyD, used = {}, {}, set()
    _SYN = {"gray": "grey", "grey": "gray"}        # ColorSwap spells the key one way and the caption the other
    for obj, tv in tk.items():
        iv = ik.get(obj, tv)
        # an object whose NAME contains a value word (VCF's "west highland white terrier") cannot be moved to a
        # third value without the name still asserting one; the evaluator skips these objects, so do we
        if str(obj).lower() != "object" and any(w in vocab for w in str(obj).lower().split()):
            return None
        alt = _SYN.get(str(tv).lower())
        pat = _re.compile(rf"\b(?:{_re.escape(str(tv))}" + (rf"|{_re.escape(alt)}" if alt else "") + r")\b", _re.IGNORECASE)
        if len(pat.findall(capC)) != 1:
            return None
        cands = [v for v in vocab if v not in {str(iv).lower(), str(tv).lower()} and v not in used]
        if len(cands) < 2:
            return None
        C = rng.choice(cands); D = rng.choice([v for v in cands if v != C]); used |= {C, D}
        capC = pat.sub(C, capC); capD = pat.sub(D, capD)
        keyC[obj] = C; keyD[obj] = D
    # the moved caption must not state any source value anywhere else in its text (a caption that names the
    # image's value in another slot is ambiguous by construction and falls back to the shipped pair)
    for v in list(ik.values()) + list(tk.values()):
        if _re.search(rf"\b{_re.escape(str(v))}\b", capC, _re.IGNORECASE) or _re.search(rf"\b{_re.escape(str(v))}\b", capD, _re.IGNORECASE):
            return None
    return capC, keyC, capD, keyD


def build_ctm_pair(processor, ds, spec, arm, op, n, device, template=None, ctrl_style="neutral", task="color", method="neo"):
    if method in ("pt", "lora"):   # lora admitted 2026-09-14: its carrier is the English request (_ctrl_user)
        # pt runs carry arm="na"; their pair semantics ARE the ctrl semantics
        # (the prefix is the request carrier -- see _ctrl_user).
        arm = "ctrl"
    """COUNTERFACTUAL-TWIN MARGIN: the symmetric two-sided completion of VGM.

    VGM's one-sided margin (cont fixed, source swapped) can be satisfied by
    suppressing the continuation under the swap without ever strengthening the
    true-source pathway -- which is why VGM taxes look-control without buying
    flips. CTM uses BOTH twins and BOTH continuations:
      look: A) logP(v |I ) > logP(v |I')   B) logP(v'|I') > logP(v'|I )
      read: same with the caption twins and text values.
    Neither side can be won by global suppression; the only solution is answer-
    tracks-source. This trains exactly what flip/stay measures; tested on
    held-out twins/values/datasets. ctrl look/read, conflict items only."""
    if arm == "link" and op in ("look", "read"):
        _task = getattr(spec, 'task', task)
        from .prompts import natural_gen_content as _ngc, gen_final
        user = _ctrl_user(op, n, spec.fed_caption, template, method)
        gt = trigger(f"{op}_gen", n)
        blk = lambda v: f"{gt} {_ngc(op, v, _task)} {gt} {gen_final(op, v)}"
        if op == "look":
            img_a, img_b = get_image(ds, spec), ds[spec.ex_index][f"image_{3 - spec.img_idx}"]
            va = fmt_key_task(spec.image_key, _task)
            vb = fmt_key_task({k: v for k, v in zip(spec.image_key, spec.text_key.values())}
                              if False else spec.text_key, _task)
            # twins: value of image A vs value of image B == the text twin's key
            return {"a_pos": _seq(processor, img_a, user, blk(va), device),
                    "a_neg": _seq(processor, img_b, user, blk(va), device),
                    "b_pos": _seq(processor, img_b, user, blk(vb), device),
                    "b_neg": _seq(processor, img_a, user, blk(vb), device)}
        else:
            img = get_image(ds, spec)
            cap_b = ds[spec.ex_index][f"caption_{spec.img_idx}"]
            user_b = _ctrl_user(op, n, cap_b, template, method)
            va = fmt_key_task(spec.text_key, _task)
            vb = fmt_key_task(spec.image_key, _task)
            return {"a_pos": _seq(processor, img, user, blk(va), device),
                    "a_neg": _seq(processor, img, user_b, blk(va), device),
                    "b_pos": _seq(processor, img, user_b, blk(vb), device),
                    "b_neg": _seq(processor, img, user, blk(vb), device)}
    if arm != "ctrl" or op not in ("look", "read", "compare") \
            or spec.cond != "conflict":
        return None
    _task = getattr(spec, "task", task)
    from .prompts import natural_ctrl_content as _ncc
    if op == "compare":
        # COMPARE-CTM: the twin structure makes the verdict itself a
        # counterfactual. fed_caption describes the TWIN image, so
        #   (I ,c ) conflict   (I',c ) agree
        #   (I ,c') agree      (I',c') conflict
        # A) logP(conflict | I ,c ) > logP(conflict | I',c )
        # B) logP(agree    | I',c ) > logP(agree    | I ,c )
        # Neither side is winnable by a class prior -- which is exactly the
        # failure mode of SFT-only compare under look/read margin training
        # (the verdict collapses to 'agree').
        ik, tk = spec.image_key, spec.text_key
        c_con = _ncc("compare", "conflict", _task,
                     fmt_key_task(ik, _task), fmt_key_task(tk, _task))
        c_agr = _ncc("compare", "agree", _task,
                     fmt_key_task(ik, _task), fmt_key_task(ik, _task))
        if method == "lora":
            return None          # LoRA has no compare carrier (LORA_INSTR has look/read only)
        if method == "pt":
            # the prefix's own compare row, in the prefix's template (2026-09-21; the neo string
            # was rendered here before, so PT compare-CTM pairs carried no trainable request)
            from .prompts import pt_template as _ptt
            user = trigger("pt_compare", n) + " " + (template if template and "{cap}" in template
                                                     else _ptt(0)).format(cap=spec.fed_caption)
        else:
            user = render_ctrl(template or ctrl_canonical(),
                               trigger("compare_ctrl", n), spec.fed_caption)
        img = get_image(ds, spec)
        imgp = ds[spec.ex_index][f"image_{3 - spec.img_idx}"]
        return {"a_pos": _seq(processor, img, user, c_con, device),
                "a_neg": _seq(processor, imgp, user, c_con, device),
                "b_pos": _seq(processor, imgp, user, c_agr, device),
                "b_neg": _seq(processor, img, user, c_agr, device)}
    if op == "look":
        # twin images: shown image (value = image_key) vs the pair's other
        # image, whose value is exactly text_key (fed_caption describes it)
        # 2026-09-21: the METHOD-AWARE carrier (word / pt prefix / LoRA's English instruction), as
        # build_pref_pair always used. Until now this line rendered the neologism strings for every
        # method, so a PT or LoRA run's CTM pairs carried no trainable request at all and _bw skipped
        # the term silently (requires_grad False). Every ladder PT+CTM / LoRA+CTM run before this
        # date is VOID as a CTM comparison (STATUS 2026-09-21 v1.47).
        _cp = cp_captions(ds, spec, "look")
        # conflict-preserving: the caption names C in BOTH contexts, so the image edit A -> B keeps the sources apart
        user = _ctrl_user("look", n, _cp[0] if _cp else spec.fed_caption, template, method)
        v, vp = fmt_key_task(spec.image_key, _task), fmt_key_task(spec.text_key, _task)
        cont_v = _ncc("look", v, _task, style=ctrl_style)
        cont_vp = _ncc("look", vp, _task, style=ctrl_style)
        img, imgp = get_image(ds, spec), ds[spec.ex_index][f"image_{3 - spec.img_idx}"]
        return {"a_pos": _seq(processor, img, user, cont_v, device),
                "a_neg": _seq(processor, imgp, user, cont_v, device),
                "b_pos": _seq(processor, imgp, user, cont_vp, device),
                "b_neg": _seq(processor, img, user, cont_vp, device)}
    # read: twin captions on the SAME image
    _cp = cp_captions(ds, spec, "read")
    if _cp:
        # conflict-preserving: the caption moves B -> D while the image still shows A, so both contexts conflict
        v, vp = fmt_key_task(spec.text_key, _task), fmt_key_task(_cp[3], _task)
        cap, capp = spec.fed_caption, _cp[2]
    else:
        v, vp = fmt_key_task(spec.text_key, _task), fmt_key_task(spec.image_key, _task)
        cap, capp = spec.fed_caption, ds[spec.ex_index][f"caption_{spec.img_idx}"]
    cont_v = _ncc("read", v, _task, style=ctrl_style)
    cont_vp = _ncc("read", vp, _task, style=ctrl_style)
    img = get_image(ds, spec)
    u_cap = _ctrl_user("read", n, cap, template, method)      # method-aware carrier (2026-09-21)
    u_capp = _ctrl_user("read", n, capp, template, method)
    return {"a_pos": _seq(processor, img, u_cap, cont_v, device),
            "a_neg": _seq(processor, img, u_capp, cont_v, device),
            "b_pos": _seq(processor, img, u_capp, cont_vp, device),
            "b_neg": _seq(processor, img, u_cap, cont_vp, device)}


def build_sm_pair(processor, ds, spec, arm, op, n, device, template=None, ctrl_style="neutral", task="color", method="neo"):
    """STAY MARGIN (2026-09-21): the training analogue of conjunctive Stay. CTM trains the requested
    source's row of the do-table (answer tracks an edit to the REQUESTED source); nothing trains the
    competing row. Here the COMPETING source is edited exactly as the evaluator edits it and the
    requested value must still beat the alternative:
      look : caption swapped to the shown image's own caption (the evaluator's capswap);
             logP(v_img | I, cap') > logP(v_txt | I, cap')
      read : image swapped to the twin whose value the caption asserts (the evaluator's imgswap);
             logP(v_txt | I', cap) > logP(v_img | I', cap)
    On a two-value ladder item the competing edit turns the conflict into agreement, so the negative
    is the value no source asserts any more and the term is weak by construction; it exists here so
    one flag (--lam_sm) means the same thing on every benchmark. The three-value Ground-CF version
    (build_clevr3_sm_pair) is where the term has something to buy. ctrl look/read, conflict only."""
    if method in ("pt", "lora"):
        arm = "ctrl"
    if arm != "ctrl" or op not in ("look", "read") or spec.cond != "conflict":
        return None
    _task = getattr(spec, "task", task)
    from .prompts import natural_ctrl_content as _ncc
    _cp = cp_captions(ds, spec, op)
    if op == "look":
        # competing edit = caption; conflict-preserving: caption -> D (a value a source now asserts), negative = D
        v = fmt_key_task(spec.image_key, _task)
        vp = fmt_key_task(_cp[3], _task) if _cp else fmt_key_task(spec.text_key, _task)
        capp = _cp[2] if _cp else ds[spec.ex_index][f"caption_{spec.img_idx}"]
        user = _ctrl_user(op, n, capp, template, method)
        img = get_image(ds, spec)
        return {"pos": _seq(processor, img, user, _ncc("look", v, _task, style=ctrl_style), device),
                "neg": _seq(processor, img, user, _ncc("look", vp, _task, style=ctrl_style), device)}
    # competing edit = image (the pair image, value B); conflict-preserving: caption says C, so pos = C, neg = B
    if _cp:
        v, vp = fmt_key_task(_cp[1], _task), fmt_key_task(spec.text_key, _task)
        user = _ctrl_user(op, n, _cp[0], template, method)
    else:
        v, vp = fmt_key_task(spec.text_key, _task), fmt_key_task(spec.image_key, _task)
        user = _ctrl_user(op, n, spec.fed_caption, template, method)
    imgp = ds[spec.ex_index][f"image_{3 - spec.img_idx}"]
    return {"pos": _seq(processor, imgp, user, _ncc("read", v, _task, style=ctrl_style), device),
            "neg": _seq(processor, imgp, user, _ncc("read", vp, _task, style=ctrl_style), device)}


def seq_logprob(model, ex, no_grad=False):
    ctx = torch.no_grad() if no_grad else nullcontext()
    with ctx:
        pv = vision_kwargs(ex)
        out = model(input_ids=ex["input_ids"], attention_mask=ex["attention_mask"], **pv)
        logits = out.logits[:, :-1, :].float()
        targets = ex["input_ids"][:, 1:]
        logp = F.log_softmax(logits, dim=-1)
        tok = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        idx = torch.arange(tok.shape[1], device=tok.device)
        start = ex.get("score_start", ex["cont_start"])
        mask = (idx >= (start - 1)).to(tok.dtype)
        return (tok * mask).sum()


def seq_logprob_batch(model, exs, no_grad=False):
    """Continuation log-probabilities of SEVERAL examples in ONE forward pass (2026-09-22, STATUS v1.99).

    Right-padded batch over the example list; padded positions are masked out of attention and out of the
    score, so each row's value is the same per-token computation as seq_logprob(). This turns the four CTM
    sequences (and the two Stay sequences, and APO's two) into one forward and one backward each instead of
    one per sequence. The sequential path stays as the reference; SC_BATCH_CHECK=1 prints the difference
    on the first steps of a run. Requires the memory for the whole batch (80 GB cards); train.py's
    --batch_terms auto turns it on only there. token_type_ids is not passed, exactly as vision_kwargs()
    never passed it in the single-sequence path."""
    ctx = torch.no_grad() if no_grad else nullcontext()
    with ctx:
        B = len(exs); lens = [int(e["input_ids"].shape[1]) for e in exs]; T = max(lens)
        dev = exs[0]["input_ids"].device
        ids = torch.zeros((B, T), dtype=exs[0]["input_ids"].dtype, device=dev)
        att = torch.zeros((B, T), dtype=exs[0]["attention_mask"].dtype, device=dev)
        for b, e in enumerate(exs):
            ids[b, :lens[b]] = e["input_ids"][0]; att[b, :lens[b]] = e["attention_mask"][0]
        kw = {}
        for k in exs[0]:
            if k in _CORE_KEYS:
                continue
            v = exs[0][k]
            if torch.is_tensor(v):
                kw[k] = torch.cat([e[k] for e in exs], dim=0)
        out = model(input_ids=ids, attention_mask=att, **kw)
        logits = out.logits[:, :-1, :].float()
        targets = ids[:, 1:]
        tok = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)     # [B, T-1]
        idx = torch.arange(tok.shape[1], device=dev).unsqueeze(0)
        starts = torch.tensor([int(e.get("score_start", e["cont_start"])) for e in exs], device=dev).unsqueeze(1)
        ends = torch.tensor(lens, device=dev).unsqueeze(1)
        mask = ((idx >= (starts - 1)) & (idx < (ends - 1))).to(tok.dtype)
        return (tok * mask).sum(1)


_BATCH_CHECKS = {"n": 0, "max": 0.0}
_BATCH_TOL = 0.5      # bf16 batched vs sequential on |log p| ~ 10-30; the eval smoke measured <= .125


def _batch_check(model, term, exs, lps):
    """In-run guard (STATUS v1.99): on the first 8 batched terms of a run, compare with the sequential
    reference; print the difference; if it ever exceeds _BATCH_TOL, switch the run back to the sequential
    path for good and say so. Always on when batching is on (SC_BATCH_CHECK=0 disables the print only)."""
    if _BATCH_CHECKS["n"] >= 8:
        return
    with torch.no_grad():
        ref = torch.stack([seq_logprob(model, e, True) for e in exs])
    d = (lps.detach().float() - ref.float()).abs()
    _BATCH_CHECKS["n"] += 1; _BATCH_CHECKS["max"] = max(_BATCH_CHECKS["max"], float(d.max()))
    if os.environ.get("SC_BATCH_CHECK", "1") != "0":
        print(f"[batch-check] {term} {_BATCH_CHECKS['n']}/8: max|Δlogp|={float(d.max()):.4f} on |logp|~{float(ref.abs().mean()):.1f} "
              f"(seq {[round(float(x), 3) for x in ref]}, batched {[round(float(x), 3) for x in lps.detach()]})", flush=True)
    if float(d.max()) > _BATCH_TOL:
        os.environ["SC_BATCH_TERMS"] = "0"
        print(f"[batch-check] FALLBACK: |Δlogp| {float(d.max()):.3f} > {_BATCH_TOL}; this run continues on the "
              f"sequential path from here", flush=True)


def _batched_terms():
    return os.environ.get("SC_BATCH_TERMS", "") == "1" and os.environ.get("SC_LOWMEM", "") != "1"


def _batch_or_none(model, exs, no_grad=False):
    """seq_logprob_batch with an out-of-memory fallback: on OOM the run drops to the sequential path for good."""
    try:
        return seq_logprob_batch(model, exs, no_grad)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        os.environ["SC_BATCH_TERMS"] = "0"
        print("[batch-terms] out of memory on a batched pass; this run continues on the sequential path", flush=True)
        return None


@contextmanager
def reference_rows(model, ids, ref_rows):
    """APO-zero reference = the model as it was before this interface was trained.
    Embedding interfaces (word, prefix): the learned rows are swapped back to their initialisation.
    LoRA (2026-09-22, STATUS v1.96): there are no learned rows; the reference is the base model with the
    adapter disabled, which is the same statement -- the pretrained model without this interface. Before this
    train.py forced APO off for LoRA ("[obj] APO/VGM only apply to method=neo/pt"), so every LoRA+APO run was
    SFT with a different name; the two Ground-CF adapters were byte-identical.
"""
    if not ids or ref_rows is None:
        _d = getattr(model, "disable_adapter", None)
        if callable(_d):
            with _d():
                yield
            return
        raise RuntimeError("reference_rows: no learned rows and no LoRA adapter to disable -- "
                           "APO has no reference model for this interface")
    emb = model.get_input_embeddings().weight
    cur = emb.data[ids].clone()
    emb.data[ids] = ref_rows.to(emb.device, emb.dtype)
    try:
        yield
    finally:
        emb.data[ids] = cur


def objective_step(model, rec, eff, all_new_ids=None, ref_rows=None,
                   accum=1):
    """When accum > 0, each loss term is backwarded IMMEDIATELY (scaled by
    1/accum) so its graph is freed before the next term's forward runs --
    mathematically identical to summing then backwarding once, but the peak
    memory holds ONE vision graph instead of three (SFT + VGM pos + VGM neg
    OOMed a 44G L40S on Qwen). Returns (float_total, comps); train.py skips
    its own backward when the return is not a tensor."""
    def _bw(term, lam):
        if torch.is_tensor(term) and term.requires_grad:
            (lam * term / max(accum, 1)).backward()
        return float(term) * lam
    return _objective_terms(model, rec, eff, all_new_ids, ref_rows, _bw)


def _margin_lowmem(model, pos, neg, gamma, lam, _bw):
    """-logsigmoid(gamma*(lp_pos - lp_neg)) with ONE vision graph alive at a time.

    The pairwise margin does not separate the way APO does -- the sigmoid couples
    the two sequences -- so the naive form must hold both graphs. But the
    gradient does separate once the sigmoid is evaluated. With
    d = lp_pos - lp_neg and s = sigmoid(-gamma*d),

        dL/dlp_pos = -gamma*s        dL/dlp_neg = +gamma*s

    so the surrogate  -gamma*s*lp_pos + gamma*s*lp_neg  with s DETACHED has
    exactly the true gradients. Evaluate s under no_grad (two cheap forwards
    that build no graph), then forward+backward each side alone. Peak memory
    goes from two vision graphs to one, at the cost of two extra no-grad
    forwards -- sequence-level gradient checkpointing, in effect.

    This is what lets MultiVENT-Raw run the causal arms at n_frames 2 instead of
    dropping to 1 and making the SFT comparison a different experiment.
    """
    with torch.no_grad():
        d0 = seq_logprob(model, pos, True) - seq_logprob(model, neg, True)
        s = torch.sigmoid(-gamma * d0)
        loss_val = float(-F.logsigmoid(gamma * d0))
    _bw(-gamma * s * seq_logprob(model, pos), lam)
    _bw(gamma * s * seq_logprob(model, neg), lam)
    return loss_val * lam


def _objective_terms(model, rec, eff, all_new_ids, ref_rows, _bw):
    """eff is the per-op flat coefficient dict. Returns (loss_or_0, comps)."""
    comps, total = {}, 0.0
    if eff["lam_sft"] > 0 and rec.get("sft") is not None:
        sft = rec["sft"]
        pv = vision_kwargs(sft)
        if eff.get("vds_weight", 0) and "pixel_values" in sft:
            # VDS-weighted CE: weight label tokens by KL(p(.|I,x) || p(.|blank,x)),
            # down-weighting language-prior/syntax gradient (VGS-style reweighting).
            out = model(input_ids=sft["input_ids"], attention_mask=sft["attention_mask"], **pv)
            with torch.no_grad():
                blank = dict(pv)   # keep grid/shape kwargs; ablate content only
                blank["pixel_values"] = torch.zeros_like(sft["pixel_values"])
                out0 = model(input_ids=sft["input_ids"], attention_mask=sft["attention_mask"], **blank)
            lg, lg0 = out.logits[:, :-1], out0.logits[:, :-1]
            lab = sft["labels"][:, 1:]
            mask = lab != -100
            lp, lp0 = F.log_softmax(lg, -1), F.log_softmax(lg0, -1)
            vds = (lp.exp() * (lp - lp0)).sum(-1).clamp_min(0)           # KL per position
            w = (vds * mask).detach()
            w = w / w.sum().clamp_min(1e-6) * mask.sum().clamp_min(1)    # renormalize to CE scale
            ce_tok = F.cross_entropy(lg.transpose(1, 2), lab.clamp_min(0), reduction="none")
            loss = (ce_tok * w * mask).sum() / mask.sum().clamp_min(1)
            comps["ce"] = float(loss); comps["vds_mean"] = float((vds * mask).sum() / mask.sum().clamp_min(1))
            total += _bw(loss, eff["lam_sft"])
        elif eff.get("delim_w", 1.0) != 1.0 and eff.get("delim_ids"):
            # DELIMITER DOWN-WEIGHTING. In blocks/link targets the CE at the
            # delimiter positions competes with the CE at the grounded content:
            # in the LINK arm those positions must hit FROZEN gen-row directions
            # (hard, uninformative gradient) and the first link result suggests
            # this taxes grounding (.400 vs ctrl-sft .679 look_flip). delim_w<1
            # keeps the format supervision but caps how much of the budget it
            # can take. delim_w=0 removes format pressure entirely (the model
            # then only sees delimiters as CONTEXT).
            out = model(input_ids=sft["input_ids"], attention_mask=sft["attention_mask"], **pv)
            lg = out.logits[:, :-1]
            lab = sft["labels"][:, 1:]
            mask = lab != -100
            ce_tok = F.cross_entropy(lg.transpose(1, 2), lab.clamp_min(0), reduction="none")
            w = torch.ones_like(ce_tok)
            did = torch.tensor(sorted(eff["delim_ids"]), device=lab.device)
            w = torch.where(torch.isin(lab.clamp_min(0), did),
                            torch.full_like(w, float(eff["delim_w"])), w)
            loss = (ce_tok * w * mask).sum() / (w * mask).sum().clamp_min(1)
            comps["ce"] = float(loss)
            comps["delim_frac"] = float((torch.isin(lab.clamp_min(0), did) & mask).float().sum()
                                        / mask.sum().clamp_min(1))
            total += _bw(loss, eff["lam_sft"])
        else:
            out = model(input_ids=sft["input_ids"], attention_mask=sft["attention_mask"],
                        labels=sft["labels"], **pv)
            comps["ce"] = float(out.loss); total += _bw(out.loss, eff["lam_sft"])
    # KEY MISMATCH, fixed 2026-07-30. The generic ObjectiveDataset (color/vcf/coco)
    # stores the APO pair under "pref", but every task with its own dataset class
    # -- clevr3.py:323, mvg.py:436, mvraw.py:348, temporal.py:189 -- stores it
    # under "apo". Reading only "pref" meant lam_apo was silently a no-op on
    # Ground-CF, MVG, MultiVENT-Raw and the temporal set: `--lam_apo 1` trained
    # pure SFT. Proven on disk -- clevr3_f_5tok_sft_seed0 and
    # clevr3_f_5tok_apo_seed0 have byte-identical (15, 2560) rows, while ctm/v3
    # (whose keys DO match) differ by mean .044. vgm/tc/ctm/ecc were never
    # affected: each reads the key its dataset writes.
    _apo = rec.get("apo") or rec.get("pref")
    if eff.get("lam_apo", 0) > 0 and _apo is not None:
        # ONE GRAPH AT A TIME. The APO loss is
        #     -logsigmoid(b*(lp_c - lp_c0)) - logsigmoid(-b*(lp_r - lp_r0))
        # and the two halves are INDEPENDENT: the first depends only on the
        # chosen sequence, the second only on the rejected one. Computing both
        # log-probs before backwarding therefore held two full vision graphs
        # simultaneously for no reason -- which is what OOMed MultiVENT-Raw on a
        # 139.8 GiB H200, not just on the 44 GiB L40S. Backwarding each half as
        # soon as its forward is done frees the first graph before the second is
        # built. Mathematically identical; the reference passes stay under
        # no_grad and build no graph at all.
        ch, rj = _apo["chosen"], _apo["rejected"]
        lp0 = None
        if _batched_terms():
            with reference_rows(model, all_new_ids, ref_rows):
                lp0 = _batch_or_none(model, [ch, rj], True)
        lps = _batch_or_none(model, [ch, rj]) if lp0 is not None else None
        if lps is not None:
            _batch_check(model, "apo", [ch, rj], lps)
            a_ = -F.logsigmoid(eff["beta"] * (lps[0] - lp0[0])) - F.logsigmoid(-eff["beta"] * (lps[1] - lp0[1]))
            comps["apo"] = float(a_); total += _bw(a_, eff["lam_apo"])
        else:
            with reference_rows(model, all_new_ids, ref_rows):
                lp_c0, lp_r0 = seq_logprob(model, ch, True), seq_logprob(model, rj, True)
            a_c = -F.logsigmoid(eff["beta"] * (seq_logprob(model, ch) - lp_c0))
            comps["apo"] = float(a_c); total += _bw(a_c, eff["lam_apo"])
            a_r = -F.logsigmoid(-eff["beta"] * (seq_logprob(model, rj) - lp_r0))
            comps["apo"] += float(a_r); total += _bw(a_r, eff["lam_apo"])
    if eff.get("lam_vgm", 0) > 0 and rec.get("vgm") is not None:
        if os.environ.get("SC_LOWMEM", "") == "1":
            comps["vgm"] = total_add = _margin_lowmem(
                model, rec["vgm"]["pos"], rec["vgm"]["neg"], eff["gamma"],
                eff["lam_vgm"], _bw)
            total += total_add
        else:
            lp_p, lp_n = seq_logprob(model, rec["vgm"]["pos"]), seq_logprob(model, rec["vgm"]["neg"])
            vgm = -F.logsigmoid(eff["gamma"] * (lp_p - lp_n))
            comps["vgm"] = float(vgm); total += _bw(vgm, eff["lam_vgm"])
    if eff.get("lam_tc", 0) > 0 and rec.get("tc") is not None:
        if os.environ.get("SC_LOWMEM", "") == "1":
            comps["tc"] = t_ = _margin_lowmem(model, rec["tc"]["pos"], rec["tc"]["neg"],
                                              eff["gamma"], eff["lam_tc"], _bw)
            total += t_
        else:
            lp_o, lp_x = seq_logprob(model, rec["tc"]["pos"]), seq_logprob(model, rec["tc"]["neg"])
            tc = -F.logsigmoid(eff["gamma"] * (lp_o - lp_x))
            comps["tc"] = float(tc); total += _bw(tc, eff["lam_tc"])
    if eff.get("lam_ctm", 0) > 0 and rec.get("ctm") is not None:
        # two one-sided margins, backwarded separately so peak memory holds
        # two vision graphs at a time (same budget as APO/VGM)
        c = rec["ctm"]
        _q = [c["a_pos"], c["a_neg"], c["b_pos"], c["b_neg"]]
        lps = _batch_or_none(model, _q) if _batched_terms() else None
        if lps is not None:
            # v1.99: the four twin sequences in one forward; one backward for both margins
            _batch_check(model, "ctm", _q, lps)
            la = -F.logsigmoid(eff["gamma"] * (lps[0] - lps[1]))
            lb = -F.logsigmoid(eff["gamma"] * (lps[2] - lps[3]))
            comps["ctm"] = float(la) + float(lb); total += _bw(la + lb, eff["lam_ctm"])
        elif os.environ.get("SC_LOWMEM", "") == "1":
            comps["ctm"] = _margin_lowmem(model, c["a_pos"], c["a_neg"],
                                          eff["gamma"], eff["lam_ctm"], _bw)
            comps["ctm"] += _margin_lowmem(model, c["b_pos"], c["b_neg"],
                                           eff["gamma"], eff["lam_ctm"], _bw)
            total += comps["ctm"]
        else:
            la = -F.logsigmoid(eff["gamma"] * (seq_logprob(model, c["a_pos"])
                                               - seq_logprob(model, c["a_neg"])))
            comps["ctm"] = float(la); total += _bw(la, eff["lam_ctm"])
            lb = -F.logsigmoid(eff["gamma"] * (seq_logprob(model, c["b_pos"])
                                               - seq_logprob(model, c["b_neg"])))
            comps["ctm"] += float(lb); total += _bw(lb, eff["lam_ctm"])
    if eff.get("lam_sft_twin", 0) > 0 and rec.get("ctm") is not None:
        # paired-SFT control (2026-09-26, reviewers R2-Q1 / R3-W1): the edited context (C', v') = the CTM record's
        # b_pos gets plain MEAN-token cross-entropy over its continuation (the SFT loss form on the twin), with no
        # cross-context coupling. Same twins and the same warm-up ramp as CTM, so only the coupling differs.
        c = rec["ctm"]; ex = c["b_pos"]
        out_t = model(input_ids=ex["input_ids"], attention_mask=ex["attention_mask"], **vision_kwargs(ex))
        lg = out_t.logits[:, :-1, :].float(); tg = ex["input_ids"][:, 1:]
        tokp = F.log_softmax(lg, dim=-1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)
        st = ex.get("score_start", ex["cont_start"])
        mk = (torch.arange(tokp.shape[1], device=tokp.device) >= (st - 1)).to(tokp.dtype)
        ce_t = -(tokp * mk).sum() / mk.sum().clamp_min(1)
        comps["ce_twin"] = float(ce_t); total += _bw(ce_t, eff["lam_sft_twin"])
    if eff.get("lam_sft_stay", 0) > 0 and rec.get("sm") is not None:
        # context-matched SFT control (2026-10-05, advisor note 4 item 2): the competing-edit context with the selected
        # source's answer, (C_u', v) = the Stay record's positive sequence, gets plain mean-token cross-entropy and no
        # margin. CTM+Stay versus this arm isolates the Stay MARGIN from exposure to the competing-edit contexts.
        ex = rec["sm"]["pos"]
        out_s = model(input_ids=ex["input_ids"], attention_mask=ex["attention_mask"], **vision_kwargs(ex))
        lg = out_s.logits[:, :-1, :].float(); tg = ex["input_ids"][:, 1:]
        tokp = F.log_softmax(lg, dim=-1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)
        st = ex.get("score_start", ex["cont_start"])
        mk = (torch.arange(tokp.shape[1], device=tokp.device) >= (st - 1)).to(tokp.dtype)
        ce_s = -(tokp * mk).sum() / mk.sum().clamp_min(1)
        comps["ce_stay"] = float(ce_s); total += _bw(ce_s, eff["lam_sft_stay"])
    if eff.get("lam_sm", 0) > 0 and rec.get("sm") is not None:
        # STAY MARGIN: one margin per item, requested value over the alternative under an edit to
        # the COMPETING source (build_sm_pair / build_clevr3_sm_pair). Same gamma as CTM.
        s_ = rec["sm"]
        lps = _batch_or_none(model, [s_["pos"], s_["neg"]]) if _batched_terms() else None
        if lps is not None:
            _batch_check(model, "sm", [s_["pos"], s_["neg"]], lps)
            lsm = -F.logsigmoid(eff["gamma"] * (lps[0] - lps[1]))
            comps["sm"] = float(lsm); total += _bw(lsm, eff["lam_sm"])
        elif os.environ.get("SC_LOWMEM", "") == "1":
            comps["sm"] = _margin_lowmem(model, s_["pos"], s_["neg"], eff["gamma"], eff["lam_sm"], _bw)
            total += comps["sm"]
        else:
            lsm = -F.logsigmoid(eff["gamma"] * (seq_logprob(model, s_["pos"])
                                                - seq_logprob(model, s_["neg"])))
            comps["sm"] = float(lsm); total += _bw(lsm, eff["lam_sm"])
    if eff.get("lam_ecc", 0) > 0 and rec.get("ecc") is not None:
        # EVIDENCE-CROP CONSISTENCY: side A is the FULL frame, side B the bbox
        # CROP of the same context, same two candidates. The crop margin is the
        # teacher (detached): the full-frame margin is pushed up to meet it via a
        # hinge, so the token is trained to LOCALISE rather than to guess from
        # global scene gist. Detaching the teacher stops the trivial solution of
        # degrading the crop margin to close the gap.
        e = rec["ecc"]
        m_full = seq_logprob(model, e["a_pos"]) - seq_logprob(model, e["a_neg"])
        with torch.no_grad():
            m_crop = (seq_logprob(model, e["b_pos"])
                      - seq_logprob(model, e["b_neg"])).detach()
        le = F.relu(m_crop - m_full) / max(float(eff.get("gamma", 1.0)), 1e-6)
        comps["ecc"] = float(le); total += _bw(le, eff["lam_ecc"])
        comps["ecc_n"] = 1.0          # 2026-10-07 activation gate: fraction of records that carried an ECC pair
    if eff.get("lam_aat", 0) > 0 and rec.get("aat") is not None and rec.get("sft") is not None:
        # ATTENTION-ALIGNMENT: teach the token to allocate GAZE. Cross-entropy
        # between the decision positions' attention over image patches and the
        # evidence-object's patch mask (exact bbox from the renderer). Gradient
        # reaches only the token rows -- via the residual-stream contribution
        # to the answer positions' attention queries. Requires eager attention.
        sft = rec["aat"]
        ex, bbox, W_img = sft["ex"], sft["bbox"], sft.get("img_size", 336)
        pv = vision_kwargs(ex)
        out = model(input_ids=ex["input_ids"], attention_mask=ex["attention_mask"],
                    **pv, output_attentions=True)
        layer = int(eff.get("aat_layer", 12))
        att = out.attentions[layer][0]                       # [H, S, S]
        ids = ex["input_ids"][0]
        img_pos = (ids == sft["image_token_id"]).nonzero().squeeze(-1)
        lab = ex["labels"][0]
        ans_pos = (lab != -100).nonzero().squeeze(-1)[-6:]   # the answer tail
        g_side = int(len(img_pos) ** 0.5)
        if len(img_pos) and len(ans_pos) and g_side * g_side == len(img_pos):
            p = att[:, ans_pos][:, :, img_pos].mean(0).mean(0)
            p = p / p.sum().clamp_min(1e-8)
            x0, y0, x1, y1 = bbox
            mask = torch.zeros(g_side, g_side, device=p.device)
            j0, j1 = max(0, int(x0 / W_img * g_side)), min(g_side - 1, int(x1 / W_img * g_side))
            i0, i1 = max(0, int(y0 / W_img * g_side)), min(g_side - 1, int(y1 / W_img * g_side))
            mask[i0:i1 + 1, j0:j1 + 1] = 1
            mask = mask.flatten() / mask.sum().clamp_min(1)
            aat = -(mask * p.clamp_min(1e-8).log()).sum()
            comps["aat"] = float(aat); total += _bw(aat, eff["lam_aat"])
    if eff.get("lam_drift", 0) > 0 and all_new_ids is not None and ref_rows is not None:
        emb = model.get_input_embeddings().weight
        e = emb[all_new_ids].float(); r0 = ref_rows.to(e.device).float()
        d = (e - r0).norm(dim=-1)
        thr = eff.get("rho_drift", 0.5) * r0.norm(dim=-1).clamp_min(1e-6)
        drift = (F.relu(d - thr) ** 2).mean()
        comps["drift"] = float(drift); total += _bw(drift, eff["lam_drift"])
    if all_new_ids is not None:
        with torch.no_grad():
            _e = model.get_input_embeddings().weight[all_new_ids].float()
            comps["emb_maxnorm"] = float(_e.norm(dim=-1).max())
            if ref_rows is not None:
                comps["emb_maxdrift"] = float((_e - ref_rows.to(_e.device).float()).norm(dim=-1).max())
    return total, comps
