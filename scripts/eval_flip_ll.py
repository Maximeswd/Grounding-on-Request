#!/usr/bin/env python
"""LL-margin flip/stay battery -- causal grounding, generation-free, ALL methods.

For each atomic conflict item with twin image I' and twin caption cap':
  look_flip : argmax_v LL(v|I,cap) == v_img  AND  argmax_v LL(v|I',cap) == v_txt
              (the answer TRACKS the pixels; v_txt is the twin image's value)
  look_stay : <look>'s choice is UNCHANGED under the caption swap cap->cap'
  read_flip : <read>'s choice tracks the caption swap
  read_stay : <read>'s choice unchanged under the image swap
Margins (LL difference of the two candidates) are recorded so the same run
yields the continuous version. Teacher-forced scoring only -- no decode, so
format drift cannot contaminate the causal read-out. Methods: neo-ctrl (user
or decision site), pt (soft prefix), lora (English-instructed prompts).
Chance flip = .25 for an evidence-blind chooser.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger
from scsrc import prompts as P, score as SC

from evaluate import apply_checkpoint

# VCF's own 13 colour words (all key values across train/ID/OOD, 2026-09-21); the third-colour
# twins (STATUS v1.66) draw the conflict caption's colour C and the twin caption's colour D from here.
TC_VOCAB = {"green","brown","pink","purple","black","red","white","yellow","blue","orange","silver","grey","gold"}
COLORS = {"red","orange","yellow","green","blue","purple","pink","brown","black",
          "white","gray","grey","gold","silver","tan","beige","violet","cyan",
          "maroon","navy"}
INSTR_VARIANTS = [
    # v0 = the paper's original phrasing
    {"look": "Answer using ONLY what you can see in the image; ignore the caption.",
     "read": "Answer using ONLY what the caption states; ignore the image."},
    {"look": "Base your answer solely on the visual content of the image. Do not use the caption.",
     "read": "Base your answer solely on the caption text. Do not use the image."},
    {"look": "Look at the image and answer from it alone, even if the caption disagrees.",
     "read": "Read the caption and answer from it alone, even if the image disagrees."},
    {"look": "Your answer must reflect what the image shows. Disregard any text.",
     "read": "Your answer must reflect what the caption says. Disregard the picture."},
    {"look": "IMPORTANT: use the image as your only source of information; the caption may be wrong.",
     "read": "IMPORTANT: use the caption as your only source of information; the image may be wrong."},
    {"look": "According to the image (and only the image), what is the answer?",
     "read": "According to the caption (and only the caption), what is the answer?"},
]
INSTR = INSTR_VARIANTS[int(__import__("os").environ.get("INSTR_VARIANT", "0"))]

# 2026-09-23 (author, STATUS v1.119): HELD-OUT REQUEST TEMPLATE. The headline Flip/Stay are scored on
# CTRL_CANONICAL (neologism) and pt_template(0) (prompt tuning) -- both TRAINING templates -- so they never
# test template transfer. SC_TEMPLATE=heldout0|heldout1 swaps in the eval-only pools
# (P.CTRL_TEMPLATES_HELDOUT / P.pt_template_heldout) and writes flip_ll_heldout<k>_step_*.json, a name the
# harvester's exact globs do not match, so it can never overwrite the in-template cell.
_TPL = os.environ.get("SC_TEMPLATE", "")
# SC_TEMPLATE=heldout  -> per-item rotation over BOTH eval-only templates (crc32 of the caption picks one, so the
#                         choice is deterministic without plumbing an item index); ONE file, flip_ll_heldout_step_*.json
# SC_TEMPLATE=heldoutK -> template K only (diagnostic), flip_ll_heldoutK_step_*.json
# LoRA (request-conditioned, trained on INSTR_VARIANTS[0]) is held out onto variants 1 and 3, which it never saw.
# The frozen English baseline has no training phrasing and is untouched (INSTR_VARIANT selects it as before).
import zlib as _zlib
_TPLK = (int(_TPL[len("heldout"):]) if _TPL.startswith("heldout") and _TPL[len("heldout"):].isdigit()
         else None)
_TPL_ROT = _TPL == "heldout"
_tpltag = ("_heldout" if _TPL_ROT else f"_heldout{_TPLK}" if _TPLK is not None else "")
# v1.153 (2026-09-24): SC_LADDER_PROTO=2 = the FIXED protocol. The held-out template used to be chosen by hashing the
# caption of EACH context, so a caption edit could also switch the instruction (75/68/56 of 140/140/120 E4B items on
# COCO-CF/ColorSwap/VCF). Under protocol 2 the key is fixed per original example (the base context's caption), so every
# version of an example gets the same template; base-context prompts are identical to protocol 1.
_P2 = os.environ.get("SC_LADDER_PROTO", "1") == "2"
# v1.158 (author: check, then decide): SC_LADDER_LEADIN=1 scores the four protocol-2 candidates AFTER the training
# lead-in, exactly as Ground-CF scores its answers (scsrc.clevr3.C3_CONT). The bare "object: X" continuation puts every
# model at ~-25 log-prob on the first answer token (the models always begin with the lead-in); the lead-in is identical
# across candidates, so it cancels in the comparison. Files tagged _lead.
_LEADIN = os.environ.get("SC_LADDER_LEADIN", "0") == "1"
LEAD_TXT = "I used the requested source to answer the question. final answer: "
_TPL_KEY = [None]
def _k(cap):
    if _TPLK is not None:
        return _TPLK
    key = _TPL_KEY[0] if (_P2 and _TPL_KEY[0] is not None) else cap
    return _zlib.crc32(key.encode()) % 2
def _ctrl_tpl(cap=""):
    return P.CTRL_TEMPLATES_HELDOUT[_k(cap) % len(P.CTRL_TEMPLATES_HELDOUT)] if (_TPL_ROT or _TPLK is not None) else P.CTRL_CANONICAL
def _pt_tpl(cap=""):
    return P.pt_template_heldout(_k(cap)) if (_TPL_ROT or _TPLK is not None) else P.pt_template(0)
_INSTR_HO = (INSTR_VARIANTS[1], INSTR_VARIANTS[3])
def _lora_instr(op, cap):
    return _INSTR_HO[_k(cap) % 2][op] if (_TPL_ROT or _TPLK is not None) else INSTR[op]

# ---- ZERO-SHOT NEGATION (2026-09-17) ---------------------------------------------------------
# SC_REQ_FRAME rewrites the request slot to test whether the model's own language machinery can
# scope a negation over a word it never saw in pretraining. The frame is a MINIMAL edit of the
# canonical training template 'Give me a {op} answer.' -- only "Do not" is added and the word is
# swapped for the OTHER source -- so a null result cannot be blamed on a phrasing shift.
#   neg_word     Caption: "...". Do not give me a <look_ctrl_0..4> answer.   (requires method=neo)
#   neg_english  Caption: "...". Do not give me an image-based answer.       (any method)
#   pos_english  Caption: "...". Give me a caption-based answer.             (matched positive)
# The target is left IMPLICIT: with two sources, "not the image" determines the caption, and that
# inference is exactly what is being tested. Scoring is unchanged, so read_acc under a neg frame
# reads as "gave the caption value when told not to use the image", and 1-read_acc is the
# lexical-matching failure rate (the negated prompt still CONTAINS the other source's word).
NEG_OTHER = {"look": "read", "read": "look"}
NEG_ENGLISH = {"look": "image-based", "read": "caption-based"}
REQ_FRAME = __import__("os").environ.get("SC_REQ_FRAME", "").strip()


def norm(k):
    return {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
            for o, v in (k or {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--dump_records", type=int, default=1,
                    help="1: include per-item 0/1 vectors in the output json "
                         "(item order is deterministic across runs on the same "
                         "split -> enables paired McNemar in stats_pass.py)")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=80)
    ap.add_argument("--third_value", type=int, default=0,
                    help="conflict-preserving twins on ANY ladder benchmark (2026-09-22): the conflict caption's "
                         "value for the queried object is replaced by a third value C drawn from the split's own "
                         "value vocabulary (colours on VCF/ColorSwap, entities on COCO-CF), the twin caption by a "
                         "fourth value D; candidates follow the context. Same construction as --third_colour, "
                         "which it supersedes; --third_colour is kept as an alias for VCF.")
    ap.add_argument("--third_colour", type=int, default=0,
                    help="VCF third-colour twins (pre-registered STATUS v1.66): the conflict caption names a colour C "
                         "not in {image A, pair B}, the twin caption names D; the twin image stays the pair; candidates "
                         "follow the context, so Flip and Stay are no longer identical to accuracy.")
    ap.add_argument("--out_task", default="",
                    help="eval-dir base name override (default: basename of --data_dir); the third-colour VCF battery "
                         "writes vcf3_<split> so it never overwrites the two-colour files.")
    ap.add_argument("--trigger_repeat", type=int, default=0,
                    help="CAPACITY CONTROL (repeated-1tok): build the trigger by "
                         "repeating the run's OWN rows until K positions total "
                         "(use on an L=1 ckpt with K=5/30). Same positions and "
                         "repetition as a wide word, no new parameters. Output "
                         "goes to flip_ll_rep<K>_step_*.json.")
    ap.add_argument("--permute_seed", type=int, default=0,
                    help="shuffle seed for --trigger_permute (run several to "
                         "distinguish ordered specialization from one unlucky "
                         "permutation)")
    ap.add_argument("--trigger_drop", type=int, default=-1,
                    help=">=0: omit position i from the trigger (individual "
                         "position ablation; pair with permutation -- one "
                         "dominant drop = a functional anchor, homogeneous "
                         "drops + permutation damage = ordered specialization)")
    ap.add_argument("--trigger_permute", type=int, default=0,
                    help="POSITION CONTROL: shuffle the order of the word's L "
                         "rows in the trigger (seed=0 shuffle). If flip is "
                         "unchanged, positions are interchangeable -- width is "
                         "a bag of rows, not a sequence program. Output "
                         "flip_ll_perm_step_*.json.")
    ap.add_argument("--head_scale", default="",
                    help="attention-head scaling baseline (Hua et al. 2025): 'L:H:ALPHA' or a comma "
                         "separated list, e.g. '11:14:10'. Applies for the whole evaluation; the file "
                         "name gains the tag so a scaled run never overwrites the plain one.")
    ap.add_argument("--frozen_model_id", default=None,
                    help="skip checkpoints: FROZEN base model + English-instructed "
                         "prompts (the tier-2 baseline row, unified protocol). "
                         "--run_dir then only names the output location.")
    a = ap.parse_args()

    if a.frozen_model_id is not None:
        from scsrc import model as M
        model, proc = M.load(model_id=(a.frozen_model_id or None) if a.frozen_model_id != "base" else None)
        model.eval()
        _noop = os.environ.get("NOOP_MODE", "0") == "1"
        n, method, meta = 0, ("noop" if _noop else "frozen"), \
            {"n_tokens": 0, "method": "noop" if _noop else "frozen"}
    else:
        model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
        n, method = meta["n_tokens"], meta["method"]
    # v1.149 (2026-09-24): matched word-vs-prefix checks (STATUS v1.148/v1.149). Each switch is valid for one method
    # only and gets its own filename tag, so a diagnostic can never overwrite a banked evaluation.
    _CAP_FIRST = os.environ.get("SC_CAP_FIRST", "1") != "0"      # 0: request BEFORE the caption (word, or inserted prefix)
    _PT_NOPREFIX = os.environ.get("SC_PT_NOPREFIX", "0") == "1"  # prefix run, soft prefix removed: its instruction alone
    _WORD_PTFMT = os.environ.get("SC_WORD_PTFMT", "0") == "1"    # word run, the prefix's slot and wording (the mirror)
    _PT_INS = os.environ.get("PT_INSERTED", "0") == "1"
    _SWEEP = os.environ.get("SC_SWEEP", "").strip()             # v1.150: template-sensitivity sweep id (P.SWEEP_TEMPLATES)
    if _SWEEP and (_SWEEP not in P.SWEEP_TEMPLATES or method not in ("neo", "pt") or not _CAP_FIRST or _PT_INS
                   or _WORD_PTFMT or _TPL):
        raise SystemExit(f"[v1.150] SC_SWEEP={_SWEEP}: needs a word or prefix run (SC_PT_NOPREFIX=1 allowed) and no "
                         "other template/placement switch (SC_TEMPLATE, SC_CAP_FIRST, PT_INSERTED, SC_WORD_PTFMT)")
    if (_PT_NOPREFIX and method != "pt") or (_WORD_PTFMT and method != "neo") or \
            (not _CAP_FIRST and not (method == "neo" or (method == "pt" and _PT_INS))):
        raise SystemExit(f"[v1.149] diagnostic switch does not apply to method={method}: SC_CAP_FIRST=0 needs a word "
                         "or PT_INSERTED=1; SC_PT_NOPREFIX=1 needs a prefix run; SC_WORD_PTFMT=1 needs a word run")
    try:
        site = json.load(open(os.path.join(a.run_dir, "config.json"))) \
            .get("args", {}).get("ctrl_site", "user")
    except Exception:
        site = "user"

    def prompt_and_cands(op, cap, c1, c2):
        """method-faithful prompt + candidate strings (shared-prefix tricks
        cancel in the argmax)."""
        if REQ_FRAME:
            other = NEG_OTHER[op]
            if REQ_FRAME == "neg_word":
                if method != "neo":
                    raise SystemExit("SC_REQ_FRAME=neg_word needs a neologism checkpoint "
                                     "(there is no word to negate otherwise)")
                return (f'Caption: "{cap}". Do not give me a {trigger(f"{other}_ctrl", n)} answer.',
                        (c1, c2))
            if REQ_FRAME == "neg_english":
                return (f'Caption: "{cap}". Do not give me a{"n" if NEG_ENGLISH[other][0] in "aeiou" else ""} '
                        f'{NEG_ENGLISH[other]} answer.', (c1, c2))
            if REQ_FRAME == "pos_word":
                # the plain training request, re-run in THIS wave under its own filename. The 2x2
                # must come from one checkpoint: the grid's banked plain number for a cell can come
                # from a different run dir (the harvester picks seed 0 then the newest eval), so
                # borrowing it as the ceiling would make the comparison unmatched.
                if method != "neo":
                    raise SystemExit("SC_REQ_FRAME=pos_word needs a neologism checkpoint")
                return (P.render_ctrl(P.CTRL_CANONICAL, trigger(f"{op}_ctrl", n), cap), (c1, c2))
            if REQ_FRAME == "pos_english":
                return (f'Caption: "{cap}". Give me a{"n" if NEG_ENGLISH[op][0] in "aeiou" else ""} '
                        f'{NEG_ENGLISH[op]} answer.', (c1, c2))
            raise SystemExit(f"unknown SC_REQ_FRAME={REQ_FRAME}")
        if method == "neo" and site == "decision":
            trig = trigger(f"{op}_ctrl", n)
            return P.render_decision(cap), (f"{trig} {c1}", f"{trig} {c2}")
        if method == "neo":
            from scsrc.tokens import token_strings
            toks = list(token_strings(f"{op}_ctrl", n))   # exact special tokens
            if a.trigger_repeat and a.trigger_repeat > n:
                toks = (toks * ((a.trigger_repeat // len(toks)) + 1))[: a.trigger_repeat]
            elif a.trigger_permute and n > 1:
                import random as _rnd
                _rnd.Random(a.permute_seed).shuffle(toks)
            elif a.trigger_drop >= 0 and n > 1 and a.trigger_drop < len(toks):
                toks = toks[:a.trigger_drop] + toks[a.trigger_drop + 1:]
            if _SWEEP:        # v1.150: the sweep's fixed text, the word as the control
                return P.render_sweep(_SWEEP, "".join(toks), cap), (c1, c2)
            if _WORD_PTFMT:   # v1.149 mirror: the word where the prefix sits, with the prefix's instruction
                return "".join(toks) + " " + _pt_tpl(cap).format(cap=cap), (c1, c2)
            return P.render_ctrl(_ctrl_tpl(cap), "".join(toks), cap, cap_first=_CAP_FIRST), (c1, c2)
        if method == "pt":
            if _SWEEP:        # v1.150: the same text, the prefix as the control (or none: SC_PT_NOPREFIX=1)
                return P.render_sweep(_SWEEP, "" if _PT_NOPREFIX else trigger(f"pt_{op}", n), cap), (c1, c2)
            if _PT_NOPREFIX:  # v1.149: the prefix's own instruction, soft prefix removed (= the frozen model)
                return _pt_tpl(cap).format(cap=cap), (c1, c2)
            if _PT_INS:
                # inserted-PT control: same rows, the WORD's slot (and the word's carrier / held-out carriers)
                return P.render_ctrl(_ctrl_tpl(cap), trigger(f"pt_{op}", n), cap, cap_first=_CAP_FIRST), (c1, c2)
            return trigger(f"pt_{op}", n) + " " + _pt_tpl(cap).format(cap=cap), (c1, c2)
        if os.environ.get("NOOP_MODE", "0") == "1":
            # no-request baseline: caption + question, NO source instruction --
            # measures the model's DEFAULT per-channel dependence
            return f'Caption: "{cap}".', (c1, c2)
        if method == "lora" and os.environ.get("SC_LORA_ADAPTER", "0") == "1":
            # v1.125: adapter per request -- the prefix's neutral prompt (training or held-out template), no prefix
            return _pt_tpl(cap).format(cap=cap), (c1, c2)
        if method == "lora":
            return f'Caption: "{cap}". {_lora_instr(op, cap)}', (c1, c2)   # LoRA: held-out variants under SC_TEMPLATE
        return f'Caption: "{cap}". {INSTR[op]}', (c1, c2)   # frozen English baseline

    from scsrc.headscale import parse_spec, scaled_heads
    _hspec = parse_spec(a.head_scale)
    _hctx = scaled_heads(model, _hspec)
    _hctx.__enter__()
    if _hspec:
        print(f"[headscale] {_hspec} active for this evaluation (Hua et al. 2025 baseline)")

    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",))
             if s.cond == "conflict"]
    is_vcf = "vcf" in a.data_dir
    if a.third_value:
        a.third_colour = 1                       # one code path; the vocabulary is what differs
    # value vocabulary of THIS split: every key value across both sides of every pair (lower-cased)
    _split_vocab = set()
    for _k in ("key_1", "key_2"):          # column access, no image decoding
        try:
            for _x in ds[_k]:
                try:
                    _split_vocab.update(str(v).strip().lower() for v in json.loads(_x).values())
                except Exception:
                    pass
        except Exception:
            pass
    if is_vcf:
        _split_vocab |= TC_VOCAB                 # VCF's 13 colours, so small splits still have choices
    res = {k: [] for k in ("look_acc", "read_acc", "look_flip", "look_stay",
                           "read_flip", "read_stay", "look_margin", "read_margin",
                           "cmp_acc", "cmp_flip_img", "cmp_flip_cap", "cmp_stay_both",
                           "cmp_margin_gap", "cmp_disc", "cmp_auc_pairs",
                           "cmp_abstain_on_conflict", "cmp_answer_on_agree",
                           "cmp_joint")}
    _ITEMS = []   # v1.141: item-level log
    _AUDIT = {}   # v1.153: protocol-2 invariance checks (counts per check; any failure stops the run)
    _P2PICKS = []  # v1.153: per item, the six picks (image request base/imgswap/capswap, then caption request) as letters
    if _LEADIN and not _P2:
        raise SystemExit("[v1.158] SC_LADDER_LEADIN=1 is defined for protocol 2 only (SC_LADDER_PROTO=2)")
    if _P2 and not a.third_value:
        raise SystemExit("[v1.153] SC_LADDER_PROTO=2 is the conflict-preserving fixed-candidate protocol: needs --third_value 1")
    _SCORE = os.environ.get('SC_SCORE', 'turn')   # v1.143: 'turn' (banked) | 'answer'
    from scripts.tokscore import split_spans as _split_spans
    n_seen = 0
    n_skipped_tc = 0; tc_log = []
    for sp in tqdm(specs, desc=f"flipLL({os.path.basename(a.run_dir)})"):
        ik, tk = norm(sp.image_key), norm(sp.text_key)
        if not ik or set(ik) != set(tk):
            continue
        conf = [o for o in ik if ik[o] != tk[o]
                and not (is_vcf and any(c in o.split() for c in COLORS))]
        if not conf:
            continue
        if n_seen >= a.max_n:
            break
        n_seen += 1
        # PER-OBJECT: multi-object items (ColorSwap) probe one conflicting
        # object per item, rotating -- candidates name the object so the
        # forced choice is unambiguous (the skyline/eval_ladder fix)
        obj = conf[n_seen % len(conf)]
        v_i, v_t = ik[obj], tk[obj]
        img = get_image(ds, sp)
        imgp = ds[sp.ex_index][f"image_{3 - sp.img_idx}"]     # twin image: value v_t
        capp = ds[sp.ex_index][f"caption_{sp.img_idx}"]       # twin caption: asserts v_i
        ci = P.fmt_key_task({obj: v_i}, a.task)
        ct = P.fmt_key_task({obj: v_t}, a.task)

        if a.third_colour:
            # ---- third-colour twins (STATUS v1.66). A = image value, B = pair image value (= the original
            # conflicting caption's value), C = conflict caption colour, D = twin caption colour.
            import re as _re, random as _rnd
            _vocab = sorted(_split_vocab - {v_i, v_t})
            _rng = _rnd.Random(2026 + sp.ex_index)
            C = _rng.choice(_vocab); D = _rng.choice([c for c in _vocab if c != C])
            _pat = _re.compile(rf"\b{_re.escape(v_t)}\b", _re.IGNORECASE)
            if len(_pat.findall(sp.fed_caption)) != 1:   # the value must appear exactly once to be replaceable
                n_skipped_tc += 1; n_seen -= 1
                continue
            cap_C = _pat.sub(C, sp.fed_caption); cap_D = _pat.sub(D, sp.fed_caption)
            cA, cB_, cC, cD = (P.fmt_key_task({obj: v}, a.task) for v in (v_i, v_t, C, D))
            if _P2:
                # ---- v1.153 FIXED-CANDIDATE PROTOCOL. One candidate set per original example, {A, B, C, D}
                # (A = image value, B = the image twin's value, C = conflict caption value, D = edited caption value),
                # scored in EVERY version and for BOTH requests; the pick is the argmax over all four. Protocol 1 scored
                # each version with its own pair ({A,C}, {B,C}, {A,D}), so an input-insensitive ranking could earn Flip.
                #   image request:   Acc = A at base;  Flip = Acc and B after the image edit;  Stay = Acc and A after the caption edit
                #   caption request: Acc = C at base;  Flip = Acc and D after the caption edit; Stay = Acc and C after the image edit
                _TPL_KEY[0] = cap_C
                V4 = {"A": v_i, "B": v_t, "C": C, "D": D}
                K4 = {"A": cA, "B": cB_, "C": cC, "D": cD}

                def pick4(op, image, cap, ctx):
                    u, _ = prompt_and_cands(op, cap, cA, cC)
                    texts = {k: (LEAD_TXT if _LEADIN else "") + prompt_and_cands(op, cap, c, c)[1][0] for k, c in K4.items()}
                    det = {k: SC.loglik(model, proc, image, u, t, "cuda", tokens=True) for k, t in texts.items()}
                    _sp4 = _split_spans({k: (d[3], d[4]) for k, d in det.items()})
                    sc = {k: (_sp4[k]["answer_sum"] if _SCORE == "answer" else det[k][0]) for k in det}
                    best = max(sc, key=sc.get)
                    try:
                        ptok = len(proc.tokenizer(u)["input_ids"])
                    except Exception:
                        ptok = None
                    _ITEMS.append({"item": n_seen - 1, "request": op, "ctx": ctx, "protocol": 2, "prompt": u, "caption": cap,
                                   "prompt_ntok": ptok, "values": dict(V4),
                                   "candidates": {k: {"text": texts[k], "sum": round(det[k][0], 4), "mean": round(det[k][1], 4),
                                                      "ntok": det[k][2], **_sp4[k]} for k in det},
                                   "pick": best, "pick_value": V4[best]})
                    return best, sc, u, texts

                CX = {"base": (img, cap_C), "imgswap": (imgp, cap_C), "capswap": (img, cap_D)}
                G = {(op, cx): pick4(op, im, cp, cx) for op in ("look", "read") for cx, (im, cp) in CX.items()}
                # ---- fail-fast audit (every item; counts are written to the output)
                t0 = G[("look", "base")][3]
                chk = {"candidates_fixed": all(g[3] == t0 for g in G.values()) and len(set(t0.values())) == 4,
                       "values_distinct": len(set(V4.values())) == 4,
                       "conflict_preserved": v_i != C and v_t != C and v_i != D,
                       "caption_edit_value_only": _re.sub(rf"\b{_re.escape(C)}\b", "<V>", cap_C, count=1)
                                                  == _re.sub(rf"\b{_re.escape(D)}\b", "<V>", cap_D, count=1) != cap_C,
                       "image_edit_image_only": imgp is not img}
                for op in ("look", "read"):
                    mb = G[(op, "base")][2].replace(cap_C, "<CAP>")
                    chk[f"prompt_fixed_{op}"] = ("<CAP>" in mb and mb == G[(op, "imgswap")][2].replace(cap_C, "<CAP>")
                                                 == G[(op, "capswap")][2].replace(cap_D, "<CAP>"))
                for k_, ok_ in chk.items():
                    _AUDIT[k_] = _AUDIT.get(k_, 0) + int(ok_)
                _AUDIT["n"] = _AUDIT.get("n", 0) + 1
                if not all(chk.values()):
                    raise SystemExit(f"[protocol 2 audit FAILED] item {n_seen - 1}: {[k_ for k_, ok_ in chk.items() if not ok_]}")
                bl, il, cl = (G[("look", c)][0] for c in ("base", "imgswap", "capswap"))
                br, ir, cr = (G[("read", c)][0] for c in ("base", "imgswap", "capswap"))
                res["look_acc"].append(int(bl == "A")); res["look_flip"].append(int(bl == "A" and il == "B"))
                res["look_stay"].append(int(cl == bl))
                res["read_acc"].append(int(br == "C")); res["read_flip"].append(int(br == "C" and cr == "D"))
                res["read_stay"].append(int(ir == br))
                mg = lambda g, tgt: g[1][tgt] - max(v for kk, v in g[1].items() if kk != tgt)   # target minus best other
                res["look_margin"].append(mg(G[("look", "base")], "A")); res["read_margin"].append(mg(G[("read", "base")], "C"))
                for k_, v_ in (("look_margin_imgswap", mg(G[("look", "imgswap")], "B")), ("look_margin_capswap", mg(G[("look", "capswap")], "A")),
                               ("read_margin_capswap", mg(G[("read", "capswap")], "D")), ("read_margin_imgswap", mg(G[("read", "imgswap")], "C"))):
                    res.setdefault(k_, []).append(float(v_))
                _P2PICKS.append("".join(G[(op, c)][0] for op in ("look", "read") for c in ("base", "imgswap", "capswap")))
                tc_log.append(f"{v_i}|{v_t}|{C}|{D}")
                continue

            def pick3(op, image, cap, c_img, c_cap, ctx="?"):
                u, (x, y) = prompt_and_cands(op, cap, c_img, c_cap)
                la, ma, na, ia, pa_ = SC.loglik(model, proc, image, u, x, "cuda", tokens=True)
                lb, mb, nb, ib, pb_ = SC.loglik(model, proc, image, u, y, "cuda", tokens=True)
                _sp = _split_spans({"image_side": (ia, pa_), "caption_side": (ib, pb_)})
                if _SCORE == "answer":   # v1.143: answer tokens only
                    la, lb = _sp["image_side"]["answer_sum"], _sp["caption_side"]["answer_sum"]
                _ITEMS.append({"item": n_seen - 1, "request": op, "ctx": ctx, "prompt": u, "caption": cap,
                               "values": {"image": v_i, "caption_orig": v_t, "C": C, "D": D},
                               "candidates": {"image_side": {"text": x, "sum": round(la, 4), "mean": round(ma, 4), "ntok": na, **_sp["image_side"]},
                                              "caption_side": {"text": y, "sum": round(lb, 4), "mean": round(mb, 4), "ntok": nb, **_sp["caption_side"]}},
                               "pick": "i" if la > lb else "t"})
                return ("i" if la > lb else "t"), la - lb

            li, m_li = pick3("look", img, cap_C, cA, cC, "base")            # base: {A, C}
            lp, m_lp = pick3("look", imgp, cap_C, cB_, cC, "imgswap")       # image swap: {B, C}
            lc, m_lc = pick3("look", img, cap_D, cA, cD, "capswap")         # caption swap: {A, D}
            ri, m_ri = pick3("read", img, cap_C, cA, cC, "base")
            rc, m_rc = pick3("read", img, cap_D, cA, cD, "capswap")         # caption swap
            rp, m_rp = pick3("read", imgp, cap_C, cB_, cC, "imgswap")       # image swap
            for k_, v_ in (("look_margin_imgswap", m_lp), ("look_margin_capswap", m_lc),
                           ("read_margin_capswap", m_rc), ("read_margin_imgswap", m_rp)):
                res.setdefault(k_, []).append(float(v_))
            res["look_acc"].append(int(li == "i"))
            res["read_acc"].append(int(ri == "t"))
            res["look_flip"].append(int(li == "i" and lp == "i"))   # A at base AND B after the image swap
            res["look_stay"].append(int(li == lc))                   # unchanged under the caption swap
            res["read_flip"].append(int(ri == "t" and rc == "t"))   # C at base AND D after the caption swap
            res["read_stay"].append(int(ri == rp))                   # unchanged under the image swap
            res["look_margin"].append(m_li)
            res["read_margin"].append(-m_ri)
            tc_log.append(f"{v_i}|{v_t}|{C}|{D}")
            continue

        def pick(op, image, cap, ctx="?"):
            u, (cA, cB) = prompt_and_cands(op, cap, ci, ct)
            la, ma, na, ia, pa_ = SC.loglik(model, proc, image, u, cA, "cuda", tokens=True)
            lb, mb, nb, ib, pb_ = SC.loglik(model, proc, image, u, cB, "cuda", tokens=True)
            _sp = _split_spans({"image_side": (ia, pa_), "caption_side": (ib, pb_)})
            if _SCORE == "answer":   # v1.143: answer tokens only
                la, lb = _sp["image_side"]["answer_sum"], _sp["caption_side"]["answer_sum"]
            _ITEMS.append({"item": n_seen - 1, "request": op, "ctx": ctx, "prompt": u, "caption": cap,
                           "candidates": {"image_side": {"text": cA, "sum": round(la, 4), "mean": round(ma, 4), "ntok": na, **_sp["image_side"]},
                                          "caption_side": {"text": cB, "sum": round(lb, 4), "mean": round(mb, 4), "ntok": nb, **_sp["caption_side"]}},
                           "pick": "i" if la > lb else "t"})
            return ("i" if la > lb else "t"), la - lb

        li, m_li = pick("look", img, sp.fed_caption, "base")
        lp, m_lp = pick("look", imgp, sp.fed_caption, "imgswap")         # image swap
        lc, m_lc = pick("look", img, capp, "capswap")                     # caption swap
        ri, m_ri = pick("read", img, sp.fed_caption, "base")
        rc, m_rc = pick("read", img, capp, "capswap")                     # caption swap
        rp, m_rp = pick("read", imgp, sp.fed_caption, "imgswap")          # image swap
        # 2026-09-14: margins in the twin contexts too (log-odds image value minus caption value under
        # each request), so detectors can be scored continuously (ROC) instead of by argmax only
        for k_, v_ in (("look_margin_imgswap", m_lp), ("look_margin_capswap", m_lc),
                       ("read_margin_capswap", m_rc), ("read_margin_imgswap", m_rp)):
            res.setdefault(k_, []).append(float(v_))
        res["look_acc"].append(int(li == "i"))
        res["read_acc"].append(int(ri == "t"))
        res["look_flip"].append(int(li == "i" and lp == "t"))
        res["look_stay"].append(int(li == lc))
        res["read_flip"].append(int(ri == "t" and rc == "i"))
        res["read_stay"].append(int(ri == rp))
        res["look_margin"].append(m_li)
        res["read_margin"].append(-m_ri)
        # ---- <compare>: relation is XOR-like of the channels; fed_caption
        # describes the TWIN image, so (I',c)=agree, (I,c')=agree, (I',c')=conflict
        if method == "neo" and site == "user":
            from scsrc.prompts import natural_ctrl_content as _ncc
            # COMPARE v2 targets are CONSEQUENTIAL, not labels: on conflict the
            # gold declines to answer, on agree it states the agreed value. The
            # 4-context battery is unchanged (that is what makes it calibration-
            # free), but the two candidate continuations are now the two ACTIONS
            # -- abstain vs answer -- so the margin measures the decision the
            # word actually drives instead of a verdict token.
            _vi = P.fmt_key_task(sp.image_key, a.task)
            _vt = P.fmt_key_task(sp.text_key, a.task)
            c_con = _ncc("compare", "conflict", a.task, _vi, _vt)
            c_agr = _ncc("compare", "agree", a.task, _vi, _vi)
            def cmarg(image, cap):
                """LL(conflict verdict) - LL(agree verdict): >0 = says conflict."""
                u = P.render_ctrl(P.CTRL_CANONICAL, trigger(f"compare_ctrl", n), cap)
                lc_, _, _ = SC.loglik(model, proc, image, u, c_con, "cuda")
                la_, _, _ = SC.loglik(model, proc, image, u, c_agr, "cuda")
                return lc_ - la_
            m_conf = cmarg(img, sp.fed_caption)          # TRUE conflict
            m_ag_i = cmarg(imgp, sp.fed_caption)         # agree (image swapped)
            m_ag_c = cmarg(img, capp)                    # agree (caption swapped)
            m_conf2 = cmarg(imgp, capp)                  # conflict again (both swapped)
            res["cmp_acc"].append(int(m_conf > 0))
            res["cmp_flip_img"].append(int(m_ag_i < 0))
            res["cmp_flip_cap"].append(int(m_ag_c < 0))
            res["cmp_stay_both"].append(int(m_conf2 > 0))
            # CALIBRATION-FREE: within-item ordering conflict-vs-agree contexts.
            # cmp_disc=1 iff BOTH agree-contexts score lower than BOTH conflict
            # contexts -> discriminative even if the argmax threshold is biased.
            res["cmp_margin_gap"].append(
                ((m_conf + m_conf2) / 2) - ((m_ag_i + m_ag_c) / 2))
            res["cmp_disc"].append(int(min(m_conf, m_conf2) > max(m_ag_i, m_ag_c)))
            res["cmp_auc_pairs"].append(int(m_conf > m_ag_i) + int(m_conf > m_ag_c)
                                        + int(m_conf2 > m_ag_i) + int(m_conf2 > m_ag_c))
            # ABSTENTION IS GAMEABLE: a policy that always declines is perfect on
            # conflict and worthless on agree. Score both sides and report the
            # JOINT rate, which only a genuinely conditional word can win.
            res["cmp_abstain_on_conflict"].append(int(m_conf > 0))
            res["cmp_answer_on_agree"].append(int(m_ag_i < 0 and m_ag_c < 0))
            res["cmp_joint"].append(int(m_conf > 0 and m_ag_i < 0 and m_ag_c < 0))

    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    if res["cmp_auc_pairs"]:
        res["cmp_auc"] = [x / 4.0 for x in res["cmp_auc_pairs"]]
    # CONDITIONAL flips: P(track swap | base pick correct) -- pure sensitivity,
    # decoupled from base accuracy (unconditional flip conflates the two)
    lc = [f for f, a_ in zip(res["look_flip"], res["look_acc"]) if a_]
    rc = [f for f, a_ in zip(res["read_flip"], res["read_acc"]) if a_]
    # CONJUNCTIVE STAY (2026-09-13): plain stay asks only whether the answer is unchanged when a
    # competing modality is edited, so a stably WRONG answer scores a point. flip is already a
    # conjunction (correct AND follows the edit); stay_c is its mirror (correct AND unchanged), and
    # stay_cond is the same quantity given a correct base answer. Reported alongside, never instead
    # of, the historical field, so banked numbers keep their meaning.
    res["look_stay_c"] = [int(a_ and s_) for a_, s_ in zip(res["look_acc"], res["look_stay"])]
    res["read_stay_c"] = [int(a_ and s_) for a_, s_ in zip(res["read_acc"], res["read_stay"])]
    ls = [s_ for s_, a_ in zip(res["look_stay"], res["look_acc"]) if a_]
    rs = [s_ for s_, a_ in zip(res["read_stay"], res["read_acc"]) if a_]
    # 2026-09-21: the weight dtype goes into every record, and a non-bf16 evaluation gets its own
    # filename tag, so a float32 file can never be harvested as if it were the banked bf16 one
    # (the V100 float32 agreement check missed its pre-registered bar; STATUS v1.70).
    import torch as _t
    _dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    twins = ("third-colour caption (image pair kept)" if a.third_colour else
             "cross-pair(approximate)" if "coco" in a.data_dir else "exact-minimal-pair")
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step, "request_template": (_TPL or "train"),
           "n": n_seen, "method": method, "site": site, "twins": twins,
           "head_scale": a.head_scale or None,
           "weight_dtype": _dtype,
           "look_flip_cond": m(lc), "read_flip_cond": m(rc),
           "look_stay_cond": m(ls), "read_stay_cond": m(rs),
           "n_look_base_correct": len(lc), "n_read_base_correct": len(rc),
           **{k: m(v) for k, v in res.items()}}
    out["protocol"] = 2 if _P2 else 1
    if _P2:
        import collections as _col
        out["candidate_set"] = ("fixed per original example: {A image value, B image-twin value, C conflict caption value, "
                                "D edited caption value}, scored in every version and for both requests")
        out["audit_p2"] = _AUDIT
        out["p2_pick_patterns"] = _col.Counter(_P2PICKS).most_common(20)   # 'ABA' + 'CCD' = the correct pattern
        # share of picks supported by NEITHER source of that version (sources: base {A,C}, imgswap {B,C}, capswap {A,D})
        _src = ("AC", "BC", "AD")
        out["p2_distractor_rate"] = {rq: round(sum(p_[o + j] not in _src[j] for p_ in _P2PICKS for j in range(3))
                                               / max(3 * len(_P2PICKS), 1), 4) for rq, o in (("look", 0), ("read", 3))}
    if a.dump_records:
        out["records"] = {k: list(map(int, v)) for k, v in res.items()
                          if k in ("look_flip", "read_flip", "look_acc", "read_acc",
                                   "look_stay", "read_stay", "look_stay_c", "read_stay_c")}
        # margins too: per-item FLOATS enable function-space similarity
        # (margin correlation across runs) beyond 0/1 agreement
        out["records"]["look_margin"] = [round(float(v), 4) for v in res["look_margin"]]
        out["records"]["read_margin"] = [round(float(v), 4) for v in res["read_margin"]]
        for k_ in ("look_margin_imgswap", "look_margin_capswap", "read_margin_capswap", "read_margin_imgswap"):
            if k_ in res:
                out["records"][k_] = [round(float(v), 4) for v in res[k_]]
    if a.third_colour:
        out["third_colour"] = {"n_skipped_no_colour_word": n_skipped_tc, "vocab_size": len(_split_vocab),
                               "vocab": sorted(_split_vocab)[:60],
                               "draw": "random.Random(2026 + item index)"}
        if a.dump_records:
            out.setdefault("records", {})["tc_colours"] = tc_log
    print(json.dumps({k: v for k, v in out.items() if k != "records"}))
    outd = os.path.join(a.run_dir, "eval", f"{a.out_task or os.path.basename(a.data_dir)}_{a.split}")
    os.makedirs(outd, exist_ok=True)
    _iv = os.environ.get("INSTR_VARIANT", "0")
    _ivtag = f"_iv{_iv}" if _iv != "0" else ""
    if _dtype != "bfloat16":
        _ivtag += f"_{_dtype}"
    if os.environ.get("NOOP_MODE", "0") == "1":
        _ivtag = "_noop"
    if REQ_FRAME:
        _ivtag = f"_{REQ_FRAME}"       # never overwrite the plain request's banked eval
        out["req_frame"] = REQ_FRAME
    if _hspec:   # after the noop branch, which reassigns the tag
        _ivtag += "_head" + a.head_scale.replace(":", "-").replace(",", "_").replace(".", "p")
    # A TRANSPLANTED or ALIASED eval must never overwrite the run's own banked eval: the run_dir and
    # data_dir are identical to the in-distribution case, so only the filename can separate them.
    _xp = os.environ.get("SC_TRANSPLANT_MODEL", "").strip()
    if _xp:
        _ivtag += "_xplant_" + _xp.split("/")[-1].replace(".", "p")
    if os.environ.get("SC_WORD_ALIAS", "").strip():
        out["word_alias"] = os.environ["SC_WORD_ALIAS"]
    if _xp:
        out["transplant_host"] = _xp
        out["donor_model"] = meta.get("model_id")
    _diag = ((["reqfirst"] if not _CAP_FIRST else []) + (["noprefix"] if _PT_NOPREFIX else [])
             + (["ptslot"] if (_PT_INS and "_pti_" not in os.path.basename(a.run_dir.rstrip("/"))) else [])
             + (["ptfmt"] if _WORD_PTFMT else []) + ([f"sw{_SWEEP}"] if _SWEEP else []) + (["lead"] if _LEADIN else [])
             + (["p2"] if _P2 else []))
    if _diag:   # v1.149: the pti runs were TRAINED inserted, so their plain files keep the historical name
        _ivtag += "_" + "_".join(_diag); out["diagnostic"] = _diag
    _hctx.__exit__(None, None, None)
    _fname = (f"flip_ll_rep{a.trigger_repeat}_step_{a.step}.json" if a.trigger_repeat else f"flip_ll_perm{a.permute_seed}_step_{a.step}.json" if a.trigger_permute else f"flip_ll_drop{a.trigger_drop}_step_{a.step}.json" if a.trigger_drop >= 0 else f"flip_ll{_ivtag}{_tpltag}{'_ans' if _SCORE == 'answer' else ''}_step_{a.step}.json")
    json.dump(out, open(os.path.join(outd, _fname), "w"), indent=1)
    if _ITEMS:   # v1.141: every scored context -- prompt, both candidates' sum/mean/ntok, pick
        import gzip as _gz
        with _gz.open(os.path.join(outd, _fname.replace('.json', '_items.jsonl.gz')), 'wt') as _f:
            _f.write(json.dumps({'_meta': {'run': os.path.basename(a.run_dir), 'split': a.split, 'task': a.task,
                                           'template': os.environ.get('SC_TEMPLATE', 'train'), 'method': method,
                                           'diagnostic': _diag,
                                           'scoring': 'sum log-likelihood (mean, ntok recorded)'}}) + '\n')
            for _r in _ITEMS:
                _f.write(json.dumps(_r) + '\n')


if __name__ == "__main__":
    main()
