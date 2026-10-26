#!/usr/bin/env python
"""Full evaluation protocol over a checkpoint. Exact string-match on parsed
{object: color}. Saves per-example records (output, correct, distractor, EM) to
runs/<run>/eval/records_step_<n>.jsonl and logs scalars + a table to wandb so you
can verify every answer yourself.

ctrl invoked as the trained USER instruction ('give me a {op} answer'); ctrl
tokens forbidden from output (input-only). gen tokens emitted. Tests:
controllability, generation, swaps (test split=generalization), no-pixels probe,
template-generalization, token-identity, verbalization + plug-in."""
import argparse, os, json, glob
from collections import defaultdict
from tqdm import tqdm
import torch

from scsrc import paths
from scsrc.data import load_colorswap, build_specs, load_legible_ids, get_image, load_task
from scsrc.tokens import default_specs, add_neologisms, trigger
from scsrc import model as M
from scsrc import metrics as Me
from scsrc import prompts as P
from scsrc.infer import generate
from scsrc.swaps import conflict_examples, flip_score
from scsrc.verbalize import verbalize_operator, plugin_test, synonyms_operator, eval_synonyms


TASK = "color"   # module-global; set per-invocation in main() so eval fns share it


def _m(x):
    return sum(x) / len(x) if x else None


def _ks(d):  # {obj:color} -> "obj: color, ..."
    return P.fmt_key(d) if d else ""


def apply_checkpoint(run_dir, step, device):
    meta = json.load(open(os.path.join(run_dir, "ckpt", f"step_{step}", "meta.json")))
    # 2026-09-22 (STATUS v1.94): a run trained under the two-modality Ground-CF protocol is evaluated with the same
    # prompt (no on-screen-text block); scsrc.tokens.c3_sources() reads this at call time.
    try:
        _args = (json.load(open(os.path.join(run_dir, "config.json"))).get("args") or {})
        _c3 = _args.get("c3_sources")
        if _c3:
            os.environ["SC_C3_SOURCES"] = _c3
            if _c3 != "look,read,ocr":
                print(f"[clevr3] source protocol from config: {_c3}")
        # 2026-09-23 (STATUS v1.123): the word's carrier form is a property of the RUN. Restore it here so every
        # evaluator renders the prompt the run was trained with (bare for banked runs, hewitt templates for the
        # new ones); the training-only instruction rotation must never leak into evaluation.
        os.environ["SC_NEO_TEMPLATE"] = _args.get("neo_template") or "bare"
        # 2026-09-29 protocol-v2 switches travel with the checkpoint (train.py records them in config.json args)
        os.environ["SC_CTRL_TEMPLATE_SET"] = _args.get("ctrl_template_set") or "legacy"
        os.environ["SC_GCF_FORMAT"] = _args.get("gcf_format") or "legacy"
        os.environ["SC_GCF_CONT"] = _args.get("gcf_cont") or "leadin"
        os.environ["SC_GCF_PT_CARRIER"] = str(_args.get("gcf_pt_carrier") or 0)
        if not os.environ.get("SC_MVG_SOURCES_OVERRIDE"):
            os.environ["SC_MVG_SOURCES"] = _args.get("mvg_sources") or "desc,ocr,vis"   # 2026-10-02: MVG blocks travel with the checkpoint
        # 2026-10-19 trained positions travel with the checkpoint; an explicit SC_GCF_CTRL_POS in the environment (relocation
        # evaluations) is kept for native runs, and SC_GCF_CTRL_POS_OVERRIDE wins over the config for trained-position runs
        _gp = _args.get("gcf_ctrl_pos") or "native"
        if os.environ.get("SC_GCF_CTRL_POS_OVERRIDE"):
            os.environ["SC_GCF_CTRL_POS"] = os.environ["SC_GCF_CTRL_POS_OVERRIDE"]; os.environ["SC_GCF_CTRL_ROTATE"] = "1"
        elif _gp != "native":
            os.environ["SC_GCF_CTRL_POS"] = _gp; os.environ["SC_GCF_CTRL_ROTATE"] = "1"
            print(f"[prompt] control position from config: {_gp}")
        _lp = _args.get("ladder_ctrl_pos") or "native"
        if _lp != "native":
            os.environ["SC_LADDER_CTRL_POS"] = _lp
        else:
            os.environ.pop("SC_LADDER_CTRL_POS", None)
        if _args.get("instr_rotate"):
            os.environ["SC_INSTR_TRAIN_POOL"] = str(_args.get("instr_rotate"))   # design v2 reference condition (English wording pool)
        else:
            os.environ.pop("SC_INSTR_TRAIN_POOL", None)
        if (os.environ["SC_CTRL_TEMPLATE_SET"], os.environ["SC_GCF_FORMAT"], os.environ["SC_GCF_CONT"],
                os.environ["SC_GCF_PT_CARRIER"]) != ("legacy", "legacy", "leadin", "0"):
            print(f"[prompt] v2 switches from config: set={os.environ['SC_CTRL_TEMPLATE_SET']} "
                  f"format={os.environ['SC_GCF_FORMAT']} cont={os.environ['SC_GCF_CONT']} "
                  f"pt_carrier={os.environ['SC_GCF_PT_CARRIER']}")
        os.environ["SC_CAPTION_STYLE"] = os.environ.get("SC_CAPTION_STYLE_OVERRIDE") or _args.get("caption_style") or "template"
        os.environ.pop("SC_INSTR_ROTATE", None); os.environ.pop("SC_Q_ROTATE", None); os.environ.pop("SC_TRAIN_SOURCES", None)
        os.environ["SC_LORA_ADAPTER"] = "1" if (_args.get("method") == "lora" and _args.get("ctrl_style") == "adapter") else "0"
        os.environ["SC_MVG_PT_PREFIX"] = str(int(_args.get("mvg_pt_prefix") or 0))     # 2026-10-24: MVG standard prompt tuning
        os.environ["SC_EVAL_REQUESTS"] = _args.get("train_sources") or ""
        if os.environ["SC_CAPTION_STYLE"] != "template":
            print(f"[prompt] caption style: {os.environ['SC_CAPTION_STYLE']}")
        if os.environ["SC_NEO_TEMPLATE"] != "bare":
            print(f"[prompt] word carrier templates from config: {os.environ['SC_NEO_TEMPLATE']}")
    except Exception:
        os.environ["SC_NEO_TEMPLATE"] = "bare"
        os.environ["SC_CAPTION_STYLE"] = os.environ.get("SC_CAPTION_STYLE_OVERRIDE") or "template"
    # CROSS-MODEL TRANSPLANT (2026-09-16): load the DONOR's trained rows into a DIFFERENT model.
    # Only the embedding width has to match; Gemma-3-4B, Qwen-3.5-4B and Gemma-4-E4B are all 2560,
    # so rows can cross model families. load_neologisms_by_name hard-fails on a width mismatch.
    _host = os.environ.get("SC_TRANSPLANT_MODEL", "").strip()
    model, processor = M.load(model_id=_host or meta.get("model_id"))
    method, arm, n = meta["method"], meta["arm"], meta["n_tokens"]
    forbid = None; all_new_ids = None
    if method in ("neo", "pt"):
        # SOURCE OF TRUTH = the checkpoint's own name_to_ids. Reconstructing
        # token strings from default_specs() silently BROKE every task whose
        # token set differs (mvg desc/ocr/vis, clevr3 src_*/op_*/fx_*): the
        # saved rows landed at ids whose STRINGS were look/read/compare, so
        # trigger("<desc_ctrl>") tokenized as junk text and the trained rows
        # never entered the prompt (-> the fake 'null selectivity').
        import torch as _t
        from scsrc.tokens import token_strings
        neo_path = os.path.join(run_dir, "ckpt", f"step_{step}", "neo.pt")
        ck = _t.load(neo_path, map_location="cpu")
        # WORD ALIASING (2026-09-16): SC_WORD_ALIAS="look_ctrl=src_look,read_ctrl=src_read" lets rows
        # trained under one task's vocabulary be TESTED under another's, with no retraining. The rows
        # are the learned object; the name is only the dictionary key that puts them in the prompt.
        _alias = dict(kv.split("=", 1) for kv in os.environ.get("SC_WORD_ALIAS", "").split(",")
                      if "=" in kv)
        _orig = list(ck["name_to_ids"].keys())
        names = [_alias.get(nm, nm) for nm in _orig]
        _len = {_alias.get(nm, nm): len(ck["name_to_ids"][nm]) for nm in _orig}
        all_strings = []
        for nm in names:   # per-name count (bundled clevr3 tokens have 2n rows)
            all_strings.extend(token_strings(nm, _len[nm]))
        processor.tokenizer.add_special_tokens({"additional_special_tokens": all_strings})
        name_to_ids_ck = {nm: processor.tokenizer.convert_tokens_to_ids(
                              token_strings(nm, _len[nm]))
                          for nm in names}
        all_new_ids = [i for ids in name_to_ids_ck.values() for i in ids]
        # seeded resize: E4B's per-layer rows for the new ids are then the training draw (meta seed), and
        # checkpoints that stored them get them restored below (2026-09-21, Gate 3)
        M.resize_for_new_tokens(model, processor.tokenizer, seed=meta.get("seed", 0))
        loaded = M.load_neologisms_by_name(model, neo_path, name_to_ids_ck, alias=_alias)
        assert set(loaded) == set(names), \
            f"checkpoint rows not fully restored: {set(names) - set(loaded)}"
        if meta.get("act"):                                   # 2026-10-19 activation control: delta + hooks
            import scsrc.act as A
            _ctrl, _ad = A.load(os.path.join(run_dir, "ckpt", f"step_{step}", "act.pt"), device)
            A.install(model, _ctrl, int(_ad["layer"]), name_to_ids_ck, processor.tokenizer, A.image_token_id(model, processor.tokenizer))
            model._act_ctrl = _ctrl
            print(f"[act] activation control installed at {A.STATE['layer_path']} (|delta| look {_ctrl.delta[0].norm():.3f}, read {_ctrl.delta[1].norm():.3f})")
        _blocks = False
        try:
            _blocks = json.load(open(os.path.join(run_dir, "config.json"))) \
                .get("args", {}).get("ctrl_style") == "blocks"
        except Exception:
            pass
        if arm == "ctrl" and _blocks:
            # ctrl_style=blocks: the CONTROL word stays input-only, but the
            # CITATION delimiters (<look>/<read>) MUST be emittable -- they are
            # the provenance marks the answer is supposed to carry.
            forbid = [i for nm, ids in name_to_ids_ck.items()
                      if nm.endswith("_ctrl") for i in ids]
        elif method == "pt" or arm == "ctrl":    # input-only tokens: never emit
            forbid = all_new_ids
        elif arm == "link":                       # ctrl input-only; gen MUST be emittable
            forbid = [i for nm, ids in name_to_ids_ck.items() if nm.endswith("_ctrl") for i in ids]
    elif method == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.path.join(run_dir, "ckpt", f"step_{step}"))
    model.eval()
    model._neo_all_ids = list(all_new_ids) if all_new_ids is not None else None
    # MVG eval scripts render the request through scsrc.mvg.mvg_word, which needs the method:
    # a PT checkpoint owns pt_<channel> rows, a neo checkpoint owns <channel>_ctrl, and LoRA owns
    # neither (its request is literal text). Before 2026-09-17 every MVG site rendered
    # <channel>_ctrl regardless, so PT runs never saw their own rows.
    from scsrc.mvg import set_word_method
    set_word_method(meta.get("method", "neo"))
    return model, processor, meta, forbid


def _neo_user(op_name, n, tpl, cap):
    return P.render_ctrl(tpl, trigger(f"{op_name}_ctrl", n), cap)


def _pt_user(op_name, n, tpl, cap):
    # learned soft PREFIX + source-neutral template; the prefix carries steering
    return trigger(f"pt_{op_name}", n) + " " + tpl.format(cap=cap)


EVAL_CODE_VERSION = "r8-audit-2026-07-18"
_MAXN = 120


def eval_dual(model, processor, ds, specs, n, device, forbid, rec, max_n=None):
    """Both <look_ctrl> and <read_ctrl> in ONE prompt (never trained this way --
    a zero-shot composition test). Metrics: is the image value present, is the
    caption value present, are BOTH present (conflict items only, where the two
    values differ and presence is unambiguous)."""
    max_n = _MAXN if max_n is None else max_n
    lt, rt = trigger("look_ctrl", n), trigger("read_ctrl", n)
    r = defaultdict(list)
    for s_ in tqdm(specs[:max_n], desc="dual"):
        if s_.cond != "conflict":
            continue
        img = get_image(ds, s_)
        user = (f'Caption: "{s_.fed_caption}". Give one answer for each requested '
                f'source, one per line: {lt} then {rt}.')
        out = generate(model, processor, img, user, device, 224, forbid_ids=forbid,
                       stop_anchor=None)
        lo = out.lower()
        iv = all(v.lower() in lo for v in s_.image_key.values())
        tv = all(v.lower() in lo for v in s_.text_key.values())
        r["dual_img_val_present"].append(int(iv))
        r["dual_txt_val_present"].append(int(tv))
        r["dual_both_present"].append(int(iv and tv))
        rec.append({"test": "dual_tokens", "cond": s_.cond, "output": out[:260],
                    "correct": {**s_.image_key, "_txt": list(s_.text_key.values())},
                    "exact_match": int(iv and tv)})
    return {k: sum(v) / max(len(v), 1) for k, v in r.items()}


def eval_controllability(model, processor, ds, specs, n, device, forbid, rec, max_n=None,
                         user_fn=_neo_user, canon=None):
    max_n = _MAXN if max_n is None else max_n
    canon = canon or P.CTRL_CANONICAL
    r = defaultdict(list)
    for s in tqdm(specs[:max_n], desc="controllability"):
        img = get_image(ds, s)
        def go(op_name, maxtok=160):
            return generate(model, processor, img, user_fn(op_name, n, canon, s.fed_caption),
                            device, maxtok, forbid_ids=forbid, stop_anchor="final answer:")
        o_look = go("look"); p_look = Me.parse_pred(o_look)
        o_read = go("read"); p_read = Me.parse_pred(o_read)
        o_cmp = go("compare"); rel = Me.parse_relation(o_cmp)
        o_noop = generate(model, processor, img, P.ctrl_noop(TASK).format(cap=s.fed_caption), device, 160, forbid_ids=forbid, stop_anchor="final answer:")
        a_look = Me.object_accuracy(p_look, s.image_key); a_read = Me.object_accuracy(p_read, s.text_key)
        a_noop = Me.object_accuracy(Me.parse_pred(o_noop), s.image_key)
        r["look_img_acc"].append(a_look); r["read_txt_acc"].append(a_read)
        r["compare_correct"].append(int(rel == s.relation)); r["noop_img_acc"].append(a_noop)
        for _nm, _o in (("look", o_look), ("read", o_read), ("compare", o_cmp)):
            r[f"trunc_{_nm}"].append(int("final answer:" not in _o.lower()))
        p_noop = Me.parse_pred(o_noop)
        for test, out, pd_, gold, other, correct, distract, em in [
            ("ctrl_look", o_look, p_look, s.image_key, s.text_key, _ks(s.image_key), _ks(s.text_key), a_look),
            ("ctrl_read", o_read, p_read, s.text_key, s.image_key, _ks(s.text_key), _ks(s.image_key), a_read),
            ("ctrl_compare", o_cmp, None, None, None, s.relation, ("conflict" if s.relation == "agree" else "agree"), int(rel == s.relation)),
            ("ctrl_noop", o_noop, p_noop, s.image_key, s.text_key, _ks(s.image_key), _ks(s.text_key), a_noop)]:
            row = {"eid": s.example_id, "cond": s.cond, "test": test, "output": out[:200],
                   "correct": correct, "distractor": distract, "exact_match": round(float(em), 3)}
            if pd_ is not None:
                sc = Me.slot_scores(pd_, gold, other)
                row.update({"containment": sc["containment"], "jaccard": sc["jaccard"],
                            "intrusion": sc["intrusion"]})
                r[f"{test}_containment"].append(sc["containment"])
                if s.cond == "conflict":
                    r[f"{test}_containment_conflict"].append(sc["containment"])
                    r[f"{test}_intrusion_conflict"].append(sc["intrusion"])
            rec.append(row)
    out = {k: _m(v) for k, v in r.items()}
    out["controllability_delta_img"] = (out.get("look_img_acc") or 0) - (out.get("noop_img_acc") or 0)
    return out


def eval_templates(model, processor, ds, specs, n, device, forbid, max_n=60,
                   user_fn=_neo_user, tset=None, hset=None):
    tset = tset or P.CTRL_TEMPLATES; hset = hset or P.CTRL_TEMPLATES_HELDOUT
    per_t = {}
    for tpl in list(tset) + list(hset):
        accs = [Me.object_accuracy(Me.parse_pred(generate(model, processor, get_image(ds, s),
                user_fn("look", n, tpl, s.fed_caption), device, 160, forbid_ids=forbid, stop_anchor="final answer:")),
                s.image_key) for s in specs[:max_n]]
        per_t[tpl] = _m(accs)
    tm = _m([per_t[t] for t in tset]); hm = _m([per_t[t] for t in hset])
    return {"per_template": per_t, "train_mean": tm, "heldout_mean": hm,
            "generalization_drop": (tm or 0) - (hm or 0)}


ABSTAIN_MARKERS = ("not available", "unavailable", "no image", "cannot see",
                   "can't see", "not provided", "missing", "unable to see",
                   "don't have access", "do not have access")


def eval_image_ablation(model, processor, ds, specs, n, device, forbid, rec, max_n=80):
    """Ask <look> with the image REMOVED. Three separable behaviors:
      abstain_no_image     -- names the missing source (the HONEST behavior;
                              only absence-trained runs are expected nonzero)
      confident_emit_...   -- emits some parseable value anyway (confabulation
                              of ANY kind, caption-derived or invented)
      launder_to_caption.. -- specifically the caption's values (source leak)
    APO lesson (records audit): low launder + zero abstain + degenerate prose =
    string-suppression, NOT honesty. Report all three, never just launder."""
    emit, launder, abstain = [], [], []
    for s in tqdm(specs[:max_n], desc="no-pixels"):
        out = generate(model, processor, None, P.render_ctrl(P.CTRL_CANONICAL, trigger("look_ctrl", n), s.fed_caption),
                       device, 160, forbid_ids=forbid, stop_anchor="final answer:")
        parsed = Me.parse_pred(out); lc = Me.object_accuracy(parsed, s.text_key)
        ab = int(any(m in out.lower() for m in ABSTAIN_MARKERS))
        emit.append(int(len(parsed) > 0)); launder.append(lc); abstain.append(ab)
        rec.append({"eid": s.example_id, "test": "no_pixels_look", "output": out[:200],
                    "correct": "(should abstain: no image)", "distractor": _ks(s.text_key),
                    "abstained": ab, "exact_match": round(float(lc), 3)})
    return {"confident_emit_no_image": _m(emit), "launder_to_caption_no_image": _m(launder),
            "abstain_no_image": _m(abstain)}


def eval_swaps(model, processor, ds, legset, n, device, forbid, max_n=80):
    look, read = trigger("look_ctrl", n), trigger("read_ctrl", n)
    g = lambda im, op, cap: Me.parse_pred(generate(model, processor, im,
        P.render_ctrl(P.CTRL_CANONICAL, op, cap), device, 160, forbid_ids=forbid, stop_anchor="final answer:"))
    iflip, istay, cflip, cstay = [], [], [], []
    for idx, row, k1, k2 in tqdm(conflict_examples(ds, legset, max_n), desc="swaps"):
        i1, i2, c1, c2 = row["image_1"], row["image_2"], row["caption_1"], row["caption_2"]
        look0 = g(i1, look, c2)
        lookI, readI = g(i2, look, c2), g(i2, read, c2)
        f, _ = flip_score(look0, lookI, k1, k2); iflip.append(int(f)); istay.append(int(Me.key_match(readI, k2)))
        read0 = g(i1, read, c2); lookC, readC = g(i1, look, c1), g(i1, read, c1)
        f2, _ = flip_score(read0, readC, k2, k1); cflip.append(int(f2)); cstay.append(int(Me.key_match(lookC, k1)))
    return {"img_swap_look_flip": _m(iflip), "img_swap_read_stay": _m(istay),
            "cap_swap_read_flip": _m(cflip), "cap_swap_look_stay": _m(cstay)}


def eval_linked(model, processor, ds, specs, n, device, forbid, rec, max_n=None):
    """ARM L: ctrl token in the prompt should INVOKE the op's gen block.
    Measures per op: block emission rate, final-answer accuracy vs the op's
    source, and (compare) whether BOTH stated values are correct."""
    max_n = _MAXN if max_n is None else max_n
    gts = {op: trigger(f"{op}_gen", n) for op in ("look", "read", "compare")}

    def _block(text, open_tok, others):
        if open_tok not in text:
            return ""
        after = text.split(open_tok, 1)[1]
        ends = [e for e in (after.find(x) for x in ([open_tok] + list(others))) if e >= 0]
        return after[:min(ends)] if ends else after

    r = defaultdict(list)
    for s in tqdm(specs[:max_n], desc="linked"):
        img = get_image(ds, s)
        for op in ("look", "read", "compare"):
            user = P.render_ctrl(P.CTRL_CANONICAL, trigger(f"{op}_ctrl", n), s.fed_caption)
            out = generate(model, processor, img, user, device, 160, forbid_ids=forbid,
                           stop_token_ids=getattr(model, "_neo_all_ids", None),
                           stop_token_count=2 * n, stop_anchor="final answer:")
            others = [v for k, v in gts.items() if k != op]
            seg = _block(out, gts[op], others)
            emitted = int(gts[op] in out)
            r[f"link_{op}_emit"].append(emitted)
            r[f"link_{op}_wrong_block"].append(int(any(o in out for o in others)))
            r[f"link_{op}_truncated"].append(int("final answer:" not in out.lower()))
            if op == "compare":
                ok = int((Me.parse_relation(seg or out) or "") == s.relation)
                both = int(_ks(s.image_key) in (seg or "") and _ks(s.text_key) in (seg or "")) \
                       if s.cond == "conflict" else None
                if both is not None:
                    r["link_compare_values_stated"].append(both)
            else:
                gold = s.image_key if op == "look" else s.text_key
                ok = int(Me.object_accuracy(Me.parse_final_answer(seg or out, TASK), gold) >= 1.0)
            r[f"link_{op}_acc"].append(ok)
            rec.append({"eid": s.example_id, "cond": s.cond, "test": f"link_{op}",
                        "output": out[:220], "correct": _ks(s.image_key) if op == "look"
                        else (_ks(s.text_key) if op == "read" else s.relation),
                        "distractor": "", "exact_match": ok, "emitted_block": emitted})
    return {k: _m(v) for k, v in r.items()}


def eval_prefill_conflict(model, processor, ds, specs, n, device, forbid, rec, max_n=60):
    """MECHANISM DISSOCIATION: put <look_ctrl>/<read_ctrl> in the prompt but
    FORCE the answer to begin with a CONGRUENT or INCONGRUENT sourced phrase.
    If the reported value follows the TOKEN despite an incongruent phrase, the
    token routes to the source directly; if it follows the PHRASE, the token is
    (at least partly) a phrase-trigger and the self-generated words carry the
    grounding. Informative for BOTH neutral- and sourced-trained tokens."""
    PRE = {"look": "Based on the image, I observe that the answer is",
           "read": "Based on the text, I read that the answer is"}
    r = defaultdict(list)
    conf = [s_ for s_ in specs if s_.cond == "conflict"][:max_n]
    for s_ in tqdm(conf, desc="prefill-conflict"):
        img = get_image(ds, s_)
        for op in ("look", "read"):
            gold = s_.image_key if op == "look" else s_.text_key
            other = s_.text_key if op == "look" else s_.image_key
            user = P.render_ctrl(P.CTRL_CANONICAL, trigger(f"{op}_ctrl", n), s_.fed_caption)
            for tag, phrase_op in (("congruent", op), ("incongruent", "read" if op == "look" else "look")):
                out = generate(model, processor, img, user, device, 160, stop_anchor="final answer:",
                               assistant_prefill=PRE[phrase_op] + " ", forbid_ids=forbid)
                pred = Me.parse_final_answer(out, TASK) or Me.parse_pred(out)
                r[f"prefill_{tag}_{op}_token_follow"].append(int(Me.object_accuracy(pred, gold) >= 1.0))
                if tag == "incongruent":
                    r[f"prefill_incongruent_{op}_phrase_follow"].append(int(Me.object_accuracy(pred, other) >= 1.0))
                rec.append({"eid": s_.example_id, "cond": s_.cond, "test": f"prefill_{tag}_{op}",
                            "output": out[:160], "correct": _ks(gold), "distractor": _ks(other),
                            "exact_match": int(Me.object_accuracy(pred, gold) >= 1.0)})
    return {k: _m(v) for k, v in r.items()}


def eval_token_identity(model, processor, ds, specs, n, device, forbid, max_n=60):
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<rand_probe_0>"]})
    model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    r = defaultdict(list)
    for s in specs[:max_n]:
        img = get_image(ds, s)
        u_read = P.render_ctrl(P.CTRL_CANONICAL, trigger("read_ctrl", n), s.fed_caption)
        r["read_request_follows_text"].append(Me.object_accuracy(Me.parse_pred(generate(model, processor, img, u_read, device, 160, forbid_ids=forbid, stop_anchor="final answer:")), s.text_key))
        u_rand = P.render_ctrl(P.CTRL_CANONICAL, "<rand_probe_0>", s.fed_caption)
        r["rand_token_img_acc"].append(Me.object_accuracy(Me.parse_pred(generate(model, processor, img, u_rand, device, 160, forbid_ids=forbid, stop_anchor="final answer:")), s.image_key))
    return {k: _m(v) for k, v in r.items()}


def eval_generation(model, processor, ds, specs, n, device, rec, max_n=None):
    max_n = _MAXN if max_n is None else max_n
    r = defaultdict(list)
    lt, rt, ct = trigger("look_gen", n), trigger("read_gen", n), trigger("compare_gen", n)
    for s in tqdm(specs[:max_n], desc="generation"):
        img = get_image(ds, s)
        _gids = getattr(model, "_neo_all_ids", None)
        out = generate(model, processor, img, P.gen_template(TASK, 0).format(cap=s.fed_caption), device, 224,
                       stop_token_ids=_gids, stop_token_count=(6 * n) if _gids else None,
                       stop_anchor=None)
        emit = int(lt in out and rt in out and ct in out)
        def _block(text, open_tok, others):
            # content after the first open tag, terminated by the earliest of:
            # the closing (same) tag or any other operator delimiter -> robust to
            # paired blocks AND to a missing close tag.
            if open_tok not in text:
                return ""
            after = text.split(open_tok, 1)[1]
            ends = [after.find(t) for t in ([open_tok] + list(others))]
            ends = [e for e in ends if e >= 0]
            return after[:min(ends)] if ends else after
        look_seg = _block(out, lt, (rt, ct))
        read_seg = _block(out, rt, (lt, ct))
        cmp_seg = _block(out, ct, (lt, rt))
        pi, pt = Me.parse_final_answer(look_seg, TASK), Me.parse_final_answer(read_seg, TASK)
        prel = Me.parse_relation(cmp_seg or out)
        al, ar = Me.object_accuracy(pi, s.image_key), Me.object_accuracy(pt, s.text_key)
        r["emit_all_tags"].append(emit); r["gen_look_faithful"].append(al); r["gen_read_faithful"].append(ar)
        r["gen_rel_correct"].append(int(prel == s.relation))
        if s.cond == "conflict":
            r["gen_collapse"].append(int(Me.is_collapse(pi, pt, prel, s.image_key, s.text_key)))
            lr = Me.laundering_rate(pi, s.image_key, s.text_key)
            if lr is not None:
                r["gen_laundering"].append(lr)
        rec.append({"eid": s.example_id, "cond": s.cond, "test": "gen_full", "output": out[:300],
                    "correct": "IMAGE %s | TEXT %s | %s" % (_ks(s.image_key), _ks(s.text_key), s.relation),
                    "distractor": "(collapse: both=%s)" % _ks(s.image_key),
                    "exact_match": round(float((al + ar) / 2), 3), "emit_all_tags": emit})
    o = {k: _m(v) for k, v in r.items()}
    return o


def eval_generation_lora(model, processor, ds, specs, device, rec, max_n=None):
    max_n = _MAXN if max_n is None else max_n
    r = defaultdict(list)
    for s in tqdm(specs[:max_n], desc="generation(lora)"):
        img = get_image(ds, s)
        # task-aware wording, mirroring the lora TRAINING prompt in
        # scsrc/collate.py (r7 hardcoded "color" here for all tasks, so the
        # counting/spatial lora cells were evaluated with a color-phrased
        # prompt -- caught by scripts/audit_render.py)
        from scsrc.prompts import slot_word as _sw
        _w = _sw(TASK)
        u = ("Report each object's %s according to the image and the text, and whether they "
             "agree or conflict. Caption: \"%s\". Format: IMAGE: obj: %s, ...; TEXT: obj: %s, "
             "...; RELATION: agree/conflict." % (_w, s.fed_caption, _w, _w))
        out = generate(model, processor, img, u, device, 192)
        img_seg = out.split("IMAGE:", 1)[1].split("TEXT:", 1)[0] if ("IMAGE:" in out and "TEXT:" in out) else ""
        txt_seg = out.split("TEXT:", 1)[1].split("RELATION:", 1)[0] if "TEXT:" in out else ""
        pi, pt, prel = Me.parse_pred(img_seg), Me.parse_pred(txt_seg), Me.parse_relation(out)
        al, ar = Me.object_accuracy(pi, s.image_key), Me.object_accuracy(pt, s.text_key)
        r["lora_look_faithful"].append(al); r["lora_read_faithful"].append(ar)
        r["lora_rel_correct"].append(int(prel == s.relation))
        if s.cond == "conflict":
            r["lora_collapse"].append(int(Me.is_collapse(pi, pt, prel, s.image_key, s.text_key)))
        rec.append({"eid": s.example_id, "cond": s.cond, "test": "lora_full", "output": out[:300],
                    "correct": "IMAGE %s | TEXT %s | %s" % (_ks(s.image_key), _ks(s.text_key), s.relation),
                    "distractor": "(collapse)", "exact_match": round(float((al + ar) / 2), 3)})
    return {k: _m(v) for k, v in r.items()}


def _flatten(d, pre=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(_flatten(v, pre + k + "."))
        elif isinstance(v, (int, float)):
            o[pre + k] = v
    return o



def _phase(out, name, fn, *a, **k):
    """Run one eval phase with a flushed banner, wall-clock, and isolation:
    a crash records the error under out['errors'][name] instead of killing the
    whole checkpoint. Silent phases and lost partial results were the failure
    modes behind 'prints nothing' / 'dies after controllability'."""
    import time as _t
    t0 = _t.time()
    print(f"[phase] {name} ...", flush=True)
    try:
        out[name] = fn(*a, **k)
        print(f"[phase] {name} done in {_t.time()-t0:.0f}s", flush=True)
    except Exception as e:
        import traceback
        out.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"
        print(f"[phase] {name} FAILED after {_t.time()-t0:.0f}s: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--step", default="all")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--task", default="color", choices=["color", "counting", "spatial", "attr"])
    ap.add_argument("--data_dir", default=None, help="processed task dir; eval on a prepared task")
    ap.add_argument("--split", default="test", help="split (processed tasks: test_id | test_ood)")
    ap.add_argument("--eval_max_n", type=int, default=120,
                    help="specs per heavy battery phase (raise for tighter CIs; 999=full split)")
    ap.add_argument("--cross_desc", default=None,
                    help="path to ANOTHER run's step_400.json; its verbalization_text.look_ctrl "
                         "is plugged in for the token here (cross-run/cross-model)")
    args = ap.parse_args()
    device = "cuda"
    global TASK
    TASK = args.task
    if args.data_dir:
        ds = load_task(args.data_dir, args.split)
        legset = None   # task-generic path has no legibility list; None = no swap filter
        specs = build_specs(ds, args.split, conditions=("agree", "conflict"))
        print(f"[eval] CODE VERSION {EVAL_CODE_VERSION}")
        global _MAXN
        _MAXN = args.eval_max_n
        print(f"[eval] task={TASK} from {args.data_dir}/{args.split} ({len(ds)} examples)")
    else:
        ds = load_colorswap("test", human_only=True)
        Me.parse_pred = lambda t: Me.parse_final_answer(t, "color")
        legset = load_legible_ids("test") or set()
        specs = build_specs(ds, "test", legible_ids=legset, conditions=("agree", "conflict"))
    # task-aware parsing for ALL helpers (recursion-safe: these don't call parse_pred)
    if TASK != "color":
        Me.parse_pred = lambda t: Me.parse_final_answer(t, TASK)
        Me.first_color = lambda t: Me.first_value(t, TASK)
        print(f"[eval] NOTE: swaps.py / some metrics are color-derived; validate the "
              f"{TASK} eval matrix on-cluster before trusting every sub-metric.")
    _JUNK = {"ish", "colored", "color", "thing"}
    def _sane(sp):
        return all(o not in _JUNK and len(o) > 1
                   for o in list(sp.image_key) + list(sp.text_key))
    n0 = len(specs); specs = [sp for sp in specs if _sane(sp)]
    if len(specs) != n0:
        print(f"[eval] filtered {n0 - len(specs)} specs with junk gold objects "
              f"(caption artifacts like '-ish'); {len(specs)} remain")
    run_name = os.path.basename(os.path.normpath(args.run_dir))
    try:
        import wandb
        wb = wandb.init(project="source-control", name=f"eval_{run_name}", dir=paths.WANDB_DIR,
                        mode=os.environ.get("WANDB_MODE", "offline"))
    except Exception as e:
        print("[wandb] disabled:", e); wb = None

    steps = ([args.step] if args.step != "all" else
             sorted(int(os.path.basename(p).split("_")[1]) for p in glob.glob(os.path.join(args.run_dir, "ckpt", "step_*"))))
    only = {x.strip() for x in os.environ.get("SC_EVAL_STEPS", "").split(",") if x.strip()}
    if only:
        steps = [st for st in steps if str(st) in only]
        print(f"[eval] SC_EVAL_STEPS={sorted(only)} -> evaluating {steps}")
    import faulthandler
    wd = int(os.environ.get("SC_WATCHDOG_SEC", "0"))  # faulthandler dumps can segfault mid-CUDA; opt-in only
    if wd > 0:
        faulthandler.dump_traceback_later(wd, repeat=True)
    full_all = os.environ.get("SC_EVAL_FULL_ALL") == "1"
    # SC_EVAL_FULL_STEPS="100" -> full battery at listed steps + the last step; light elsewhere
    full_steps = {int(x) for x in os.environ.get("SC_EVAL_FULL_STEPS", "").split(",") if x.strip()}
    last_step = max(int(x) for x in steps)
    for step in steps:
        full = full_all or int(step) == last_step or int(step) in full_steps
        _tag = f"{TASK}_{args.split}" + ("" if not args.data_dir else
                ("" if os.path.normpath(args.data_dir).endswith(TASK) else
                 "_" + os.path.basename(os.path.normpath(args.data_dir))))
        _done = os.path.join(args.run_dir, "eval", _tag, f"step_{step}.json")
        if os.path.exists(_done) and os.environ.get("SC_EVAL_RESUME", "1") == "1":
            try:
                _prev = json.load(open(_done))
                if _prev.get("eval_code_version") == EVAL_CODE_VERSION and (not full or "verbalization_text" in _prev or _prev.get("meta", {}).get("arm") != "ctrl"):
                    print(f"[eval] step {step}: current-version result exists -- skipping (SC_EVAL_RESUME=0 to force)", flush=True)
                    continue
            except Exception:
                pass
        print(f"[eval] ===== checkpoint step {step} ({'FULL' if full else 'light'} battery) =====", flush=True)
        model, processor, meta, forbid = apply_checkpoint(args.run_dir, int(step), device)
        n, arm, method = meta["n_tokens"], meta["arm"], meta["method"]
        out = {"meta": meta, "step": int(step), "eval_code_version": EVAL_CODE_VERSION}; rec = []
        if method == "neo" and arm == "ctrl":
            _phase(out, "controllability", eval_controllability, model, processor, ds, specs, n, device, forbid, rec)
            if full:
                _phase(out, "templates", eval_templates, model, processor, ds, specs, n, device, forbid)
                _phase(out, "no_pixels", eval_image_ablation, model, processor, ds, specs, n, device, forbid, rec)
                _phase(out, "swaps", eval_swaps, model, processor, ds, legset, n, device, forbid)
                _phase(out, "token_identity", eval_token_identity, model, processor, ds, specs, n, device, forbid)
                _phase(out, "prefill_conflict", eval_prefill_conflict, model, processor, ds, specs, n, device, forbid, rec)
                def _verbs(m, pr, nn, fb):
                    return {nm: verbalize_operator(m, pr, nm, nn, device, "ctrl", forbid_ids=fb)
                            for nm in ("look_ctrl", "read_ctrl", "compare_ctrl")}
                _phase(out, "verbalization_text", _verbs, model, processor, n, forbid)
                if "verbalization_text" in out:
                    _phase(out, "plugin_look", plugin_test, model, processor, specs[:60], ds, "look_ctrl",
                           out["verbalization_text"]["look_ctrl"], device, n, "image_key",
                           forbid_ids=forbid, task=TASK, rec=rec)
                    _phase(out, "dual", eval_dual, model, processor, ds, specs, n, device, forbid, rec)
                    def _syn_phase(model=model, processor=processor):
                        syns, raw = synonyms_operator(model, processor, "look_ctrl", n, device, forbid_ids=forbid)
                        out["synonyms_text"] = {"look_ctrl": {"raw": raw[:300], "parsed": syns}}
                        return eval_synonyms(model, processor, specs, ds, "look_ctrl", syns, device, n,
                                             "image_key", forbid_ids=forbid, task=TASK, rec=rec)
                    _phase(out, "plugin_synonyms", lambda *a_, **k_: _syn_phase())
                    def _neighbors(model=model, processor=processor, n=n):
                        import torch as _t
                        emb = model.get_input_embeddings().weight.detach().float()
                        tok = processor.tokenizer
                        names = ("look_ctrl", "read_ctrl", "compare_ctrl")
                        nid = {nm: tok.encode(trigger(nm, n), add_special_tokens=False) for nm in names}
                        trained = {i for ids in nid.values() for i in ids}
                        cand = emb[:len(tok)]
                        res = {}
                        for nm, ids in nid.items():
                            row = emb[ids[0]]
                            sims = _t.nn.functional.cosine_similarity(cand, row.unsqueeze(0).expand_as(cand), dim=1)
                            for ti in trained:
                                sims[ti] = -1.0   # a trained row is never its own neighbor
                            top = _t.topk(sims, 32)
                            out5 = []
                            for v, i in zip(top.values, top.indices):
                                w = tok.decode([int(i)]).strip()
                                if not w or w.startswith("<unused") or w.startswith("<pad"):
                                    continue      # reserved slots are noise, not words
                                out5.append((w, round(float(v), 3)))
                                if len(out5) == 5:
                                    break
                            res[nm] = out5
                        return res
                    _phase(out, "embedding_neighbors", lambda *a_, **k_: _neighbors())
                    if args.cross_desc:
                        import json as _json
                        _cd = _json.load(open(args.cross_desc)).get("verbalization_text", {}).get("look_ctrl", "")
                        if _cd:
                            _phase(out, "plugin_cross", plugin_test, model, processor, specs[:60], ds, "look_ctrl",
                                   _cd, device, n, "image_key", forbid_ids=forbid, task=TASK, rec=rec)
                            out["plugin_cross_source"] = args.cross_desc
        elif method == "neo" and arm == "link":
            _phase(out, "linked", eval_linked, model, processor, ds, specs, n, device, forbid, rec)
        elif method == "neo" and arm == "gen":
            _phase(out, "generation", eval_generation, model, processor, ds, specs, n, device, rec)
            _vf = getattr(model, "_neo_all_ids", None) or forbid
            if not full:
                _vf = None   # sentinel: skip verbalization on light passes
            def _gverbs(m, pr, nn, fb):
                return {nm: verbalize_operator(m, pr, nm, nn, device, "gen", forbid_ids=fb)
                        for nm in ("look_gen", "read_gen", "compare_gen")}
            if _vf is not None or os.environ.get("SC_EVAL_FULL_ALL") == "1":
                _phase(out, "verbalization_text", _gverbs, model, processor, n, _vf)
        elif method == "pt":
            _phase(out, "controllability", eval_controllability, model, processor, ds, specs, n, device, forbid, rec,
                   user_fn=_pt_user, canon=P.pt_template(0))
            if full:
                _phase(out, "templates", eval_templates, model, processor, ds, specs, n, device, forbid,
                       user_fn=_pt_user, tset=[P.pt_template(i) for i in range(4)],
                       hset=[P.pt_template_heldout(0), P.pt_template_heldout(1)])
            # NOTE: self-verbalization, plug-in, interchange, swap-signature are
            # neo-only (a soft prefix has no symbol to verbalize/swap/insert).
        elif method == "lora":
            _phase(out, "generation_lora", eval_generation_lora, model, processor, ds, specs, device, rec)

        # one subdir PER (task, split) so ID / OOD / cross-dataset / cross-task
        # evals of the same run never overwrite each other
        eval_tag = f"{TASK}_{args.split}" + ("" if not args.data_dir else
                    ("" if os.path.normpath(args.data_dir).endswith(TASK) else
                     "_" + os.path.basename(os.path.normpath(args.data_dir))))
        d = os.path.join(args.run_dir, "eval", eval_tag); os.makedirs(d, exist_ok=True)
        out["meta"] = {**out["meta"], "eval_task": TASK, "eval_split": args.split,
                       "eval_data_dir": args.data_dir or "colorswap_builtin"}
        with open(os.path.join(d, f"step_{step}.json"), "w") as f:
            json.dump(out, f, indent=2)
        with open(os.path.join(d, f"records_step_{step}.jsonl"), "w") as f:
            for r in rec:
                f.write(json.dumps(r) + "\n")
        print(f"[eval] step {step}: {len(rec)} records -> {d}/records_step_{step}.jsonl")
        # print a few so you see outputs in the log
        for r in rec[:6]:
            print(f"   [{r['test']}] out='{str(r['output'])[:70]}' correct='{r['correct']}' "
                  f"distractor='{r['distractor']}' EM={r['exact_match']}")
        if wb is not None:
            wb.log({**{f"eval/{k}": v for k, v in _flatten({k: v for k, v in out.items() if k not in ('meta', 'verbalization_text')}).items()}, "step": int(step)})
            if rec:
                cols = sorted({k for x in rec[:300] for k in x})
                tbl = wandb.Table(columns=cols)
                for x in rec[:300]:
                    tbl.add_data(*[x.get(c) for c in cols])
                wb.log({f"eval/records_step_{step}": tbl})
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
