#!/usr/bin/env python
"""METALINGUISTIC ACCESS to the learned requests on Ground-CF (author, 2026-09-23, STATUS v1.131).

Can the model TALK ABOUT the words it was given, not only use them? Seven prompt families. A-F contain the request
itself as an object of discussion and need no image; G asks, inside a real Ground-CF prompt, which source the active
request asks for.

  A same_yn      "Does <look> mean the same thing as <read>? Answer yes or no."      gold no  (yes for <look>/<look>)
  B same_sd      "Are <look> and <read> the same or different? Answer same or different."   gold different / same
  C mc_word      "Which instruction asks you to answer from the image?  A. <look>  B. <read>"   both option orders
  D mc_source    "What does <look> ask you to use when answering?  A. the image  B. the caption"  both orders
  E open_diff    "To answer this question, what is the difference between answering in a <look> way and a <read> way?"
  F open_mean    "What does it mean to answer in a <look> way?"
  G self_report  a Ground-CF conflict item (image + caption) with the interface's held-out request carrier; the question
                 is replaced by "Which source does this request ask you to use? A. the image B. the caption" (both orders)

Controls per interface: word -> its tokens; prefix -> its soft tokens placed inline where the word would be (a prefix
has no other way to be an object of a sentence); English (--mode instructed) -> "according to the image" / "according to
the caption" in quotes, the ceiling reference; LoRA adapter and no request -> A-F not expressible (N/A records), G only
(the adapter, or nothing, is the request; for no request G has no gold and the answer distribution is reported).

Scoring: closed families (A-D, G) are deterministic forced choices -- the log-likelihood of every candidate is recorded
under two continuations, anchored "final answer: <c>" (PRIMARY, prespecified 2026-09-24) and bare "<c>" -- plus a greedy free generation that is
kept verbatim and parsed deterministically (the letter or word after the last "final answer:", else the first candidate
in the reply). E-F are generation only; the raw text is kept and tagged with keyword flags (image / caption vocabulary)
for later annotation. Every record carries the rendered prompt.

Output: runs/<run>/eval/clevr3_<split>/meta_words_step_<step><tag>.json  (use a _meta_* tag: the keeper matches *<tag>.json)
Usage: python scripts/eval_meta_words.py --run_dir runs/<run> --tag _meta_fc [--mode instructed|noop --tag _meta_fc_instr] [--n_items 40]
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scsrc.clevr3 import clevr3_token, clevr3_user_content, set_c3_vocab, heldout_question, eval_design
from scsrc.tokens import trigger
from evaluate import apply_checkpoint
from scripts.eval_clevr3 import loglik
from scripts.eval_clevr3_generate import generate_free, parse_answer, parse_mc, ANCHOR

SRC_NAME = {"look": "the image", "read": "the caption"}
IMG_WORDS = ("image", "picture", "visual", "photo", "see", "seen", "look at", "depicted", "shown")
CAP_WORDS = ("caption", "text", "written", "read", "stated", "says", "description", "words")


def _first(low, words):
    hits = [m.start() for w in words for m in [re.search(rf"\b{re.escape(w)}\b", low)] if m]
    return min(hits) if hits else -1


def kw_flags(text):
    """Keyword tags for later annotation only (word-boundary matches; a heuristic, never a headline number)."""
    low = text.lower()
    fi, fc = _first(low, IMG_WORDS), _first(low, CAP_WORDS)
    first = ("image" if fi >= 0 and (fc < 0 or fi < fc) else "caption" if fc >= 0 else None)
    return {"mentions_image": fi >= 0, "mentions_caption": fc >= 0, "first_mentioned": first}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id"); ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--mode", default=None, help="None (the run's own interface) | instructed | noop")
    ap.add_argument("--n_items", type=int, default=40, help="conflict items for family G (balanced over skills)")
    ap.add_argument("--max_new", type=int, default=64); ap.add_argument("--max_new_open", type=int, default=160)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode needs --tag")
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    method = "english" if a.mode == "instructed" else "no_request" if a.mode == "noop" else cfg.get("method", "neo")
    model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    if forbid is None:
        forbid = getattr(model, "_neo_all_ids", None)
    n = meta["n_tokens"]; fm = cfg.get("factor_mode", "srconly")
    probe_item = {"skill": "color", "id": "meta"}
    if method == "neo":
        W = {s: clevr3_token(probe_item, s, n, fm) for s in ("look", "read")}
    elif method == "pt":
        W = {s: trigger(f"pt_{s}", n) for s in ("look", "read")}
    elif method == "english":
        W = {s: f'"according to {SRC_NAME[s]}"' for s in ("look", "read")}
    else:
        W = None                                          # LoRA adapter / no request: no word to talk about
    recs = []
    forbid_holder = [forbid]

    def anchored(c):
        """The anchored forced-choice continuation in the run's own answer format (2026-10-02): v2 runs were trained on
        'Final answer: <v>' (SC_GCF_CONT=final, restored from config by apply_checkpoint), bare runs on '<v>'; banked
        runs keep the legacy anchored short form 'final answer: <c>'."""
        fmt = os.environ.get("SC_GCF_CONT", "leadin")
        return f"Final answer: {c}" if fmt == "final" else (f"{c}" if fmt == "bare" else f"{ANCHOR} {c}")

    def run_closed(family, para, text, cands, gold, opts=None, extra=None, content=None):
        um = [{"role": "user", "content": content or [{"type": "text", "text": text}]}]
        from scripts.eval_clevr3_generate import loglik_both as _lb
        from scripts.tokscore import split_spans as _ss
        _det = {c: _lb(model, proc, um, anchored(c), "cuda", tokens=True) for c in cands}
        _spd = _ss({c: (d[3], d[4]) for c, d in _det.items()})              # v1.144: token-level detail
        s_anc = {c: round(float(_spd[c]["answer_sum"] if os.environ.get("SC_SCORE", "turn") == "answer" else _det[c][0]), 4) for c in cands}
        s_bare = {c: round(float(loglik(model, proc, um, c, "cuda")), 4) for c in cands}
        p_anc = max(s_anc, key=s_anc.get); p_bare = max(s_bare, key=s_bare.get)
        txt, nnew = generate_free(model, proc, um[0]["content"], "cuda", a.max_new, forbid_holder[0], 6)
        if opts:
            # parse against display names, so a reply naming the source instead of the letter still parses
            disp = {k: {"look": "image", "read": "caption"}.get(v, v) for k, v in opts.items()}
            lab, val, _route = parse_mc(txt, disp); parsed = next((k for k, v in disp.items() if v == val), None)
        else:
            lab, parsed = parse_answer(txt, tuple(cands))
        ptxt = " ".join("[IMAGE]" if c.get("type") == "image" else c.get("text", "") for c in um[0]["content"])
        recs.append({"family": family, "paraphrase": para, "prompt": ptxt, "candidates": list(cands), "gold": gold,
                     "options": opts, "scores_anchored": s_anc, "scores_bare": s_bare,
                     "anchored_detail": {c: {"sum": round(_det[c][0], 4), **_spd[c]} for c in _det},
                     "pick_anchored": p_anc, "pick_bare": p_bare, "raw_generation": txt, "gen_ntok": nnew,
                     "parse_label": lab, "parsed": parsed,
                     "correct_ll": (int(p_anc == gold) if gold is not None else None),
                     "correct_gen": (int(parsed == gold) if gold is not None else None), **(extra or {})})

    def run_open(family, para, text, extra=None):
        um = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        txt, nnew = generate_free(model, proc, um[0]["content"], "cuda", a.max_new_open, forbid_holder[0], 6)
        recs.append({"family": family, "paraphrase": para, "prompt": text, "raw_generation": txt, "gen_ntok": nnew,
                     **kw_flags(txt), **(extra or {})})

    def closed_families(Wd, sfx=""):
        """A-D with the given request strings; sfx names a control set (v1.141)."""
        combos = [("look", "read", "no"), ("read", "look", "no"), ("look", "look", "yes"), ("read", "read", "yes")]
        for p, tpl in enumerate(["Does {x} mean the same thing as {y}? Answer yes or no.",
                                 "Do {x} and {y} ask for the same thing? Answer yes or no."]):
            for x, y, g in combos:
                run_closed("A_same_yn" + sfx, p, tpl.format(x=Wd[x], y=Wd[y]), ("yes", "no"), g, extra={"words": [x, y]})
        for p, tpl in enumerate(["Are {x} and {y} the same or different? Answer same or different.",
                                 "Is {x} the same as {y}, or different? Answer same or different."]):
            for x, y, g in combos:
                run_closed("B_same_sd" + sfx, p, tpl.format(x=Wd[x], y=Wd[y]), ("same", "different"),
                           "same" if g == "yes" else "different", extra={"words": [x, y]})
        for p, tpl in enumerate(["Which instruction asks you to answer from {s}?\nA. {a}\nB. {b}\nAnswer with the letter.",
                                 "Which of these asks you to use {s}?\nA. Answer in a {a} way.\nB. Answer in a {b} way.\n"
                                 "Answer with the letter."]):
            for target in ("look", "read"):
                for first in ("look", "read"):
                    second = "read" if first == "look" else "look"
                    opts = {"A": first, "B": second}
                    run_closed("C_mc_word" + sfx, p, tpl.format(s=SRC_NAME[target], a=Wd[first], b=Wd[second]), ("A", "B"),
                               "A" if first == target else "B", opts=opts, extra={"target": target})
        for p, tpl in enumerate(["What does {w} ask you to use when answering?\nA. {a}\nB. {b}\nAnswer with the letter.",
                                 "If a question says \"Answer in a {w} way\", where should the answer come from?\n"
                                 "A. {a}\nB. {b}\nAnswer with the letter."]):
            for wsrc in ("look", "read"):
                for first in ("look", "read"):
                    second = "read" if first == "look" else "look"
                    opts = {"A": first, "B": second}
                    run_closed("D_mc_source" + sfx, p, tpl.format(w=Wd[wsrc], a=SRC_NAME[first], b=SRC_NAME[second]),
                               ("A", "B"), "A" if first == wsrc else "B", opts=opts, extra={"word": wsrc})

    if W is not None:
        closed_families(W)
        # v1.141 (reviewer): CONTROLS for A-D. The Ground-CF words start from the mean embedding of "image" / "caption"
        # (prefix rows: "image" / "text"), so an untrained word already leans toward its source. Two control sets, added
        # as NEW special tokens with the trainer's word_mean init (+0.01 noise, fixed seed):
        #   _ctlinit : initialised exactly like the learned rows before training (init words of this run's specs);
        #   _ctlnul  : both initialised from "thing" -- tests whether "different" is just "two distinct tokens".
        # The gold of a control prompt keeps the learned-word gold (distinct pair -> "different"); for _ctlnul the
        # C/D golds are meaningless and are reported only as the pick distribution.
        if method in ("neo", "pt"):
            import torch as _t
            from scsrc import model as _M
            from scsrc.tokens import token_strings as _ts
            # 2026-10-02 (advisor): the initialization words come from the RUN (word: Ground-CF specs image/caption;
            # prefix: image / --pt_read_init, "caption" in the protocol-v2 pilot, "text" in banked runs)
            init_words = ({"look": "image", "read": "caption"} if method == "neo"
                          else {"look": "image", "read": cfg.get("pt_read_init") or "text"})
            print(f"[meta] initialization-control words: {init_words}")
            ctl = {"ctlinit_look": init_words["look"], "ctlinit_read": init_words["read"], "ctlnul_a": "thing", "ctlnul_b": "thing"}
            strs = [t for nm in ctl for t in _ts(nm, n)]
            proc.tokenizer.add_special_tokens({"additional_special_tokens": strs})
            _M.resize_for_new_tokens(model, proc.tokenizer, seed=0)
            emb = model.get_input_embeddings().weight; g = _t.Generator(device="cpu").manual_seed(12345)
            with _t.no_grad():
                for nm, wd in ctl.items():
                    sub = proc.tokenizer(wd, add_special_tokens=False)["input_ids"]; base = emb[sub].mean(0)
                    for tid in proc.tokenizer.convert_tokens_to_ids(_ts(nm, n)):
                        emb[tid] = base + 0.01 * _t.randn(emb.shape[1], generator=g).to(emb.device, emb.dtype)
            ctl_ids = proc.tokenizer.convert_tokens_to_ids(strs)
            nonlocal_forbid = list(forbid or []) + list(ctl_ids)
            forbid_holder[0] = nonlocal_forbid
            closed_families({"look": trigger("ctlinit_look", n), "read": trigger("ctlinit_read", n)}, "_ctlinit")
            closed_families({"look": trigger("ctlnul_a", n), "read": trigger("ctlnul_b", n)}, "_ctlnul")
    if W is not None:
        for p, (x, y) in enumerate([("look", "read"), ("read", "look")]):
            run_open("E_open_diff", p, "To answer this question, what is the difference between answering in a "
                     f"{W[x]} way and a {W[y]} way?", extra={"order": [x, y]})
        for wsrc in ("look", "read"):
            run_open("F_open_mean", 0, f"What does it mean to answer in a {W[wsrc]} way?", extra={"word": wsrc})

    # G: self-report inside a real Ground-CF prompt, for every interface
    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    set_c3_vocab(rows)
    conf = sorted([r for r in rows if str(r["channels"]["look"]) != str(r["channels"]["read"])], key=lambda r: r["id"])
    per = {}; items = []
    for r in conf:
        if per.get(r["skill"], 0) < max(a.n_items // 4, 1):
            per[r["skill"]] = per.get(r["skill"], 0) + 1; items.append(r)
    lora_src = (cfg.get("train_sources") or "").strip() if method == "lora" else None
    g_sources = [lora_src] if lora_src else ["look", "read"]
    cmode = a.mode or fm
    meth_ct = "neo" if a.mode in ("instructed", "noop") else cfg.get("method", "neo")
    for k, it in enumerate(items):
        for src in g_sources:
            for first in ("look", "read"):
                second = "read" if first == "look" else "look"
                q = (f"Which source does this request ask you to use to answer? Options: A. {SRC_NAME[first]} "
                     f"B. {SRC_NAME[second]}. Answer with the letter.")
                # render the REAL item (the fact caption is derived from its question), then swap only the question
                content = clevr3_user_content(it, src, n, cmode, k, method=meth_ct, question_style="heldout")
                q_ho = heldout_question(it, eval_design(it, k, "heldout")[0])   # the formulation the builder used
                last = content[-1]["text"]
                assert q_ho in last, (q_ho, last)
                content = content[:-1] + [{"type": "text", "text": last.replace(q_ho, q, 1)}]
                gold = None if method == "no_request" else ("A" if first == src else "B")
                run_closed("G_self_report", 0, None, ("A", "B"), gold, opts={"A": first, "B": second},
                           content=content, extra={"request": src, "item_id": it["id"], "skill": it["skill"]})

    fam = {}
    for r in recs:
        f = fam.setdefault(r["family"], {"n": 0, "ll": [], "gen": [], "parsed": [], "yes": []})
        f["n"] += 1
        if r.get("correct_ll") is not None:
            f["ll"].append(r["correct_ll"]); f["gen"].append(r["correct_gen"])
        if "parse_label" in r:
            f["parsed"].append(int(r["parse_label"] == "cand"))
        if r["family"].startswith("A_same_yn"):
            f["yes"].append(int(r["pick_anchored"] == "yes"))
        if r["family"] == "G_self_report":
            f.setdefault("picked_image", []).append(int(r["options"][r["pick_anchored"]] == "look"))
    m = lambda v: round(sum(v) / len(v), 4) if v else None
    summary = {k: {"n": v["n"], "acc_ll": m(v["ll"]), "acc_gen": m(v["gen"]), "parse_rate": m(v["parsed"]),
                   **({"p_yes_ll": m(v["yes"])} if v["yes"] else {}),
                   **({"p_pick_image_ll": m(v["picked_image"])} if v.get("picked_image") else {}),
                   "chance": (0.5 if k[0] in "ABCDG" else None)} for k, v in fam.items()}
    for k in ("E_open_diff", "F_open_mean"):
        if k in summary:
            rs = [r for r in recs if r["family"] == k]
            summary[k]["keyword_first_mentioned"] = [r["first_mentioned"] for r in rs]
    od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}"); os.makedirs(od, exist_ok=True)   # where the keeper looks
    fp = os.path.join(od, f"meta_words_step_{a.step}{a.tag}.json")
    out = {"run": os.path.basename(a.run_dir), "model": cfg.get("model_id"), "method": method, "mode": a.mode,
           "step": a.step, "words": W, "applicable_A_to_F": W is not None, "g_requests": g_sources,
           "caption_style": os.environ.get("SC_CAPTION_STYLE", "template"), "summary": summary, "records": recs,
           "cont_format": os.environ.get("SC_GCF_CONT", "leadin"), "init_control_words": (init_words if method in ("neo", "pt") else None),
           "note": "A-D, G forced choice over the listed candidates (anchored continuation in the run's answer format, primary; bare recorded) "
                   "plus greedy generation parsed deterministically; E-F generation only with keyword flags. Chance .5."}
    json.dump(out, open(fp, "w"), indent=1)
    print(json.dumps(summary, indent=1)); print("[saved]", fp)


if __name__ == "__main__":
    main()
