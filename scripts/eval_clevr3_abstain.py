#!/usr/bin/env python
"""ABSTENTION on Ground-CF: does the requested word abstain when its source lacks the answer, answer
when its source has it, and never take the answer from the other source?

Rows come from data/clevr3/absence_<split>_eval.jsonl (scripts/clevr3_absence.py): per base item and
requested source three typed rows,
  req_absent    requested source lacks it, the other has it   -> gold ABSTAIN
  other_absent  requested source has it, the other lacks it   -> gold the requested value
  conflict      both have it and disagree                     -> gold the requested value
Every row is a forced choice over THREE candidates: the requested value (or, when absent, the other
source's value = the LEAK candidate), a same-type distractor absent from the scene, and the ABSTAIN
string of the requested source. Reported per source (and per skill):
  abstain_when_absent   req_absent rows answered ABSTAIN
  leak_rate             req_absent rows answered with the other source's value  (the failure that matters)
  answer_when_present   other_absent rows answered with the requested value
  false_abstain         other_absent + conflict rows answered ABSTAIN
  conflict_acc          conflict rows answered with the requested value
  balanced              (abstain_when_absent + answer_when_present) / 2   (always-abstain = .5)
Output: runs/<run>/eval/clevr3_<split>/abstain_step_<step><tag>.json
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from scsrc.clevr3 import clevr3_user_content, ABSTAIN
from scripts.eval_clevr3_generate import generate_free, parse_answer
from evaluate import apply_checkpoint
from scripts.eval_clevr3 import cand, loglik, loglik_batch
from scsrc.collate import _encode, _prompt_len


@torch.no_grad()   # the original scorer is decorated; without this the Gemma-3 rows OOM a 44 GB L40S (2026-09-21)
def loglik_norm(model, proc, umsg, candidate, device):
    """Length-normalised twin of eval_clevr3.loglik: mean token log-prob of the candidate. The forced
    choice here pits 1-token values against 5-token abstain strings, so the unnormalised sum penalises
    abstaining by four extra terms (STATUS 2026-09-21 v1.45 F3). Ladder candidates are all 1 token, so
    this only matters for abstention."""
    fmsg = umsg + [{"role": "assistant", "content": [{"type": "text", "text": candidate}]}]
    enc = _encode(proc, fmsg, device); plen = _prompt_len(proc, umsg, device)
    ids = enc["input_ids"]; n = ids.shape[1] - plen
    if n <= 0:
        return -1e9
    kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    lg = model(**kw, use_cache=False).logits.float()
    tgt = ids[0, plen:]
    lp = torch.log_softmax(lg[0, plen - 1:ids.shape[1] - 1, :], -1)
    return float(lp.gather(-1, tgt.unsqueeze(-1)).sum()) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id"); ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=100, help="base items per source (x3 typed rows)")
    ap.add_argument("--mode", default=None, help="None (the run's own request) | instructed | noop")
    ap.add_argument("--tag", default="")
    ap.add_argument("--generate", type=int, default=1, help="v1.130: also greedy-generate at every context of the twin battery; raw text kept, parsed deterministically")
    ap.add_argument("--abstain_neutral", action="store_true",
                    help="2026-09-23 (v1.130, author): ZERO-SHOT policy with no modality information anywhere: the policy "
                         "'If there is not enough information to determine the answer according to that request, answer "
                         "unknown.' is appended and the abstain candidate is 'unknown' for BOTH requests (the default "
                         "abstain strings 'not visible in the image' / 'not stated in the caption' name the modality). "
                         "Needs --tag; the question style follows SC_QSTYLE (default explicit).")
    ap.add_argument("--abstain_instr", action="store_true",
                    help="2026-09-23 (author's zero-shot design): APPEND an explicit response option to the user turn -- "
                         "'If the requested source does not contain the answer, reply <ABSTAIN string of the source>.' "
                         "The battery otherwise never tells the model that abstaining is an option (pure forced choice), "
                         "so this is the prompt-POLICY test: the trained word is unchanged, the policy is new. Needs --tag.")
    ap.add_argument("--lennorm", action="store_true",
                    help="score candidates by MEAN token log-prob instead of the sum (see loglik_norm)")
    ap.add_argument("--twins", action="store_true",
                    help="ABSTENTION FLIP/STAY (2026-09-22, STATUS v1.83): read data/clevr3/abstain_twins_<split>.jsonl "
                         "(scripts/clevr3_abstain_twins.py). Per req_absent row: base pick (gold abstain); FLIP = the "
                         "requested source gains the answer with a value the competitor does not state (gold that value, "
                         "candidates abstain / competitor's value / new value); STAY = the competitor's value changes while "
                         "the requested source stays silent (gold abstain). Both twins keep the sources in conflict. "
                         "Writes abstain_twins_step_<step><tag>.json; requires --tag.")
    ap.add_argument("--hard", action="store_true",
                    help="absence_<split>_hard_eval.jsonl: the uninformative caption keeps the TRAINED\n                         template and carries an off-type value, so abstention cannot be read off the\n                         surface form. Those rows get a fourth candidate, the off-type value itself.")
    a = ap.parse_args()
    if a.abstain_instr and not a.tag:
        raise SystemExit("[STOP] --abstain_instr needs --tag so the forced-choice file is not overwritten")
    if a.abstain_neutral and not a.tag:
        raise SystemExit("[STOP] --abstain_neutral needs --tag so the forced-choice file is not overwritten")
    _QS = os.environ.get("SC_QSTYLE", "explicit")

    def ABS(src_):
        return "unknown" if a.abstain_neutral else ABSTAIN[src_]

    def _uc(*args_, **kw_):
        """clevr3_user_content, plus the explicit abstain instruction when --abstain_instr is set. The
        announced option is the SAME string the forced choice scores, so policy and candidate agree."""
        kw_.setdefault("question_style", _QS)
        c = clevr3_user_content(*args_, **kw_)
        if a.abstain_neutral:
            c = list(c) + [{"type": "text", "text": " If there is not enough information to determine the answer according to that request, answer unknown."}]
        elif a.abstain_instr:
            src_ = args_[1]
            c = list(c) + [{"type": "text", "text": f" If the requested source does not contain the answer, reply \"{ABSTAIN[src_]}\"."}]
        return c
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode needs --tag so the trained arm's file is not overwritten")
    if a.lennorm and not a.tag:
        raise SystemExit("[STOP] --lennorm needs --tag so the sum-scored file is not overwritten")
    if a.hard and not a.tag:
        raise SystemExit("[STOP] --hard needs --tag so the easy-caption file is not overwritten")
    if a.twins and not a.tag:
        raise SystemExit("[STOP] --twins needs --tag")
    model, proc, meta, _fb = apply_checkpoint(a.run_dir, a.step, "cuda")
    _forbid = _fb if _fb is not None else getattr(model, "_neo_all_ids", None)
    n = meta["n_tokens"]
    try:
        _cfg = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    except Exception:
        _cfg = {}
    meth = "neo" if a.mode in ("instructed", "noop") else _cfg.get("method", "neo")
    cfgm = a.mode or _cfg.get("factor_mode", "srconly")
    if a.twins:
        tw = [json.loads(l) for l in open(os.path.join(a.data_dir, f"abstain_twins_{a.split}.jsonl"))]
        per_src = collections.Counter(); keep = []
        for r in tw:
            per_src[r["source"]] += 1
            if per_src[r["source"]] <= a.max_n: keep.append(r)
        tw = keep; _ll = loglik_norm if a.lennorm else loglik
        _bchk = {"n": 0, "agree": 0, "maxdiff": 0.0}
        # ANSWER -> ABSTAIN (2026-09-22, STATUS v1.91): the answerable state A of the same item is the split item
        # itself (object present in the image / value stated in the caption). look: A = (original image, the row's
        # caption); read: A = (the row's image, the original caption). U = the row's own base context. The requested
        # source LOSES its evidence while the competitor keeps stating its value.
        base_items = {r_["id"]: r_ for r_ in (json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl")))}
        picks = []
        for i, r in enumerate(tqdm(tw, desc="c3abstain-twins")):
            src = r["source"]
            def ctx(image, caption):
                rr = dict(r); rr["image"] = image; rr["caption"] = caption
                return [{"role": "user", "content": _uc(rr, src, n, cfgm, i, method=meth)}]
            def choose(um, cands):
                # v1.95: one batched pass per pick; the first 12 picks are checked against single-sequence scoring
                strs = {k: cand(ABS(src) if k == "abstain" else v) for k, v in cands.items()}
                # 2026-09-23 (STATUS v1.104): sequential scoring only (the batched pass is not equivalent)
                sc = {k: float(_ll(model, proc, um, strs[k], "cuda")) for k in cands}
                if _bchk["n"] < 12:
                    sc1 = {k: float(_ll(model, proc, um, strs[k], "cuda")) for k in cands}
                    _bchk["n"] += 1; _bchk["agree"] += int(max(sc, key=sc.get) == max(sc1, key=sc1.get))
                    _bchk["maxdiff"] = max(_bchk["maxdiff"], max(abs(sc[k] - sc1[k]) for k in cands))
                return max(sc, key=sc.get), sc
            def gpick(um, cands):
                """v1.130: greedy generation at the same context; the parse is mapped back to a candidate KEY."""
                if not a.generate:
                    return None
                txt, nnew = generate_free(model, proc, um[0]["content"], "cuda", 64, _forbid, 8)
                vals = {k: (ABS(src) if k == "abstain" else str(v)) for k, v in cands.items()}
                single = [v for v in vals.values() if " " not in v]
                lab, val = parse_answer(txt, tuple(single))
                key = next((k for k, v in vals.items() if lab == "cand" and v == val), None)
                if key is None and lab == "abstain" and "abstain" in vals:
                    key = "abstain"                 # multi-word abstain strings / natural refusals
                return {"raw_generation": txt, "gen_ntok": nnew, "parse_label": lab, "parsed_key": key}
            b_pick, b_sc = choose(ctx(r["image"], r["caption"]), {"abstain": None, "competitor": r["leak"], "distractor": r["distractor"]})
            f_pick, f_sc = choose(ctx(r["flip"]["image"], r["flip"]["caption"]), r["flip"]["cands"])
            s_pick, s_sc = choose(ctx(r["stay"]["image"], r["stay"]["caption"]), r["stay"]["cands"])
            a_pick = None
            bi = base_items.get(r.get("base_id"))
            if bi is not None and str(bi["channels"][src]) == str(r["req_value"]):
                a_img = bi["image"] if src == "look" else r["image"]
                a_cap = r["caption"] if src == "look" else bi["caption"]
                a_pick, _ = choose(ctx(a_img, a_cap), {"abstain": None, "value": r["req_value"], "competitor": r["leak"]})
            _g = {}
            if a.generate:
                _g["base"] = gpick(ctx(r["image"], r["caption"]), {"abstain": None, "competitor": r["leak"], "distractor": r["distractor"]})
                _g["flip"] = gpick(ctx(r["flip"]["image"], r["flip"]["caption"]), r["flip"]["cands"])
                _g["stay"] = gpick(ctx(r["stay"]["image"], r["stay"]["caption"]), r["stay"]["cands"])
                if a_pick is not None:
                    _g["answerable"] = gpick(ctx(a_img, a_cap), {"abstain": None, "value": r["req_value"], "competitor": r["leak"]})
            picks.append({"id": r["id"], "source": src, "skill": r["skill"], "base": b_pick, "flip": f_pick, "stay": s_pick,
                          "flip_gold_value": r["flip"]["gold"], "answerable": a_pick,
                          "scores": {"base": {k: round(v, 4) for k, v in b_sc.items()}, "flip": {k: round(v, 4) for k, v in f_sc.items()},
                                     "stay": {k: round(v, 4) for k, v in s_sc.items()}},
                          "gen": _g})
        def rate(src, fn, skill=None):
            xs = [p for p in picks if p["source"] == src and (skill is None or p["skill"] == skill)]
            return round(sum(fn(p) for p in xs) / len(xs), 3) if xs else None
        out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step, "mode": a.mode, "abstain_instr": bool(a.abstain_instr), "abstain_neutral": bool(a.abstain_neutral), "question_style": _QS, "twins": True,
               "lennorm": a.lennorm, "n_rows": len(picks), "per_source": {}, "per_skill": {},
               "batched_scoring": True, "batched_check": {"n": _bchk["n"], "pick_agreement": (round(_bchk["agree"] / _bchk["n"], 3) if _bchk["n"] else None),
                                                           "max_abs_loglik_diff": round(_bchk["maxdiff"], 4)}}
        for src in ("look", "read"):
            d = {"abstain_base": rate(src, lambda p: p["base"] == "abstain"),
                 "leak_base": rate(src, lambda p: p["base"] == "competitor"),
                 # FLIP: abstained at base AND answered the NEW value when the requested source gained it
                 "flip": rate(src, lambda p: p["base"] == "abstain" and p["flip"] == "value"),
                 "flip_cond": None,
                 "flip_leak": rate(src, lambda p: p["flip"] == "competitor"),
                 # STAY: abstained at base AND still abstains when only the competitor changed
                 "stay": rate(src, lambda p: p["base"] == "abstain" and p["stay"] == "abstain"),
                 "stay_leak": rate(src, lambda p: p["stay"] == "competitor")}
            ab = [p for p in picks if p["source"] == src and p["base"] == "abstain"]
            d["flip_cond"] = round(sum(p["flip"] == "value" for p in ab) / len(ab), 3) if ab else None
            d["stay_cond"] = round(sum(p["stay"] == "abstain" for p in ab) / len(ab), 3) if ab else None
            # answer -> abstain: answered the requested value in the answerable state AND abstains once the
            # requested source loses its evidence (the competitor keeps its value); leak = answered the competitor
            an = [p for p in picks if p["source"] == src and p["answerable"] is not None]
            d["answer_a"] = round(sum(p["answerable"] == "value" for p in an) / len(an), 3) if an else None
            d["a2u"] = round(sum(p["answerable"] == "value" and p["base"] == "abstain" for p in an) / len(an), 3) if an else None
            av = [p for p in an if p["answerable"] == "value"]
            d["a2u_cond"] = round(sum(p["base"] == "abstain" for p in av) / len(av), 3) if av else None
            d["a2u_leak"] = round(sum(p["base"] == "competitor" for p in av) / len(av), 3) if av else None
            d["a2u_n"] = len(an)
            out["per_source"][src] = d
            if a.generate:
                _gk = lambda p, c: ((p.get("gen") or {}).get(c) or {}).get("parsed_key")
                d.update({"gen_abstain_base": rate(src, lambda p: _gk(p, "base") == "abstain"),
                          "gen_leak_base": rate(src, lambda p: _gk(p, "base") == "competitor"),
                          "gen_flip": rate(src, lambda p: _gk(p, "base") == "abstain" and _gk(p, "flip") == "value"),
                          "gen_answer_a": (lambda xs: round(sum(_gk(p, "answerable") == "value" for p in xs) / len(xs), 3) if xs else None)(
                              [p for p in picks if p["source"] == src and (p.get("gen") or {}).get("answerable")]),
                          "gen_parse_rate": rate(src, lambda p: _gk(p, "base") is not None)})
            for sk in sorted({p["skill"] for p in picks}):
                out["per_skill"][f"{src}|{sk}"] = {"abstain_base": rate(src, lambda p: p["base"] == "abstain", sk),
                                                   "flip": rate(src, lambda p: p["base"] == "abstain" and p["flip"] == "value", sk),
                                                   "stay": rate(src, lambda p: p["base"] == "abstain" and p["stay"] == "abstain", sk)}
        out["picks"] = picks
        od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}"); os.makedirs(od, exist_ok=True)
        dst = os.path.join(od, f"abstain_twins_step_{a.step}{a.tag}.json")
        json.dump(out, open(dst, "w"), indent=1)
        print(json.dumps({k: v for k, v in out.items() if k not in ("picks", "per_skill")}, indent=1)); print("[saved]", dst)
        return
    _stem = f"absence_{a.split}_hard_eval.jsonl" if a.hard else f"absence_{a.split}_eval.jsonl"
    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, _stem))]
    # cap by base items per source, keeping the three types of each base item together
    keep, per_src = [], collections.Counter()
    for r in rows:
        if r["type"] == "req_absent":
            per_src[r["source"]] += 1
        if per_src[r["source"]] <= a.max_n:
            keep.append(r)
    rows = keep
    picks = []
    for i, r in enumerate(tqdm(rows, desc="c3abstain")):
        src = r["source"]
        value_cand = r["leak"] if r["type"] == "req_absent" else r["req_value"]
        cands = {"value": value_cand, "distractor": r["distractor"], "abstain": ABS(src)}
        # the off-type value that the rewritten caption states: a model that just repeats the value
        # word it can see in the caption is scored as copying, not silently counted as abstaining.
        if r.get("copy"):
            cands["copy"] = r["copy"]
        um = [{"role": "user", "content": _uc(r, src, n, cfgm, i, method=meth)}]
        _ll = loglik_norm if a.lennorm else loglik
        sc = {k: float(_ll(model, proc, um, cand(v), "cuda")) for k, v in cands.items()}
        pick = max(sc, key=sc.get)
        picks.append({"id": r["id"], "source": src, "type": r["type"], "skill": r["skill"], "pick": pick,
                      "margin_abstain": round(sc["abstain"] - max(sc["value"], sc["distractor"]), 4)})
    def rate(src, typ, pick, skill=None):
        xs = [p for p in picks if p["source"] == src and p["type"] == typ and (skill is None or p["skill"] == skill)]
        return round(sum(p["pick"] == pick for p in xs) / len(xs), 3) if xs else None
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step, "mode": a.mode, "abstain_instr": bool(a.abstain_instr), "abstain_neutral": bool(a.abstain_neutral), "question_style": _QS, "hard": a.hard, "lennorm": a.lennorm,
           "n_rows": len(picks), "per_source": {}, "per_skill": {}, "note": __doc__.strip().split("\n\n")[0]}
    for src in ("look", "read"):
        d = {"abstain_when_absent": rate(src, "req_absent", "abstain"), "leak_rate": rate(src, "req_absent", "value"),
             "answer_when_present": rate(src, "other_absent", "value"),
             "false_abstain_present": rate(src, "other_absent", "abstain"),
             "conflict_acc": rate(src, "conflict", "value"), "false_abstain_conflict": rate(src, "conflict", "abstain"),
             "copy_when_absent": rate(src, "req_absent", "copy"),
             "copy_when_present": rate(src, "other_absent", "copy")}
        d["balanced"] = (round((d["abstain_when_absent"] + d["answer_when_present"]) / 2, 3)
                         if d["abstain_when_absent"] is not None and d["answer_when_present"] is not None else None)
        out["per_source"][src] = d
        for sk in ("color", "object"):
            out["per_skill"][f"{src}|{sk}"] = {"abstain_when_absent": rate(src, "req_absent", "abstain", sk),
                                               "leak_rate": rate(src, "req_absent", "value", sk),
                                               "answer_when_present": rate(src, "other_absent", "value", sk)}
    out["records"] = picks
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=1))
    od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"abstain_step_{a.step}{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
