#!/usr/bin/env python
"""COMPOSED CONFLICT DETECTION on Ground-CF: two modality requests in ONE prompt.

Prompt and target follow scsrc.prompts.compose_* exactly as in training
(ctrl_style="compose" / --compose_frac), so a compose-trained run is scored in its own
format and a parity run (trained on single requests only) is scored ZERO-SHOT in the
same format. Nothing is generated or parsed:

  answers  the four two-answer prefixes (v1, v2) over the two sources' values are scored
           jointly by teacher forced log likelihood; the best pair is the model's answer
             detect  = the two answers differ on a CONFLICT item
             attrib  = and each answer is its own source's value
             false   = the two answers differ on the matching AGREE twin (the text member
                       of the pair edited to the other source's value)
  verdict  given the model's own two answers, "The requested sources agree/conflict.
           final answer: agree/conflict" is a forced choice; reported as recall on conflict
           and on agree items, and balanced (chance .5 for any constant policy)

Interfaces differ only in the request carrier (scsrc.prompts.compose_carrier):
  neo words inline | pt prefixes prepended | lora and --mode instructed English names |
  --mode noop a neutral phrase (the position-effect null).
--template heldout scores with a request template never used in training
(format-generalisation test).

Usage:
  python scripts/eval_compose_detect.py --run_dir runs/clevr3_f_5tok_ctm_compose_seed0 --split test_id
  python scripts/eval_compose_detect.py --run_dir runs/clevr3_f_5tok_sft_seed0 --split test_id --mode instructed --tag _instr
Output: runs/<run>/eval/clevr3_<split>/compose_detect_step_<step><tag>.json
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from scsrc.clevr3 import _edited_item, clevr3_compose_content, COMPOSE_PAIRS
from scripts.eval_clevr3_generate import generate_free, parse_compose, verdict_loose
from scsrc.tokens import c3_sources
from scsrc.prompts import compose_answers, compose_verdict
from evaluate import apply_checkpoint
from scripts.eval_clevr3 import loglik


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=150)
    ap.add_argument("--mode", default=None, help="srconly (default: the run's) | instructed | noop")
    ap.add_argument("--template", default="train", choices=["train", "heldout", "explicit2"],
                    help="explicit2 (v1.141): 'Answer the question twice. First, answer according to X. Second, ...' -- "
                         "both-correct is primary, conflict is derived from the two answers, no verdict is asked")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pt_start", type=int, default=0,
                    help="v1.161 (author): explicit2 + prefix runs only -- STANDARD prefix composition: both source prefixes "
                         "at the start of the user turn in request order; the instruction maps them to the two answers in order")
    ap.add_argument("--both_orders", type=int, default=1)
    ap.add_argument("--generate", type=int, default=1, help="v1.130: also greedy-generate the composed answer; raw text kept, parsed deterministically")
    ap.add_argument("--max_new", type=int, default=96)
    a = ap.parse_args()
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode instructed/noop needs --tag (would overwrite the trained arm's file)")
    _QS = os.environ.get("SC_QSTYLE", "explicit")     # v1.131: "heldout" = held-out question formulations
    if _QS != "explicit" and not a.tag:
        raise SystemExit("[STOP] SC_QSTYLE needs --tag")
    try:
        _cfg0 = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    except Exception:
        _cfg0 = {}
    if a.template == "explicit2" and a.mode == "noop":
        od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}"); os.makedirs(od, exist_ok=True)
        fp = os.path.join(od, f"compose_detect_step_{a.step}{a.tag}.json")
        json.dump({"meta": {"run": a.run_dir, "applicable": False, "template": "explicit2",
                            "note": "no request: there is nothing to compose"}}, open(fp, "w"), indent=1)
        print("[saved]", fp, "(not applicable)"); return
    if a.mode is None and _cfg0.get("method") == "lora":
        # v1.131 (author): one adapter per request is global model state -- two requests in one prompt are not
        # expressible. An explicit N/A record instead of a number (the two-separate-prompts detection covers LoRA).
        od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}"); os.makedirs(od, exist_ok=True)
        fp = os.path.join(od, f"compose_detect_step_{a.step}{a.tag}.json")
        json.dump({"meta": {"run": a.run_dir, "applicable": False, "note": "LoRA adapter per request: a composed "
                   "two-request prompt is not expressible; see gen_step_*_detect2 (two separate prompts)."}},
                  open(fp, "w"), indent=1)
        print("[saved]", fp, "(not applicable)"); return
    model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    if forbid is None:
        forbid = getattr(model, "_neo_all_ids", None)
    n = meta["n_tokens"]
    try:
        cfg = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    except Exception:
        cfg = {}
    meth = "neo" if a.mode in ("instructed", "noop") else cfg.get("method", "neo")
    cfgm = a.mode or cfg.get("factor_mode", "srconly")
    trained_compose = float(cfg.get("compose_frac") or 0) > 0
    heldout = a.template == "heldout"
    if a.template == "explicit2":
        return run_explicit2(a, model, proc, n, meth, cfgm, forbid, cfg)
    print(f"[prompt] method={meth} mode={cfgm} L={n} compose-trained={trained_compose} template={a.template}")
    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]

    def um(item, srcs, seed):
        return [{"role": "user", "content": clevr3_compose_content(
            item, srcs, n, cfgm, seed, method=meth, heldout=heldout, question_style=_QS)}]

    def pick_two(item, srcs, cands, seed):
        u = um(item, srcs, seed)
        sc = {(v1, v2): loglik(model, proc, u, compose_answers(v1, v2), "cuda")
              for v1 in cands for v2 in cands}
        pick_two.last = {f"{v1}; {v2}": round(float(x), 4) for (v1, v2), x in sc.items()}   # v1.131: every pair's score
        return max(sc, key=sc.get)

    def verdict(item, srcs, p1, p2, seed):
        u = um(item, srcs, seed); pre = compose_answers(p1, p2)
        sc = {r: loglik(model, proc, u, pre + compose_verdict(r), "cuda") for r in ("agree", "conflict")}
        verdict.last = {r: round(float(x), 4) for r, x in sc.items()}
        return max(sc, key=sc.get)

    def gen(item, srcs, cands, seed):
        """v1.130: free generation of the composed answer + deterministic parse (strict verdict after 'final answer:')."""
        if not a.generate:
            return {}
        txt, nnew = generate_free(model, proc, um(item, srcs, seed)[0]["content"], "cuda", a.max_new, forbid, 8)
        lab, v1, v2, vd = parse_compose(txt, tuple(dict.fromkeys(cands)))
        return {"raw_generation": txt, "gen_ntok": nnew, "parse_label": lab, "parsed": [v1, v2],
                "verdict_strict": vd, "verdict_loose": verdict_loose(txt)}

    orders = []
    for s1, s2 in [p for p in COMPOSE_PAIRS if p[0] in c3_sources() and p[1] in c3_sources()]:
        orders.append((s1, s2))
        if a.both_orders:
            orders.append((s2, s1))
    R = {f"{s1}|{s2}": {"detect": [], "attrib": [], "false": [], "v_conflict": [], "v_agree": [],
                        "consistent": []} for s1, s2 in orders}
    SK = {}                     # per-skill (colour / shape / count / spatial), STATUS v1.102
    recs, seen = [], 0
    for i, it in enumerate(tqdm(rows, desc="compose")):
        if seen >= a.max_n:
            break
        ch = {k: str(v) for k, v in it["channels"].items()}
        used = False
        for s1, s2 in orders:
            if ch[s1] == ch[s2]:
                continue
            used = True
            key = f"{s1}|{s2}"
            p1, p2 = pick_two(it, (s1, s2), (ch[s1], ch[s2]), i); sc_c = pick_two.last
            R[key]["detect"].append(int(p1 != p2))
            R[key]["attrib"].append(int(p1 == ch[s1] and p2 == ch[s2]))
            # v1.102 (author): the detector is reported per question type as well as pooled
            _sk = SK.setdefault(f'{key}|{it.get("skill", "?")}', {k: [] for k in ("detect", "attrib", "false", "v_conflict", "v_agree", "consistent")})
            _sk["detect"].append(int(p1 != p2)); _sk["attrib"].append(int(p1 == ch[s1] and p2 == ch[s2]))
            vc = verdict(it, (s1, s2), p1, p2, i); vs_c = verdict.last
            R[key]["v_conflict"].append(int(vc == "conflict"))
            R[key]["consistent"].append(int((vc == "conflict") == (p1 != p2)))
            _sk["v_conflict"].append(int(vc == "conflict")); _sk["consistent"].append(int((vc == "conflict") == (p1 != p2)))
            text_member = s2 if s2 in ("read", "ocr") else s1
            other = s1 if text_member == s2 else s2
            it_ag = _edited_item(it, text_member, ch[other])
            q1, q2 = pick_two(it_ag, (s1, s2), (ch[other], ch[text_member]), i); sc_a = pick_two.last
            R[key]["false"].append(int(q1 != q2)); _sk["false"].append(int(q1 != q2))
            va = verdict(it_ag, (s1, s2), q1, q2, i); vs_a = verdict.last
            R[key]["v_agree"].append(int(va == "agree")); _sk["v_agree"].append(int(va == "agree"))
            g_c = gen(it, (s1, s2), (ch[s1], ch[s2]), i); g_a = gen(it_ag, (s1, s2), (ch[other], ch[text_member]), i)
            if g_c:
                pc = g_c["parsed"]; pa = g_a["parsed"]
                R[key].setdefault("g_parsed", []).append(int(None not in pc))
                R[key].setdefault("g_detect", []).append(int(None not in pc and pc[0] != pc[1]))
                R[key].setdefault("g_attrib", []).append(int(pc == [ch[s1], ch[s2]]))
                R[key].setdefault("g_false", []).append(int(None not in pa and pa[0] != pa[1]))
                R[key].setdefault("g_vconf_strict", []).append(int(g_c["verdict_strict"] == "conflict"))
                R[key].setdefault("g_vagree_strict", []).append(int(g_a["verdict_strict"] == "agree"))
                R[key].setdefault("g_vconf_loose", []).append(int(g_c["verdict_loose"] == "conflict"))
                R[key].setdefault("g_vagree_loose", []).append(int(g_a["verdict_loose"] == "agree"))
            recs.append({"id": it.get("id"), "pair": key, "p1": p1, "p2": p2, "gold": [ch[s1], ch[s2]],
                         "agree_p": [q1, q2], "verdict_conflict": vc, "verdict_agree": va,
                         "gen_conflict_item": g_c, "gen_agree_twin": g_a,
                         "scores_conflict": sc_c, "scores_agree": sc_a,          # v1.131: log-lik of every answer pair
                         "verdict_scores_conflict": vs_c, "verdict_scores_agree": vs_a,
                         "values": {"conflict": {s1: ch[s1], s2: ch[s2]}, "agree": {s1: ch[other], s2: ch[other]}},
                         "prompt_conflict": (" ".join("[IMAGE]" if c.get("type") == "image" else c.get("text", "")
                                                      for c in um(it, (s1, s2), i)[0]["content"]) if len(recs) < 50 else None)})
        seen += used
    out = {"meta": {"run": a.run_dir, "step": a.step, "mode": cfgm, "method": meth, "n_items": seen,
                    "split": a.split, "template": a.template, "compose_trained": trained_compose,
                    "question_style": _QS, "caption_style": os.environ.get("SC_CAPTION_STYLE", "template"), "applicable": True},
           "pairs": {}}
    for key, d in R.items():
        if not d["detect"]:
            continue
        f = lambda k: sum(d[k]) / len(d[k])
        det, fa, vc, va = f("detect"), f("false"), f("v_conflict"), f("v_agree")
        out["pairs"][key] = {"detect": round(det, 4), "attrib": round(f("attrib"), 4), "false": round(fa, 4),
                             "balanced": round((det + 1 - fa) / 2, 4),
                             "verdict_conflict_recall": round(vc, 4), "verdict_agree_recall": round(va, 4),
                             "verdict_balanced": round((vc + va) / 2, 4),
                             "verdict_consistent": round(f("consistent"), 4), "n": len(d["detect"])}
        if d.get("g_detect"):
            g = lambda k: round(sum(d[k]) / len(d[k]), 4)
            out["pairs"][key].update({"gen_parse_rate": g("g_parsed"), "gen_detect": g("g_detect"), "gen_attrib": g("g_attrib"),
                                      "gen_false": g("g_false"), "gen_balanced": round((g("g_detect") + 1 - g("g_false")) / 2, 4),
                                      "gen_verdict_conflict_strict": g("g_vconf_strict"), "gen_verdict_agree_strict": g("g_vagree_strict"),
                                      "gen_verdict_balanced_strict": round((g("g_vconf_strict") + g("g_vagree_strict")) / 2, 4),
                                      "gen_verdict_balanced_loose": round((g("g_vconf_loose") + g("g_vagree_loose")) / 2, 4)})
        o = out["pairs"][key]
        print(f"  {key:10s} detect {det:.3f} attrib {o['attrib']:.3f} false {fa:.3f} bal {o['balanced']:.3f} | "
              f"verdict conflict {vc:.2f} agree {va:.2f} bal {o['verdict_balanced']:.2f} | n={o['n']}")
    out["records"] = recs
    od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, f"compose_detect_step_{a.step}{a.tag}.json")
    out["per_skill"] = {k: {kk: (round(sum(vv) / len(vv), 4) if vv else None) for kk, vv in v.items()}
                          for k, v in SK.items()}
    out["per_skill_n"] = {k: {kk: len(vv) for kk, vv in v.items()} for k, v in SK.items()}
    json.dump(out, open(fp, "w"), indent=1)
    print("[saved]", fp)


def run_explicit2(a, model, proc, n, meth, cfgm, forbid, cfg):
    """v1.141 (reviewer, author): zero-shot composition with an explicit instruction and the answer PAIR as the
    primary result. Prompt (blocks in the item's balanced order, held-out question):
        <q>
        Answer the question twice. First, answer according to <c1>. Second, answer according to <c2>. Return the two
        answers in that order, separated by a semicolon.
    c = the word (neo), the prefix's soft tokens placed inline (pt; a new insertion regime), "the image"/"the caption"
    (English). All 9 pairs "v1; v2" over {image value, caption value, a third value} are scored (chance 1/9) and a
    greedy reply is parsed as a pair. both_correct (primary), conflict = the two answers differ (derived, never
    asked); on the agree twin (caption set to the image value) a difference is a false alarm."""
    import itertools
    from PIL import Image
    from scsrc import clevr3 as C
    from scsrc.tokens import trigger
    from scripts.eval_clevr3_generate import parse_pair
    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    C.set_c3_vocab(rows)
    conf = [r for r in rows if str(r["channels"]["look"]) != str(r["channels"]["read"])][:a.max_n]

    def ctl(item, src):
        if a.mode == "instructed":
            return C.INSTR[src]
        if meth == "pt":
            return trigger(f"pt_{src}", n)
        return C.clevr3_token(item, src, n, cfgm)

    if a.pt_start and (meth != "pt" or a.mode or not a.tag):
        raise SystemExit("[STOP] --pt_start 1 needs a prefix run, no --mode, and its own --tag")

    def content(item, srcs, seed):
        s2, cap_first = C.eval_design(item, seed, "heldout")
        blocks = C._blocks(item); order = C._block_order(s2, cap_first)
        img = Image.open(item["image"]).convert("RGB")
        out, buf = [], []
        for b in order:
            if b not in blocks:
                continue
            if b == "look":
                if buf:
                    out.append({"type": "text", "text": "\n".join(buf)}); buf = []
                out.append({"type": "image", "image": img})
            else:
                buf.append(blocks[b])
        q = C.heldout_question(item, s2)
        if a.pt_start:   # the two prefixes lead the turn; the instruction ties them to the answers in order
            buf.append(f"{q}\nAnswer the question twice, once for each request given at the start, in the same order. "
                       f"Return the two answers in that order, separated by a semicolon.")
        else:
            buf.append(f"{q}\nAnswer the question twice. First, answer according to {ctl(item, srcs[0])}. Second, answer "
                       f"according to {ctl(item, srcs[1])}. Return the two answers in that order, separated by a semicolon.")
        out.append({"type": "text", "text": "\n".join(buf)})
        if a.pt_start:
            out.insert(0, {"type": "text", "text": trigger(f"pt_{srcs[0]}", n) + trigger(f"pt_{srcs[1]}", n)})
        return out, cap_first

    ptxt = lambda c: " ".join("[IMAGE]" if x.get("type") == "image" else x.get("text", "") for x in c)
    R = {}; recs = []
    for i, it in enumerate(tqdm(conf, desc="compose-x2")):
        ch = {k: str(v) for k, v in it["channels"].items()}
        A, B = ch["look"], ch["read"]
        w = C.cp_third_value(it, exclude={A, B}, salt="compose") or A
        cands = list(dict.fromkeys([A, B, w]))
        agree = C._edited_item(it, "read", A)
        for srcs in (("look", "read"), ("read", "look")):
            key = f"{srcs[0]}|{srcs[1]}"; d = R.setdefault(key, collections.defaultdict(list))
            rec = {"id": it["id"], "skill": it.get("skill"), "order": list(srcs), "candidates": cands}
            for ctxname, item_c, gold in (("conflict", it, [ch[srcs[0]], ch[srcs[1]]]), ("agree", agree, [A, A])):
                c, cap_first = content(item_c, srcs, i)
                um = [{"role": "user", "content": c}]
                from scripts.eval_clevr3_generate import loglik_both as _lb
                from scripts.tokscore import split_spans as _ss
                det = {f"{v1}; {v2}": _lb(model, proc, um, f"{v1}; {v2}", "cuda", tokens=True) for v1, v2 in itertools.product(cands, cands)}
                spd = _ss({k: (d[3], d[4]) for k, d in det.items()})         # v1.144: token-level detail
                sc = ({k: spd[k]["answer_sum"] for k in det} if os.environ.get("SC_SCORE", "turn") == "answer" else {k: float(d[0]) for k, d in det.items()})
                best = max(sc, key=sc.get); p1, p2 = best.split("; ")
                txt, nnew = generate_free(model, proc, c, "cuda", a.max_new, forbid, 8)
                glab, g1, g2 = parse_pair(txt, tuple(cands), tuple(cands))
                both = int(p1 == gold[0] and p2 == gold[1]); diff = int(p1 != p2)
                gboth = int(glab == "pair" and g1 == gold[0] and g2 == gold[1]); gdiff = int(glab == "pair" and g1 != g2)
                if ctxname == "conflict":
                    d["both_correct"].append(both); d["q1"].append(int(p1 == gold[0])); d["q2"].append(int(p2 == gold[1]))
                    d["detect"].append(diff); d["gen_both_correct"].append(gboth); d["gen_detect"].append(gdiff)
                    d["gen_parsed"].append(int(glab == "pair"))
                else:
                    d["false"].append(diff); d["agree_both_correct"].append(both); d["gen_false"].append(gdiff)
                rec[ctxname] = {"prompt": ptxt(c), "cap_first": cap_first, "gold": gold,
                                "values": {"image": str(item_c["channels"]["look"]), "caption": str(item_c["channels"]["read"])},
                                "candidate_scores": {k: round(v, 4) for k, v in sc.items()}, "prediction": [p1, p2],
                                "candidate_detail": {k: {"sum": round(det[k][0], 4), **spd[k]} for k in det},
                                "raw_generation": txt, "gen_ntok": nnew, "parse_label": glab, "parsed": [g1, g2]}
            recs.append(rec)
    m = lambda v: round(sum(v) / len(v), 4) if v else None
    pairs = {}
    for key, d in R.items():
        o = {k: m(v) for k, v in d.items()}
        o["balanced"] = round((o["detect"] + 1 - o["false"]) / 2, 4) if o.get("detect") is not None and o.get("false") is not None else None
        o["gen_balanced"] = (round((o["gen_detect"] + 1 - o["gen_false"]) / 2, 4)
                             if o.get("gen_detect") is not None and o.get("gen_false") is not None else None)
        o["n"] = len(d["both_correct"]); pairs[key] = o
        print(f"  {key}: both_correct {o['both_correct']} detect {o['detect']} false {o['false']} balanced {o['balanced']} | gen both {o['gen_both_correct']}")
    od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}"); os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, f"compose_detect_step_{a.step}{a.tag}.json")
    json.dump({"meta": {"run": a.run_dir, "step": a.step, "mode": cfgm, "method": meth, "template": "explicit2",
                        "split": a.split, "applicable": True, "n_items": len(conf), "chance_both": round(1 / 9, 4),
                        "primary": "both_correct (conflict derived from the two answers; no verdict asked)",
                        "caption_style": os.environ.get("SC_CAPTION_STYLE", "template")},
               "pairs": pairs, "records": recs}, open(fp, "w"), indent=1)
    print("[saved]", fp)


if __name__ == "__main__":
    main()
