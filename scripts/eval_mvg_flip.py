#!/usr/bin/env python
"""Real-video flip/stay on the mined within-video counterfactual pairs.

For each pair (prepare_mvg_ctm.py) the channel's token is queried under
context A and context B (only that channel's evidence differs):
  vis_value_swap / desc_swap : forced-choice {gold_a, gold_b} in both contexts;
        acc_a, acc_b, and FLIP = correct in both (the answer tracked the
        evidence change). Chance flip = .25 for an evidence-blind scorer.
  ocr_ablate / vis_presence  : gold has no alternative value -> report the
        LL margin DROP of gold_a from ctx_a to ctx_b (mean drop + frac>0);
        an evidence-reading token loses confidence when evidence is removed.
STAY control: the OTHER channels' tokens on the same pair should NOT flip
(their evidence is unchanged) -- reported as stay_acc for one other channel.
Prompt structure identical to training (mvg_user_content + MVG_CONT).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from tqdm import tqdm
from scsrc.mvg import mvg_user_content, MVG_CONT
INSTR = {"VIS":  ("Answer using ONLY the video imagery. Ignore the description "
                  "and the on-screen text."),
         "OCR":  ("Answer using ONLY the on-screen text. Ignore the video "
                  "imagery and the description."),
         "DESC": ("Answer using ONLY the description. Ignore the video imagery "
                  "and the on-screen text.")}
NOOP = "Answer the question."
from scsrc.tokens import trigger
from scsrc.mvg import mvg_word, assert_words_registered
from evaluate import apply_checkpoint
from scripts.eval_mvg import loglik_mvg

CH_TOK = {"VIS": "vis_ctrl", "DESC": "desc_ctrl", "OCR": "ocr_ctrl"}
OTHER = {"VIS": "DESC", "DESC": "VIS", "OCR": "DESC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--pairs", default="data/mvg_ctm/pairs.jsonl")
    ap.add_argument("--mvg", default="data/multiventG/multivent_g.json")
    ap.add_argument("--claims", default="data/multiventG/claims.jsonl")
    ap.add_argument("--frames", default="data/multiventG/frames_manifest.jsonl")
    ap.add_argument("--max_per_type", type=int, default=100)
    ap.add_argument("--mvg_sources", default=None,
                    help="override the evidence blocks (desc,vis = two-modality) -- for frozen baselines, which have no "
                         "training config; trained runs restore their own setting from config.json")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filename (e.g. _clean) so a run "
                         "on a filtered pair set cannot overwrite the original")
    ap.add_argument("--style", default="neo", choices=["neo", "instr", "noop"],
                    help="THE MISSING CAUSAL BASELINE on real video: instr = an "
                         "English source-naming instruction in place of the word; "
                         "noop = no source named. Without these, a vis-flip near "
                         "chance cannot be told from 'the task is impossible'.")
    ap.add_argument("--generate", type=int, default=0,
                    help="v1.141: also greedy-generate in all four prompts of the first --gen_max pairs per type and read the "
                         "answer by exact phrase matching after 'final answer:'; per-token mean scores are always recorded")
    ap.add_argument("--gen_max", type=int, default=150)
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--frozen_model_id", default=None,
                    help="skip checkpoints: frozen base model (requires --style "
                         "instr|noop; a frozen model has no word rows to trigger)")
    ap.add_argument("--crop_bbox", type=float, default=0.0,
                    help="ATTRIBUTION TEST for the weak visual channel. The mined "
                         "evidence occupies a MEDIAN 6% of the frame (p10 1%), so a "
                         "whole-frame forced choice between two annotator phrases "
                         "conflates CONTROL with LOCALISATION. >0 crops each context "
                         "to its own evidence bbox, expanded by this margin (e.g. "
                         "0.25 = 25% padding). If flips jump, the .13-.19 ceiling is "
                         "perception at 6% scale, not a failure of the token.")
    a = ap.parse_args()
    if a.mvg_sources:
        os.environ["SC_MVG_SOURCES"] = a.mvg_sources; os.environ["SC_MVG_SOURCES_OVERRIDE"] = "1"

    if a.frozen_model_id is not None:
        assert a.style != "neo", "--frozen_model_id requires --style instr|noop"
        from scsrc import model as M
        model, proc = M.load(model_id=(a.frozen_model_id or None)
                             if a.frozen_model_id != "base" else None)
        model.eval(); meta = {"n_tokens": 0}; forbid = None
    else:
        model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    if a.style == "neo":
        assert_words_registered(proc.tokenizer, n)
    mvg = json.load(open(a.mvg))
    ocr_by_vid = defaultdict(set)
    for l in open(a.claims):
        c = json.loads(l)
        if c["channel"] == "OCR":
            ocr_by_vid[c["video_id"]].add(c.get("source_text", ""))
    frames = defaultdict(list)
    for l in open(a.frames):
        r = json.loads(l); frames[r["video_id"]].append((r["t"], r["path"]))

    pairs = [json.loads(l) for l in open(a.pairs)]
    from scsrc.mvg import mvg_sources as _msrc
    if "ocr" not in _msrc():      # 2026-10-02 two-modality MVG: no on-screen-text block -> the OCR ablation is undefined
        _n0 = len(pairs); pairs = [p for p in pairs if p["pair_type"] != "ocr_ablate"]
        print(f"[mvg] sources={','.join(_msrc())}: ocr_ablate pairs skipped ({_n0 - len(pairs)})")
    by_type = defaultdict(list)
    for p in pairs:
        by_type[p["pair_type"]].append(p)

    def base_item(p):
        v = p["video_id"]
        # 2026-10-05 (advisor audit): deterministic block order -- Python's hash() is salted per process, so different
        # evaluation jobs (methods) saw different source orders; crc32 gives every method the identical prompt per pair
        return {"video_id": v, "role": p["role"], "perm6": __import__("zlib").crc32((v + p["role"]).encode()) % 6,
                "desc_text": mvg[v]["description"][:1200],
                "ocr_strings": sorted(ocr_by_vid[v])[:8],
                "vis_frame": frames[v][0][1] if frames[v] else None,
                "role_question": mvg[v]["template"].get(p["role"], "")}

    def _crop(path, bbox, pad):
        """bbox is [x0,y0,x1,y1] in PERCENT of the frame."""
        from PIL import Image
        im = Image.open(path).convert("RGB")
        W, H = im.size
        x0, y0, x1, y1 = [max(0.0, min(100.0, float(c))) for c in bbox]
        x0, x1 = sorted((x0 * W / 100.0, x1 * W / 100.0))
        y0, y1 = sorted((y0 * H / 100.0, y1 * H / 100.0))
        dw, dh = (x1 - x0) * pad, (y1 - y0) * pad
        box = (max(0, int(x0 - dw)), max(0, int(y0 - dh)),
               min(W, int(x1 + dw)), min(H, int(y1 + dh)))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return im
        return im.crop(box)

    def ov(p, side):
        pt = p["pair_type"]; ctx = p[f"ctx_{side}"]
        # vis_easy_control is a FRAME pair too (cross-video), so it must take the
        # image branch. Without it the frames are never loaded and the "control"
        # silently becomes a text-only comparison -- which would look like a pass.
        if pt in ("vis_value_swap", "vis_presence", "vis_easy_control"):
            bb = p.get(f"bbox_{side}")
            if a.crop_bbox > 0 and bb:
                return {"vis_frame": _crop(ctx["frame"], bb, a.crop_bbox)}
            return {"vis_frame": ctx["frame"]}
        if pt == "desc_swap":
            return {"desc_text": ctx["desc"]}
        return {"ocr_strings": ctx["ocr_strings"]}

    res = defaultdict(lambda: defaultdict(list))
    # v1.141: item-level log (inputs, every candidate's sum / mean / token count, raw generations, derived labels)
    import gzip as _gz, re as _re
    from scripts.eval_clevr3_generate import generate_free
    _side = []
    _norm = lambda t: _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9 ]+", " ", t.lower())).strip()
    def _ptxt(u):
        return " ".join("[IMAGE]" if c.get("type") == "image" else c.get("text", "") for c in u[0]["content"])
    from scripts.tokscore import split_spans as _split_spans
    def _sc2(u, ca_, cb_):
        """both candidates of one prompt: whole-turn sum/ntok + the answer-span split (v1.142)"""
        r = {g: loglik_mvg(model, proc, u, c, "cuda", tokens=True) for g, c in (("a", ca_), ("b", cb_))}
        sp = _split_spans({g: (v[2], v[3]) for g, v in r.items()})
        return {g: {"sum": round(v[0], 4), "ntok": v[1], **sp[g]} for g, v in r.items()}
    def _parse(txt, ga, gb):
        """exact phrase match after the LAST 'final answer:' (else the whole reply): 'exact' = the answer equals one
        phrase, 'contains' = exactly one phrase occurs, 'ambiguous' = both occur, 'other' = neither, 'empty'."""
        if not txt or not txt.strip():
            return "empty", None
        low = txt.lower(); seg = txt[low.rfind("final answer:") + len("final answer:"):] if "final answer:" in low else txt
        seg, A, B = _norm(seg), _norm(ga), _norm(gb)
        if seg in (A, B):
            return "exact", ("a" if seg == A else "b")
        ha, hb = (A in seg), (B in seg)
        if ha and hb and A != B:          # one phrase inside the other ("people" / "group of people"): the longer wins
            if A in B: return "contains", "b"
            if B in A: return "contains", "a"
            return "ambiguous", None
        if ha or hb:
            return "contains", ("a" if ha else "b")
        return "other", None
    for pt, ps in by_type.items():
        for _pi, p in enumerate(tqdm(ps[:a.max_per_type], desc=pt)):
            it = base_item(p)
            if it["vis_frame"] is None:
                continue
            if a.style == "neo":
                tok = mvg_word(CH_TOK[p["channel"]], n)
            else:
                # the word slot carries English (instr) or nothing (noop);
                # everything else about the prompt is identical
                tok = (INSTR[p["channel"]] if a.style == "instr" else NOOP)
            ua = [{"role": "user", "content": mvg_user_content(it, tok, ov(p, "a"))}]
            ub = [{"role": "user", "content": mvg_user_content(it, tok, ov(p, "b"))}]
            ca, cb = MVG_CONT.format(v=p["gold_a"]), MVG_CONT.format(v=p["gold_b"])
            if pt in ("vis_value_swap", "desc_swap", "vis_easy_control"):
                sa = {g: loglik_mvg(model, proc, ua, c, "cuda")
                      for g, c in (("a", ca), ("b", cb))}
                sb = {g: loglik_mvg(model, proc, ub, c, "cuda")
                      for g, c in (("a", ca), ("b", cb))}
                hit_a = int(sa["a"] > sa["b"]); hit_b = int(sb["b"] > sb["a"])
                res[pt]["acc_a"].append(hit_a); res[pt]["acc_b"].append(hit_b)
                res[pt]["flip"].append(int(hit_a and hit_b))
                # STAY: another channel's token, same contexts, should keep
                # preferring whatever it preferred (its evidence is unchanged)
                # 2026-10-05 (advisor audit): the frozen baselines get the OTHER source's request in their own form (the
                # English instruction / no request), not the learned-word string they never saw
                otok = (INSTR[OTHER[p["channel"]]] if a.style == "instr" else NOOP if a.style == "noop"
                        else mvg_word(CH_TOK[OTHER[p["channel"]]], n))
                oa = [{"role": "user", "content": mvg_user_content(it, otok, ov(p, "a"))}]
                ob = [{"role": "user", "content": mvg_user_content(it, otok, ov(p, "b"))}]
                pa = loglik_mvg(model, proc, oa, ca, "cuda") > loglik_mvg(model, proc, oa, cb, "cuda")
                pb = loglik_mvg(model, proc, ob, ca, "cuda") > loglik_mvg(model, proc, ob, cb, "cuda")
                res[pt]["stay_other"].append(int(pa == pb))
                # v1.141: per-token mean scoring (the phrases differ in length) + exact-match generation
                S = {k: _sc2(u, ca, cb) for k, u in (("own_a", ua), ("own_b", ub), ("other_a", oa), ("other_b", ob))}
                # length normalisation over the ANSWER tokens only (the shared lead-in is excluded, v1.142)
                mpick = {k: ("a" if v["a"]["answer_mean"] > v["b"]["answer_mean"] else "b") for k, v in S.items()}
                res[pt]["acc_a_mean"].append(int(mpick["own_a"] == "a")); res[pt]["acc_b_mean"].append(int(mpick["own_b"] == "b"))
                res[pt]["flip_mean"].append(int(mpick["own_a"] == "a" and mpick["own_b"] == "b"))
                res[pt]["stay_other_mean"].append(int(mpick["other_a"] == mpick["other_b"]))
                apick = {k: ("a" if v["a"]["answer_sum"] > v["b"]["answer_sum"] else "b") for k, v in S.items()}   # v1.143
                res[pt]["acc_a_ans"].append(int(apick["own_a"] == "a")); res[pt]["acc_b_ans"].append(int(apick["own_b"] == "b"))
                res[pt]["flip_ans"].append(int(apick["own_a"] == "a" and apick["own_b"] == "b"))
                res[pt]["stay_other_ans"].append(int(apick["other_a"] == apick["other_b"]))
                rec = {"pair_type": pt, "video_id": p["video_id"], "role": p["role"], "channel": p["channel"],
                       "gold_a": p["gold_a"], "gold_b": p["gold_b"], "crop_bbox": a.crop_bbox, "style": a.style,
                       "prompts": {k: _ptxt(u) for k, u in (("own_a", ua), ("own_b", ub), ("other_a", oa), ("other_b", ob))},
                       "scores": S, "pick_sum": {"own_a": "a" if hit_a else "b", "own_b": "b" if hit_b else "a",
                                                 "other_a": "a" if pa else "b", "other_b": "a" if pb else "b"},
                       "pick_mean": mpick, "pick_answer": apick}
                if a.generate and _pi < a.gen_max:
                    G = {}
                    for k, u in (("own_a", ua), ("own_b", ub), ("other_a", oa), ("other_b", ob)):
                        txt, nn = generate_free(model, proc, u[0]["content"], "cuda", a.max_new, forbid, 8)
                        lab, val = _parse(txt, p["gold_a"], p["gold_b"])
                        G[k] = {"raw_generation": txt, "gen_ntok": nn, "parse_label": lab, "parsed": val}
                    ok = lambda k, want: int(G[k]["parsed"] == want)
                    res[pt]["gen_flip"].append(ok("own_a", "a") * ok("own_b", "b"))
                    res[pt]["gen_acc_a"].append(ok("own_a", "a")); res[pt]["gen_acc_b"].append(ok("own_b", "b"))
                    res[pt]["gen_stay_other"].append(int(G["other_a"]["parsed"] is not None and G["other_a"]["parsed"] == G["other_b"]["parsed"]))
                    res[pt]["gen_parsed"].append(int(all(G[k]["parsed"] is not None for k in G)))
                    rec["generation"] = G
                _side.append(rec)
            else:   # evidence-removal: margin drop of gold_a
                la = loglik_mvg(model, proc, ua, ca, "cuda")
                lb = loglik_mvg(model, proc, ub, ca, "cuda")
                res[pt]["drop"].append(la - lb)
                res[pt]["drop_pos"].append(int(la > lb))

    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    out = {"run": os.path.basename(a.run_dir), "step": a.step,
           "crop_bbox": a.crop_bbox, "block_order": "crc32(video_id+role) % 6 (deterministic since 2026-10-05)", "mvg_sources": ",".join(__import__("scsrc.mvg", fromlist=["mvg_sources"]).mvg_sources()),
           **{pt: {k: m(v) for k, v in d.items()} | {"n": len(next(iter(d.values())))}
              for pt, d in res.items()}}
    print(json.dumps(out, indent=1))
    outd = os.path.join(a.run_dir, "eval", "mvg_flip"); os.makedirs(outd, exist_ok=True)
    _tag = (a.tag + ("" if a.crop_bbox <= 0 else f"_crop{a.crop_bbox}")
            + ("" if a.style == "neo" else f"_{a.style}"))
    out["scoring"] = {"primary": "sum log-likelihood of '... final answer: <phrase>' (as before)",
                      "also": "answer-token mean (acc/flip/stay_other *_mean; lead-in excluded) and, with --generate, exact phrase match of a "
                              "greedy reply after 'final answer:' (gen_*) on the first --gen_max pairs per type"}
    json.dump(out, open(os.path.join(outd, f"flip_step_{a.step}{_tag}.json"), "w"), indent=1)
    with _gz.open(os.path.join(outd, f"flip_step_{a.step}{_tag}_items.jsonl.gz"), "wt") as _f:
        for _r in _side:
            _f.write(json.dumps(_r) + "\n")
    print(f"[saved] {outd}/flip_step_{a.step}{_tag}.json")


if __name__ == "__main__":
    main()
