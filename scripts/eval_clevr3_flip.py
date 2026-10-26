#!/usr/bin/env python
"""Ground-CF (clevr3) flip/stay -- the channel-edit interventional eval.

The text counterfactual is a CHANNEL EDIT: rewrite one text channel to assert a
different value, all else bit-identical, which realises do(channel = v') exactly.

For a long time this file also said Ground-CF had no image twin, so <look> got
only a STAY side plus a gray-image presence test. That was a build gap, not a
property of the dataset: prepare_clevr3.py renders every scene itself, so an
exact one-attribute twin is constructible, and clevr3_image_twins.py now emits
774 of them (colour recolour, shape redraw). <look> therefore has a real FOLLOWS
here, scored from data/clevr3/pairs_look_<split>.jsonl.

Per item (only items where the probed channel disagrees with some other
channel, so candidates are well-defined):
  read_flip : <read> answer = caption value AND, after the caption is edited to
              v', = v'  (both directions; forced choice over {v, v'})
  read_stay : <read> unchanged when the OCR channel is edited instead
  ocr_flip / ocr_stay : symmetric
  look_stay : <look> unchanged under a caption edit AND under an OCR edit
  look_presence : LL(look-gold) drop when the image is replaced by gray
                  (drop_pos = fraction of items where it drops)

Output: runs/<run>/eval/clevr3_<split>/flip_step_<STEP>.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from tqdm import tqdm
from scsrc.clevr3 import clevr3_user_content, _edited_item, set_c3_vocab, cp_text_edit_value, cp_fix_look_collision
# 2026-09-23 (author, STATUS v1.119): QUESTION-PHRASING SHIFT. Ground-CF has no request-template pool; the
# only eval-time variation is block order. SC_QSTYLE=generic scores an explicit-trained run under the
# skill-free GENERIC_QS questions (the word must carry the skill) -- the Ground-CF analogue of the ladder's
# held-out template. Requires --tag so the in-phrasing file is never overwritten.
_QS = os.environ.get("SC_QSTYLE", "explicit")
from scsrc.tokens import c3_sources
from scsrc.collate import _encode, _prompt_len
from evaluate import apply_checkpoint
from scripts.eval_clevr3 import cand, loglik, loglik_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--dump_records", type=int, default=1,
                    help="1: per-item 0/1 vectors in the json (deterministic "
                         "order -> paired McNemar)")
    ap.add_argument("--dump_picks", type=int, default=0,
                    help="1: also store the per-item chosen candidates (picks) so a "
                         "non-flip can be split into stale old value vs a switch "
                         "to another modality's value")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=150)
    ap.add_argument("--stride", type=int, default=1,
                    help="v1.211: keep every k-th item (the split files hold equal per-skill blocks, so k | 200 keeps the "
                         "skills balanced); recorded in the output. Default 1 = every item.")
    ap.add_argument("--mode", default=None,
                    help="factored/srconly/bundled/instructed/noop; default = the "
                         "run's own mode. `instructed` swaps the learned trigger "
                         "for the English parity prompt and `noop` for nothing, "
                         "which is how the untrained baselines get a flip/stay "
                         "number -- clevr3_token() already implements both.")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filename. REQUIRED when --mode "
                         "overrides the run's own mode: the filename carries only "
                         "the step, so `--mode instructed` without a tag would "
                         "OVERWRITE the trained arm's flip_step_<step>.json.")
    ap.add_argument("--pairs_file", default=None,
                    help="image-twin pairs file overriding data_dir/pairs_look_<split>.jsonl (e.g. the "
                         "four-skill pairs_look_all_<split>.jsonl from clevr3_twins_count_spatial.py); "
                         "use with --tag so the banked two-skill file is not overwritten (2026-09-21)")
    ap.add_argument("--presence", type=int, default=0, help="gray-image presence diagnostic under --cp (2 extra passes per item; not printed)")
    ap.add_argument("--batched", type=int, default=0,
                    help="score all candidates of a pick in one forward pass. OFF since 2026-09-23 (STATUS v1.104): "
                         "the batched pass is NOT numerically equivalent -- right padding changed summed "
                         "log-likelihoods by up to 8.7 nats and flipped a pick in one cell. Only the sequential "
                         "path produces printed numbers.")
    ap.add_argument("--batched_check", type=int, default=12, help="picks compared against single-sequence scoring; recorded")
    ap.add_argument("--cands", default="twin", choices=["twin", "full"],
                    help="conflict-preserving candidate set: twin = the item's image/caption/edit values (banked); full = the "
                         "whole skill inventory (2026-10-02, separate evaluation; chance 1/|V|). Needs --cp 1 and --tag.")
    ap.add_argument("--cp", type=int, default=0,
                    help="conflict-preserving edits (STATUS v1.86): text channels move to a third value no channel "
                         "asserts (fallback: the other text channel's value, counted); read Stay = image twin; look "
                         "twins whose value a text channel asserts get that channel moved in both contexts. Needs --tag.")
    a = ap.parse_args()

    if a.cp and not a.tag:
        raise SystemExit("[STOP] --cp 1 needs --tag (e.g. _cp4) so the pre-fix file is not overwritten")
    if _QS != "explicit" and not a.tag:
        raise SystemExit("[STOP] SC_QSTYLE needs --tag so the in-phrasing file is not overwritten")
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit(f"[STOP] --mode {a.mode} with no --tag would overwrite "
                         f"{a.run_dir}/eval/clevr3_{a.split}/flip_step_{a.step}.json "
                         f"(the trained arm's result). Pass e.g. --tag _{a.mode}.")
    if a.cands == "full":
        if not a.cp or not a.tag:
            raise SystemExit("[STOP] --cands full needs --cp 1 and a --tag (separate evaluation, not the banked candidate set)")
        os.environ["SC_CP_CANDS"] = "full"; print("[cp] candidate set = FULL skill inventory (chance 1/|V|)")
    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
    if a.mode in ("instructed", "noop"):
        # The base weights are untouched by neologism training -- only the added
        # embedding rows are learned -- and an English/empty prompt references
        # none of them, so this measures the FROZEN backbone. Same convention the
        # *_explicit baselines already use (they live in the trained run's dir).
        print(f"[baseline] mode={a.mode}: trigger replaced, trained rows unused "
              f"-> this is the frozen backbone with an English/empty prompt")
    n = meta["n_tokens"]
    # PT runs carry the request as a learned prefix, not as an inline src token
    # (clevr3_user_content(method=..., question_style=_QS)). Reading it from the run config keeps
    # train and eval on the same prompt; a baseline mode overrides it because
    # instructed/noop reference no learned rows at all.
    try:
        _cfg_method = json.load(open(os.path.join(a.run_dir, "config.json")))["args"].get("method", "neo")
    except Exception:
        _cfg_method = "neo"
    meth = "neo" if a.mode in ("instructed", "noop") else _cfg_method
    print(f"[prompt] method={meth} (run method={_cfg_method}, mode={a.mode})")
    cfgm = a.mode
    if cfgm is None:
        try:
            cfgm = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]["factor_mode"]
        except Exception:
            cfgm = "srconly"

    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    if a.stride > 1:
        rows = rows[::a.stride]
        print(f"[stride] every {a.stride}th item: {len(rows)} items", flush=True)
    if a.cp:
        print(f"[cp] vocabulary {set_c3_vocab(rows)}")
    gray = Image.new("RGB", (336, 336), (128, 128, 128))
    R = {k: [] for k in ("read_stay_c", "ocr_stay_c", "look_stay_c",
                         "read_flip", "read_stay", "ocr_flip", "ocr_stay",
                         "look_stay", "look_presence_drop", "look_acc",
                         "read_acc", "ocr_acc",
                         "look_flip", "look_twin_acc", "read_stay_img",
                         "look_leak_text")}
    # per-skill, because Ground-CF twins are half colour and half SHAPE and the
    # two are not equally hard: a recolour keeps the silhouette, a shape redraw
    # changes it. Pooling them hides which one carries the result.
    import collections as _c
    SK = _c.defaultdict(lambda: _c.defaultdict(list))
    PK = {"look": [], "read": [], "ocr": []}   # per-item picks (only stored with --dump_picks 1)

    _bchk = {"n": 0, "agree": 0}
    def pick(item, source, cands, seed):
        um = [{"role": "user",
               "content": clevr3_user_content(item, source, n, cfgm, seed,
                                              method=meth, question_style=_QS)}]
        if a.dump_picks and len(pick.last_prompts) < 400:
            # v1.131: the first rendered prompt per (request, item) -- the base context -- kept for the audit
            pick.last_prompts.setdefault((source, item["id"]), " ".join(
                "[IMAGE]" if c.get("type") == "image" else c.get("text", "") for c in um[0]["content"]))
        if a.batched:
            bs = loglik_batch(model, proc, um, [cand(v) for v in cands], "cuda")
            sc = {v: bs[cand(v)][0] for v in cands}
            if _bchk["n"] < a.batched_check:
                # STATUS v1.95: the batched pass must reproduce the single-sequence pick; recorded, never silent
                sc1 = {v: loglik(model, proc, um, cand(v), "cuda") for v in cands}
                _bchk["n"] += 1; _bchk["agree"] += int(max(sc, key=sc.get) == max(sc1, key=sc1.get))
                _bchk.setdefault("max_abs_diff", 0.0)
                _bchk["max_abs_diff"] = max(_bchk["max_abs_diff"], max(abs(sc[v] - sc1[v]) for v in cands))
            return max(sc, key=sc.get), sc
        if a.dump_picks:
            # v1.141 (author: save inputs, raw evidence and labels): the same forward pass also gives the token count
            det = {v: _loglik_both(model, proc, um, cand(v), "cuda", tokens=True) for v in cands}
            _sp = _split_spans({v: (d[3], d[4]) for v, d in det.items()})   # v1.142: lead-in / answer / end of turn
            # v1.143: SC_SCORE picks the DECISION score -- "turn" (sum over the whole assistant turn, the banked rule) or
            # "answer" (answer tokens only: no lead-in, no end-of-turn). Both are always recorded in the item log.
            sc = ({v: _sp[v]["answer_sum"] for v in det} if _SCORE == "answer" else {v: d[0] for v, d in det.items()})
            best = max(sc, key=sc.get)
            best_turn = max(det, key=lambda v: det[v][0]); best_ans = max(det, key=lambda v: _sp[v]["answer_sum"])
            from scsrc.clevr3 import eval_design as _ed
            _cf = _ed(item, seed, _QS)[1]
            _SIDE.append({"request": source, "ctx": getattr(pick, "ctx", None), "id": item.get("id"), "skill": item.get("skill"), "image": item.get("image"),
                          "caption": item.get("caption"), "values": {k: str(v) for k, v in item.get("channels", {}).items()},
                          "cap_first": _cf, "question_style": _QS,
                          "prompt": " ".join("[IMAGE]" if c.get("type") == "image" else c.get("text", "") for c in um[0]["content"]),
                          "candidates": {v: {"sum": round(d[0], 4), "mean": round(d[1], 4), "ntok": d[2], **_sp[v]} for v, d in det.items()},
                          "pick": best, "pick_turn": best_turn, "pick_answer": best_ans, "decision_score": _SCORE})
            return best, sc
        sc = {v: loglik(model, proc, um, cand(v), "cuda") for v in cands}
        return max(sc, key=sc.get), sc

    pick.last_prompts = {}
    _SIDE = []
    _SCORE = os.environ.get("SC_SCORE", "turn")   # v1.143: "turn" (banked) | "answer" (answer tokens only)
    if _SCORE == "answer" and not a.dump_picks:
        raise SystemExit("[STOP] SC_SCORE=answer needs --dump_picks 1 (the token-level path)")
    from scripts.eval_clevr3_generate import loglik_both as _loglik_both
    from scripts.tokscore import split_spans as _split_spans
    seen = 0
    if a.cp:
        # STATUS v1.94: one candidate set per item across every compared context (scripts/clevr3_cp_eval.py)
        from scripts.clevr3_cp_eval import cp_protocol
        _twin_path = a.pairs_file or os.path.join(a.data_dir, f"pairs_look_all_{a.split.replace('test_', '')}.jsonl")
        _pairs = [json.loads(l) for l in open(_twin_path)] if os.path.exists(_twin_path) else []
        print(f"[cp] protocol: sources={c3_sources()} twins={len(_pairs)} from {_twin_path}")
        def _presence(item):
            um_ = [{"role": "user", "content": clevr3_user_content(item, "look", n, cfgm, 0, method=meth, question_style=_QS)}]
            it_g = dict(item); it_g["image"] = gray
            umg_ = [{"role": "user", "content": clevr3_user_content(it_g, "look", n, cfgm, 0, method=meth, question_style=_QS)}]
            g_ = str(item["channels"]["look"])
            return loglik(model, proc, um_, cand(g_), "cuda") - loglik(model, proc, umg_, cand(g_), "cuda")
        # v1.95: the gray-image presence diagnostic is not printed anywhere in the paper; off by default (--presence 1)
        seen = cp_protocol(rows, _pairs, pick, R, SK, PK, max_n=a.max_n, presence=(_presence if a.presence else None))
    for i, it in enumerate(tqdm([] if a.cp else rows, desc="c3flip")):
        if seen >= a.max_n:
            break
        ch = {k: str(v) for k, v in it["channels"].items()}
        # need a disagreeing alternative per text channel
        alt = {}
        for c in ("read", "ocr"):
            if a.cp:
                alt[c], _fb = cp_text_edit_value(it, c)
                R.setdefault(f"cp_fallback_{c}", []).append(int(_fb))
            else:
                alt[c] = next((v for k, v in ch.items() if k != c and v != ch[c]), None)
        if alt["read"] is None and alt["ocr"] is None:
            continue
        seen += 1
        # ---------- text-channel flips (exact do() via channel edit)
        # v1.93: under --cp the on-screen text is edited only as a competitor, never requested (the paper has two
        # modalities), so its own Flip/Stay passes are skipped: ~40% fewer forward passes per item.
        for c in (("read",) if a.cp else ("read", "ocr")):
            if alt[c] is None:
                continue
            cands = (ch[c], alt[c])
            p0, s0 = pick(it, c, cands, i)
            it2 = _edited_item(it, c, alt[c])
            p1, s1 = pick(it2, c, cands, i)
            # 2026-09-14: margins (log-lik of the original gold minus the alternative) before and after
            # the edit, so detectors can be scored continuously
            R.setdefault(f"{c}_margin", []).append(float(s0[ch[c]] - s0[alt[c]]))
            R.setdefault(f"{c}_margin_edit", []).append(float(s1[ch[c]] - s1[alt[c]]))
            R[f"{c}_acc"].append(int(p0 == ch[c]))
            R[f"{c}_flip"].append(int(p0 == ch[c] and p1 == alt[c]))
            # 2026-09-15: per-skill for the text channels too (count/color/object/spatial), so the
            # appendix can report Ground-CF per question type instead of pooled
            SK[it.get("skill", "?")][f"{c}_acc"].append(int(p0 == ch[c]))
            SK[it.get("skill", "?")][f"{c}_flip"].append(int(p0 == ch[c] and p1 == alt[c]))
            PK[c].append({"id": it.get("id"), "p0": p0, "p1": p1, "old": ch[c], "new": alt[c]})
            # stay: edit the OTHER text channel; this channel's pick must hold
            other = "ocr" if c == "read" else "read"
            if alt[other] is not None and not a.cp:
                it3 = _edited_item(it, other, alt[other])
                p2, _ = pick(it3, c, cands, i)
                R[f"{c}_stay"].append(int(p2 == p0))
                # conjunctive stay: correct in the original context AND unchanged (mirror of flip)
                R[f"{c}_stay_c"].append(int(p0 == ch[c] and p2 == p0))
                SK[it.get("skill", "?")][f"{c}_stay"].append(int(p2 == p0))
                SK[it.get("skill", "?")][f"{c}_stay_c"].append(int(p0 == ch[c] and p2 == p0))
        # ---------- look: stay under either text edit + presence
        lk_alt = next((v for k, v in ch.items() if k != "look" and v != ch["look"]),
                      None)
        if lk_alt is not None:
            cands = (ch["look"], lk_alt)
            p0, s0 = pick(it, "look", cands, i)
            R.setdefault("look_margin", []).append(float(s0[ch["look"]] - s0[lk_alt]))
            R["look_acc"].append(int(p0 == ch["look"]))
            SK[it.get("skill", "?")]["look_acc"].append(int(p0 == ch["look"]))
            stays = []
            for c in (("read",) if a.cp else ("read", "ocr")):
                if alt[c] is not None:
                    p_, _ = pick(_edited_item(it, c, alt[c]), "look", cands, i)
                    stays.append(int(p_ == p0))
            if stays:
                R["look_stay"].append(min(stays))
                R["look_stay_c"].append(int(p0 == ch["look"] and min(stays) == 1))
                SK[it.get("skill", "?")]["look_stay"].append(min(stays))
                SK[it.get("skill", "?")]["look_stay_c"].append(int(p0 == ch["look"] and min(stays) == 1))
            # presence: gold LL with the real image vs gray
            um = [{"role": "user",
                   "content": clevr3_user_content(it, "look", n, cfgm, i,
                                                  method=meth, question_style=_QS)}]
            l_real = loglik(model, proc, um, cand(ch["look"]), "cuda")
            it_g = dict(it); it_g["image"] = gray
            umg = [{"role": "user",
                    "content": clevr3_user_content(it_g, "look", n, cfgm, i,
                                                    method=meth, question_style=_QS)}]
            l_gray = loglik(model, proc, umg, cand(ch["look"]), "cuda")
            R["look_presence_drop"].append(l_real - l_gray)

    # ---------- look FLIP on the exact image twins
    #
    # This script shipped asserting "look has NO flip: Ground-CF has no image
    # twin". That was true only until clevr3_image_twins.py rendered them: a
    # colour recolour and a shape redraw are in-place edits of our OWN render, so
    # image B differs from image A in the probed value and in nothing else --
    # positions, other objects, palette and resolution stay byte-identical. 774
    # twins (374 id / 400 ood) were sitting on disk unread because no code path
    # here opened the file, so <look> FOLLOWS was not merely unmeasured on the
    # anchor dataset, it was unmeasurable.
    twin = a.pairs_file or os.path.join(a.data_dir,
                                        f"pairs_look_{a.split.replace('test_', '')}.jsonl")
    if (not a.cp) and os.path.exists(twin):
        by_id = {r["id"]: r for r in rows}
        prs = [json.loads(l) for l in open(twin)][:a.max_n]
        for j, pr in enumerate(tqdm(prs, desc="c3look")):
            base = by_id.get(pr["id"])
            if base is None:
                continue
            # CROSS-CHANNEL DISTRACTORS. Scored over (gold_a, gold_b) alone this
            # is not a choice between SOURCES at all: the caption's value and the
            # OCR's value are not even on the ballot, so a model that has never
            # heard of source control answers "what colour do I see" and scores
            # 1.000. Prompt tuning, LoRA and both neologisms all did exactly that.
            # Putting the other channels' values in the candidate set makes
            # FOLLOWING the image edit require IGNORING the text -- which is the
            # thing the word is supposed to do.
            rv_, ov_ = pr.get("read_value"), pr.get("ocr_value")
            if a.cp and pr["gold_b"] in (rv_, ov_):
                # v1.86: the twin value is asserted by a text channel; move that channel (both contexts)
                fixed, _moved = cp_fix_look_collision(base, pr["gold_b"])
                if fixed is None:
                    R.setdefault("look_cp_unfixable", []).append(1); continue
                R.setdefault("look_cp_moved", []).append(1)
                base = fixed
                rv_, ov_ = str(base["channels"]["read"]), str(base["channels"]["ocr"])
            cands = tuple(dict.fromkeys(
                [pr["gold_a"], pr["gold_b"]]
                + [x for x in (rv_, ov_) if x]))
            ia = dict(base); ia["image"] = pr["ctx_a"]["image"]
            ib = dict(base); ib["image"] = pr["ctx_b"]["image"]
            pa, _ = pick(ia, "look", cands, j)
            pb, _ = pick(ib, "look", cands, j)
            R["look_twin_acc"].append(int(pa == pr["gold_a"]))
            R["look_flip"].append(int(pa == pr["gold_a"] and pb == pr["gold_b"]))
            PK["look"].append({"id": pr["id"], "pa": pa, "pb": pb, "gold_a": pr["gold_a"],
                               "gold_b": pr["gold_b"], "read": pr.get("read_value"),
                               "ocr": pr.get("ocr_value")})
            SK[pr.get("skill", "?")]["look_flip"].append(
                int(pa == pr["gold_a"] and pb == pr["gold_b"]))
            SK[pr.get("skill", "?")]["look_twin_acc"].append(int(pa == pr["gold_a"]))
            # did it answer from a TEXT channel instead? the diagnostic that
            # distinguishes "cannot see" from "read the caption".
            R["look_leak_text"].append(int(pa not in (pr["gold_a"], pr["gold_b"])))
            # an image edit leaves BOTH text channels untouched, so <read> must
            # not move -- the IGNORES side of the same intervention, and the
            # check that a "flip" is not just the model tracking whichever value
            # was mentioned last.
            rv = rv_
            if rv and rv != pr["gold_b"]:
                rc = (rv, pr["gold_b"])
                ra, _ = pick(ia, "read", rc, j)
                rb, _ = pick(ib, "read", rc, j)
                R["read_stay_img"].append(int(ra == rb))
                # conjunctive form + per skill (v1.86): under --cp this IS read Stay (competitor = image)
                R.setdefault("read_stay_img_c", []).append(int(ra == rv and ra == rb))
                SK[pr.get("skill", "?")]["read_stay_img"].append(int(ra == rb))
                SK[pr.get("skill", "?")]["read_stay_img_c"].append(int(ra == rv and ra == rb))
    elif not a.cp:
        print(f"[look] no image twins at {twin} -- look_flip will be null. "
              f"Build them with scripts/clevr3_image_twins.py --split {a.split}")

    m = lambda v: round(sum(v) / len(v), 3) if v else None
    from scripts.clevr3_cp_eval import macro_over_skills as _macro
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "mode": cfgm, "n": seen, "cp": int(a.cp), "cands": a.cands,
           "cp_fallback_read": m(R.get("cp_fallback_read", [])), "cp_fallback_ocr": m(R.get("cp_fallback_ocr", [])),
           "look_cp_moved_n": len(R.get("look_cp_moved", [])), "look_cp_unfixable_n": len(R.get("look_cp_unfixable", [])),
           "read_stay_ocr": m(R.get("read_stay_ocr", [])), "read_stay_ocr_c": m(R.get("read_stay_ocr_c", [])),
           "read_stay_n": len(R.get("read_stay", [])),
           "cand_n_mean": m(R.get("cand_n", [])), "chance": (round(1.0 / (sum(R["cand_n"]) / len(R["cand_n"])), 3) if R.get("cand_n") else None),
           "cp_skipped_n": len(R.get("cp_skipped", [])), "sources": list(c3_sources()), "question_style": _QS, "stride": a.stride,
           "tpl_pt": os.environ.get("SC_TPL_PT", ""),
           # 2026-09-23 (v1.114): record WHICH English instruction was used. The harvester used to infer it
           # from the filename (`_instr_iv5` -> instr, bare `_instr` -> instr_v0), so an INSTR_VARIANT=5 run
           # tagged `_cp4_instr` was filed as v0 and never matched the ladder's `instr` row.
           "instr_variant": (int(os.environ.get("INSTR_VARIANT", "0")) if a.mode == "instructed" else None),
           "macro_skill": _macro(SK) if a.cp else None,
           "batched_scoring": bool(a.batched), "batched_check": ({"n": _bchk["n"], "pick_agreement": (round(_bchk["agree"] / _bchk["n"], 3) if _bchk["n"] else None),
                                                                 "max_abs_loglik_diff": round(_bchk.get("max_abs_diff", 0.0), 4)} if a.batched else None),
           "read_stay_c": m(R["read_stay_c"]), "ocr_stay_c": m(R["ocr_stay_c"]),
           "look_stay_c": m(R["look_stay_c"]),
           "read_acc": m(R["read_acc"]), "read_flip": m(R["read_flip"]),
           "read_stay": m(R["read_stay"]),
           "ocr_acc": m(R["ocr_acc"]), "ocr_flip": m(R["ocr_flip"]),
           "ocr_stay": m(R["ocr_stay"]),
           "look_acc": m(R["look_acc"]), "look_stay": m(R["look_stay"]),
           "look_presence_drop": m(R["look_presence_drop"]),
           "look_presence_pos": m([int(x > 0) for x in R["look_presence_drop"]]),
           "look_flip": m(R["look_flip"]),
           "look_twin_acc": m(R["look_twin_acc"]),
           "look_twin_n": len(R["look_flip"]),
           # v1.161: old-style (agreement-twin) Flip on the same items, when SC_GCF_AGREE_TWIN=1
           **({"look_flip_agree": m(R["look_flip_agree"]), "look_agree_n": len(R["look_flip_agree"])} if R.get("look_flip_agree") else {}),
           "read_stay_img": m(R["read_stay_img"]),
           "look_leak_text": m(R["look_leak_text"]),
           "per_skill": {k: {kk: m(vv) for kk, vv in v.items()}
                         for k, v in SK.items()},
           "per_skill_n": {k: {kk: len(vv) for kk, vv in v.items()} for k, v in SK.items()},
           "note": ("look_flip is scored on exact in-place image twins "
                    "(colour recolour / shape redraw); the text channels are "
                    "untouched by that edit, so read_stay_img is its IGNORES "
                    "side. look_stay/presence come from the channel edits.")}
    if a.dump_records:
        out["records"] = {k: list(map(int, v)) for k, v in R.items()
                          if k.endswith(("_flip", "_acc", "_stay", "_stay_c")) and v}
        out["records"].update({k: [round(float(x), 4) for x in v] for k, v in R.items()
                               if "_margin" in k and v})
    if a.dump_picks:
        out["picks"] = PK
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=1))
    od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}")
    os.makedirs(od, exist_ok=True)
    # non-bf16 evaluations get their own filename and a recorded dtype (STATUS v1.70)
    import torch as _t
    _dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    out["weight_dtype"] = _dtype
    out["decision_score"] = os.environ.get("SC_SCORE", "turn")   # v1.143
    _dt_tag = "" if _dtype == "bfloat16" else f"_{_dtype}"
    fname = f"flip_step_{a.step}{a.tag}{_dt_tag}.json"
    json.dump(out, open(os.path.join(od, fname), "w"), indent=1)
    if _SIDE:
        import gzip as _gz
        _meta = {"run": os.path.basename(a.run_dir), "step": a.step, "split": a.split, "tag": a.tag, "mode": a.mode,
                 "method": (json.load(open(os.path.join(a.run_dir, "config.json"))).get("args", {}).get("method")
                            if os.path.exists(os.path.join(a.run_dir, "config.json")) else None),
                 "continuation": cand("<v>"), "decision_score": _SCORE,
                 "scoring": "argmax of the summed log-likelihood of the whole assistant turn (lead-in + answer + end of turn); "
                            "per candidate also answer_sum/answer_mean/answer_ntok, lead_sum, end_sum and per-token log-probs"}
        with _gz.open(os.path.join(od, fname.replace(".json", "_items.jsonl.gz")), "wt") as _f:
            _f.write(json.dumps({"_meta": _meta}) + "\n")
            for _r in _SIDE:
                _f.write(json.dumps(_r) + "\n")
        print("[saved] item log", fname.replace(".json", "_items.jsonl.gz"), len(_SIDE), "scored contexts")
    print(f"[saved] {od}/{fname}")


if __name__ == "__main__":
    main()
