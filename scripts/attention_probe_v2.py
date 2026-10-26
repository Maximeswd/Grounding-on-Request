#!/usr/bin/env python
"""PER-ITEM ATTENTION WITH A CORRECTNESS SPLIT (2026-09-17), on Ground-CF and MultiVENT-G.

Why this exists. attention_probe2.py records attention on the ladder but summarises before it knows
whether the answer was right; attention_probe_gcf.py adds the evidence-box split of the image tokens but
TEACHER-FORCES the gold target, so a correct and an incorrect item look identical and
"correct minus incorrect" cannot be formed. Here correctness is decided FIRST, by the same two-way
forced choice that defines Accuracy and Flip everywhere else in the paper, and the attention is then
recorded on the continuation the model actually PREFERS (its own pick, not the gold). So:
    correct item   -> attention while committing to the right value
    incorrect item -> attention while committing to the wrong value
and the difference is a within-cell contrast at matched positions.

Positions (query rows), the four the analysis asks for plus three that are free:
    first   the first generated token          -- "beginning of the answer generation"
    ans     mean over the whole answer
    final   mean over "final answer: <value>"  -- the committed span
    preval  the position whose prediction is the first value token -- "start of the final answer value"
    ctrl    the request word's own positions; lastp the last prompt token; value the value tokens.
Targets (key columns): img, and its partition into imgbox (the annotated evidence box) and imgout;
cap and capval (the requested value's occurrences inside the caption/description); ocr; the request
word; bos; and carrier (the remaining prompt text). PRECISION is not stored separately: it is the
ratio imgbox/img or capval/cap at plot time, as scripts/attn_layer_profiles.py already does.

Output: runs/<run>/attention2/<ds>_<split>_v2_step<N>/{correct,incorrect}_<op>.npz (items x L x H x CH,
float16), items.json (one record per item: op, correct, pick, gold, span sizes), summary.json.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import scripts.attention_probe2 as AP
from scripts.attention_probe_gcf import token_grid, box_cells
from scsrc.clevr3 import clevr3_user_content, _blocks as c3_blocks
from scsrc.collate import _encode, _prompt_len
from scripts.eval_clevr3 import cand as c3_cand, loglik as c3_loglik

QUERIES = ("ctrl", "lastp", "first", "preval", "value", "ans", "final")
KEYS = ("img", "imgbox", "imgout", "cap", "capval", "ocr", "ctrlk", "bos", "carrier")
CH = [f"{q}_{k}" for q in QUERIES for k in KEYS]
FINAL_MARK = "final answer:"


def locate_block(ids, tok, txt, lo=0):
    """Token positions of a rendered text block inside the encoded sequence.

    A block cannot be found by tokenizing it standalone: in context it is preceded by a space or a
    newline, so its first token differs ("Description" vs " Description") and a subsequence search
    fails silently -- which is how the MultiVENT-G description block came back EMPTY on the first
    dry run. Try the spaced and newline-prefixed forms, then fall back to anchoring on an interior
    slice and extending by the block's token length."""
    txt = (txt or "").strip()
    if not txt:
        return []
    n_tok = len(tok(txt, add_special_tokens=False)["input_ids"])
    for probe in (txt, " " + txt, "\n" + txt, '"' + txt, ' "' + txt):
        sp = AP._find_sub(ids, tok(probe, add_special_tokens=False)["input_ids"], lo)
        if sp:
            return list(range(sp[0], sp[1]))
    mid = txt[8:88] if len(txt) > 96 else txt[2:]
    for probe in (mid, " " + mid):
        sp = AP._find_sub(ids, tok(probe, add_special_tokens=False)["input_ids"], lo)
        if sp:
            # the anchor sits inside the block; take n_tok tokens around it, clipped to the sequence
            start = max(lo, sp[0] - 12)
            return list(range(start, min(len(ids), start + n_tok + 12)))
    return []


def _img_token_ids(tok):
    ids = set()
    for tk in ("image_token", "boi_token"):
        t = getattr(tok, tk, None)
        if t:
            try: ids.add(tok.convert_tokens_to_ids(t))
            except Exception: pass
    return ids


def _spans(ids, tok, plen, T, value, img_pos, cap_pos, ocr_pos, ctrl_pos, imgbox, imgout, capval):
    """Query and key position sets for one encoded (prompt + chosen answer) sequence."""
    vpos = AP._tok_spans_in(ids, tok, str(value), plen, T)
    # "final answer:" must be found WITH its leading space: in context the answer reads
    # "... the question. final answer: yellow", which tokenizes as [" final", " answer", ":"], while
    # tok("final answer:") gives ["final", ...] and never matches. Try the spaced form first.
    fmark = None
    for probe in (" " + FINAL_MARK, FINAL_MARK, " answer:", "answer:"):
        fmark = AP._find_sub(ids, tok(probe, add_special_tokens=False)["input_ids"], plen)
        if fmark:
            break
    # trailing structural tokens (<end_of_turn>, newline) are not part of the committed answer
    special = set(getattr(tok, "all_special_ids", []) or [])
    end = T
    while end > plen and ids[end - 1] in special:
        end -= 1
    final = list(range(fmark[0], end)) if fmark else (list(range(vpos[0], end)) if vpos else [])
    known = set(img_pos) | set(cap_pos) | set(ocr_pos) | set(ctrl_pos) | {0}
    carrier = [i for i in range(1, plen) if i not in known]
    q = {"ctrl": ctrl_pos, "lastp": [plen - 1], "first": [plen] if plen < T else [],
         "preval": [vpos[0] - 1] if vpos and vpos[0] - 1 >= 0 else [],
         "value": vpos, "ans": list(range(plen, end)), "final": final}
    k = {"img": img_pos, "imgbox": imgbox, "imgout": imgout, "cap": cap_pos, "capval": capval,
         "ocr": ocr_pos, "ctrlk": ctrl_pos, "bos": [0], "carrier": carrier}
    return q, k


@torch.no_grad()
def record(model, processor, umsg, answer, device, spans_fn):
    """One forward pass over prompt + `answer`; returns (L x H x CH, record dict)."""
    fmsg = umsg + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    enc = _encode(processor, fmsg, device)
    plen = _prompt_len(processor, umsg, device)
    o = model(**{k: v for k, v in enc.items() if k != "labels"}, output_attentions=True)
    ids = enc["input_ids"][0].tolist(); T = len(ids)
    q_sets, k_sets = spans_fn(ids, plen, T, enc)
    L, Hh = len(o.attentions), o.attentions[0].shape[1]
    arr = np.full((L, Hh, len(CH)), np.nan, dtype=np.float32)
    qk = [(q_sets[qq], k_sets[kk]) for qq in QUERIES for kk in KEYS]
    for li, A in enumerate(o.attentions):
        am = A[0].float().cpu().numpy()
        for ci, (qp, kp) in enumerate(qk):
            if qp and kp:
                arr[li, :, ci] = AP.mass(am, qp, kp)
    rec = {"T": T, "plen": plen, **{f"n_{kk}": len(vv) for kk, vv in k_sets.items()},
           "has_final": bool(q_sets["final"]), "n_ans": len(q_sets["ans"])}
    return arr.astype(np.float16), rec


# ----------------------------------------------------------------- Ground-CF
def gcf_items(a):
    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    return [it for it in rows if it.get("bbox")]


def gcf_cell(model, processor, meta, a, it, src, device, img_tok_ids):
    """Returns (arr, rec) or None. Two-way forced choice, exactly as eval_clevr3_flip.pick."""
    tok = processor.tokenizer
    method, n = meta["method"], meta["n_tokens"]
    ch = {k: str(v) for k, v in it["channels"].items()}
    alt = next((v for k, v in ch.items() if k != src and v != ch[src]), None)
    if alt is None:
        return None
    img = Image.open(it["image"]).convert("RGB"); W, H = img.size
    it2 = dict(it); it2["image"] = img
    content = clevr3_user_content(it2, src, n, a.mode, a.seed, method=method)
    umsg = [{"role": "user", "content": content}]
    sc = {v: c3_loglik(model, processor, umsg, c3_cand(v), device) for v in (ch[src], alt)}
    pick = max(sc, key=sc.get)
    blocks = c3_blocks(it)

    def spans_fn(ids, plen, T, enc):
        img_pos = [i for i, t in enumerate(ids) if t in img_tok_ids]
        gh, gw, off = token_grid(enc, len(img_pos))
        grid = img_pos[off:off + gh * gw]
        cells = sorted(set(box_cells(it["bbox"], W, H, gh, gw)))
        imgbox = [grid[c] for c in cells if c < len(grid)]
        imgout = [p for p in img_pos if p not in set(imgbox)]
        cap_pos = locate_block(ids, tok, blocks.get("read"))
        ocr_pos = locate_block(ids, tok, blocks.get("ocr"))
        capval = sorted(set(AP._tok_spans_in(ids, tok, ch[src], cap_pos[0], cap_pos[-1] + 1))) if cap_pos else []
        ctrl_pos = [i for i, t in enumerate(ids) if t in set(getattr(model, "_neo_all_ids", []) or [])]
        return _spans(ids, tok, plen, T, pick, img_pos, cap_pos, ocr_pos, ctrl_pos, imgbox, imgout, capval)

    arr, rec = record(model, processor, umsg, c3_cand(pick), device, spans_fn)
    rec |= {"id": it["id"], "skill": it.get("skill"), "op": src, "gold": ch[src], "pick": pick,
            "correct": int(pick == ch[src]), "alt": alt,
            "margin": round(float(sc[ch[src]] - sc[alt]), 4)}
    return arr, rec


# ----------------------------------------------------------------- MultiVENT-G
def mvg_setup(a):
    from collections import defaultdict
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
    pairs = [p for p in pairs if p["pair_type"] == "vis_value_swap" and p.get("bbox_a")]
    return mvg, ocr_by_vid, frames, pairs


def mvg_cell(model, processor, meta, a, p, device, img_tok_ids, ctx):
    from scsrc.mvg import mvg_user_content, MVG_CONT, mvg_word, _blocks as mvg_blocks
    from scripts.eval_mvg import loglik_mvg
    from scripts.eval_mvg_flip import INSTR, NOOP
    mvg, ocr_by_vid, frames, _ = ctx
    tok = processor.tokenizer; n = meta["n_tokens"]
    v = p["video_id"]
    if not frames[v]:
        return None
    it = {"video_id": v, "role": p["role"], "perm6": hash(v + p["role"]) % 6,
          "desc_text": mvg[v]["description"][:1200], "ocr_strings": sorted(ocr_by_vid[v])[:8],
          "vis_frame": frames[v][0][1], "role_question": mvg[v]["template"].get(p["role"], "")}
    chan = p["channel"]                                  # VIS for vis_value_swap
    if a.mode == "srconly":
        req = mvg_word(f"{chan.lower()}_ctrl", n)
    else:
        req = INSTR[chan] if a.mode == "instructed" else NOOP
    frame = Image.open(p["ctx_a"]["frame"]).convert("RGB"); W, H = frame.size
    content = mvg_user_content(it, req, {"vis_frame": frame})
    umsg = [{"role": "user", "content": content}]
    ga, gb = str(p["gold_a"]), str(p["gold_b"])
    sc = {g: loglik_mvg(model, processor, umsg, MVG_CONT.format(v=g), device) for g in (ga, gb)}
    pick = max(sc, key=sc.get)

    def spans_fn(ids, plen, T, enc):
        img_pos = [i for i, t in enumerate(ids) if t in img_tok_ids]
        gh, gw, off = token_grid(enc, len(img_pos))
        grid = img_pos[off:off + gh * gw]
        # MultiVENT-G bboxes are in PERCENT of the frame (eval_mvg_flip._crop divides by 100);
        # box_cells expects PIXELS, as Ground-CF stores them. Converting here was missing on the
        # first dry run and an 81% x 28% box came out as ONE grid cell.
        x0, y0, x1, y1 = [max(0.0, min(100.0, float(c))) for c in p["bbox_a"]]
        bbox_px = [x0 * W / 100.0, y0 * H / 100.0, x1 * W / 100.0, y1 * H / 100.0]
        cells = sorted(set(box_cells(bbox_px, W, H, gh, gw)))
        imgbox = [grid[c] for c in cells if c < len(grid)]
        imgout = [q for q in img_pos if q not in set(imgbox)]
        # locate the RENDERED blocks ('Description: "..."', 'On-screen text: "..."'), not the raw
        # field values: the raw description is preceded by a quote in context, so its standalone
        # first token never matches and the search fell back to a PADDED window that then swallowed
        # the neighbouring OCR block (both found on the dry run).
        mblocks = mvg_blocks(it | {"vis_frame": frame})
        cap_pos = locate_block(ids, tok, mblocks["desc"])
        ocr_pos = sorted(set(locate_block(ids, tok, mblocks["ocr"])) - set(cap_pos))
        capval = sorted(set(AP._tok_spans_in(ids, tok, ga, cap_pos[0], cap_pos[-1] + 1))) if cap_pos else []
        ctrl_pos = [i for i, t in enumerate(ids) if t in set(getattr(model, "_neo_all_ids", []) or [])]
        return _spans(ids, tok, plen, T, pick, img_pos, cap_pos, ocr_pos, ctrl_pos, imgbox, imgout, capval)

    arr, rec = record(model, processor, umsg, MVG_CONT.format(v=pick), device, spans_fn)
    rec |= {"id": f"{v}|{p['role']}", "op": chan.lower(), "gold": ga, "pick": pick,
            "correct": int(pick == ga), "alt": gb, "margin": round(float(sc[ga] - sc[gb]), 4),
            "bbox_share_of_image": round(rec["n_imgbox"] / max(1, rec["n_img"]), 4)}
    return arr, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ds", default="clevr3", choices=["clevr3", "mvg"])
    ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=150,
                    help="items per (run, op); the correctness split is whatever falls out")
    ap.add_argument("--mode", default="srconly", choices=["srconly", "instructed", "noop"])
    ap.add_argument("--ops", default="look,read", help="clevr3 only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frozen_model_id", default=None,
                    help="no checkpoint: the base model (requires --mode instructed|noop)")
    ap.add_argument("--pairs", default="data/mvg_ctm/pairs_clean_rb.jsonl")
    ap.add_argument("--mvg", default="data/multiventG/multivent_g_rebuilt.json")
    ap.add_argument("--claims", default="data/multiventG/claims.jsonl")
    ap.add_argument("--frames", default="data/multiventG/frames_manifest.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda",
                    help="cpu runs the whole path on a login node for a 1-item validation")
    a = ap.parse_args()
    device = a.device
    if a.frozen_model_id:
        assert a.mode != "srconly", "--frozen_model_id needs --mode instructed|noop"
        from scsrc import model as M
        model, processor = M.load(model_id=None if a.frozen_model_id == "base" else a.frozen_model_id)
        if device == "cpu": model = model.to("cpu")
        model.eval(); meta = {"n_tokens": 0, "method": "neo"}
        run_out = a.run_dir
    else:
        import evaluate as EV
        model, processor, meta, _ = EV.apply_checkpoint(a.run_dir, a.step, device)
        run_out = a.run_dir
    tok = processor.tokenizer
    img_tok_ids = _img_token_ids(tok)
    # the mode is part of the directory name: noop and instructed share one frozen run_dir and
    # would otherwise overwrite each other.
    out_dir = os.path.join(run_out, "attention2",
                           f"{a.ds}_{a.split}_v2_{a.mode}{a.tag}_step{a.step}")
    os.makedirs(out_dir, exist_ok=True)

    groups, recs = {}, []
    if a.ds == "clevr3":
        items = gcf_items(a)[:a.max_n]
        ops = [o for o in a.ops.split(",") if o]
        for it in tqdm(items, desc=f"gcf/{a.mode}"):
            for src in ops:
                r = gcf_cell(model, processor, meta, a, it, src, device, img_tok_ids)
                if r is None: continue
                arr, rec = r
                groups.setdefault((src, rec["correct"]), []).append(arr); recs.append(rec)
    else:
        ctx = mvg_setup(a)
        for p in tqdm(ctx[3][:a.max_n], desc=f"mvg/{a.mode}"):
            r = mvg_cell(model, processor, meta, a, p, device, img_tok_ids, ctx)
            if r is None: continue
            arr, rec = r
            groups.setdefault((rec["op"], rec["correct"]), []).append(arr); recs.append(rec)

    for (op, ok), arrs in groups.items():
        np.savez_compressed(os.path.join(out_dir, f"{'correct' if ok else 'incorrect'}_{op}.npz"),
                            a=np.stack(arrs))
    json.dump(recs, open(os.path.join(out_dir, "items.json"), "w"))
    n_ok = sum(r["correct"] for r in recs)
    summary = {"channels": CH, "queries": list(QUERIES), "keys": list(KEYS), "ds": a.ds,
               "mode": a.mode, "method": meta.get("method"), "run": os.path.basename(a.run_dir),
               "split": a.split, "n_items": len(recs), "n_correct": n_ok,
               "n_incorrect": len(recs) - n_ok,
               "per_op": {f"{op}|{'correct' if ok else 'incorrect'}": len(v) for (op, ok), v in groups.items()},
               "note": "correctness from the same two-way forced choice as Accuracy/Flip; attention "
                       "recorded on the model's OWN preferred continuation, not the gold."}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("run", "ds", "mode", "n_items", "n_correct", "n_incorrect", "per_op")}))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
