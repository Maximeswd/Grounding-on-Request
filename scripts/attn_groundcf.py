#!/usr/bin/env python
"""WHICH modality, WHERE inside it, and WHEN (which layers) does the answer attend, per
request and per interface, on Ground-CF. Attention is read at the position that emits
the first value token of the gold answer (teacher forced), averaged over heads.

Per (item, requested source) the probe records attention mass onto
    img      the image tokens                 cap     the caption block
    capval   the caption's value token(s)      ocr     the on-screen text block
    ocrval   the on-screen value token(s)      ctrl    the learned word's positions
    bbox     image tokens whose grid cell overlaps the evidence box (items with a box)
    bos      position 0 (attention sink)
and, from them,
    share    mass on each modality / total mass on the three modalities  (WHICH)
    prec_img = bbox / img, prec_cap = capval / cap, prec_ocr = ocrval / ocr   (WHERE)
    layer profiles of img / cap / ocr mass                                    (WHEN)
plus a mean image heat map per source over the middle third of layers (Gemma 3: 16x16;
Qwen 3.5: the processor's grid; Gemma 4 E4B: mass only, its soft token layout is not a
grid we can map a box onto).

Usage: python scripts/attn_groundcf.py --run_dir runs/clevr3_f_5tok_ctm_seed0 --split test_id
Output: runs/<run>/eval/clevr3_<split>/attn_groundcf_step_<step><tag>.json (+ .npz heat maps)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
EVID = None
import torch
from tqdm import tqdm
from scsrc.clevr3 import clevr3_user_content
from scsrc.collate import _encode, _prompt_len
from evaluate import apply_checkpoint
from scripts.eval_clevr3 import cand

SRC = ("look", "read", "ocr")
IMG_W = 336


def find_sub(ids, sub, start=0):
    if not sub:
        return None
    for i in range(start, len(ids) - len(sub) + 1):
        if ids[i:i + len(sub)] == sub:
            return i, i + len(sub)
    return None


def find_any(ids, tok, text, start, end):
    """Positions of `text` inside ids[start:end]. Tries with and without a leading space
    and without trailing punctuation: inside a quoted block the final '.' merges with the
    closing quote into one token, so the exact string never matches."""
    core = text.rstrip(".!?")
    for t in (text, " " + text, core, " " + core):
        sub = tok(t, add_special_tokens=False)["input_ids"]
        sp = find_sub(ids[:end], sub, start)
        if sp:
            return list(range(*sp))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=150)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump_items", type=int, default=12,
                    help="store the per-item mid-layer image map (and the item's evidence boxes) for the first N "
                         "box items of each request in the .npz, so one example can be drawn across interfaces (2026-09-21)")
    ap.add_argument("--heads", type=int, default=0,
                    help="2026-09-22 (author): keep EVERY head at EVERY layer, at two query positions -- the last "
                         "prompt token (the moment generation starts) and the token before the first gold value "
                         "(commitment, right after 'final answer:') -- with image/caption mass and box/value "
                         "precision per head, accumulated per request and per skill (colour, object, count, "
                         "spatial). Written to the npz as heads__<pos>__<source>__<skill>__<key> [L,H] means.")
    ap.add_argument("--evidence_boxes", default=None,
                    help="JSON from the scene replay: {items: {id: {boxes: [[x0,y0,x1,y1], ...]}}}; colour/object = target, "
                         "spatial = target + anchor, count = every object. The union of their cells is the evidence "
                         "(2026-09-21); without it only the item's single target bbox is used.")
    a = ap.parse_args()
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode instructed/noop needs --tag")
    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
    global SRC
    from scsrc.tokens import c3_sources as _c3s
    SRC = tuple(_c3s())          # STATUS v1.94: probe only the run's own sources (two-modality protocol runs: look, read)
    print(f"[attn] sources probed: {SRC}")
    tok = proc.tokenizer
    n = meta["n_tokens"]
    try:
        cfg = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    except Exception:
        cfg = {}
    meth = "neo" if a.mode in ("instructed", "noop") else cfg.get("method", "neo")
    cfgm = a.mode or cfg.get("factor_mode", "srconly")
    print(f"[prompt] method={meth} mode={cfgm} L={n}")

    # ONLY the soft image tokens: the begin/end-of-image markers are not grid cells
    # (including <start_of_image> gave Gemma 3 257 positions and no 16x16 map).
    img_ids = set()
    for attr in ("image_token_index", "image_token_id"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int):
            img_ids.add(v)
    t = getattr(tok, "image_token", None)
    if t:
        try:
            img_ids.add(tok.convert_tokens_to_ids(t))
        except Exception:
            pass
    ctrl_ids = set(getattr(model, "_neo_all_ids", []) or [])
    mid = getattr(model.config, "model_type", "")
    n_hidden = getattr(model.config, "num_hidden_layers", None) or \
        getattr(getattr(model.config, "text_config", None), "num_hidden_layers", None)
    print(f"[probe] image ids {sorted(img_ids)[:3]} ctrl ids {len(ctrl_ids)} model_type={mid} "
          f"hidden layers={n_hidden}")

    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    global EVID
    EVID = (json.load(open(a.evidence_boxes))["items"] if a.evidence_boxes else None)
    if EVID: print(f"[attn] evidence boxes for {len(EVID)} items from {a.evidence_boxes}")
    recs, heat, heat_n, seen = [], {s: None for s in SRC}, {s: 0 for s in SRC}, 0
    item_maps, item_dumped = {}, {}
    # box-centred maps (items with a box): each 16x16 map is shifted so the box centre sits at
    # the canvas centre before averaging, so evidence does not smear out across items; the
    # per-item attention SINK (the highest image cell, when it lies outside the box) is masked.
    cent, cent_n = {s: None for s in SRC}, {s: 0 for s in SRC}
    L_layers = None
    HEADS, HEADS_N = {}, {}          # (pos, source, skill, key) -> summed [L,H] array, count
    for i, it in enumerate(tqdm(rows, desc="attn")):
        if seen >= a.max_n:
            break
        ch = {k: str(v) for k, v in it["channels"].items()}
        seen += 1
        for s in SRC:
            um = [{"role": "user", "content": clevr3_user_content(it, s, n, cfgm, i, method=meth)}]
            fmsg = um + [{"role": "assistant", "content": [{"type": "text", "text": cand(ch[s])}]}]
            enc = _encode(proc, fmsg, "cuda")
            plen = _prompt_len(proc, um, "cuda")
            kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
            with torch.no_grad():
                o = model(**kw, output_attentions=True, use_cache=False)
            if not o.attentions or o.attentions[0] is None:
                raise SystemExit("[STOP] no attentions returned; load the model with attn='eager'")
            ids = enc["input_ids"][0].tolist(); T = len(ids)
            L_layers = len(o.attentions)
            img_pos = [k for k, t in enumerate(ids) if t in img_ids]
            cap_pos = find_any(ids, tok, it["caption"], 0, plen)
            ocr_pos = find_any(ids, tok, it["ocr_text"], 0, plen)
            capval = find_any(ids, tok, ch["read"], cap_pos[0], cap_pos[-1] + 1) if cap_pos else []
            ocrval = find_any(ids, tok, ch["ocr"], ocr_pos[0], ocr_pos[-1] + 1) if ocr_pos else []
            ctrl_pos = [k for k, t in enumerate(ids) if t in ctrl_ids and k < plen]
            vpos = find_any(ids, tok, ch[s], plen, T)
            q = (vpos[0] - 1) if vpos else (T - 1)
            # bbox -> image token cells
            bbox_pos = []
            grid = None
            if img_pos:
                nimg = len(img_pos)
                if "image_grid_thw" in enc:
                    thw = enc["image_grid_thw"][0].tolist()
                    merge = getattr(getattr(model.config, "vision_config", None), "spatial_merge_size", 2) or 2
                    gh, gw = thw[1] // merge, thw[2] // merge
                    if gh * gw == nimg:
                        grid = (gh, gw)
                if grid is None and nimg == 256:
                    grid = (16, 16)      # Gemma 3 and Gemma 4: 256 soft tokens on a square grid
                _boxes = ((EVID.get(it["id"]) or {}).get("boxes") if EVID else None) or ([it["bbox"]] if it.get("bbox") else [])
                if grid and _boxes:
                    gh, gw = grid
                    for (x0, y0, x1, y1) in _boxes:
                        for r in range(gh):
                            for c in range(gw):
                                cx0, cy0 = c * IMG_W / gw, r * IMG_W / gh
                                cx1, cy1 = (c + 1) * IMG_W / gw, (r + 1) * IMG_W / gh
                                if cx0 < x1 and cx1 > x0 and cy0 < y1 and cy1 > y0 and img_pos[r * gw + c] not in bbox_pos:
                                    bbox_pos.append(img_pos[r * gw + c])
            spans = {"img": img_pos, "cap": cap_pos, "capval": capval, "ocr": ocr_pos,
                     "ocrval": ocrval, "ctrl": ctrl_pos, "bbox": bbox_pos, "bos": [0]}
            prof = {k: [] for k in ("img", "cap", "ocr", "ctrl", "bbox", "capval", "ocrval")}   # capval/ocrval: per-layer text precision (2026-09-21)
            mass = {k: 0.0 for k in spans}
            hm = None
            capm = None                                     # per-token caption attention, mid layers (2026-09-21)
            lo, hi = L_layers // 3, 2 * L_layers // 3
            if a.heads:
                # per-head, two positions: q_first predicts the first continuation token; q (commitment) the first gold value token
                for pname, qq in (("first", plen - 1), ("commit", q)):
                    stack = torch.stack([o.attentions[l][0][:, qq, :].float() for l in range(L_layers)])   # [L,H,T]
                    tot_img = stack[:, :, img_pos].sum(-1) if img_pos else None
                    tot_cap = stack[:, :, cap_pos].sum(-1) if cap_pos else None
                    tot_all = stack.sum(-1).clamp_min(1e-9)
                    feats = {}
                    if tot_img is not None:
                        feats["img_share"] = (tot_img / tot_all)
                        if bbox_pos: feats["bbox_prec"] = (stack[:, :, bbox_pos].sum(-1) / tot_img.clamp_min(1e-9))
                    if tot_cap is not None:
                        feats["cap_share"] = (tot_cap / tot_all)
                        if capval: feats["capval_prec"] = (stack[:, :, capval].sum(-1) / tot_cap.clamp_min(1e-9))
                    for key, val in feats.items():
                        for sk in (it.get("skill") or "all", "all"):
                            kk = (pname, s, sk, key)
                            arr = val.cpu().numpy()
                            HEADS[kk] = arr if kk not in HEADS else HEADS[kk] + arr
                            HEADS_N[kk] = HEADS_N.get(kk, 0) + 1
                    del stack
            for l in range(L_layers):
                A = o.attentions[l][0].float().mean(0)[q]          # (T,)
                for k, pos in spans.items():
                    m = float(A[pos].sum()) if pos else 0.0
                    mass[k] += m / L_layers
                    if k in prof:
                        prof[k].append(round(m, 5))
                if grid and lo <= l < hi:
                    g = A[img_pos].reshape(grid).cpu().numpy()
                    hm = g if hm is None else hm + g
                if cap_pos and lo <= l < hi:
                    cv = A[cap_pos].float().cpu().numpy()
                    capm = cv if capm is None else capm + cv
            sink_cell, sink_mass, prec_nosink, img_nosink = None, None, None, None
            if hm is not None:
                hm = hm / max(1, hi - lo)
                heat[s] = hm if heat[s] is None else heat[s] + hm
                heat_n[s] += 1
                if bbox_pos and item_dumped.get(s, 0) < a.dump_items:
                    item_maps[f"item__{s}__{it['id']}"] = hm.astype(np.float32)
                    item_maps[f"boxes__{it['id']}"] = np.asarray(_boxes, dtype=np.float32)
                    if capm is not None:
                        item_maps[f"captok__{s}__{it['id']}"] = (capm / max(1, hi - lo)).astype(np.float32)
                        item_maps[f"captxt__{it['id']}"] = np.asarray("\x1f".join(tok.convert_ids_to_tokens([ids[p] for p in cap_pos])))
                        item_maps[f"capval__{s}__{it['id']}"] = np.asarray([int(p in set(capval)) for p in cap_pos], dtype=np.int8)
                    item_dumped[s] = item_dumped.get(s, 0) + 1
                gh, gw = grid
                # sink: the strongest image cell over the mid layers, if it is not evidence
                r_, c_ = divmod(int(hm.argmax()), gw)
                in_box = bool(bbox_pos) and img_pos[r_ * gw + c_] in bbox_pos
                if not in_box:
                    sink_cell = [r_, c_]
                    # its full-depth mass (same averaging as `mass`)
                    sm = 0.0
                    for l in range(L_layers):
                        sm += float(o.attentions[l][0].float().mean(0)[q][img_pos[r_ * gw + c_]]) / L_layers
                    sink_mass = sm
                    img_nosink = mass["img"] - sm
                    if bbox_pos and img_nosink > 0:
                        prec_nosink = mass["bbox"] / img_nosink
                if it.get("bbox") and bbox_pos:
                    x0, y0, x1, y1 = it["bbox"]
                    cx, cy = (x0 + x1) / 2 / IMG_W * gw, (y0 + y1) / 2 / IMG_W * gh
                    g16 = hm.copy()
                    if sink_cell is not None:
                        g16[r_, c_] = 0.0
                    canvas = np.zeros((3 * gh, 3 * gw), dtype=np.float64)
                    oy, ox = int(round(1.5 * gh - cy)), int(round(1.5 * gw - cx))
                    oy, ox = min(max(oy, 0), 2 * gh), min(max(ox, 0), 2 * gw)
                    canvas[oy:oy + gh, ox:ox + gw] += g16
                    c16 = canvas[gh:2 * gh, gw:2 * gw]
                    cent[s] = c16 if cent[s] is None else cent[s] + c16
                    cent_n[s] += 1
            tot = mass["img"] + mass["cap"] + mass["ocr"]
            recs.append({"id": it.get("id"), "skill": it.get("skill"), "source": s,
                         "has_bbox": bool(bbox_pos), "T": T, "q": q, "n_img": len(img_pos),
                         "mass": {k: round(v, 5) for k, v in mass.items()},
                         "share": {k: round(mass[k] / tot, 4) if tot else None for k in ("img", "cap", "ocr")},
                         "prec_img": round(mass["bbox"] / mass["img"], 4) if bbox_pos and mass["img"] else None,
                         "prec_img_nosink": None if prec_nosink is None else round(prec_nosink, 4),
                         "sink_cell": sink_cell, "sink_mass": None if sink_mass is None else round(sink_mass, 5),
                         "prec_cap": round(mass["capval"] / mass["cap"], 4) if capval and mass["cap"] else None,
                         "prec_ocr": round(mass["ocrval"] / mass["ocr"], 4) if ocrval and mass["ocr"] else None,
                         "profile": prof})
            del o
    # aggregates
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None
    agg = {}
    for s in SRC:
        rs = [r for r in recs if r["source"] == s]
        if not rs:
            continue
        agg[s] = {"n": len(rs),
                  "share": {k: mean([r["share"][k] for r in rs]) for k in ("img", "cap", "ocr")},
                  "mass": {k: mean([r["mass"][k] for r in rs]) for k in ("img", "cap", "ocr", "ctrl", "bos")},
                  "prec_img": mean([r["prec_img"] for r in rs]), "prec_cap": mean([r["prec_cap"] for r in rs]),
                  "prec_img_nosink": mean([r.get("prec_img_nosink") for r in rs]),
                  "sink_mass": mean([r.get("sink_mass") for r in rs]),
                  "sink_share_of_img": mean([(r["sink_mass"] / r["mass"]["img"]) if r.get("sink_mass") is not None and r["mass"]["img"] else None for r in rs]),
                  "prec_ocr": mean([r["prec_ocr"] for r in rs]),
                  "layer_profile": {k: [round(float(np.mean([r["profile"][k][l] for r in rs])), 5)
                                        for l in range(L_layers)] for k in ("img", "cap", "ocr", "ctrl")}}
        d = agg[s]
        print(f"  request {s:4s}: share img {d['share']['img']} cap {d['share']['cap']} ocr {d['share']['ocr']} | "
              f"prec img {d['prec_img']} (no sink {d['prec_img_nosink']}, sink share {d['sink_share_of_img']}) "
              f"cap {d['prec_cap']} ocr {d['prec_ocr']} | ctrl mass {d['mass']['ctrl']}")
    # Qwen 3.5 is a hybrid stack: only its full-attention layers return weights (8 of 32),
    # so `layers` counts layers WITH attention and `hidden_layers` the whole stack.
    out = {"meta": {"run": a.run_dir, "step": a.step, "mode": cfgm, "method": meth, "n_items": seen,
                    "layers": L_layers, "hidden_layers": n_hidden, "model_type": mid,
                    "instr_variant": os.environ.get("INSTR_VARIANT")},
           "by_source": agg, "records": recs}
    od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}")
    os.makedirs(od, exist_ok=True)
    stem = os.path.join(od, f"attn_groundcf_step_{a.step}{a.tag}")
    json.dump(out, open(stem + ".json", "w"), indent=1)
    hm = {s: (heat[s] / heat_n[s]) for s in SRC if heat[s] is not None and heat_n[s]}
    for s in SRC:
        if cent[s] is not None and cent_n[s]:
            hm[f"{s}_centred"] = cent[s] / cent_n[s]
    if a.heads:
        for (pname, src_, sk, key), arr in HEADS.items():
            item_maps[f"heads__{pname}__{src_}__{sk}__{key}"] = (arr / max(1, HEADS_N[(pname, src_, sk, key)])).astype(np.float32)
        item_maps["heads__meta"] = np.asarray(json.dumps({"positions": ["first", "commit"], "keys": ["img_share", "bbox_prec", "cap_share", "capval_prec"],
                                                          "L": L_layers, "n": {"|".join(k): v for k, v in HEADS_N.items()}}))
    if hm or item_maps:
        np.savez_compressed(stem + ".npz", **{s_: v.astype(np.float32) for s_, v in hm.items()}, **item_maps)
    print("[saved]", stem + ".json", "(+ .npz)" if hm else "(no grid heat maps for this model)")


if __name__ == "__main__":
    main()
