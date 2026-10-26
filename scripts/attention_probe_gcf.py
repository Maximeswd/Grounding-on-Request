#!/usr/bin/env python
"""ATTENTION PROBE FOR GROUND-CF WITH AN EVIDENCE-BOX CHANNEL (2026-09-17).

Why this exists. scripts/attention_probe2.py records attention as WHOLE-SPAN masses -- `img` is the
entire image -- so image-side PRECISION (the share of image attention that lands where the answer is)
was blank everywhere: the ladder has no per-patch annotation. Ground-CF carries a bbox per colour /
object / spatial item, so here every image token is assigned to the evidence box or to the rest, and
two keys are added:  imgbox = image tokens whose patch intersects the annotated box,
                    imgout = the other image tokens.
    precision(q) = mass(q -> imgbox) / mass(q -> img)   is then computable for the image side,
which is the measurement S10 needs and the author asked for ("do the words ground on the part of the
image that matters?").

Patch geometry. Gemma-3 encodes every image as a fixed 16x16 grid (256 tokens, sometimes +1 marker) in
raster order; Qwen exposes image_grid_thw = (t, h, w) in 14px patches and merges 2x2, so its token grid
is (h//2, w//2). The box is normalised by the ORIGINAL image size; both families preserve relative
position when they resize, so a normalised box maps straight onto the token grid.

Answers are TEACHER-FORCED to the gold target the words were trained on, so `preval` / `value` are
always defined and no generation is needed. Correct/incorrect splitting is deliberately left to a later
pass; this pass measures allocation and precision.

Output mirrors attention_probe2 exactly (summary.json with 'channels', <op>_<all>.npz, records.jsonl)
under runs/<run>/attention2/clevr3_<split>_gcf_step<N>/ so attn_precision.py-style consumers can read it.
Usage: python scripts/attention_probe_gcf.py --run_dir runs/clevr3_f_5tok_ctm_seed0 --split test_id
       [--max_n 40] [--mode srconly|instructed|noop] [--step 800]
"""
import argparse, glob, json, math, os, sys
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import attention_probe2 as AP                       # mass, _find_sub, _tok_spans_in, QUERIES
from scsrc.clevr3 import clevr3_user_content, _blocks, _encode, _prompt_len

QUERIES = AP.QUERIES                                 # ctrl, lastp, first, preval, value, ans
KEYS = ("img", "imgbox", "imgout", "cap", "capval", "ctrlk", "bos", "carrier")
CH = [f"{q}_{k}" for q in QUERIES for k in KEYS]
SOURCES = ("look", "read")


def token_grid(enc, n_img):
    """(rows, cols, offset) of the image-token grid; offset skips a leading marker token."""
    thw = enc.get("image_grid_thw")
    if thw is not None:
        t, h, w = [int(x) for x in thw[0].tolist()]
        gh, gw = h // 2, w // 2                      # Qwen merges 2x2 patches into one token
        off = n_img - gh * gw
        return gh, gw, max(0, off)
    side = int(round(math.sqrt(n_img)))
    if side * side == n_img:
        return side, side, 0
    side = int(round(math.sqrt(n_img - 1)))
    if side * side == n_img - 1:                     # Gemma: 256 patches + 1 marker
        return side, side, 1
    raise RuntimeError(f"cannot infer an image-token grid from n_img={n_img}")


def box_cells(bbox, W, H, gh, gw):
    x0, y0, x1, y1 = bbox
    c0, c1 = int(math.floor(x0 / W * gw)), int(math.ceil(x1 / W * gw)) - 1
    r0, r1 = int(math.floor(y0 / H * gh)), int(math.ceil(y1 / H * gh)) - 1
    c0, c1 = max(0, c0), min(gw - 1, max(c0, c1))
    r0, r1 = max(0, r0), min(gh - 1, max(r0, r1))
    return [r * gw + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=40)
    ap.add_argument("--mode", default="srconly", choices=["srconly", "instructed", "noop"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = "cuda"
    import evaluate as EV
    model, processor, meta, _ = EV.apply_checkpoint(a.run_dir, a.step, device)
    method, n = meta["method"], meta["n_tokens"]
    tok = processor.tokenizer
    img_tok_ids = set()
    for tk in ("image_token", "boi_token"):
        t = getattr(tok, tk, None)
        if t:
            try: img_tok_ids.add(tok.convert_tokens_to_ids(t))
            except Exception: pass

    items = [json.loads(l) for l in open(os.path.join(ROOT, a.data_dir, f"{a.split}.jsonl"))]
    items = [it for it in items if it.get("bbox")][:a.max_n]       # colour / object / spatial only
    out_dir = os.path.join(a.run_dir, "attention2", f"clevr3_{a.split}_gcf_step{a.step}")
    os.makedirs(out_dir, exist_ok=True)
    groups, recs = {}, []
    for it in items:
        img = Image.open(it["image"]).convert("RGB"); W, H = img.size
        it2 = dict(it); it2["image"] = img
        for src in SOURCES:
            content = clevr3_user_content(it2, src, n, a.mode, a.seed, method=method)
            gold = it["channels"][src]
            target = f"I used the requested source to answer the question. final answer: {gold}"
            umsg = [{"role": "user", "content": content}]
            fmsg = umsg + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]
            enc = _encode(processor, fmsg, device)
            plen = _prompt_len(processor, umsg, device)
            with torch.no_grad():
                o = model(**{k: v for k, v in enc.items() if k != "labels"}, output_attentions=True)
            ids = enc["input_ids"][0].tolist(); T = len(ids)
            img_pos = [i for i, t in enumerate(ids) if t in img_tok_ids]
            gh, gw, off = token_grid(enc, len(img_pos))
            grid_pos = img_pos[off:off + gh * gw]
            cells = set(box_cells(it["bbox"], W, H, gh, gw))
            imgbox = [grid_pos[c] for c in sorted(cells) if c < len(grid_pos)]
            imgout = [p for p in img_pos if p not in set(imgbox)]
            cap_ids = tok(_blocks(it)["read"], add_special_tokens=False)["input_ids"]
            csp = AP._find_sub(ids, cap_ids)
            cap_pos = list(range(*csp)) if csp else []
            capval = sorted(set(AP._tok_spans_in(ids, tok, str(it["channels"]["read"]),
                                                 cap_pos[0], cap_pos[-1] + 1))) if cap_pos else []
            ctrl_pos = [i for i, t in enumerate(ids) if t in set(getattr(model, "_neo_all_ids", []) or [])]
            vpos = AP._tok_spans_in(ids, tok, str(gold), plen, T)
            known = set(img_pos) | set(cap_pos) | set(ctrl_pos) | {0}
            carrier = [i for i in range(1, plen) if i not in known]
            q_sets = {"ctrl": ctrl_pos, "lastp": [plen - 1], "first": [plen] if plen < T else [],
                      "preval": [vpos[0] - 1] if vpos and vpos[0] - 1 >= 0 else [],
                      "value": vpos, "ans": list(range(plen, T))}
            k_sets = {"img": img_pos, "imgbox": imgbox, "imgout": imgout, "cap": cap_pos,
                      "capval": capval, "ctrlk": ctrl_pos, "bos": [0], "carrier": carrier}
            L = len(o.attentions); Hh = o.attentions[0].shape[1]
            arr = np.zeros((L, Hh, len(CH)), dtype=np.float32)
            qk = [(q_sets[q], k_sets[k]) for q in QUERIES for k in KEYS]
            for li, A in enumerate(o.attentions):
                am = A[0].float().cpu().numpy()          # heads x T x T
                for ci, (qp, kp) in enumerate(qk):
                    arr[li, :, ci] = AP.mass(am, qp, kp) if (qp and kp) else np.nan
            groups.setdefault(src, []).append(arr)
            recs.append({"id": it["id"], "skill": it["skill"], "op": src, "T": T, "plen": plen,
                         "n_img": len(img_pos), "n_imgbox": len(imgbox), "n_cap": len(cap_pos),
                         "n_capval": len(capval), "n_ctrl": len(ctrl_pos), "grid": [gh, gw],
                         "bbox_share_of_image": round(len(imgbox) / max(1, len(img_pos)), 4)})
    summary = {"channels": CH, "mode": a.mode, "method": method, "n_items": len(items),
               "note": "Ground-CF attention with an evidence-box split of the image tokens; "
                       "answers teacher-forced to the trained gold target; imgbox/imgout partition img."}
    for src, arrs in groups.items():
        A_ = np.stack(arrs)                               # items x L x H x CH
        np.savez(os.path.join(out_dir, f"{src}_all.npz"), attn=A_)
        mean = np.nanmean(A_, axis=(0, 2))                # L x CH (mean over items and heads)
        summary[f"{src}_all"] = {"n": int(A_.shape[0]),
                                 "layer_mean": {ch: [round(float(x), 5) for x in mean[:, ci]]
                                                for ci, ch in enumerate(CH)}}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"[attn-gcf] wrote {out_dir} ({len(recs)} records, {len(items)} items, grid {recs[0]['grid'] if recs else '?'})")


if __name__ == "__main__":
    main()
