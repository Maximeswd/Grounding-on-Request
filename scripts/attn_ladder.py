#!/usr/bin/env python
"""Attention allocation on the ladder (VCF / ColorSwap / COCO-CF): the two-source version of
scripts/attn_groundcf.py with the same output schema. At the position that emits the value
of the gold answer (teacher forced), mean over heads: mass on the image tokens, the caption
span, the caption's value token(s), the learned word's positions and position 0; the
modality SHARE (image vs caption), caption PRECISION (value token share of caption mass),
per-layer profiles, and a mean 16x16 image heat map per request (Qwen grids are resampled
to 16x16 before averaging). The ladder has no evidence boxes, so no image precision.

Usage: python scripts/attn_ladder.py --run_dir runs/color_neo_ctrl_5tok_ctm_seed0 --task color \
           --data_dir data/color --split test_id [--mode instructed|noop --tag _x]
Output: runs/<run>/eval/<ds>_<split>/attn_ladder_step_<step><tag>.json (+ .npz)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc import prompts as P
from scsrc.collate import _user_content, _encode, _prompt_len
from scsrc.tokens import trigger, token_strings
from evaluate import apply_checkpoint
from scripts.eval_flip_ll import norm, COLORS, INSTR
from scripts.attn_groundcf import find_any

SRC = ("look", "read")


def resample16(g):
    """Nearest-neighbour resample of an (h, w) grid to 16x16."""
    h, w = g.shape
    ri = (np.arange(16) * h / 16).astype(int).clip(0, h - 1)
    ci = (np.arange(16) * w / 16).astype(int).clip(0, w - 1)
    return g[np.ix_(ri, ci)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", default="data/color")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=140)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode instructed/noop needs --tag")
    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
    tok = proc.tokenizer
    n, method = meta["n_tokens"], meta["method"]
    meth = "neo" if a.mode in ("instructed", "noop") else method
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
    print(f"[probe] method={meth} mode={a.mode} L={n} image ids {sorted(img_ids)} model_type={mid} layers={n_hidden}")

    def user(op, cap):
        if a.mode == "noop":
            return f'Caption: "{cap}".'
        if a.mode == "instructed" or meth == "lora":
            return f'Caption: "{cap}". {INSTR[op]}'
        if meth == "pt":
            return trigger(f"pt_{op}", n) + " " + P.pt_template(0).format(cap=cap)
        return P.render_ctrl(P.CTRL_CANONICAL, "".join(token_strings(f"{op}_ctrl", n)), cap)

    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",)) if s.cond == "conflict"]
    is_vcf = "vcf" in a.data_dir
    recs, heat, heat_n, n_seen, L_layers = [], {s: None for s in SRC}, {s: 0 for s in SRC}, 0, None
    for sp in tqdm(specs, desc="attn-ladder"):
        ik, tk = norm(sp.image_key), norm(sp.text_key)
        if not ik or set(ik) != set(tk):
            continue
        conf = [o for o in ik if ik[o] != tk[o] and not (is_vcf and any(c in o.split() for c in COLORS))]
        if not conf:
            continue
        if n_seen >= a.max_n:
            break
        n_seen += 1
        obj = conf[n_seen % len(conf)]
        vals = {"look": str(ik[obj]), "read": str(tk[obj])}
        cand = {"look": P.fmt_key_task({obj: ik[obj]}, a.task), "read": P.fmt_key_task({obj: tk[obj]}, a.task)}
        img = get_image(ds, sp)
        for s in SRC:
            um = [{"role": "user", "content": _user_content(img, user(s, sp.fed_caption))}]
            fmsg = um + [{"role": "assistant", "content": [{"type": "text", "text": cand[s]}]}]
            enc = _encode(proc, fmsg, "cuda"); plen = _prompt_len(proc, um, "cuda")
            kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
            with torch.no_grad():
                o = model(**kw, output_attentions=True, use_cache=False)
            ids = enc["input_ids"][0].tolist(); T = len(ids); L_layers = len(o.attentions)
            img_pos = [k for k, t_ in enumerate(ids) if t_ in img_ids]
            cap_pos = find_any(ids, tok, sp.fed_caption, 0, plen)
            capval = find_any(ids, tok, vals["read"], cap_pos[0], cap_pos[-1] + 1) if cap_pos else []
            ctrl_pos = [k for k, t_ in enumerate(ids) if t_ in ctrl_ids and k < plen]
            vpos = find_any(ids, tok, vals[s], plen, T)
            q = (vpos[0] - 1) if vpos else (T - 1)
            grid = None
            if img_pos:
                if "image_grid_thw" in enc:
                    thw = enc["image_grid_thw"][0].tolist()
                    merge = getattr(getattr(model.config, "vision_config", None), "spatial_merge_size", 2) or 2
                    gh, gw = thw[1] // merge, thw[2] // merge
                    if gh * gw == len(img_pos):
                        grid = (gh, gw)
                if grid is None and len(img_pos) == 256:
                    grid = (16, 16)
            spans = {"img": img_pos, "cap": cap_pos, "capval": capval, "ctrl": ctrl_pos, "bos": [0]}
            prof = {k: [] for k in ("img", "cap", "ctrl")}
            mass = {k: 0.0 for k in spans}
            hm = None; lo, hi = L_layers // 3, 2 * L_layers // 3
            for l in range(L_layers):
                A = o.attentions[l][0].float().mean(0)[q]
                for k, pos in spans.items():
                    m = float(A[pos].sum()) if pos else 0.0
                    mass[k] += m / L_layers
                    if k in prof:
                        prof[k].append(round(m, 5))
                if grid and lo <= l < hi:
                    g = resample16(A[img_pos].reshape(grid).cpu().numpy())
                    hm = g if hm is None else hm + g
            sink_mass = None
            if hm is not None:
                hm = hm / max(1, hi - lo); heat[s] = hm if heat[s] is None else heat[s] + hm; heat_n[s] += 1
                # sink = the strongest image cell (no boxes on the ladder, so always masked)
                gh, gw = grid; r_, c_ = divmod(int(A[img_pos].reshape(grid).cpu().numpy().argmax()), gw)
                sm = 0.0
                for l in range(L_layers):
                    sm += float(o.attentions[l][0].float().mean(0)[q][img_pos[r_ * gw + c_]]) / L_layers
                sink_mass = sm
            tot = mass["img"] + mass["cap"]
            recs.append({"ex": sp.ex_index, "source": s, "T": T, "q": q, "n_img": len(img_pos),
                         "mass": {k: round(v, 5) for k, v in mass.items()},
                         "share": {"img": round(mass["img"] / tot, 4) if tot else None,
                                   "cap": round(mass["cap"] / tot, 4) if tot else None},
                         "prec_cap": round(mass["capval"] / mass["cap"], 4) if capval and mass["cap"] else None,
                         "sink_mass": None if sink_mass is None else round(sink_mass, 5),
                         "share_img_nosink": round((mass["img"] - sink_mass) / (tot - sink_mass), 4)
                         if sink_mass is not None and (tot - sink_mass) > 0 else None,
                         "profile": prof})
            del o
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None
    agg = {}
    for s in SRC:
        rs = [r for r in recs if r["source"] == s]
        if not rs:
            continue
        agg[s] = {"n": len(rs), "share": {k: mean([r["share"][k] for r in rs]) for k in ("img", "cap")},
                  "mass": {k: mean([r["mass"][k] for r in rs]) for k in ("img", "cap", "ctrl", "bos")},
                  "prec_cap": mean([r["prec_cap"] for r in rs]),
                  "share_img_nosink": mean([r.get("share_img_nosink") for r in rs]),
                  "sink_share_of_img": mean([(r["sink_mass"] / r["mass"]["img"]) if r.get("sink_mass") is not None and r["mass"]["img"] else None for r in rs]),
                  "layer_profile": {k: [round(float(np.mean([r["profile"][k][l] for r in rs])), 5)
                                        for l in range(L_layers)] for k in ("img", "cap", "ctrl")}}
        d = agg[s]
        print(f"  request {s:4s}: share img {d['share']['img']} cap {d['share']['cap']} | prec cap {d['prec_cap']} | "
              f"ctrl mass {d['mass']['ctrl']}")
    out = {"meta": {"run": a.run_dir, "step": a.step, "mode": a.mode, "method": meth, "n_items": n_seen,
                    "layers": L_layers, "hidden_layers": n_hidden, "model_type": mid, "task": a.task,
                    "data_dir": a.data_dir, "split": a.split, "instr_variant": os.environ.get("INSTR_VARIANT")},
           "by_source": agg, "records": recs}
    dsname = os.path.basename(a.data_dir.rstrip("/"))
    od = os.path.join(a.run_dir, "eval", f"{dsname}_{a.split}")
    os.makedirs(od, exist_ok=True)
    stem = os.path.join(od, f"attn_ladder_step_{a.step}{a.tag}")
    json.dump(out, open(stem + ".json", "w"), indent=1)
    hm = {s: heat[s] / heat_n[s] for s in SRC if heat[s] is not None and heat_n[s]}
    if hm:
        np.savez_compressed(stem + ".npz", **{s: v.astype(np.float32) for s, v in hm.items()})
    print("[saved]", stem + ".json")


if __name__ == "__main__":
    main()
