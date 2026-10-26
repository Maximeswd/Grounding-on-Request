#!/usr/bin/env python
"""Attention flip/stay: attention and causal margins from the SAME forwards.

The puzzle: causal-objective tokens (ctm/apo) beat English instructions on
flip/stay, yet instructions show LARGER measured attention contrast (even at
max-head granularity). Two hypotheses:
  H_narr : prior attention numbers are narration artifacts (whole-answer
           averages over format-mismatched generations)
  H_use  : attention MASS is intake, not use -- the token gates the readout of
           already-attended content, so its answers can flip under evidence
           swaps while its attention stays put.
This probe decides between them by teacher-forcing MATCHED candidates (the
flip_ll protocol) and recording, per forward, the candidate log-likelihood AND
attention at the commitment positions. Contexts per item (exact twins):
  A = (I, c)   B = (I', c)  [image swap]   C = (I, c')  [caption swap]
Ops: tok_look/tok_read (method-faithful) + instr_look/instr_read (English),
same items, same candidates.

Per (context, op, candidate) forward we record (L, H, 10):
  queries {preval, value} x keys {img, cap, capval, ctrlk, bos}
  preval = position emitting the candidate's value token (causal query)
  capval = the token span of the value the CURRENT caption asserts
plus the continuation log-likelihood. Pre-registered read-outs:
  attn_stay_img   : |mass_img(B) - mass_img(A)| at preval, look op -- LOW while
                    the margin flips supports H_use for the token
  retarget_capval : mass_capval(A) vs mass_capval(C), read op -- does the
                    commitment query follow the caption's asserted value?
  coupling        : per-item Spearman of d(margin) vs d(mass_img) under image
                    swap -- is behavior change mediated by attention change?
Outputs: runs/<run>/attention_flip/<ds>_<split>_step<STEP>/
  {records.jsonl, attn.npz, summary.json}
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from tqdm import tqdm

from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger
from scsrc import prompts as P
from scsrc.collate import _encode, _prompt_len
from evaluate import apply_checkpoint

COLORS = {"red","orange","yellow","green","blue","purple","pink","brown","black",
          "white","gray","grey","gold","silver","tan","beige","violet","cyan",
          "maroon","navy"}
INSTR = {"look": "Answer using ONLY what you can see in the image; ignore the caption.",
         "read": "Answer using ONLY what the caption states; ignore the image."}
QUERIES = ("preval", "value")
KEYS = ("img", "cap", "capval", "ctrlk", "bos", "evid", "nonevid")
CH = [f"{q}_{k}" for q in QUERIES for k in KEYS]


def evidence_patches(img_a, img_b, grid=16, side=896, thresh=12.0):
    """FOCUS vs MASS: the region where exact twins differ IS the evidence for
    the probed object. Map |I - I'| onto the vision-token grid (gemma-3:
    896x896 -> 16x16 pooled tokens, 56px blocks) -> evidence token indices."""
    import numpy as _np
    a = _np.asarray(img_a.convert("RGB").resize((side, side)), dtype=_np.float32)
    b = _np.asarray(img_b.convert("RGB").resize((side, side)), dtype=_np.float32)
    d = _np.abs(a - b).mean(-1)
    blk = side // grid
    cell = d.reshape(grid, blk, grid, blk).mean((1, 3))     # (16,16)
    idx = _np.where(cell.flatten() > thresh)[0].tolist()
    return idx, round(len(idx) / (grid * grid), 3)


def norm(k):
    return {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
            for o, v in (k or {}).items()}


def _find_sub(hay, needle, start=0):
    for i in range(start, len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i, i + len(needle)
    return None


def _val_spans(ids, tokenizer, text, lo, hi):
    out = []
    for cand in (text, " " + text, text.capitalize(), " " + text.capitalize()):
        t = tokenizer(cand, add_special_tokens=False)["input_ids"]
        j = lo
        while True:
            sp = _find_sub(ids, t, start=j)
            if not sp or sp[0] >= hi:
                break
            out.extend(range(*sp)); j = sp[1]
    return sorted(set(out))


def mass(a, q, k):
    if not q or not k:
        return None
    return a[:, q][:, :, k].sum(-1).mean(-1)


@torch.no_grad()
def forward_attn(model, proc, image, user, cont, device, img_tok_ids, new_ids,
                 cap_text, capval_text, val_text, tokenizer, evid_idx=None):
    umsg = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": user}]}]
    fmsg = umsg + [{"role": "assistant", "content": [{"type": "text", "text": cont}]}]
    enc = _encode(proc, fmsg, device)
    plen = _prompt_len(proc, umsg, device)
    ids = enc["input_ids"][0].tolist()
    T = len(ids)
    kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    o = model(**kw, output_attentions=True, use_cache=False)
    lg = o.logits.float()
    tgt = enc["input_ids"][0, plen:]
    lp = torch.log_softmax(lg[0, plen - 1:T - 1, :], -1)
    ll = float(lp.gather(-1, tgt.unsqueeze(-1)).sum())
    img_pos = [i for i, t in enumerate(ids) if t in img_tok_ids]
    cap_ids = tokenizer(cap_text, add_special_tokens=False)["input_ids"]
    csp = _find_sub(ids, cap_ids)
    cap_pos = list(range(*csp)) if csp else []
    capval = (_val_spans(ids, tokenizer, capval_text, cap_pos[0], cap_pos[-1] + 1)
              if cap_pos and capval_text else [])
    ctrlk = [i for i, t in enumerate(ids) if t in set(new_ids or [])]
    vpos = _val_spans(ids, tokenizer, val_text, plen, T) if val_text else []
    q_sets = {"preval": [vpos[0] - 1] if vpos else [],
              "value": vpos}
    evid_pos, nonevid_pos = [], []
    if evid_idx and len(img_pos) >= 4:
        # map grid indices onto image token positions (raster order); rescale
        # if the token count differs from 256 (defensive for other models)
        scale = len(img_pos) / 256.0
        e = sorted({min(len(img_pos) - 1, int(i * scale)) for i in evid_idx})
        evid_pos = [img_pos[i] for i in e]
        nonevid_pos = [p for p in img_pos if p not in set(evid_pos)]
    k_sets = {"img": img_pos, "cap": cap_pos, "capval": capval,
              "ctrlk": ctrlk, "bos": [0], "evid": evid_pos,
              "nonevid": nonevid_pos}
    L = len(o.attentions); H = o.attentions[0].shape[1]
    arr = np.zeros((L, H, len(CH)), dtype=np.float32)
    for li, A in enumerate(o.attentions):
        a = A[0].float().cpu().numpy()
        for gi, (q, k) in enumerate((q_sets[q_], k_sets[k_])
                                    for q_ in QUERIES for k_ in KEYS):
            m = mass(a, q, k)
            if m is not None:
                arr[li, :, gi] = m
    del o
    torch.cuda.empty_cache()
    return ll, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=40)
    a = ap.parse_args()
    device = "cuda"

    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, device)
    n, method = meta["n_tokens"], meta["method"]
    tokenizer = proc.tokenizer
    img_tok_ids = set()
    for tk in ("image_token", "boi_token"):
        t = getattr(tokenizer, tk, None)
        if t:
            try:
                img_tok_ids.add(tokenizer.convert_tokens_to_ids(t))
            except Exception:
                pass
    new_ids = getattr(model, "_neo_all_ids", None)

    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",))
             if s.cond == "conflict"]
    is_vcf = "vcf" in a.data_dir

    def prompt(kind, op, cap):
        if kind == "tok":
            return P.render_ctrl(P.CTRL_CANONICAL, trigger(f"{op}_ctrl", n), cap)
        if kind == "comb":   # additivity: token prompt + English instruction
            return (P.render_ctrl(P.CTRL_CANONICAL, trigger(f"{op}_ctrl", n), cap)
                    + " " + INSTR[op])
        return f'Caption: "{cap}". {INSTR[op]}'

    recs, arrs = [], []
    n_seen = 0
    for sp_ in tqdm(specs, desc=f"attnflip({os.path.basename(a.run_dir)})"):
        ik, tk = norm(sp_.image_key), norm(sp_.text_key)
        if not ik or set(ik) != set(tk):
            continue
        conf = [o for o in ik if ik[o] != tk[o]
                and not (is_vcf and any(c in o.split() for c in COLORS))]
        if not conf:
            continue
        if n_seen >= a.max_n:
            break
        n_seen += 1
        obj = conf[n_seen % len(conf)]
        v_i, v_t = ik[obj], tk[obj]
        img = get_image(ds, sp_)
        imgp = ds[sp_.ex_index][f"image_{3 - sp_.img_idx}"]
        capp = ds[sp_.ex_index][f"caption_{sp_.img_idx}"]
        ci = P.fmt_key_task({obj: v_i}, a.task)
        ct = P.fmt_key_task({obj: v_t}, a.task)
        evid_idx, evid_frac = evidence_patches(img, imgp)
        # asserted caption value: fed caption asserts v_t; twin caption asserts v_i
        ctxs = {"A": (img, sp_.fed_caption, v_t), "B": (imgp, sp_.fed_caption, v_t),
                "C": (img, capp, v_i)}
        for kind in ("tok", "instr", "comb"):
            for op in ("look", "read"):
                for cname, (image, cap, capval_v) in ctxs.items():
                    for cand, vv in (("ci", v_i), ("ct", v_t)):
                        u = prompt(kind, op, cap)
                        ll, arr = forward_attn(model, proc, image, u,
                                               ci if cand == "ci" else ct,
                                               device, img_tok_ids, new_ids,
                                               cap, capval_v, vv, tokenizer,
                                               evid_idx=evid_idx)
                        recs.append({"i": n_seen - 1, "obj": obj, "kind": kind,
                                     "op": op, "ctx": cname, "cand": cand,
                                     "evid_frac": evid_frac,
                                     "ll": round(ll, 4)})
                        arrs.append(arr)

    out_dir = os.path.join(a.run_dir, "attention_flip",
                           f"{os.path.basename(a.data_dir)}_{a.split}_step{a.step}")
    os.makedirs(out_dir, exist_ok=True)
    A = np.stack(arrs)                                   # (R, L, H, 10)
    np.savez_compressed(os.path.join(out_dir, "attn.npz"), attn=A)
    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    # ---------- in-script summary ----------
    L = A.shape[1]
    mid = slice(L // 3, 2 * L // 3)
    idx = {}
    for j, r in enumerate(recs):
        idx[(r["i"], r["kind"], r["op"], r["ctx"], r["cand"])] = j

    def m_ch(j, ch):
        return float(A[j, mid, :, CH.index(ch)].mean())

    def margin(i, kind, op, ctx):
        """LL(ci) - LL(ct) in this context."""
        ja, jb = idx.get((i, kind, op, ctx, "ci")), idx.get((i, kind, op, ctx, "ct"))
        return None if ja is None or jb is None else recs[ja]["ll"] - recs[jb]["ll"]

    def cmass(i, kind, op, ctx, ch):
        js = [idx.get((i, kind, op, ctx, c)) for c in ("ci", "ct")]
        js = [j for j in js if j is not None]
        return None if not js else float(np.mean([m_ch(j, ch) for j in js]))

    items = sorted({r["i"] for r in recs})
    summ = {"run": os.path.basename(a.run_dir), "n_items": len(items),
            "channels": CH}
    for kind in ("tok", "instr", "comb"):
        # behavioral anchors (same-forward margins)
        lookA = [margin(i, kind, "look", "A") for i in items]
        lookB = [margin(i, kind, "look", "B") for i in items]
        readA = [margin(i, kind, "read", "A") for i in items]
        readC = [margin(i, kind, "read", "C") for i in items]
        ok = [i for i, (x, y) in enumerate(zip(lookA, lookB)) if x is not None and y is not None]
        flipL = [lookA[i] > 0 and lookB[i] < 0 for i in ok]
        okr = [i for i, (x, y) in enumerate(zip(readA, readC)) if x is not None and y is not None]
        flipR = [readA[i] < 0 and readC[i] > 0 for i in okr]
        # attention stay / retarget at preval
        dimg = [abs(cmass(i, kind, "look", "B", "preval_img") - cmass(i, kind, "look", "A", "preval_img"))
                for i in items if cmass(i, kind, "look", "B", "preval_img") is not None
                and cmass(i, kind, "look", "A", "preval_img") is not None]
        lvl = [cmass(i, kind, "look", "A", "preval_img") for i in items]
        lvl = [x for x in lvl if x is not None]
        rvA = [cmass(i, kind, "read", "A", "preval_capval") for i in items]
        rvC = [cmass(i, kind, "read", "C", "preval_capval") for i in items]
        rvA = [x for x in rvA if x is not None]; rvC = [x for x in rvC if x is not None]
        # coupling: d(margin) vs d(mass_img) under image swap, look op
        dm, da = [], []
        for i in items:
            mA, mB = margin(i, kind, "look", "A"), margin(i, kind, "look", "B")
            aA = cmass(i, kind, "look", "A", "preval_img")
            aB = cmass(i, kind, "look", "B", "preval_img")
            if None not in (mA, mB, aA, aB):
                dm.append(mB - mA); da.append(aB - aA)
        coup = None
        if len(dm) >= 8:
            rx = np.argsort(np.argsort(dm)); ry = np.argsort(np.argsort(da))
            coup = round(float(np.corrcoef(rx, ry)[0, 1]), 3)
        # FOCUS: evidence-patch share of image attention at preval (look, ctx A)
        fnum = [cmass(i, kind, "look", "A", "preval_evid") for i in items]
        fden = [cmass(i, kind, "look", "A", "preval_img") for i in items]
        focus = [x / y for x, y in zip(fnum, fden) if x is not None and y and y > 1e-6]
        # calibration-free causal effect: margin(A) - margin(B), format cancels
        dmarg = [margin(i, kind, "look", "A") - margin(i, kind, "look", "B")
                 for i in items
                 if margin(i, kind, "look", "A") is not None
                 and margin(i, kind, "look", "B") is not None]
        summ[kind] = {
            "focus_evid_share": round(float(np.mean(focus)), 4) if focus else None,
            "look_causal_margin_effect": round(float(np.mean(dmarg)), 3) if dmarg else None,
            "look_margin_flip": round(float(np.mean(flipL)), 3) if flipL else None,
            "read_margin_flip": round(float(np.mean(flipR)), 3) if flipR else None,
            "attn_stay_img_absdelta": round(float(np.mean(dimg)), 4) if dimg else None,
            "attn_img_level_A": round(float(np.mean(lvl)), 4) if lvl else None,
            "retarget_capval_A": round(float(np.mean(rvA)), 4) if rvA else None,
            "retarget_capval_C": round(float(np.mean(rvC)), 4) if rvC else None,
            "coupling_dmargin_dattn": coup}
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(f"[attn-flip] wrote {out_dir}")


if __name__ == "__main__":
    main()
