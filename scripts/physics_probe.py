#!/usr/bin/env python
"""Three physics-style probes of a learned source-control word.

  --mode fdt         FLUCTUATION-RESPONSE. Per item, the decision variable is
                     D = LL(image value) - LL(caption value). Measured under
                     three prompts on ONE checkpoint:
                       neutral : bare question, NO control token
                       look    : <look_ctrl>
                       read    : <read_ctrl>
                     Because the method only ADDS vocabulary rows, the neutral
                     condition is bit-identical to the frozen base model -- so
                     var(D_neutral) is the model's EQUILIBRIUM FLUCTUATION and
                     mean(D_look) - mean(D_read) is the RESPONSE to the control
                     field, both from the same forward code path. FDT analogue:
                     is response predicted by baseline fluctuation? If so,
                     steerability is measurable WITHOUT training a word.

  --mode alpha       ORDER PARAMETER. Overwrite the look rows with
                     e(a) = (1-a) e_look + a e_read and sweep a in [0,1],
                     prompting with <look_ctrl> throughout. We report the full
                     per-item DISTRIBUTION of D at each a, not just the mean:
                       unimodal broadening -> continuous crossover (a dial:
                         "weight the image 70%" is an addressable control)
                       bimodal at mid-a    -> first-order transition (words are
                         discrete attractors / programs, not directions)
                     Sarle's bimodality coefficient b=(skew^2+1)/kurtosis,
                     b > 5/9 is the usual bimodality flag.

  --mode curve       STIMULUS-RESPONSE FUNCTION (the honest linear-response
                     measurement -- a DERIVATIVE, not two rates, so it needs no
                     equal-variance Gaussian assumption the way d'/criterion do).
                     Exact twins let us build a CONTINUOUS stimulus intensity by
                     blending the images themselves:
                         I(b) = (1-b) I + b I'
                     b=0: the pixels fully support the image value; b=1: they
                     fully support the caption's value. Sweeping b and measuring
                     D(b) = LL(image value) - LL(caption value) yields, per
                     control condition {neutral, instruction, token}:
                       GAIN  = dP(track image)/db at threshold (susceptibility)
                       b50   = the stimulus intensity at which the source wins
                               = HOW MUCH visual evidence a method needs to beat
                                 the text prior (an interpretable number)
                     Crossed with --mode alpha this is a 2-D phase diagram of
                     lexical control: stimulus intensity x control strength.

  --mode commutator  ORDER EFFECTS. Prompt with BOTH words, in both orders, and
                     compare to each alone:
                       D(look,read) vs D(read,look)  -> commutator
                     A nonzero commutator means the source operators do not
                     compose commutatively -- the natural formal companion to
                     the measured composition null.

Output: runs/<run>/physics/<mode>_<ds>_<split>_step<STEP>.json (+ per-item lists)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
from tqdm import tqdm

from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger, token_strings
from scsrc import prompts as P, score as SC
from evaluate import apply_checkpoint

COLORS = {"red","orange","yellow","green","blue","purple","pink","brown","black",
          "white","gray","grey","gold","silver","tan","beige","violet","cyan",
          "maroon","navy"}


def norm(k):
    return {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
            for o, v in (k or {}).items()}


def moments(v):
    n = len(v)
    if n < 3:
        return {}
    m = sum(v) / n
    d = [x - m for x in v]
    s2 = sum(x * x for x in d) / n
    sd = math.sqrt(max(s2, 1e-12))
    sk = (sum(x ** 3 for x in d) / n) / sd ** 3
    ku = (sum(x ** 4 for x in d) / n) / sd ** 4
    bc = (sk * sk + 1.0) / ku if ku > 1e-9 else float("nan")   # Sarle
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4),
            "var": round(s2, 4), "skew": round(sk, 3), "kurt": round(ku, 3),
            "bimodality_coef": round(bc, 3), "frac_positive": round(
                sum(1 for x in v if x > 0) / n, 3)}


def evid_patches(ia, ib, grid=16, side=896, thresh=10.0):
    """Vision-grid cells where exact twins differ = the discriminative evidence."""
    import numpy as np
    from PIL import Image
    A = np.asarray(ia.convert("RGB").resize((side, side)), np.float32)
    B = np.asarray(ib.convert("RGB").resize((side, side)), np.float32)
    d = np.abs(A - B).mean(-1)
    blk = side // grid
    cell = d.reshape(grid, blk, grid, blk).mean((1, 3))
    idx = list(zip(*np.where(cell > thresh)))
    return idx, grid, blk


def stim(ia, ib, b, kind, cache, idx=0):
    """Graded visual evidence.
    kind='noise' (DEFAULT): additive Gaussian noise of sd = b*128 on the image
      only; the caption is untouched. b is visual DEGRADATION, so the response
      curve measures NOISE TOLERANCE: b50 = the corruption level at which the
      method abandons the image and falls back on the text prior. HIGHER b50 =
      stronger visual grounding (note: the opposite direction to a twin blend).
      Alignment-free, which matters because our twins are NOT pixel-aligned
      (measured: 97% of vision cells differ on ColorSwap, 79% COCO, 43% VCF --
      they are independently generated scenes, so patch/blend interpolation
      would put a checkerboard or a ghost of TWO scenes on the input manifold).
      The noise realisation is seeded per ITEM, so every control condition sees
      the identical corrupted image = a properly paired comparison.
    kind='patch' (default, ON-MANIFOLD): swap a random fraction b of ONLY the
      grid cells where the twins differ -- b is literally 'what fraction of the
      discriminative evidence supports the caption'. The rest of the image is
      untouched, so we never leave the image manifold except inside the
      evidence region itself.
    kind='blend': whole-image pixel interpolation (simpler, but b in the middle
      is a semi-transparent superposition = off-manifold; kept as a robustness
      check)."""
    import numpy as np
    from PIL import Image
    if kind == "noise":
        A = np.asarray(ia.convert("RGB"), np.float32)
        if b <= 0:
            return ia.convert("RGB")
        rng = np.random.default_rng(10_000 + idx)      # per-item, shared across conditions
        x = A + rng.normal(0.0, b * 128.0, A.shape)
        return Image.fromarray(x.clip(0, 255).astype(np.uint8))
    if kind == "blend":
        A = ia.convert("RGB"); B = ib.convert("RGB").resize(A.size)
        x = (1 - b) * np.asarray(A, np.float32) + b * np.asarray(B, np.float32)
        return Image.fromarray(x.clip(0, 255).astype(np.uint8))
    key = id(ia)
    if key not in cache:
        cache[key] = evid_patches(ia, ib)
    idx, grid, blk = cache[key]
    side = grid * blk
    A = np.asarray(ia.convert("RGB").resize((side, side)), np.uint8).copy()
    B = np.asarray(ib.convert("RGB").resize((side, side)), np.uint8)
    if idx:
        rng = np.random.default_rng(1234)
        order = rng.permutation(len(idx))
        k = int(round(b * len(idx)))
        for t_ in order[:k]:
            r, c = idx[t_]
            A[r*blk:(r+1)*blk, c*blk:(c+1)*blk] = B[r*blk:(r+1)*blk, c*blk:(c+1)*blk]
    return Image.fromarray(A)


def logistic_fit(bs, ps):
    """grid-search (slope, mid) for p(b) = 1/(1+exp(slope*(b-mid))); no scipy."""
    best = (None, None, 1e18)
    for mid in [i / 200 for i in range(-40, 241)]:
        for sl in [0.5 * k for k in range(1, 61)]:
            e = sum((p - 1.0 / (1.0 + math.exp(min(60, max(-60, sl * (b - mid)))))) ** 2
                    for b, p in zip(bs, ps))
            if e < best[2]:
                best = (sl, mid, e)
    return {"slope": round(best[0], 2), "b50": round(best[1], 3),
            "sse": round(best[2], 4),
            "gain_at_b50": round(best[0] / 4.0, 3)}   # dp/db at midpoint


def boot_b50(bs, per_item, n_boot=300):
    """bootstrap CI for b50 by resampling ITEMS (paired across b levels)."""
    import random as _r
    N = len(next(iter(per_item.values())))
    rng = _r.Random(7)
    out = []
    for _ in range(n_boot):
        sel = [rng.randrange(N) for _ in range(N)]
        ps = [sum(1 for i in sel if per_item[f"{b}"][i] > 0) / N for b in bs]
        f = logistic_fit(bs, ps)
        if f["b50"] is not None:
            out.append(f["b50"])
    out.sort()
    if len(out) < 20:
        return None, None
    return round(out[int(.025 * len(out))], 3), round(out[int(.975 * len(out))], 3)


def items(ds, split, task, data_dir, max_n):
    out = []
    is_vcf = "vcf" in data_dir
    specs = [s for s in build_specs(ds, split, conditions=("conflict",))
             if s.cond == "conflict"]
    k = 0
    for sp in specs:
        ik, tk = norm(sp.image_key), norm(sp.text_key)
        if not ik or set(ik) != set(tk):
            continue
        conf = [o for o in ik if ik[o] != tk[o]
                and not (is_vcf and any(c in o.split() for c in COLORS))]
        if not conf:
            continue
        k += 1
        if k > max_n:
            break
        obj = conf[k % len(conf)]
        out.append({"img": get_image(ds, sp), "cap": sp.fed_caption,
                    "imgp": ds[sp.ex_index][f"image_{3 - sp.img_idx}"],
                    "ci": P.fmt_key_task({obj: ik[obj]}, task),
                    "ct": P.fmt_key_task({obj: tk[obj]}, task)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--mode", required=True, choices=["fdt", "alpha", "commutator", "curve", "phase"])
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=70)
    ap.add_argument("--alphas", default="0,0.15,0.3,0.4,0.5,0.6,0.7,0.85,1.0")
    ap.add_argument("--betas", default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0",
                    help="curve/phase mode: graded-evidence levels")
    ap.add_argument("--stimulus", default="noise", choices=["noise", "patch", "blend"],
                    help="noise (DEFAULT) = additive Gaussian on the image only, "
                         "b = degradation, b50 = noise tolerance; patch/blend "
                         "interpolate toward the twin and REQUIRE pixel-aligned "
                         "twins (ours are not -- kept for aligned data only)")
    a = ap.parse_args()
    dev = "cuda"

    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, dev)
    n = meta["n_tokens"]
    ds = load_task(a.data_dir, a.split)
    IT = items(ds, a.split, a.task, a.data_dir, a.max_n)
    print(f"[physics:{a.mode}] {len(IT)} conflict items")

    def D(user, it):
        """decision variable: LL(image value) - LL(caption value)"""
        la, _, _ = SC.loglik(model, proc, it["img"], user, it["ci"], dev)
        lb, _, _ = SC.loglik(model, proc, it["img"], user, it["ct"], dev)
        return la - lb

    from scsrc.prompts import slot_word as _sw
    neutral = lambda it: (f'Caption: "{it["cap"]}". '
                          f"What is each object's {_sw(a.task)}?")
    tok = lambda op, it: P.render_ctrl(P.CTRL_CANONICAL, trigger(f"{op}_ctrl", n), it["cap"])
    out = {"run": os.path.basename(a.run_dir), "mode": a.mode, "step": a.step,
           "dataset": os.path.basename(a.data_dir), "n_tokens": n}

    if a.mode == "fdt":
        rec = {"neutral": [], "look": [], "read": []}
        for it in tqdm(IT, desc="fdt"):
            rec["neutral"].append(D(neutral(it), it))
            rec["look"].append(D(tok("look", it), it))
            rec["read"].append(D(tok("read", it), it))
        out["moments"] = {k: moments(v) for k, v in rec.items()}
        out["per_item"] = {k: [round(x, 4) for x in v] for k, v in rec.items()}
        eq = out["moments"]["neutral"]
        out["equilibrium_fluctuation_var"] = eq.get("var")
        out["response_look_minus_read"] = round(
            out["moments"]["look"]["mean"] - out["moments"]["read"]["mean"], 4)
        out["response_look_minus_neutral"] = round(
            out["moments"]["look"]["mean"] - eq["mean"], 4)
        # per-item susceptibility: does a large |D_neutral| item move MORE?
        pn, pl = rec["neutral"], rec["look"]
        mid = sorted(range(len(pn)), key=lambda i: abs(pn[i]))
        lo, hi = mid[:len(mid)//2], mid[len(mid)//2:]
        f = lambda idx: round(sum(pl[i] - pn[i] for i in idx) / max(len(idx), 1), 4)
        out["response_low_fluct_half"] = f(lo)
        out["response_high_fluct_half"] = f(hi)

    elif a.mode == "alpha":
        W = model.get_input_embeddings().weight
        lid = proc.tokenizer.convert_tokens_to_ids(token_strings("look_ctrl", n))
        rid = proc.tokenizer.convert_tokens_to_ids(token_strings("read_ctrl", n))
        e_look = W.data[lid].detach().clone()
        e_read = W.data[rid].detach().clone()
        out["alpha_curve"] = {}
        for al in [float(x) for x in a.alphas.split(",")]:
            with torch.no_grad():
                W.data[lid] = (1 - al) * e_look + al * e_read
            vals = [D(tok("look", it), it) for it in tqdm(IT, desc=f"a={al}")]
            out["alpha_curve"][f"{al}"] = {**moments(vals),
                                           "per_item": [round(x, 4) for x in vals]}
        with torch.no_grad():
            W.data[lid] = e_look        # restore

    elif a.mode == "curve":
        INSTR = "Answer using ONLY what you can see in the image; ignore the caption."
        conds = {"neutral": lambda it: neutral(it),
                 "instruction": lambda it: f'Caption: "{it["cap"]}". {INSTR}',
                 "token_look": lambda it: tok("look", it),
                 "token_read": lambda it: tok("read", it)}
        betas = [float(x) for x in a.betas.split(",")]
        cache = {}
        out["stimulus"] = a.stimulus
        out["curve"], out["fit"] = {}, {}
        for cname, mk in conds.items():
            per = {}
            for b in betas:
                vals = []
                for _i, it in enumerate(tqdm(IT, desc=f"{cname} b={b}")):
                    im = stim(it["img"], it["imgp"], b, a.stimulus, cache, idx=_i)
                    u = mk(it)
                    la, _, _ = SC.loglik(model, proc, im, u, it["ci"], dev)
                    lb, _, _ = SC.loglik(model, proc, im, u, it["ct"], dev)
                    vals.append(la - lb)
                per[f"{b}"] = vals
            out["curve"][cname] = {k: {**moments(v),
                                       "per_item": [round(x, 4) for x in v]}
                                   for k, v in per.items()}
            ps = [out["curve"][cname][f"{b}"]["frac_positive"] for b in betas]
            fit = logistic_fit(betas, ps)
            lo, hi = boot_b50(betas, per)
            fit.update({"b50_ci95": [lo, hi], "frac_pos_curve": ps,
                        "betas": betas,
                        "dynamic_range": round(ps[0] - ps[-1], 3)})
            out["fit"][cname] = fit

    elif a.mode == "phase":
        # 2-D PHASE DIAGRAM: stimulus intensity (beta) x control strength (alpha)
        # alpha interpolates the look rows toward read, so alpha is the control
        # axis and beta the evidence axis. The b50(alpha) contour is the phase
        # boundary between answering-from-image and answering-from-text.
        W = model.get_input_embeddings().weight
        lid = proc.tokenizer.convert_tokens_to_ids(token_strings("look_ctrl", n))
        rid = proc.tokenizer.convert_tokens_to_ids(token_strings("read_ctrl", n))
        e_look = W.data[lid].detach().clone()
        e_read = W.data[rid].detach().clone()
        betas = [float(x) for x in a.betas.split(",")]
        alphas = [float(x) for x in a.alphas.split(",")]
        cache = {}
        out["stimulus"] = a.stimulus
        out["grid"], out["boundary"] = {}, {}
        for al in alphas:
            with torch.no_grad():
                W.data[lid] = (1 - al) * e_look + al * e_read
            per = {}
            for b in betas:
                vals = []
                for _i, it in enumerate(tqdm(IT, desc=f"a={al} b={b}")):
                    im = stim(it["img"], it["imgp"], b, a.stimulus, cache, idx=_i)
                    u = tok("look", it)
                    la, _, _ = SC.loglik(model, proc, im, u, it["ci"], dev)
                    lb, _, _ = SC.loglik(model, proc, im, u, it["ct"], dev)
                    vals.append(la - lb)
                per[f"{b}"] = vals
            out["grid"][f"{al}"] = {k: {**moments(v)} for k, v in per.items()}
            ps = [out["grid"][f"{al}"][f"{b}"]["frac_positive"] for b in betas]
            out["boundary"][f"{al}"] = {**logistic_fit(betas, ps),
                                        "frac_pos_curve": ps, "betas": betas}
        with torch.no_grad():
            W.data[lid] = e_look
        out["alphas"], out["betas_used"] = alphas, betas

    else:  # commutator
        both = lambda o1, o2, it: P.render_ctrl(
            P.CTRL_CANONICAL,
            trigger(f"{o1}_ctrl", n) + " " + trigger(f"{o2}_ctrl", n), it["cap"])
        rec = {"look": [], "read": [], "look_read": [], "read_look": []}
        for it in tqdm(IT, desc="commutator"):
            rec["look"].append(D(tok("look", it), it))
            rec["read"].append(D(tok("read", it), it))
            rec["look_read"].append(D(both("look", "read", it), it))
            rec["read_look"].append(D(both("read", "look", it), it))
        out["moments"] = {k: moments(v) for k, v in rec.items()}
        out["per_item"] = {k: [round(x, 4) for x in v] for k, v in rec.items()}
        com = [x - y for x, y in zip(rec["look_read"], rec["read_look"])]
        out["commutator"] = moments(com)
        out["commutator_mean_abs"] = round(sum(abs(x) for x in com) / max(len(com), 1), 4)
        # which single word does each ordering resemble? (last-word-wins test)
        for pair in ("look_read", "read_look"):
            dl = sum(abs(x - y) for x, y in zip(rec[pair], rec["look"])) / len(com)
            dr = sum(abs(x - y) for x, y in zip(rec[pair], rec["read"])) / len(com)
            out[f"{pair}_dist_to_look"] = round(dl, 3)
            out[f"{pair}_dist_to_read"] = round(dr, 3)

    outd = os.path.join(a.run_dir, "physics")
    os.makedirs(outd, exist_ok=True)
    p = os.path.join(outd, f"{a.mode}_{os.path.basename(a.data_dir)}_{a.split}_step{a.step}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("per_item", "alpha_curve", "curve", "grid")}, indent=1)[:1600])
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
