#!/usr/bin/env python
"""CLEVR-3channel factored eval: source routing + zero-shot composition.

For each item and each source, teacher-force the in-format candidate answers
(one per DISTINCT channel value) and argmax by total logP. Metrics:
  control acc  P(<src><op> answers with channels[src]'s value)  -- per cell + overall
  routing M    M[query src][picked src] on n_distinct==3 items (unambiguous)
  selectivity  mean(diag) - mean(off-diag) of the routing matrix
  zero-shot    control acc restricted to held-out (src,op) cells (factored only)
Run on a factored run and a bundled run; the bundled run has no token for held
cells, so its zero-shot cells are (correctly) untestable -> that is the point.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from collections import defaultdict
from scsrc.clevr3 import clevr3_user_content
from scsrc.collate import _encode, _prompt_len
from scsrc.tokens import CLEVR3_SOURCES, c3_sources
from evaluate import apply_checkpoint

SRCS = list(CLEVR3_SOURCES)   # look, read, ocr (module default; main() re-reads the run's protocol)


def cand(v):
    # 2026-09-29 (audit item 1): the evaluation candidate uses the SAME formatter as the training targets; the answer
    # format travels with the checkpoint (SC_GCF_CONT restored by evaluate.apply_checkpoint; leadin for banked runs)
    from scsrc.clevr3 import c3_cont
    return c3_cont(v)


@torch.no_grad()
def loglik(model, proc, umsg, candidate, device):
    fmsg = umsg + [{"role": "assistant", "content": [{"type": "text", "text": candidate}]}]
    enc = _encode(proc, fmsg, device); plen = _prompt_len(proc, umsg, device)
    ids = enc["input_ids"]; n = ids.shape[1] - plen
    if n <= 0:
        return -1e9
    kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    lg = model(**kw, use_cache=False).logits.float()
    tgt = ids[0, plen:]
    lp = torch.log_softmax(lg[0, plen - 1:ids.shape[1] - 1, :], -1)
    return float(lp.gather(-1, tgt.unsqueeze(-1)).sum())


@torch.no_grad()
def loglik_batch(model, proc, umsg, candidates, device):
    """Summed log-likelihood of EVERY candidate continuation in ONE forward pass (STATUS v1.95, 2026-09-22).
    The prompt (image + text) is identical across candidates, so the B sequences are right-padded and stacked;
    causal attention makes right padding inert for the real tokens. Vision tensors are concatenated along dim 0
    (Gemma: [B,C,H,W]; Qwen: [B*patches,D] with image_grid_thw [B,3]). Returns {candidate: (sum, mean, n_tokens)}.
    Numerically the same computation as loglik() per sequence; the evaluators verify pick agreement on their
    first items and record it."""
    cands = list(candidates)
    encs = [_encode(proc, umsg + [{"role": "assistant", "content": [{"type": "text", "text": c}]}], device) for c in cands]
    plen = _prompt_len(proc, umsg, device)
    lens = [e["input_ids"].shape[1] for e in encs]
    T = max(lens); B = len(encs)
    pad_id = proc.tokenizer.pad_token_id if proc.tokenizer.pad_token_id is not None else 0
    ids = torch.full((B, T), pad_id, dtype=encs[0]["input_ids"].dtype, device=device)
    att = torch.zeros((B, T), dtype=encs[0]["attention_mask"].dtype, device=device)
    for b, e in enumerate(encs):
        ids[b, :lens[b]] = e["input_ids"][0]; att[b, :lens[b]] = e["attention_mask"][0]
    kw = {"input_ids": ids, "attention_mask": att}
    for k in encs[0]:
        if k in ("input_ids", "attention_mask", "token_type_ids"):
            continue
        v = encs[0][k]
        if torch.is_tensor(v):
            kw[k] = torch.cat([e[k] for e in encs], dim=0)
    lg = model(**kw, use_cache=False).logits.float()
    lp = torch.log_softmax(lg, -1)
    out = {}
    for b, c in enumerate(cands):
        n = lens[b] - plen
        if n <= 0:
            out[c] = (-1e9, -1e9, 0); continue
        tgt = ids[b, plen:lens[b]]
        tok = lp[b, plen - 1:lens[b] - 1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        ssum = float(tok.sum()); out[c] = (ssum, ssum / n, int(n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--mode", default=None,
                    choices=[None, "factored", "bundled", "srconly",
                             "instructed", "noop"])
    ap.add_argument("--question_style", default=None,
                    choices=[None, "explicit", "generic"],
                    help="default: the style the run was trained with")
    ap.add_argument("--held_cells", default=None)
    ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=300)
    ap.add_argument("--skill_matrix", type=int, default=1,
                    help="factored/bundled only: image-only prompts, candidates "
                         "= the scene's look-values across ALL skills (all true "
                         "of the image) -> only the op token can pick the skill")
    ap.add_argument("--n_tokens", type=int, default=None)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    args = cfg.get("args", {})
    mode = a.mode or args.get("factor_mode", "factored")
    qstyle = a.question_style or args.get("question_style", "explicit")
    n_tokens = a.n_tokens or args.get("n_tokens", 5)
    held = a.held_cells if a.held_cells is not None else args.get("held_cells", "")
    held_cells = {tuple(c.split(":")) for c in held.split(",") if ":" in c}

    model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")

    global SRCS

    SRCS = list(c3_sources())   # two-modality protocol runs evaluate on their two sources (STATUS v1.94)
    items = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))][:a.max_n]

    cell_hit = defaultdict(list)          # (skill, source) -> [0/1]
    held_hit = defaultdict(list)
    M = defaultdict(lambda: defaultdict(int)); tot = defaultdict(int)
    for it in tqdm(items, desc=f"clevr3:{mode}"):
        vals = it["channels"]
        distinct = sorted(set(vals.values()))
        three_way = len(distinct) == 3
        for qs in SRCS:                    # query source
            is_held = (qs, it["skill"]) in held_cells
            if mode == "bundled" and is_held:
                continue                   # no fused token exists for a held cell
            um = [{"role": "user",
                   "content": clevr3_user_content(it, qs, n_tokens, mode, seed=0,
                                                  question_style=qstyle)}]
            scores = {v: loglik(model, proc, um, cand(v), "cuda") for v in distinct}
            pick = max(scores, key=scores.get)
            hit = int(pick == vals[qs])
            (held_hit if is_held else cell_hit)[(it["skill"], qs)].append(hit)
            if three_way:                  # unambiguous value->source for the matrix
                picked_src = next(s for s in SRCS if vals[s] == pick)
                M[qs][picked_src] += 1; tot[qs] += 1

    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    cells = {f"{s}:{sk}": m(cell_hit[(sk, s)])
             for (sk, s) in sorted(cell_hit)}
    per_src = {s: m([x for (sk, ss), v in cell_hit.items() if ss == s for x in v])
               for s in SRCS}
    per_skill = {sk: m([x for (ss, s), v in cell_hit.items() if ss == sk for x in v])
                 for sk in sorted({sk for sk, _ in cell_hit})}
    overall = m([x for v in cell_hit.values() for x in v])
    zshot = {f"{s}:{sk}": m(held_hit[(sk, s)]) for (sk, s) in sorted(held_hit)}
    d = [M[q][q] / tot[q] for q in SRCS if tot[q]]
    o = [M[q][p] / tot[q] for q in SRCS for p in SRCS if p != q and tot[q]]
    sel = round((sum(d) / max(len(d), 1)) - (sum(o) / max(len(o), 1)), 3)

    print(f"\n== CLEVR3 {mode}  {os.path.basename(a.run_dir)}  {a.split} step {a.step}")
    print(f"   overall control acc {overall}   per-source {per_src}")
    print(f"   per-skill {per_skill}")
    print("   routing M P(query picks source)  picks: look  read  ocr")
    for q in SRCS:
        row = "  ".join(f"{(M[q][p]/tot[q]):.2f}" if tot[q] else " -- " for p in SRCS)
        print(f"   <{q:4s}>({tot[q]:3d}): {row}")
    print(f"   selectivity(diag-off) {sel}")
    if zshot:
        print(f"   ZERO-SHOT held cells {zshot}  (mean {m([x for v in held_hit.values() for x in v])})")

    # ---- SKILL-SELECTIVITY matrix (the operator-denotation metric) ----------
    # Image-only prompt + generic question + <src_look><op_X>; candidates = the
    # scene's look-values across ALL FOUR skills (all TRUE of the image, typed
    # differently). Nothing but the op token indicates which skill is wanted,
    # so diag>off-diag = the op token DENOTES its skill. Only meaningful for
    # modes with an operator slot (factored/bundled).
    SK = ("count", "color", "object", "spatial")
    M2 = defaultdict(lambda: defaultdict(int)); tot2 = defaultdict(int)
    if a.skill_matrix and mode in ("factored", "bundled"):
        from collections import defaultdict as dd
        by_scene = dd(dict)
        for it in items:
            by_scene[it["id"].rsplit("_", 1)[0]][it["skill"]] = it
        scenes = [v for v in by_scene.values() if len(v) == 4][:60]
        for sc in tqdm(scenes, desc="skill-matrix"):
            looks = {sk: sc[sk]["channels"]["look"] for sk in SK}
            if len(set(looks.values())) < 4:
                continue                       # need 4 distinguishable values
            for q_sk in SK:                    # queried operator
                if ("look", q_sk) in held_cells and mode == "bundled":
                    continue
                um = [{"role": "user",
                       "content": clevr3_user_content(sc[q_sk], "look", n_tokens,
                                                      mode, seed=0,
                                                      question_style="generic",
                                                      blocks_mode="image_only")}]
                scores = {sk: loglik(model, proc, um, cand(looks[sk]), "cuda")
                          for sk in SK}
                picked = max(scores, key=scores.get)
                M2[q_sk][picked] += 1; tot2[q_sk] += 1
        if tot2:
            d2 = [M2[q][q] / tot2[q] for q in SK if tot2[q]]
            o2 = [M2[q][p] / tot2[q] for q in SK for p in SK if p != q and tot2[q]]
            sk_sel = round(sum(d2) / max(len(d2), 1) - sum(o2) / max(len(o2), 1), 3)
            print("   SKILL matrix P(op picks skill)  picks: " + "  ".join(SK))
            for q in SK:
                row = "  ".join(f"{(M2[q][p]/tot2[q]):.2f}" if tot2[q] else " -- " for p in SK)
                print(f"   <op_{q:7s}>({tot2[q]:3d}): {row}")
            print(f"   SKILL selectivity(diag-off) {sk_sel}")

    out = {"run": os.path.basename(a.run_dir), "mode": mode, "split": a.split,
           "step": a.step, "n": len(items), "n_tokens": n_tokens,
           "question_style": qstyle,
           "overall": overall, "per_source": per_src, "per_skill": per_skill,
           "cells": cells, "selectivity": sel,
           "matrix": {q: {p: round(M[q][p] / tot[q], 3) for p in SRCS if tot[q]}
                      for q in SRCS if tot[q]},
           "skill_matrix": {q: {p: round(M2[q][p] / tot2[q], 3) for p in SK}
                            for q in SK if tot2.get(q)},
           "skill_selectivity": (round(
               sum(M2[q][q] / tot2[q] for q in SK if tot2[q]) / max(sum(1 for q in SK if tot2[q]), 1)
               - sum(M2[q][p] / tot2[q] for q in SK for p in SK if p != q and tot2[q])
               / max(sum(3 for q in SK if tot2[q]), 1), 3) if tot2 else None),
           "zero_shot": zshot,
           "zero_shot_mean": m([x for v in held_hit.values() for x in v]) if held_hit else None}
    # filename encodes MODE + question_style so override-mode baselines
    # (instructed/noop/generic) never clobber the trained-mode result
    outd = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}"); os.makedirs(outd, exist_ok=True)
    fname = f"{mode}_{qstyle}_step_{a.step}.json"
    json.dump(out, open(os.path.join(outd, fname), "w"), indent=1)
    print(f"[saved] {outd}/{fname}")


if __name__ == "__main__":
    main()
