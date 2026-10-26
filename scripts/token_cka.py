#!/usr/bin/env python
"""Upgraded token-identity analysis: does FUNCTION converge when GEOMETRY doesn't?

Mean-vector cosine is insufficient (sparse/permuted wiring can implement the
same computation in different bases). Three graded measures per run pair, for
the same functional role (e.g. <look> across seeds, src_look across designs):
  1. BEHAVIORAL agreement: per-item forced-choice agreement between the two
     runs' tokens on the SAME conflict items (does function actually converge).
  2. STATE CKA: linear CKA between the hidden states the token induces at layer
     L across the SAME matched prompts -- basis-free computation similarity.
  3. Procrustes-aligned row-space cosine: optimal orthogonal alignment of the
     two L-row blocks before comparing (kills permutation/rotation artifacts).
One base model is loaded once; each run's rows are hot-swapped by name.
Outcomes: (beh high, CKA high, cos low) = same circuit / different key;
(beh high, CKA low) = multiple circuits; (beh low) = seed variance is
functional -- each outcome is a distinct paper claim.
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import token_strings
from scsrc import prompts as P, score as SC, model as M
from scsrc.collate import _encode

COLORS = {"red","orange","yellow","green","blue","purple","pink","brown","black",
          "white","gray","grey","gold","silver","tan","beige","violet","cyan"}


def norm(k):
    return {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
            for o, v in (k or {}).items()}


def load_rows(path):
    ck = torch.load(os.path.join(path, "neo.pt"), map_location="cpu")
    return ck  # rows, ids, name_to_ids


def cka(X, Y):
    """Linear CKA between [n, d] state matrices."""
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).norm() ** 2
    xx = (X.T @ X).norm(); yy = (Y.T @ Y).norm()
    return float(xy / (xx * yy).clamp_min(1e-9))


def procrustes_cos(A, B):
    """Optimal orthogonal map of A's rows onto B's, then mean cosine."""
    U, _, Vt = torch.linalg.svd(A.T @ B)
    R = U @ Vt
    return float(F.cosine_similarity(A @ R, B, dim=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="semicolon list: runA/ckpt/step_800:tokA,runB/ckpt/step_800:tokB;...")
    ap.add_argument("--data_dir", default="data/color")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--max_n", type=int, default=40)
    a = ap.parse_args()

    model, proc = M.load()
    model.eval()
    tokzr = proc.tokenizer
    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",))
             if s.cond == "conflict"]

    def install(ck, name):
        """Add this token's strings, load its rows; return the trigger string."""
        n = len(ck["name_to_ids"][name])
        strs = token_strings(name, n)
        tokzr.add_special_tokens({"additional_special_tokens": strs})
        model.resize_token_embeddings(len(tokzr), mean_resizing=False)
        ids = tokzr.convert_tokens_to_ids(strs)
        rowmap = {i: r for i, r in zip(ck["ids"], ck["rows"])}
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            for i, src_id in zip(ids, ck["name_to_ids"][name]):
                emb[i] = rowmap[src_id].to(emb.device, emb.dtype)
        return "".join(strs), ids

    @torch.no_grad()
    def probe(trig, tok_ids):
        """(choices, states): per-item image-vs-text pick + layer-L state at the
        token block's LAST position, over shared items."""
        picks, states = [], []
        seen = 0
        for sp in specs:
            ik, tk = norm(sp.image_key), norm(sp.text_key)
            if not ik or set(ik) != set(tk):
                continue
            conf = [o for o in ik if ik[o] != tk[o]]
            if not conf:
                continue
            if seen >= a.max_n:
                break
            seen += 1
            obj = conf[seen % len(conf)]   # per-object (multi-object ColorSwap)
            img = get_image(ds, sp)
            u = P.render_ctrl(P.CTRL_CANONICAL, trig, sp.fed_caption)
            ci = P.fmt_key_task({obj: ik[obj]}, "color")
            ct = P.fmt_key_task({obj: tk[obj]}, "color")
            li, _, _ = SC.loglik(model, proc, img, u, ci, "cuda")
            lt, _, _ = SC.loglik(model, proc, img, u, ct, "cuda")
            picks.append(int(li > lt))
            umsg = [{"role": "user",
                     "content": [{"type": "image", "image": img},
                                 {"type": "text", "text": u}]}]
            enc = _encode(proc, umsg, "cuda")
            out = model(**{k: v for k, v in enc.items() if k != "token_type_ids"},
                        output_hidden_states=True, use_cache=False)
            ids_row = enc["input_ids"][0]
            pos = (ids_row == tok_ids[-1]).nonzero().squeeze(-1)
            h = out.hidden_states[a.layer][0, pos[-1]]
            states.append(h.float().cpu())
        return picks, torch.stack(states)

    results = []
    for pair in a.pairs.split(";"):
        (pa, ta), (pb, tb) = [p.rsplit(":", 1) for p in pair.split(",")]
        cka_, beh, pc, cos_ = None, None, None, None
        ckA, ckB = load_rows(pa), load_rows(pb)
        trigA, idsA = install(ckA, ta)
        picksA, statesA = probe(trigA, idsA)
        trigB, idsB = install(ckB, tb)
        picksB, statesB = probe(trigB, idsB)
        beh = sum(int(x == y) for x, y in zip(picksA, picksB)) / max(len(picksA), 1)
        cka_ = cka(statesA, statesB)
        A = torch.stack([r.float() for i, r in zip(ckA["ids"], ckA["rows"])
                         if i in ckA["name_to_ids"][ta]])
        B = torch.stack([r.float() for i, r in zip(ckB["ids"], ckB["rows"])
                         if i in ckB["name_to_ids"][tb]])
        cos_ = float(F.cosine_similarity(A.mean(0, keepdim=True),
                                         B.mean(0, keepdim=True)))
        pc = procrustes_cos(A, B) if A.shape == B.shape else None
        r = {"pair": pair, "behavioral_agreement": round(beh, 3),
             "state_cka_L%d" % a.layer: round(cka_, 3),
             "mean_cosine": round(cos_, 3),
             "procrustes_cosine": (round(pc, 3) if pc is not None else None),
             "n_items": len(picksA)}
        print(json.dumps(r)); results.append(r)

    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/token_cka.json", "w"), indent=1)
    print("[saved] results/token_cka.json")


if __name__ == "__main__":
    main()
