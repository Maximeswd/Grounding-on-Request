#!/usr/bin/env python
"""Result (1): does counterfactual training (CTM) add grounding over plain SFT? (v1.170, 2026-09-24). CPU only.
Ground-CF, held-out questions, test_id, same items: word L1 trained with SFT only (`clevr3_2s_1tok_sft_fc_seed0*`) vs
SFT + CTM (`clevr3_2s_1tok_ctm_cp_fc_seed0*`), per model and request; Flip and the conjunctive Stay, with exact McNemar
on items paired by id. Writes results/sft_vs_ctm.json. Usage: python scripts/sft_vs_ctm.py
"""
import json, os, sys
from math import comb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from position_analysis import ROOT, MODELS, load

def per_item(d, req):
    out = {}
    for p in (d.get("picks") or {}).get(req) or []:
        g = p.get("gold") or {}; ok0 = p.get("p0") == g.get("base")
        tw = p.get("p_twin") if req == "look" else p.get("p1", p.get("p_capedit"))
        oth = p.get("p_capedit") if req == "look" else p.get("p_twin")
        gt = g.get("twin") if req == "look" else g.get("capedit"); go = g.get("capedit") if req == "look" else g.get("twin")
        out[p["id"]] = (int(ok0 and tw == gt) if tw is not None and gt is not None else None,
                        int(ok0 and oth == go) if oth is not None and go is not None else None)
    return out

def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y); n10 = sum(1 for x, y in zip(a, b) if y and not x); n = n01 + n10
    p = min(1.0, 2 * sum(comb(n, j) for j in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p

def main():
    bank = {}
    print("Ground-CF, word L1: SFT only vs SFT + CTM  (held-out questions, paired items, exact McNemar)")
    for sfx, ml in MODELS:
        S = load(f"runs/clevr3_2s_1tok_sft_fc_seed0{sfx}/eval/clevr3_test_id/flip_step_800_cp4_ho.json")
        C = load(f"runs/clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}/eval/clevr3_test_id/flip_step_800_cp4_ho.json")
        if S is None or C is None:
            print(f"  {ml}: pending ({'SFT' if S is None else 'CTM'} evaluation missing)"); bank[ml] = None; continue
        for req in ("look", "read"):
            s, c = per_item(S, req), per_item(C, req)
            for j, met in ((0, "Flip"), (1, "Stay")):
                ks = [k for k in s if k in c and s[k][j] is not None and c[k][j] is not None]
                a = [c[k][j] for k in ks]; b = [s[k][j] for k in ks]
                n01, n10, p = mcnemar(a, b)
                rec = {"ctm": round(sum(a) / len(a), 3), "sft": round(sum(b) / len(b), 3), "n": len(ks),
                       "ctm_only": n01, "sft_only": n10, "p": p}
                bank[f"{ml}|{req}|{met}"] = rec
                print(f"  {ml:8s} <{req}> {met:4s}  CTM {rec['ctm']:.3f}  SFT {rec['sft']:.3f}  diff {rec['ctm'] - rec['sft']:+.3f}"
                      f"  (n={len(ks)}; CTM-only {n01}, SFT-only {n10}; p={p:.2g})")
    json.dump(bank, open(os.path.join(ROOT, "results", "sft_vs_ctm.json"), "w"), indent=1)
    print("[saved] results/sft_vs_ctm.json")

if __name__ == "__main__":
    main()
