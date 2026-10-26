#!/usr/bin/env python
"""DIAGNOSTIC ONLY since v1.172 (2026-09-24 18:54, author, before any mixed result): the adoption rule of STATUS v1.169
is WITHDRAWN -- the original word stays the method and the mixed-placement word is an appendix diagnostic. This script
still reports the two paired comparisons the rule used. CPU only.
Per model, Ground-CF test_id held-out questions, <look> Flip, items paired by id:
  (a) native position: one-sided 95% lower bound of mean(mixed - native) over a 10,000-draw item bootstrap (seed 0) > -.02
  (b) early position: mixed (`_cp4_ho_reloc_first`) vs the native word moved there with identical prompts
      (`_cp4_ho_reloc_first_hs`): exact two-sided McNemar p < .05 with mixed ahead.
The mixed-placement word becomes the headline word only if (a) and (b) hold on at least 2 of the 3 models.
Writes results/adoption_rule.json. Usage: python scripts/adopt_mixed.py
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from position_analysis import ROOT, MODELS, load
from sft_vs_ctm import per_item, mcnemar

def flips(d):
    return {k: v[0] for k, v in per_item(d, "look").items() if v[0] is not None}

def main():
    E = "runs/{r}/eval/clevr3_test_id/flip_step_800{t}.json"
    out, passes, decided = {}, 0, 0
    for sfx, ml in MODELS:
        nat, mix = f"clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}", f"clevr3_2s_1tok_ctm_cp_fc_pmix_seed0{sfx}"
        f = {k: load(E.format(r=r, t=t)) for k, (r, t) in {"nat": (nat, "_cp4_ho"), "mix": (mix, "_cp4_ho"),
             "nat_first": (nat, "_cp4_ho_reloc_first_hs"), "mix_first": (mix, "_cp4_ho_reloc_first")}.items()}
        miss = [k for k, v in f.items() if v is None]
        if miss:
            out[ml] = {"pending": miss}; print(f"{ml}: PENDING ({', '.join(miss)})"); continue
        a, b = flips(f["mix"]), flips(f["nat"]); ks = sorted(k for k in a if k in b)
        diffs = [a[k] - b[k] for k in ks]; rng = random.Random(0); n = len(diffs)
        boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
        lb = boot[int(0.05 * 10000)]; ok_a = lb > -0.02
        c, d = flips(f["mix_first"]), flips(f["nat_first"]); ks2 = sorted(k for k in c if k in d)
        n01, n10, p = mcnemar([c[k] for k in ks2], [d[k] for k in ks2])
        mc, md = sum(c[k] for k in ks2) / len(ks2), sum(d[k] for k in ks2) / len(ks2)
        ok_b = p < 0.05 and mc > md
        decided += 1; passes += ok_a and ok_b
        out[ml] = {"a_native_diff": round(sum(diffs) / n, 3), "a_lower95": round(lb, 3), "a_pass": ok_a, "a_n": n,
                   "b_mixed_first": round(mc, 3), "b_native_first": round(md, 3), "b_p": p, "b_mixed_only": n01,
                   "b_native_only": n10, "b_pass": ok_b, "b_n": len(ks2)}
        print(f"{ml}: (a) native-position diff {out[ml]['a_native_diff']:+.3f}, lower 95% {lb:+.3f} -> {'pass' if ok_a else 'FAIL'};"
              f"  (b) early position mixed {mc:.3f} vs native {md:.3f}, p={p:.2g} -> {'pass' if ok_b else 'FAIL'}")
    verdict = ("ADOPT the mixed-placement word" if passes >= 2 else
               "KEEP the native word" if decided - passes > 1 or decided == 3 else "UNDECIDED (results pending)")
    out["verdict"] = verdict; out["models_passing"] = passes; out["models_decided"] = decided
    verdict = "diagnostic only (rule withdrawn v1.172); rule would say: " + verdict; out["verdict"] = verdict
    print("VERDICT:", verdict, f"({passes} of {decided} decided models pass; rule: >= 2 of 3)")
    json.dump(out, open(os.path.join(ROOT, "results", "adoption_rule.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
