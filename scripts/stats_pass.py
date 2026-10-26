#!/usr/bin/env python
"""Statistics pass for the 9-page submission (EXECUTION_SEP Track S).

Produces results/stats_pass.json plus two auto tables:
  paper/iclr2027/tables/tab_stats_main.tex   headline proportions w/ Wilson CI
  paper/iclr2027/tables/tab_mcnemar.tex      paired exact McNemar comparisons

Rules (from the plan): paired tests only where item-level pairing is real;
exact two-sided p; report n everywhere; intervals, never stars. F1 is not a
proportion -> percentile bootstrap (10k) over claims.

Pairings used (all genuinely item-level):
  M1 emitted-vs-posthoc citation faithfulness  -- same generations
     (recomputed per claim from banked citefaith records via the same
     claim_of parser the battery uses)
  M2 word_sft vs pti flips                     -- same items, same split
     (flip_ll records; item order deterministic per split)
  M3 word_sft vs best-instruction-variant flips -- same items
  M4 words vs lexical claim attribution        -- same claims (chunk_id+claim)

Usage: python scripts/stats_pass.py
"""
import glob, json, math, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_citefaith import claim_of, OPS  # same parser as the battery

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "iclr2027", "tables")


def wilson(p, n, z=1.96):
    if n == 0 or p is None:
        return None
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 4)


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


stats = {"proportions": [], "mcnemar": [], "bootstrap": []}


def prop(label, value, n):
    if value is None or not n:
        return
    stats["proportions"].append(
        {"label": label, "value": round(value, 3), "n": n,
         "ci": wilson(value, n)})


# ---------------------------------------------------------------- A. headline proportions
# Ground-CF gemma parity rows (Table 2 of the master; §Installing of main9)
for meth, run, tag in (
        ("Words SFT", "clevr3_f_5tok_sft_seed0", ""),
        ("Words +CTM", "clevr3_f_5tok_ctm_seed0", ""),
        ("Prompt tuning", "clevr3_f_pt_5tok_seed0", ""),
        ("Instruction", "clevr3_f_5tok_sft_seed0", "_instr_iv5")):   # canonical variant (Sep-3 grid); "_instructed" was the retired recipe
    d = load(f"{ROOT}/runs/{run}/eval/clevr3_test_id/flip_step_800{tag}.json")
    if d:
        n = d.get("n") or 150
        for ch in ("look", "read", "ocr"):
            prop(f"Ground-CF {meth} {ch} flip (Gemma)", d.get(f"{ch}_flip"), n)

# ladder gemma parity flips incl. the new no-request row
for meth, run, tag in (
        ("No request", "color_base_frozen_0tok_none_seed0", "_noop"),
        ("Words SFT", "color_neo_ctrl_5tok_sft_seed0", ""),
        ("Words +CTM", "color_neo_ctrl_5tok_ctm_seed0", ""),
        ("Inline PT (pti)", "color_pti_na_5tok_sft_seed0", "")):
    d = load(f"{ROOT}/runs/{run}/eval/color_test_id/flip_ll{tag}_step_800.json")
    if d:
        n = d.get("n") or 140
        for ch in ("look", "read"):
            prop(f"ColorSwap {meth} {ch} flip (Gemma)", d.get(f"{ch}_flip"), n)

# citation faithfulness bars (E4)
for lab, run, tag, key in (
        ("Frozen posthoc faithful (Gemma)", "color_base_frozen_0tok_none_seed0",
         "_frozen", "posthoc_faithful"),
        ("Frozen posthoc faithful (Qwen)", "color_base_frozen_0tok_none_seed0_qwen35",
         "_frozen", "posthoc_faithful"),
        ("Emitted faithful|supp (CTM-link Qwen)",
         "color_neo_link_5tok_ctm_seed0_qwen35_v3", "", "emitted_faithful_sup"),
        ("Emitted faithful|supp (chain Gemma)",
         "color_neo_link_5tok_ctm_genapo_seed0_v2", "", "emitted_faithful_sup")):
    d = load(f"{ROOT}/runs/{run}/eval/color_test_id/citefaith_step_800{tag}.json")
    if d:
        prop(lab, d.get(key), d.get(f"n_{key}") or d.get("n_posthoc_faithful"))

# attribution accuracies (single-attested; F1 gets bootstrap below)
for lab, run, tag in (
        ("Attribution acc (Words CTM, OOD)", "mvr_g_5tok_ctm_seed0c2h2lm", ""),
        ("Attribution acc (lexical, OOD)", "_lexical_baseline", "_lexical"),
        ("Attribution acc (instr Gemma, OOD)", "_frozen_attrib_g3", "_frozen")):
    d = load(f"{ROOT}/runs/{run}/eval/mvraw_test_ood/attrib_step_800{tag}.json")
    if d:
        prop(lab, d.get("acc_single"), d.get("n_single"))

# ---------------------------------------------------------------- B. McNemar
def mc(label, a_vec, b_vec):
    pairs = [(x, y) for x, y in zip(a_vec, b_vec)
             if x is not None and y is not None]
    b_ = sum(1 for x, y in pairs if x == 1 and y == 0)
    c_ = sum(1 for x, y in pairs if x == 0 and y == 1)
    stats["mcnemar"].append(
        {"label": label, "n_pairs": len(pairs), "a_only": b_, "b_only": c_,
         "p_exact": mcnemar_exact(b_, c_)})


# M1: emitted vs posthoc faithfulness, per claim, same generations
def faithful_pairs(path):
    d = load(path)
    if not d:
        return [], []
    em, ph = [], []
    twin_of = {"look": "img_twin", "read": "cap_twin"}
    other = {"look": "read", "read": "look"}
    for r in d["records"]:
        vi, vt = r["v_i"], r["v_t"]
        for op in OPS:
            base = r["gen"].get(f"{op}|orig")
            if base is None:
                continue
            v0, emitted = claim_of(base, op, vi, vt, r.get("obj"))
            if v0 is None or not emitted:
                continue

            def faith(label):
                tc = r["gen"].get(f"{op}|{twin_of[label]}")
                tu = r["gen"].get(f"{op}|{twin_of[other[label]]}")
                if tc is None or tu is None:
                    return None
                vc, _ = claim_of(tc, op, vi, vt, r.get("obj"))
                vu, _ = claim_of(tu, op, vi, vt, r.get("obj"))
                return int((vc is not None and vc != v0) and (vu == v0))
            em.append(faith(op))                       # the emitted label
            ph.append(faith("look" if v0 == "i" else "read"))  # support-matched
    return em, ph


for run in ("color_neo_link_5tok_ctm_seed0_qwen35_v3",
            "color_neo_link_5tok_ctm_genapo_seed0_v2",
            "color_neo_ctrl_5tok_blocks_seed0"):
    e, p = faithful_pairs(f"{ROOT}/runs/{run}/eval/color_test_id/citefaith_step_800.json")
    if e:
        mc(f"M1 emitted vs posthoc faithful ({run.split('_5tok_')[1]})", e, p)

# M2/M3: word vs pti / vs best instruction variant (same items)
def rec(path, key):
    d = load(path)
    return ((d.get("records") or {}).get(key)) if d else None


for ds, ch in (("color", "read"), ("vcf", "look")):
    w = rec(f"{ROOT}/runs/{ds}_neo_ctrl_5tok_sft_seed0/eval/{ds}_test_id/flip_ll_step_800.json",
            f"{ch}_flip")
    t = rec(f"{ROOT}/runs/{ds}_pti_na_5tok_sft_seed0/eval/{ds}_test_id/flip_ll_step_800.json",
            f"{ch}_flip")
    if w and t:
        mc(f"M2 word-SFT vs pti-SFT {ds} {ch} flip (Gemma)", w, t)
    # best instruction variant WITH records
    best, bv = None, -1
    for iv in ("_iv1", "_iv2", "_iv3", "_iv4", "_iv5"):
        d = load(f"{ROOT}/runs/{ds}_base_frozen_0tok_none_seed0/eval/{ds}_test_id/flip_ll{iv}_step_800.json")
        if d and (d.get("records") or {}).get(f"{ch}_flip") and d.get(f"{ch}_flip", -1) > bv:
            best, bv = d["records"][f"{ch}_flip"], d[f"{ch}_flip"]
    wc = rec(f"{ROOT}/runs/{ds}_neo_ctrl_5tok_ctm_seed0/eval/{ds}_test_id/flip_ll_step_800.json",
             f"{ch}_flip")
    if wc and best:
        mc(f"M3 word-CTM vs best-instruction {ds} {ch} flip (Gemma)", wc, best)

# M4: words vs lexical attribution, paired by (chunk_id, claim)
def attrib_map(path):
    d = load(path)
    if not d:
        return {}
    return {(r["chunk_id"], r["claim"]): int(r["pred"] in set(r["attested"]))
            for r in d["records"] if r.get("pred") is not None}


wa = attrib_map(f"{ROOT}/runs/mvr_g_5tok_ctm_seed0c2h2lm/eval/mvraw_test_ood/attrib_step_800.json")
lx = attrib_map(f"{ROOT}/runs/_lexical_baseline/eval/mvraw_test_ood/attrib_step_800_lexical.json")
common = sorted(set(wa) & set(lx))
if common:
    mc("M4 words-CTM vs lexical attribution hit (OOD)",
       [wa[k] for k in common], [lx[k] for k in common])

# ---------------------------------------------------------------- C. bootstrap macro-F1
def macro_f1(records):
    conf = {}
    for r in records:
        att = set(r["attested"])
        if len(att) != 1 or r.get("pred") is None:
            continue
        g = next(iter(att))
        conf.setdefault(g, {}).setdefault(r["pred"], 0)
        conf[g][r["pred"]] += 1
    f1s = []
    chans = ("VIS", "OCR", "ASR")
    for ch in chans:
        tp = conf.get(ch, {}).get(ch, 0)
        fp = sum(conf.get(g, {}).get(ch, 0) for g in chans if g != ch)
        fn = sum(v for p, v in conf.get(ch, {}).items() if p != ch)
        if tp + fn == 0:
            continue
        pr = tp / max(1, tp + fp)
        rc = tp / max(1, tp + fn)
        f1s.append(2 * pr * rc / max(1e-9, pr + rc))
    return sum(f1s) / max(1, len(f1s))


def boot_f1(label, path, tag=""):
    d = load(path)
    if not d:
        return
    recs = [r for r in d["records"] if len(set(r["attested"])) == 1]
    if not recs:
        return
    rng = random.Random(0)
    vals = []
    for _ in range(10000):
        sample = [recs[rng.randrange(len(recs))] for _ in range(len(recs))]
        vals.append(macro_f1(sample))
    vals.sort()
    stats["bootstrap"].append(
        {"label": label, "value": round(macro_f1(recs), 3), "n": len(recs),
         "ci95": (round(vals[249], 3), round(vals[9749], 3)),
         "method": "percentile bootstrap, 10k resamples over claims"})


boot_f1("Attribution macro-F1 (Words CTM, OOD)",
        f"{ROOT}/runs/mvr_g_5tok_ctm_seed0c2h2lm/eval/mvraw_test_ood/attrib_step_800.json")
boot_f1("Attribution macro-F1 (lexical, OOD)",
        f"{ROOT}/runs/_lexical_baseline/eval/mvraw_test_ood/attrib_step_800_lexical.json")
boot_f1("Attribution macro-F1 (instr Gemma, OOD)",
        f"{ROOT}/runs/_frozen_attrib_g3/eval/mvraw_test_ood/attrib_step_800_frozen.json")

# ---------------------------------------------------------------- write
json.dump(stats, open(f"{ROOT}/results/stats_pass.json", "w"), indent=1)

rows = ["\\begin{tabular}{lccc}", "\\toprule",
        "Quantity & Value & $95\\%$ CI & $n$ \\\\", "\\midrule"]
for s in stats["proportions"]:
    ci = f"[{s['ci'][0]}, {s['ci'][1]}]" if s["ci"] else "--"
    rows.append(f"{s['label'].replace(chr(95), chr(92)+chr(95))} & {s['value']} & {ci} & {s['n']} \\\\")
rows.append("\\midrule")
for s in stats["bootstrap"]:
    rows.append(f"{s['label'].replace(chr(95), chr(92)+chr(95))} & {s['value']} & [{s['ci95'][0]}, {s['ci95'][1]}]"
                f" & {s['n']} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/tab_stats_main.tex", "w").write("\n".join(rows) + "\n")

rows = ["\\begin{tabular}{lcccc}", "\\toprule",
        "Paired comparison & $n$ pairs & A-only & B-only & exact $p$ \\\\",
        "\\midrule"]
for s in stats["mcnemar"]:
    rows.append(f"{s['label'].replace(chr(95), chr(92)+chr(95))} & {s['n_pairs']} & {s['a_only']} & "
                f"{s['b_only']} & {s['p_exact']} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/tab_mcnemar.tex", "w").write("\n".join(rows) + "\n")

print(f"[stats] {len(stats['proportions'])} proportions, "
      f"{len(stats['mcnemar'])} McNemar pairs, "
      f"{len(stats['bootstrap'])} bootstrap F1s -> results/stats_pass.json")
