#!/usr/bin/env python3
"""Every paper/presentation table in one command.

  python scripts/harvest.py                 # print all tables
  python scripts/harvest.py --md out.md     # also write markdown

Tables:
  T1 method comparison   best-neo vs pt / LoRA / frozen+instruction, look & read
                         flips, per (dataset x model) + means. THE headline.
  T2 objective effect    plain SFT -> best causal objective (CTM/CTMv3/APO)
  T3 capacity            look-routing and look-flip vs L per dataset (incl MVG)
  T4 arm x capacity      does the winning arm change with L?
  T5 transfer            src -> tgt routing/flip, incl. ladder -> MVG bridge
  T6 MVG                 selectivity, channel flips, report-generation, RAG
  T7 compare             argmax acc vs calibration-free discrimination/AUC
  T8 mechanism           commitment-position attention, citation sites, SAP,
                         and the attention-vs-behaviour couplings
  T9 verbalization       token vs english vs filler routing
 T10 causal claim ledger what we may and may not assert, with the numbers
All numbers are read from disk; missing cells print as '--' (never invented).
"""
import argparse, glob, json, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ("color", "vcf", "coco")
MODELS = (("", "gemma"), ("_qwen35", "qwen"))
NEO = ("sft", "v3", "ctm", "ctmv3", "apo", "vgm", "srcsyn", "iitL12", "dsite",
       "confonly", "ctmcmp", "blocks", "blocksctm")
CAUSAL = ("ctm", "ctmv3", "apo")
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def j(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def flip(run, ds, split="test_id", step=800):
    return j(f"{ROOT}/runs/{run}/eval/{ds}_{split}/flip_ll_step_{step}.json")


def route(run, ds, split="test_id", step=800):
    return j(f"{ROOT}/runs/{run}/eval/{ds}_{split}/ctrl_ll_step_{step}.json")


def f3(x):
    return "  --" if x is None else f"{x:.3f}"


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


# ---------------------------------------------------------------- T1
def t1():
    say("=" * 104)
    say("T1  METHOD COMPARISON -- causal flip/stay (own dataset, test_id). THE headline table.")
    say("=" * 104)
    say(f"{'dataset/model':14s} {'best neo (arm)':>20s} {'pt':>7s} {'LoRA':>7s} {'frozen+instr':>13s} "
        f"| {'READ neo':>9s} {'pt':>7s} {'frozen':>7s}")
    agg = defaultdict(list)
    for ds in DS:
        for sfx, m in MODELS:
            cand = [(a, flip(f"{ds}_neo_ctrl_5tok_{a}_seed0{sfx}", ds)) for a in NEO]
            cand = [(a, d) for a, d in cand if d]
            if not cand:
                continue
            ba, bd = max(cand, key=lambda t: t[1]["look_flip"])
            ra, rd = max(cand, key=lambda t: t[1]["read_flip"])
            pt = (flip(f"{ds}_pt_na_5tok_sft_seed0{sfx}", ds)
                  or flip(f"{ds}_pt_na_4tok_sft_seed0{sfx}", ds))
            lo = flip(f"{ds}_lora_na_1tok_sft_seed0{sfx}", ds)
            fr = flip("_frozen_gemma" if m == "gemma" else "_frozen_qwen35", ds)
            g = lambda d, k: None if not d else d[k]
            say(f"{ds+'/'+m:14s} {bd['look_flip']:.3f} ({ba:9s}) {f3(g(pt,'look_flip')):>7s} "
                f"{f3(g(lo,'look_flip')):>7s} {f3(g(fr,'look_flip')):>13s} | "
                f"{rd['read_flip']:9.3f} {f3(g(pt,'read_flip')):>7s} {f3(g(fr,'read_flip')):>7s}")
            agg["neo"].append(bd["look_flip"]); agg["neoR"].append(rd["read_flip"])
            agg["pt"].append(g(pt, "look_flip")); agg["ptR"].append(g(pt, "read_flip"))
            agg["lora"].append(g(lo, "look_flip"))
            agg["fr"].append(g(fr, "look_flip")); agg["frR"].append(g(fr, "read_flip"))
    say(f"{'MEAN':14s} {f3(mean(agg['neo'])):>20s} {f3(mean(agg['pt'])):>7s} "
        f"{f3(mean(agg['lora'])):>7s} {f3(mean(agg['fr'])):>13s} | "
        f"{f3(mean(agg['neoR'])):>9s} {f3(mean(agg['ptR'])):>7s} {f3(mean(agg['frR'])):>7s}")
    if mean(agg["neo"]) and mean(agg["fr"]):
        say(f"  -> LOOK advantage over English instructions: {mean(agg['neo'])-mean(agg['fr']):+.3f}"
            f"   READ advantage: {(mean(agg['neoR']) or 0)-(mean(agg['frR']) or 0):+.3f}")
    # matched-parameter LoRA controls
    say("\n  LoRA capacity controls (is the gap budget or conditionality?)")
    for r, lbl in (("color_lora_r1_matched_seed0", "r=1  ~181k params (matched to a 30-token word)"),
                   ("color_lora_na_1tok_sft_seed0", "r=16 2.9M params"),
                   ("color_lora_r64_big_seed0", "r=64 11.6M params")):
        d = flip(r, "color")
        say(f"    {lbl:48s} look_flip={f3(None if not d else d['look_flip'])} "
            f"read_flip={f3(None if not d else d['read_flip'])}")


# ---------------------------------------------------------------- T2
def t2():
    say("\n" + "=" * 104)
    say("T2  OBJECTIVE EFFECT -- plain SFT vs best CAUSAL objective (look_flip)")
    say("=" * 104)
    for ds in DS:
        for sfx, m in MODELS:
            s = flip(f"{ds}_neo_ctrl_5tok_sft_seed0{sfx}", ds)
            c = [(a, flip(f"{ds}_neo_ctrl_5tok_{a}_seed0{sfx}", ds)) for a in CAUSAL]
            c = [(a, d) for a, d in c if d]
            if not (s and c):
                continue
            ba, bd = max(c, key=lambda t: t[1]["look_flip"])
            twins = bd.get("twins", "?")
            say(f"  {ds}/{m:6s} sft={s['look_flip']:.3f} -> {ba}={bd['look_flip']:.3f} "
                f"(delta {bd['look_flip']-s['look_flip']:+.3f})  margins {s['look_margin']:.1f}->"
                f"{bd['look_margin']:.1f}  twins={twins}")


# ---------------------------------------------------------------- T3
def t3():
    say("\n" + "=" * 104)
    say("T3  CAPACITY -- does required token count scale with dataset difficulty?")
    say("=" * 104)
    Ls = (1, 2, 4, 5, 8, 10, 15, 16, 20, 25, 30)
    for sfx, m in MODELS:
        for ds in DS:
            row = [(L, route(f"{ds}_neo_ctrl_{L}tok_v3_seed0{sfx}", ds),
                    flip(f"{ds}_neo_ctrl_{L}tok_v3_seed0{sfx}", ds)) for L in Ls]
            row = [(L, a, b) for L, a, b in row if a or b]
            if not row:
                continue
            say(f"  {ds}/{m} (v3 arm)")
            say("     L        : " + " ".join(f"{L:6d}" for L, _, _ in row))
            say("     route    : " + " ".join(f"{a['ctrl_look']:6.2f}" if a else "    --" for _, a, _ in row))
            say("     look_flip: " + " ".join(f"{b['look_flip']:6.2f}" if b else "    --" for _, _, b in row))
    # MVG capacity (3-way selectivity)
    say("  MVG (real video, selectivity; ctm arm)")
    mv = []
    for L in (1, 2, 5, 10, 20, 30):
        for nm in (f"mvg_neo_ctrl_{L}tok_sftctm_seed0", f"mvg_neo_ctrl_{L}tok_sft_seed0_cs"):
            p = glob.glob(f"{ROOT}/runs/{nm}/eval/mvg3_test_id/*.json")
            if p:
                d = j(p[0])
                if d:
                    mv.append((L, d.get("selectivity_mean", d.get("selectivity"))))
                    break
    if mv:
        say("     L        : " + " ".join(f"{L:6d}" for L, _ in mv))
        say("     selectiv.: " + " ".join(f"{v:6.3f}" if v is not None else "    --" for _, v in mv))


# ---------------------------------------------------------------- T4
def t4():
    say("\n" + "=" * 104)
    say("T4  ARM x CAPACITY -- is the winning arm stable across L?")
    say("=" * 104)
    for ds in DS:
        for sfx, m in MODELS:
            printed = False
            for L in (1, 4, 5, 10, 30):
                row = {}
                for a in ("sft", "v3", "ctm", "apo", "vgm", "srcsyn"):
                    d = flip(f"{ds}_neo_ctrl_{L}tok_{a}_seed0{sfx}", ds)
                    if d:
                        row[a] = d["look_flip"]
                if len(row) >= 2:
                    if not printed:
                        say(f"  {ds}/{m}"); printed = True
                    best = max(row, key=row.get)
                    say(f"    L={L:2d}: " + " ".join(f"{a}={v:.3f}" for a, v in sorted(row.items()))
                        + f"   -> BEST={best}")


# ---------------------------------------------------------------- T5
def t5():
    say("\n" + "=" * 104)
    say("T5  TRANSFER -- source-trained word evaluated on a DIFFERENT dataset")
    say("=" * 104)
    say(f"{'run':34s} {'->tgt':7s} {'route_look':>10s} {'route_read':>10s} {'look_flip':>10s} {'read_flip':>10s}")
    for arm in ("apo", "ctm", "v3", "sft"):
        for sfx, m in MODELS:
            for src in DS:
                r = f"{src}_neo_ctrl_5tok_{arm}_seed0{sfx}"
                native_r, native_f = route(r, src), flip(r, src)
                for tgt in DS:
                    if tgt == src:
                        continue
                    a, b = route(r, tgt), flip(r, tgt)
                    if not (a or b):
                        continue
                    say(f"{src+'_'+arm+'_'+m:34s} {tgt:7s} "
                        f"{f3(None if not a else a['ctrl_look']):>10s} {f3(None if not a else a['ctrl_read']):>10s} "
                        f"{f3(None if not b else b['look_flip']):>10s} {f3(None if not b else b['read_flip']):>10s}")
                if native_r or native_f:
                    say(f"{'  (native reference)':34s} {src:7s} "
                        f"{f3(None if not native_r else native_r['ctrl_look']):>10s} "
                        f"{f3(None if not native_r else native_r['ctrl_read']):>10s} "
                        f"{f3(None if not native_f else native_f['look_flip']):>10s} "
                        f"{f3(None if not native_f else native_f['read_flip']):>10s}")
    say("\n  SYNTHETIC -> REAL VIDEO (zero-shot bridge onto MultiVENT-G)")
    for p in sorted(glob.glob(f"{ROOT}/runs/*/eval/rung3_mvg_test_id/*.json")):
        d = j(p)
        if d:
            say(f"    {p.split('/runs/')[1].split('/')[0][:44]:44s} selectivity={d.get('selectivity')}")


# ---------------------------------------------------------------- T6
def t6():
    say("\n" + "=" * 104)
    say("T6  MULTIVENT-G (real news video)")
    say("=" * 104)
    say("  3-way channel routing (selectivity = diag - off-diag)")
    for p in sorted(glob.glob(f"{ROOT}/runs/*/eval/mvg3_test_id/*.json")):
        d = j(p)
        if d:
            say(f"    {p.split('/runs/')[1].split('/')[0][:48]:48s} sel={d.get('selectivity_mean', d.get('selectivity'))}")
    say("  counterfactual channel flips (within-video twins)")
    for p in sorted(glob.glob(f"{ROOT}/runs/*/eval/mvg_flip/*.json")):
        d = j(p)
        if not d:
            continue
        r = p.split("/runs/")[1].split("/")[0]
        bits = []
        for k in ("vis_value_swap", "desc_swap"):
            if isinstance(d.get(k), dict):
                bits.append(f"{k}.flip={d[k].get('flip')}")
        for k in ("ocr_ablate", "vis_presence"):
            if isinstance(d.get(k), dict):
                bits.append(f"{k}.drop={d[k].get('drop')}")
        say(f"    {r[:48]:48s} " + " ".join(bits))
    say("  report generation (MultiVENT-G paper metrics)")
    for p in sorted(glob.glob(f"{ROOT}/runs/*/eval/mvg_report/*.json")):
        d = j(p)
        if d:
            say(f"    {d.get('run','?')[:36]:36s} mode={d.get('mode','?'):9s} F0={d.get('F0')} "
                f"FC={d.get('FC')} prov={d.get('provenance_acc')} q/vid={d.get('queries_per_video')}")
    say("  retrieval stress (RAG)")
    for p in sorted(glob.glob(f"{ROOT}/runs/*/eval/rag_stress/*.json")):
        d = j(p)
        if d:
            say(f"    {d.get('run','?')[:36]:36s} sel={d.get('selectivity')} "
                f"assert_gold={d.get('assert_gold_rate')} margin_drop={d.get('gold_margin_drop_vs_oracle')}")


# ---------------------------------------------------------------- T7
def t7():
    say("\n" + "=" * 104)
    say("T7  <compare> -- argmax accuracy vs CALIBRATION-FREE discrimination")
    say("=" * 104)
    say(f"{'run':40s} {'cmp_acc':>8s} {'cmp_disc':>9s} {'cmp_auc':>8s} {'margin_gap':>11s}")
    for ds in DS:
        for sfx, _ in MODELS:
            for a in NEO:
                d = flip(f"{ds}_neo_ctrl_5tok_{a}_seed0{sfx}", ds)
                if d and ("cmp_disc" in d or d.get("cmp_acc") is not None):
                    say(f"{ds+'_'+a+sfx:40s} {f3(d.get('cmp_acc')):>8s} {f3(d.get('cmp_disc')):>9s} "
                        f"{f3(d.get('cmp_auc')):>8s} {f3(d.get('cmp_margin_gap')):>11s}")


# ---------------------------------------------------------------- T8
def t8():
    say("\n" + "=" * 104)
    say("T8  MECHANISM")
    say("=" * 104)
    c = j(f"{ROOT}/results/attn2_commitment.json")
    if c:
        say("  commitment-position attention (token vs instruction at MATCHED positions)")
        for r in c["rows"][:24]:
            if r.get("kind") == "ctrl":
                say(f"    {r['run'][:38]:38s} {r['dataset']:6s} tok_preval={f3(r.get('tok_preval_img_mid'))} "
                    f"instr_preval={f3(r.get('instr_preval_img_mid'))} tok_value={f3(r.get('tok_value_img_mid'))} "
                    f"instr_value={f3(r.get('instr_value_img_mid'))} citetag={f3(r.get('citetag_img_contrast_mid'))} "
                    f"citeblk={f3(r.get('citeblk_img_contrast_mid'))}")
            else:
                say(f"    GEN {r['run'][:34]:34s} lookblk_fid={r.get('lookblk_img_fid_mid')} "
                    f"readblk_fid={r.get('readblk_cap_fid_mid')}")
    else:
        say("  (attn2_commitment.json not built yet -- run scripts/attn2_summary.py)")
    say("  attention flip/stay: is behaviour change MEDIATED by attention change?")
    for p in sorted(glob.glob(f"{ROOT}/runs/*/attention_flip/*/summary.json")):
        d = j(p)
        if not d:
            continue
        say(f"    {d.get('run','?')[:34]:34s}")
        for kind in ("tok", "instr", "comb"):
            k = d.get(kind)
            if isinstance(k, dict):
                say(f"      {kind:5s} flipL={k.get('look_margin_flip')} dMargin={k.get('look_causal_margin_effect')} "
                    f"attn_stay={k.get('attn_stay_img_absdelta')} focus_evid={k.get('focus_evid_share')} "
                    f"coupling={k.get('coupling_dmargin_dattn')}")
    say("  SAP-noise ablation: is selective attention NECESSARY for flips?")
    for p in sorted(glob.glob(f"{ROOT}/runs/*/eval/*/sap_flip_step_*.json")):
        d = j(p)
        if not d:
            continue
        say(f"    {d.get('run','?')[:34]:34s}")
        for cond in ("none", "mid", "all"):
            k = d.get(cond)
            if isinstance(k, dict):
                say(f"      {cond:5s} look_acc={k.get('look_acc')} look_flip={k.get('look_flip')} "
                    f"read_flip={k.get('read_flip')} lmarg={k.get('look_margin')} hits={k.get('patch_hits')}")
    ap = j(f"{ROOT}/results/attn_vs_perf.json")
    if ap:
        say(f"  arm-level correlations: contrast~flip={ap.get('spearman_lift_vs_flip')} "
            f"contrast~routing={ap.get('spearman_lift_vs_routing')} (n={ap.get('n_arms_with_both')})")


# ---------------------------------------------------------------- T9
def t9():
    say("\n" + "=" * 104)
    say("T9  VERBALIZATION / word identity (routing under token vs english vs filler)")
    say("=" * 104)
    for p in sorted(glob.glob(f"{ROOT}/runs/*/eval/verbalize/plugin_self_step_800.json")):
        d = j(p)
        if not d:
            continue
        run = p.split("/runs/")[1].split("/")[0]
        for op in ("look", "read"):
            res = (d.get("results") or {}).get(op) or {}
            if res:
                bits = " ".join(f"{k}={v.get('routing')}" for k, v in res.items()
                                if isinstance(v, dict) and "routing" in v)
                say(f"    {run[:38]:38s} {op:5s} {bits}")


# ---------------------------------------------------------------- T10
def t10():
    say("\n" + "=" * 104)
    say("T10 CAUSAL CLAIM LEDGER -- what the evidence licenses")
    say("=" * 104)
    say("  [OK ] interventional evaluation: exact twins implement do(source value); flip/stay")
    say("        are interventional queries. Chance flip = .25.")
    say("  [OK ] interventional training: CTM/APO optimise a counterfactual-twin margin")
    say("        (empirical risk over interventional pairs) -> T2 gains on exact-twin data.")
    say("  [NEG] representational causal abstraction (IIA): reported as a NEGATIVE result.")
    for r in ("color_neo_ctrl_5tok_iitL12_seed0", "color_neo_ctrl_5tok_sft_seed0"):
        d = j(f"{ROOT}/runs/{r}/eval/color_test_id/iia_probe_step_800.json")
        if d:
            il = d.get("iia_look", {}); ic = d.get("iia_look_cond", {})
            best = max(il.values()) if il else None
            say(f"        {r[:40]:40s} iia_look(max)={f3(best)} base_donor_rate="
                f"{f3(d.get('base_look_donor_rate'))} iia_look_cond(max)="
                f"{f3(max(ic.values()) if ic else None)}")
    say("        -> raw IIA tracks the base donor-coincidence rate; competence-conditioned")
    say("           IIA ~ 0. Do NOT claim state-level causal mediation.")
    say("  [??? ] mechanism: pending T8. Until the commitment/citation/SAP numbers land,")
    say("        no 'the word re-allocates attention' claim (instructions move gaze more).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    for f in (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
        try:
            f()
        except Exception as e:
            say(f"[{f.__name__} failed: {type(e).__name__}: {e}]")
    if a.md:
        open(a.md, "w").write("```\n" + "\n".join(OUT) + "\n```\n")
        print(f"\n[saved] {a.md}")


if __name__ == "__main__":
    main()
