#!/usr/bin/env python
"""Section 1 evidence: does correctness establish modality control? (v1.189, 2026-09-25, author). CPU only, read-only.
Claim under test: correctness, and even high confidence in the requested answer, does not establish modality control.

(A) Accuracy vs Flip / Stay per method (no request, English, prefix L1, LoRA r16, word L1), 3 models x {VCF, ColorSwap,
    COCO-CF, Ground-CF}, <look> and <read>, ID and OOD (cells from scripts/core_grid.py, so every number is the grid's).
    Flip and Stay are conjunctive (correct on the original AND follows / stays), so Accuracy - Flip and Accuracy - Stay
    are the shares of items answered correctly whose answer the request does not control.
(B) Item level, same files: among ORIGINALLY CORRECT items, P(Flip fails) and P(Stay fails), overall, by quartile of the
    original margin (log-lik of the requested value minus the best other candidate) and for confident answers
    (candidate-softmax probability of the requested value >= .90 and >= .99).
(C) Examples: the most confident originally-correct answers that fail Flip (English and word; prompt, values, picks).
(D) Evaluation design: the image-blind stub (a scorer that never sees the image) under agreement-resolving twins vs
    conflict-preserving twins on the same Ground-CF items, and the same-checkpoint comparison for trained methods.
Writes results/correctness_vs_control.{md,json}. Usage: python scripts/correctness_vs_control.py
"""
import gzip, json, math, os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import core_grid as CG  # noqa: E402

MODELS, DS, METH = CG.MODELS, CG.DS, CG.METH
STUB = os.path.join(ROOT, "results", "cp_blind_stub_{split}.json")
# v1.189 twin-residue audit (scripts/audit_twin_residue.py): pixels by which a shipped colour/object twin differs from the
# inclusive-bounds regeneration; > 0 = the twin still shows a 1-px edge of the old object
RESIDUE = {}
for _sp, _t in (("test_id", "id"), ("test_ood", "ood")):
    _f = os.path.join(ROOT, "data", "clevr3", f"pairs_look_all_{_t}_clean.jsonl")
    if os.path.exists(_f):
        for _l in open(_f):
            _q = json.loads(_l)
            RESIDUE[(_sp, _q["id"])] = _q.get("residue_px_fixed", 0)


def softmax_p(scores, gold):
    m = max(scores.values()); z = sum(math.exp(v - m) for v in scores.values())
    return math.exp(scores[gold] - m) / z


def margin(scores, gold):
    oth = [v for k, v in scores.items() if k != gold]
    return scores[gold] - max(oth) if oth else None


# ---------------------------------------------------------------- item loaders
def gcf_items(sfx, meth, split):
    e = f"eval/clevr3_{split}/flip_step_800"
    if meth in ("No request", "English"):
        k = "noop" if meth == "No request" else "instr"
        f = {r: f"runs/clevr3_2s_5tok_sft_seed0{sfx}/{e}_cp4_ho_fc_{k}.json" for r in ("look", "read")}
    elif meth == "LoRA r16":
        f = {r: f"runs/clevr3_2s_lora_r16_ctm_cp_fc_{r}_seed0{sfx}/{e}_cp4_ho.json" for r in ("look", "read")}
    else:
        run = {"Prefix L1": f"clevr3_2s_pt_1tok_ctm_cp_fc_seed0{sfx}", "Word L1": f"clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}"}[meth]
        f = {r: f"runs/{run}/{e}_cp4_ho.json" for r in ("look", "read")}
    out = {}
    _pr = os.path.join(ROOT, "results", "clean_twin_promotion.json")
    promoted = json.load(open(_pr))["promoted"] if os.path.exists(_pr) else {}
    for req, p in f.items():
        d = CG.J(p)
        P = ((d or {}).get("picks") or {}).get(req)
        clean = p in promoted            # v1.191: promoted from a corrected-twin rerun -> no residue on any twin
        if not P:
            out[req] = None; continue
        rows = []
        for x in P:
            g = x["gold"]; sc = (x.get("scores") or {}).get("base") or {}
            if g["base"] not in sc:
                continue
            ok0 = x.get("p0") == g["base"]
            if req == "look":
                fl = None if x.get("p_twin") is None else (ok0 and x["p_twin"] == g["twin"])
                st = None if x.get("p_capedit") is None else (ok0 and x["p_capedit"] == g["capedit"])
                picks = {"original": x.get("p0"), "image edit": x.get("p_twin"), "caption edit": x.get("p_capedit")}
            else:
                fl = None if x.get("p1") is None else (ok0 and x["p1"] == g["capedit"])
                st = None if x.get("p_twin") is None else (ok0 and x["p_twin"] == g["twin"])
                picks = {"original": x.get("p0"), "caption edit": x.get("p1"), "image edit": x.get("p_twin")}
            rows.append({"ds": "Ground-CF", "id": x["id"], "skill": x.get("skill"), "ok0": ok0, "flip": fl, "stay": st,
                         "residue": 0 if clean else RESIDUE.get((split, x["id"]), 0),
                         "p_gold": softmax_p(sc, g["base"]), "margin": margin(sc, g["base"]), "picks": picks,
                         "values": x.get("values"), "gold": g, "prompt": x.get("prompt_base"), "file": p})
        out[req] = rows
    return out


LADDER_GOLD = {"look": {"base": "A", "flip": ("imgswap", "B"), "stay": ("capswap", "A")},
               "read": {"base": "C", "flip": ("capswap", "D"), "stay": ("imgswap", "C")}}


def ladder_items(ds, dn, sfx, meth, split):
    e = f"eval/{ds}3_{split}/flip_ll"
    fz = f"runs/{ds}_base_frozen_0tok_none_seed0{sfx}/{e}"
    if meth in ("No request", "English"):
        f = {r: f"{fz}_{'noop' if meth == 'No request' else 'iv5'}_lead_p2_step_800_items.jsonl.gz" for r in ("look", "read")}
    elif meth == "LoRA r16":
        f = {r: f"runs/{ds}_lora_r16_lr2e4_ad_ctm_cp_{r}_seed0{sfx}/{e}_lead_p2_heldout_step_800_items.jsonl.gz" for r in ("look", "read")}
    else:
        run = {"Prefix L1": f"{ds}_pt_na_1tok_ctm_cp_seed0{sfx}", "Word L1": f"{ds}_neo_ctrl_1tok_ctm_cp_seed0{sfx}"}[meth]
        f = {r: f"runs/{run}/{e}_lead_p2_heldout_step_800_items.jsonl.gz" for r in ("look", "read")}
    out = {}
    for req, p in f.items():
        fp = os.path.join(ROOT, p)
        if not os.path.exists(fp):
            out[req] = None; continue
        by = collections.defaultdict(dict)
        for l in gzip.open(fp, "rt"):
            a = json.loads(l)
            if "_meta" in a or a.get("request") != req:
                continue
            by[a["item"]][a["ctx"]] = a
        G = LADDER_GOLD[req]; rows = []
        for it, c in by.items():
            b = c.get("base")
            if not b:
                continue
            sc = {k: v["sum"] for k, v in b["candidates"].items()}
            ok0 = b["pick"] == G["base"]
            fc, fg = G["flip"]; sc_, sg = G["stay"]
            fl = None if fc not in c else (ok0 and c[fc]["pick"] == fg)
            st = None if sc_ not in c else (ok0 and c[sc_]["pick"] == sg)
            rows.append({"ds": dn, "id": it, "ok0": ok0, "flip": fl, "stay": st, "residue": 0, "p_gold": softmax_p(sc, G["base"]),
                         "margin": margin(sc, G["base"]),
                         "picks": {"original": b.get("pick_value"), **{k: c[k].get("pick_value") for k in ("imgswap", "capswap") if k in c}},
                         "values": b.get("values"), "prompt": b.get("prompt"), "file": p})
        out[req] = rows
    return out


def items(dsk, dn, sfx, meth, split):
    return gcf_items(sfx, meth, split) if dsk == "gcf" else ladder_items(dsk, dn, sfx, meth, split)


# ---------------------------------------------------------------- helpers
f3 = CG.f3


def rate(rows, key):
    v = [r for r in rows if r[key] is not None]
    return (sum(1 for r in v if not r[key]) / len(v), len(v)) if v else (None, 0)


def fmt_rate(x):
    return "-" if x[0] is None else f"{x[0]:.2f} (n={x[1]})"


def main():
    bank = {"A": {}, "B": {}, "C": {}, "D": {}}
    md = ["# Does correctness establish modality control?", "",
          "Generated by `scripts/correctness_vs_control.py` (CPU, read-only). Flip = correct on the original AND follows the "
          "requested source's edit; Stay = correct AND unchanged under the competing source's edit. Ladder trained methods on "
          "held-out templates; Ground-CF on held-out questions. Chance for Accuracy: 1/4 (ladder), 1/3 (Ground-CF).", ""]
    for split in ("test_id", "test_ood"):
        CG.SPLIT = split
        sl = "ID" if split == "test_id" else "OOD"
        # (A) cells and macro means
        md += [f"## (A) Accuracy vs Flip / Stay — {sl}", "",
               "Macro mean over the available (model, dataset) cells; gap = Accuracy − Flip / Accuracy − Stay.", "",
               "| method | request | cells | Accuracy | Flip | Stay | Acc − Flip | Acc − Stay |", "|---|---|---|---|---|---|---|---|"]
        cells = {}
        for req in ("look", "read"):
            for m in METH:
                trip = []
                for sfx, ml in MODELS:
                    for ds, dn in DS:
                        if ds == "gcf":
                            d = CG.gcf(sfx, m)[0 if req == "look" else 1]
                        else:
                            ho = m not in ("No request", "English")
                            d = CG.ladder(ds, sfx, m, ho)[0 if req == "look" else 1]
                        c = CG.trip(d, req)
                        if c and None not in c:
                            cells[f"{ml}|{dn}|{m}|{req}"] = c; trip.append(c)
                if trip:
                    a, f, s = (sum(t[i] for t in trip) / len(trip) for i in range(3))
                    bank["A"][f"{sl}|{m}|{req}"] = {"cells": len(trip), "acc": a, "flip": f, "stay": s, "gap_flip": a - f, "gap_stay": a - s}
                    md.append(f"| {m} | <{req}> | {len(trip)} | {f3(a)} | {f3(f)} | {f3(s)} | {f3(a - f)} | {f3(a - s)} |")
                else:
                    md.append(f"| {m} | <{req}> | 0 | pending | | | | |")
        bank["A"][f"{sl}|cells"] = cells
        md += ["", f"English (natural-language instruction) per cell, {sl}: Accuracy / Flip / Stay", "",
               "| model | dataset | <look> | <read> |", "|---|---|---|---|"]
        for sfx, ml in MODELS:
            for ds, dn in DS:
                cc = [cells.get(f"{ml}|{dn}|English|{r}") for r in ("look", "read")]
                md.append(f"| {ml} | {dn} | " + " | ".join("pending" if c is None else "/".join(f3(x) for x in c) for c in cc) + " |")
        # (B) item level
        md += ["", f"## (B) Among originally correct answers: how often does the request fail to control? — {sl}", "",
               "Pooled over models and datasets with item files. p = candidate-softmax probability of the requested value "
               "on the original; margin quartiles within each method x request.", "",
               "| method | request | correct items | P(Flip fails) | P(Stay fails) | P(Flip fails), p ≥ .90 | P(Flip fails), p ≥ .99 | P(Stay fails), p ≥ .99 | Flip fails by margin quartile Q1→Q4 |",
               "|---|---|---|---|---|---|---|---|---|"]
        pool = {}
        for m in METH:
            for req in ("look", "read"):
                rows = []
                for sfx, ml in MODELS:
                    for ds, dn in DS:
                        R = items(ds, dn, sfx, m, split).get(req)
                        if R:
                            for r in R:
                                r["model"] = ml
                            rows += R
                pool[(m, req)] = rows
                cor = [r for r in rows if r["ok0"] and r["margin"] is not None]
                if not cor:
                    md.append(f"| {m} | <{req}> | 0 | pending | | | | | |"); continue
                hi90 = [r for r in cor if r["p_gold"] >= .90]; hi99 = [r for r in cor if r["p_gold"] >= .99]
                srt = sorted([r for r in cor if r["flip"] is not None], key=lambda r: r["margin"]); q = len(srt) // 4
                qs = [srt[i * q:(i + 1) * q] if i < 3 else srt[3 * q:] for i in range(4)]
                qf = [rate(x, "flip") for x in qs]
                bank["B"][f"{sl}|{m}|{req}"] = {"n_correct": len(cor), "flip_fail": rate(cor, "flip"), "stay_fail": rate(cor, "stay"),
                                                "flip_fail_p90": rate(hi90, "flip"), "flip_fail_p99": rate(hi99, "flip"),
                                                "stay_fail_p99": rate(hi99, "stay"), "flip_fail_quartiles": qf,
                                                "quartile_margin_ranges": [[round(x[0]["margin"], 2), round(x[-1]["margin"], 2)] if x else None for x in qs]}
                md.append(f"| {m} | <{req}> | {len(cor)} | {fmt_rate(rate(cor, 'flip'))} | {fmt_rate(rate(cor, 'stay'))} | "
                          f"{fmt_rate(rate(hi90, 'flip'))} | {fmt_rate(rate(hi99, 'flip'))} | {fmt_rate(rate(hi99, 'stay'))} | "
                          + " → ".join("-" if x[0] is None else f"{x[0]:.2f}" for x in qf) + " |")
        # per dataset for English (the instruction baseline)
        md += ["", f"English per dataset (pooled over models), {sl}: correct items / P(Flip fails) / P(Flip fails | p ≥ .99)", "",
               "| dataset | <look> | <read> |", "|---|---|---|"]
        for ds, dn in DS:
            cc = []
            for req in ("look", "read"):
                cor = [r for r in pool[("English", req)] if r["ds"] == dn and r["ok0"] and r["margin"] is not None]
                h = [r for r in cor if r["p_gold"] >= .99]
                bank["B"][f"{sl}|English|{dn}|{req}"] = {"n_correct": len(cor), "flip_fail": rate(cor, "flip"), "flip_fail_p99": rate(h, "flip")}
                cc.append(f"{len(cor)} / {fmt_rate(rate(cor, 'flip'))} / {fmt_rate(rate(h, 'flip'))}" if cor else "pending")
            md.append(f"| {dn} | " + " | ".join(cc) + " |")
        # (C) examples (ID only)
        if split == "test_id":
            md += ["", "## (C) Confident, correct, and not controlled (ID): the most confident originally-correct answers that fail Flip", "",
                   "Residue-free twins only (twin-residue audit) and no COCO-CF items (its noun-swap captions read oddly).", ""]
            for m in ("English", "Word L1"):
                for req in ("look", "read"):
                    fails = sorted([r for r in pool[(m, req)] if r["ok0"] and r["flip"] is False and not r["residue"]
                                    and r["ds"] != "COCO-CF"], key=lambda r: -r["margin"])
                    ex = []
                    for r in fails[:3]:
                        e = {k: r.get(k) for k in ("model", "ds", "id", "skill", "p_gold", "margin", "picks", "values", "gold", "prompt", "file")}
                        ex.append(e)
                        md.append(f"- **{m}, <{req}>, {r['model']} {r['ds']} item {r['id']}**: p(requested value) = {r['p_gold']:.4f}, "
                                  f"margin {r['margin']:.1f}; picks {r['picks']}; values {r['values']}; prompt: `{(r.get('prompt') or '')[:220]}`")
                    bank["C"][f"{m}|{req}"] = ex
        # (D) evaluation design
        md += ["", f"## (D) Agreement-resolving vs conflict-preserving image twins, Ground-CF <look> — {sl}", ""]
        st = CG.J(os.path.relpath(STUB.format(split=split), ROOT))
        if st:
            bank["D"][f"{sl}|stub"] = {k: st.get(k) for k in ("look_acc", "look_flip", "look_flip_agree", "look_agree_n", "look_twin_n")}
            md.append(f"- Image-blind scorer (scores depend on the prompt TEXT only; `scripts/dryrun_eval.py`): Flip under agreement "
                      f"twins {st.get('look_flip_agree'):.3f} (n={st.get('look_agree_n')}) vs conflict-preserving {st.get('look_flip'):.3f} "
                      f"(n={st.get('look_twin_n')}). Agreement twins also set the caption to the new value, so the text changes and a "
                      "scorer that never sees the image can be credited with following it; conflict-preserving twins keep the text "
                      "identical, so any image-blind scorer gets Flip exactly 0.")
        else:
            md.append("- Image-blind scorer: pending")
        for sfx, ml in MODELS:
            row = []
            for m, run in (("Word L1", f"clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}"), ("Prefix L1", f"clevr3_2s_pt_1tok_ctm_cp_fc_seed0{sfx}"),
                           ("LoRA r16", f"clevr3_2s_lora_r16_ctm_cp_fc_look_seed0{sfx}")):
                d = CG.J(f"runs/{run}/eval/clevr3_{split}/flip_step_800_cp4_ho_agree.json")
                if d and d.get("look_flip_agree") is not None:
                    bank["D"][f"{sl}|{ml}|{m}"] = {"agree": d["look_flip_agree"], "cp": d["look_flip"], "n_agree": d.get("look_agree_n"), "n_cp": d.get("look_twin_n")}
                    row.append(f"{m} {d['look_flip_agree']:.3f} / {d['look_flip']:.3f}")
            md.append(f"- {ml}, same checkpoint, Flip agreement / conflict-preserving: " + ("; ".join(row) if row else "pending"))
        md.append("")
    json.dump(bank, open(os.path.join(ROOT, "results", "correctness_vs_control.json"), "w"), indent=1, default=str)
    open(os.path.join(ROOT, "results", "correctness_vs_control.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
