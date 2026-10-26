#!/usr/bin/env python
"""Where the request sits relative to the evidence (v1.170, 2026-09-24). CPU only, read-only; missing cells print "pending".

GROUND-CF (held-out questions, test_id, conflict-preserving twins). Every learned control and the English instruction is
placed at one of three positions (SC_GCF_CTRL_POS; native placements: word after the question, prefix first, English
after the question). "between" is split by the item's evidence order (picks[*]["cap_first"]), which gives four CAUSAL
positions per request r with requested source s(r) (image for <look>, caption for <read>):
    before both | after s(r), before the competitor | after the competitor, before s(r) | after both
Per cell: Flip = correct on the original AND the requested source's edit is followed (items with a twin); Stay =
correct on the original AND unchanged under the competing source's edit (the paper's conjunctive Stay). n per cell.
The native word (`_cp4_ho`) is recomputed from the stored picks and checked against its summary (fails loudly).

LADDER (corrected protocol, Qwen-4B / Gemma-3 / Qwen-2B as they land): the image is always first, so the only boundary a
control can cross is the caption. Cells: word native (after both), word in the prefix's prompt (_ptfmt), word's sentence
before the caption (_reqfirst); prefix native (after the image, before the caption), prefix in the word's slot (_ptslot),
prefix prompt without the prefix (_noprefix).
Writes results/position_analysis.json. Usage: python scripts/position_analysis.py
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [("", "Gemma-3"), ("_qwen35", "Qwen-4B"), ("_qwen2b", "Qwen-2B")]
CATS = ("before both", "after requested, before competitor", "after competitor, before requested", "after both")


def load(path):
    p = os.path.join(ROOT, path)
    return json.load(open(p)) if os.path.exists(p) else None


def item_scores(d, req):
    """per-item (cap_first, flip or None, stay) from the stored picks of an eval_clevr3_flip summary"""
    out = []
    for p in (d.get("picks") or {}).get(req) or []:
        g = p.get("gold") or {}
        ok0 = p.get("p0") == g.get("base")
        # <look> picks: p_twin (image edit), p_capedit (caption edit); <read> picks: p1 (caption edit), p_twin (image edit)
        tw = p.get("p_twin") if req == "look" else p.get("p1", p.get("p_capedit"))   # the requested source's edit
        oth = p.get("p_capedit") if req == "look" else p.get("p_twin")                # the competing source's edit
        gt = g.get("twin") if req == "look" else g.get("capedit")
        go = g.get("capedit") if req == "look" else g.get("twin")
        flip = (ok0 and tw == gt) if (tw is not None and gt is not None) else None
        stay = (ok0 and oth == go) if (oth is not None and go is not None) else None
        out.append((bool(p.get("cap_first")), flip, stay))
    return out


def cat_of(pos, req, cap_first):
    if pos == "first":
        return CATS[0]
    if pos == "last":
        return CATS[3]
    requested_first = (not cap_first) if req == "look" else cap_first       # which evidence block comes first
    return CATS[1] if requested_first else CATS[2]


def agg(rows):
    f = [x for _, x, _ in rows if x is not None]; s = [x for _, _, x in rows if x is not None]
    return {"flip": round(sum(f) / len(f), 3) if f else None, "n_flip": len(f),
            "stay": round(sum(s) / len(s), 3) if s else None, "n_stay": len(s)}


def gcf_cells(sfx):
    E = "runs/{r}/eval/clevr3_test_id/flip_step_800{t}.json"
    word, pre, eng, mix = (f"clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}", f"clevr3_2s_pt_1tok_ctm_cp_fc_seed0{sfx}",
                           f"clevr3_2s_5tok_sft_seed0{sfx}", f"clevr3_2s_1tok_ctm_cp_fc_pmix_seed0{sfx}")
    return [  # (method, file, position)
        ("word", E.format(r=word, t="_cp4_ho"), "last"), ("word", E.format(r=word, t="_cp4_ho_reloc_first"), "first"),
        ("word", E.format(r=word, t="_cp4_ho_reloc_between"), "between"),
        ("word (held-out lead sentence)", E.format(r=word, t="_cp4_ho_reloc_first_hs"), "first"),
        ("prefix", E.format(r=pre, t="_cp4_ho"), "first"), ("prefix", E.format(r=pre, t="_cp4_ho_reloc_between"), "between"),
        ("prefix", E.format(r=pre, t="_cp4_ho_reloc_last"), "last"),
        ("English (training questions)", E.format(r=eng, t="_cp4_fc_instr"), "last"),
        ("English (training questions)", E.format(r=eng, t="_cp4_fc_instr_reloc_first"), "first"),
        ("English (training questions)", E.format(r=eng, t="_cp4_fc_instr_reloc_between"), "between"),
        ("mixed-placement word", E.format(r=mix, t="_cp4_ho"), "last"),
        ("mixed-placement word", E.format(r=mix, t="_cp4_ho_reloc_first"), "first"),
        ("mixed-placement word", E.format(r=mix, t="_cp4_ho_reloc_between"), "between")]


def main():
    bank = {"groundcf": {}, "ladder": {}}
    print("GROUND-CF  <look> / <read>: Flip  Stay  (n)   [causal position of the control]")
    for sfx, ml in MODELS:
        print(f"\n{ml}")
        # guard: recomputation must reproduce the stored summary of the native word
        d0 = load(f"runs/clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}/eval/clevr3_test_id/flip_step_800_cp4_ho.json")
        if d0 and d0.get("picks"):
            for rq in ("look", "read"):
                a = agg(item_scores(d0, rq))
                assert abs(a["flip"] - d0[f"{rq}_flip"]) < 1e-3 and abs(a["stay"] - d0[f"{rq}_stay_c"]) < 1e-3, (ml, rq, a, d0[f"{rq}_flip"], d0[f"{rq}_stay_c"])
        for meth, f, pos in gcf_cells(sfx):
            d = load(f)
            for req in ("look", "read"):
                key = f"{ml}|{meth}|{req}"
                if d is None:
                    bank["groundcf"].setdefault(key, {})[pos] = None
                    continue
                rows = item_scores(d, req)
                if not rows:     # no stored picks (e.g. the original English run): summary only, native position
                    cell = {"flip": d.get(f"{req}_flip"), "stay": d.get(f"{req}_stay_c"), "n_flip": None, "n_stay": None}
                    bank["groundcf"].setdefault(key, {})[cat_of(pos, req, False) if pos != "between" else "between (unsplit)"] = cell
                    continue
                by = {}
                for r in rows:
                    by.setdefault(cat_of(pos, req, r[0]), []).append(r)
                for c, rr in by.items():
                    bank["groundcf"].setdefault(key, {})[c] = agg(rr)
        for meth in dict.fromkeys(m for m, _, _ in gcf_cells(sfx)):
            for req in ("look", "read"):
                cells = bank["groundcf"].get(f"{ml}|{meth}|{req}", {})
                txt = []
                for c in CATS + ("between (unsplit)",):
                    if c in cells:
                        v = cells[c]
                        fm = lambda x: "-" if x is None else f"{x:.3f}"
                        txt.append(f"{c}: " + ("pending" if v is None else
                                               f"{fm(v['flip'])} {fm(v['stay'])}" + (f" (n={v['n_flip']}/{v['n_stay']})" if v.get('n_flip') else "")))
                pend = [p for p, v in cells.items() if v is None]
                if all(v is None for v in cells.values()):
                    txt = ["pending"]
                print(f"  {meth:31s} <{req}>  " + " | ".join(t for t in txt if not t.endswith("pending") or len(txt) == 1))
    print("\nLADDER (corrected protocol, training templates)  <look> Flip Stay | <read> Flip Stay")
    for ds in ("vcf", "color", "coco"):
        for sfx, ml in MODELS:
            W, P = f"{ds}_neo_ctrl_1tok_ctm_cp_seed0{sfx}", f"{ds}_pt_na_1tok_ctm_cp_seed0{sfx}"
            cells = [("word native (after both)", W, ""), ("word in prefix prompt (before caption)", W, "_ptfmt"),
                     ("word sentence before caption", W, "_reqfirst"), ("prefix native (before caption)", P, ""),
                     ("prefix in word slot (after both)", P, "_ptslot"), ("prefix prompt, no prefix", P, "_noprefix")]
            got = []
            for lab, run, t in cells:
                d = load(f"runs/{run}/eval/{ds}3_test_id/flip_ll{t}_lead_p2_step_800.json")
                bank["ladder"][f"{ds}|{ml}|{lab}"] = None if d is None else {k: d.get(k) for k in ("look_flip", "look_stay_c", "read_flip", "read_stay_c", "n")}
                got.append((lab, d))
            if all(d is None for _, d in got):
                continue
            print(f"  {ds}/{ml}")
            for lab, d in got:
                print(f"    {lab:40s} " + ("pending" if d is None else f"{d['look_flip']:.3f} {d['look_stay_c']:.3f} | {d['read_flip']:.3f} {d['read_stay_c']:.3f}  (n={d.get('n')})"))
    json.dump(bank, open(os.path.join(ROOT, "results", "position_analysis.json"), "w"), indent=1)
    print("\n[saved] results/position_analysis.json")


if __name__ == "__main__":
    main()
