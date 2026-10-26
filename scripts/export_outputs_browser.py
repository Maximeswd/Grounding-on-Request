#!/usr/bin/env python
"""Export per-item evaluation outputs for the outputs browser page (v1.152, 2026-09-24). CPU only, read-only on runs/.

Ground-CF (test_id, 748 conflict items, the banked main-grid files):
  trained cells (word CTM, word CTM+Stay, prefix CTM, LoRA r16 CTM): the stored picks per context (original / caption
  edit / image twin) and every candidate's log-likelihood (the `picks` block of flip_step_800_cp4_ho.json; LoRA = the
  look adapter's file for the image request, the read adapter's for the text request);
  baselines (no request, English): per-item outcomes and margins (`records`; no candidate scores were stored).
  Prompts: rebuilt exactly by running the real evaluator with a stub model (scripts/dryrun_eval.py) on each cell's
  manifest line (--prompts_dir); checked against the prompts the real runs stored for their first 50 items.
Ladder (VCF / ColorSwap / COCO-CF): item logs of the trained cells (prompt, both candidates' scores, pick per
  context); baselines from `records` (outcomes + margins), aligned by item index.
Writes <out>/gcf_items.json, gcf_<model>.json, ladder_<ds>.json, manifest.json.
"""
import argparse, base64, collections, glob, gzip, io, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [("", "gemma3", "Gemma-3-4B"), ("_qwen35", "qwen4b", "Qwen-3.5-4B"), ("_qwen2b", "qwen2b", "Qwen-3.5-2B")]   # v1.185: E4B dropped
GCF_HOST = {"": "clevr3_2s_5tok_sft_seed0", "_qwen35": "clevr3_2s_5tok_sft_seed0_qwen35",
            "_qwen2b": "clevr3_2s_5tok_sft_seed0_qwen2b", "_gemma4e4b": "clevr3_2s_5tok_apo_seed0_gemma4e4b"}
GCF_METHODS = [("noop", "No request"), ("instr", "English instruction"), ("word", "Word L1 CTM"),
               ("wordearly", "Word L1 CTM, early (before the evidence)"),
               ("wordstay", "Word L1 CTM+Stay"), ("prefix", "Prefix L1 CTM"), ("lora", "LoRA r16 CTM")]
LAD_METHODS = [("noop", "No request"), ("instr", "English instruction"), ("word", "Word L1 CTM"),
               ("prefix", "Prefix L1 CTM"), ("lora", "LoRA r16 CTM")]
PROMPT_RUN = {"noop": ("clevr3_2s_5tok_sft_seed0", "_pr_noop"), "instr": ("clevr3_2s_5tok_sft_seed0", "_pr_instr"),
              "word": ("clevr3_2s_1tok_ctm_cp_fc_seed0", "_pr_ho"),
              "wordearly": ("clevr3_2s_1tok_ctm_cp_fc_pfirst_seed0", "_pr_ho_first"), "wordstay": ("clevr3_2s_1tok_ctmsm_cp_fc_seed0", "_pr_ho"),
              "prefix": ("clevr3_2s_pt_1tok_ctm_cp_fc_seed0", "_pr_ho"), "lora": ("clevr3_2s_lora_r16_ctm_cp_fc_look_seed0", "_pr_ho")}
CTXS = ("base", "capedit", "twin")


def thumb(path, size=200):
    from PIL import Image
    im = Image.open(os.path.join(ROOT, path)).convert("RGB").resize((size, size), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "WEBP", quality=80)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


def gcf_file(sfx, meth, rq):
    d = "clevr3_test_id"
    if meth in ("noop", "instr"):
        # v1.185: the prompt-matched held-out-question reruns (they store picks) when they exist
        ho = f"runs/{GCF_HOST[sfx]}/eval/{d}/flip_step_800_cp4_ho_fc_{meth}.json"
        return ho if os.path.exists(os.path.join(ROOT, ho)) else f"runs/{GCF_HOST[sfx]}/eval/{d}/flip_step_800_cp4_fc_{meth}.json"
    run = {"word": "clevr3_2s_1tok_ctm_cp_fc_seed0", "wordstay": "clevr3_2s_1tok_ctmsm_cp_fc_seed0",
           "wordearly": "clevr3_2s_1tok_ctm_cp_fc_pfirst_seed0",
           "prefix": "clevr3_2s_pt_1tok_ctm_cp_fc_seed0", "lora": f"clevr3_2s_lora_r16_ctm_cp_fc_{rq}_seed0"}[meth]
    # v1.190 the early word at its own position (word sentence before the evidence, held-out lead sentence)
    return f"runs/{run}{sfx}/eval/{d}/flip_step_800_cp4_ho{'_reloc_first' if meth == 'wordearly' else ''}.json"


def _spread(v, n, idx):
    """A record list over all items stays as is; one over the twin subset (length len(idx)) is put back in place."""
    if v is None:
        return None
    if len(v) == n:
        return v
    if len(v) != len(idx):
        return [None] * n
    o = [None] * n
    for k, x in zip(idx, v):
        o[k] = x
    return o


def pick_of(p, rq, ctx):
    return p.get({"base": "p0", "capedit": "p_capedit" if rq == "look" else "p1", "twin": "p_twin"}[ctx])


def export_gcf(out, prompts_dir):
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data/clevr3/test_id.jsonl"))}
    # v1.191: show the corrected (residue-free) twin images once every model's word cell is the promoted clean rerun
    _pr = os.path.join(ROOT, "results", "clean_twin_promotion.json")
    _prom = json.load(open(_pr))["promoted"] if os.path.exists(_pr) else {}
    _clean = all(gcf_file(s_, "word", "look") in _prom for s_ in ("", "_qwen35", "_qwen2b"))
    twins = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(
        ROOT, "data/clevr3/pairs_look_all_id_clean.jsonl" if _clean else "data/clevr3/pairs_look_all_id.jsonl"))}
    # canonical item order + values from one trained cell (identical protocol across cells; checked below)
    ref = json.load(open(os.path.join(ROOT, gcf_file("", "word", "look"))))["picks"]
    order = [p["id"] for p in ref["look"]]
    # prompts rebuilt by the stub-model run of the real evaluator
    P = {}
    # v1.185: the baselines' held-out-question reruns (prompt-matched) replace the training-question files when present
    if os.path.exists(os.path.join(ROOT, gcf_file("", "noop", "look"))) and "_cp4_ho_fc_" in gcf_file("", "noop", "look"):
        PROMPT_RUN.update({"noop": ("clevr3_2s_5tok_sft_seed0", "_pr_ho_noop"), "instr": ("clevr3_2s_5tok_sft_seed0", "_pr_ho_instr")})
    for meth, (run, tag) in PROMPT_RUN.items():
        f = os.path.join(prompts_dir, run, "eval", "clevr3_test_id", f"flip_step_800{tag}_items.jsonl.gz")
        P[meth] = {}
        for l in gzip.open(f, "rt"):
            a = json.loads(l)
            if "_meta" not in a:
                P[meth][(a["id"], a["request"], a["ctx"])] = a["prompt"]
    items, imgs = [], {}
    for i, iid in enumerate(order):
        pl, pr = ref["look"][i], ref["read"][i]
        tw = twins.get(iid); base_img = rows[iid]["image"]
        for pth in [base_img] + ([tw["ctx_b"]["image"]] if tw and "twin" in pl["values"] else []):
            if pth not in imgs:
                imgs[pth] = thumb(pth)
        items.append({"id": iid, "skill": pl["skill"], "Y": pl["Y"], "values": pl["values"], "read_values": pr["values"],
                      "img": base_img, "twin_img": tw["ctx_b"]["image"] if (tw and "twin" in pl["values"]) else None,
                      "prompts": {m: {rq: {c: P[m].get((iid, rq, c)) for c in CTXS} for rq in ("look", "read")} for m in P}})
    json.dump({"items": items, "images": imgs}, open(os.path.join(out, "gcf_items.json"), "w"), separators=(",", ":"))
    audit = collections.Counter()
    for sfx, mk, ml in MODELS:
        cells = {}
        for meth, lab in GCF_METHODS:
            cell = {"label": lab, "files": [], "look": None, "read": None}
            for rq in ("look", "read"):
                f = gcf_file(sfx, meth, rq); cell["files"].append(f)
                if not os.path.exists(os.path.join(ROOT, f)):
                    continue
                j = json.load(open(os.path.join(ROOT, f)))
                summ = {k: j.get(f"{rq}_{k}") for k in ("acc", "flip", "stay_c")}
                if meth in ("noop", "instr") and not (j.get("picks") or {}).get(rq):
                    R = j["records"]; tw_idx = [k for k, it in enumerate(items) if it["twin_img"]]
                    fl = R[f"{rq}_flip"]; st = R[f"{rq}_stay_c"]
                    # look: flip on the twin subset, stay on all; read: flip on all, stay on the twin subset
                    per = {"acc": R[f"{rq}_acc"], "margin": R.get(f"{rq}_margin")}
                    per["flip"], per["stay"] = _spread(fl, len(items), tw_idx), _spread(st, len(items), tw_idx)
                    cell[rq] = {"kind": "outcomes", "summary": summ, **per}
                else:
                    pk = j["picks"][rq]
                    assert [p["id"] for p in pk] == order, (f, "item order differs")
                    picks = [[pick_of(p, rq, c) for c in CTXS] for p in pk]
                    scores = [[[p["scores"].get(c, {}).get(y) for y in p["Y"]] if c in p["scores"] else None for c in CTXS] for p in pk]
                    gold = [[p["gold"].get(c) for c in CTXS] for p in pk]
                    for p in pk:   # prompt audit: rebuilt vs the prompt the real run stored (first 50 items)
                        if p.get("prompt_base"):
                            ok = P[meth].get((p["id"], rq, "base")) == p["prompt_base"]; audit[(meth, ok)] += 1
                    cell[rq] = {"kind": "picks", "summary": summ, "picks": picks, "scores": scores, "gold": gold}
            cells[meth] = cell
        json.dump({"model": ml, "cells": cells}, open(os.path.join(out, f"gcf_{mk}.json"), "w"), separators=(",", ":"))
    return audit, len(items), len(imgs)


def export_ladder(out):
    """v1.185: corrected protocol 2 + lead-in. Four fixed candidates per item: A = image value, B = edited image value,
    C = caption value, D = edited caption value. Trained methods on held-out templates (the headline), frozen rows as
    evaluated. Gold per version: <look> A / B (image edited) / A (caption edited); <read> C / C / D."""
    res = {}
    for ds, lab in (("vcf", "VCF"), ("color", "ColorSwap"), ("coco", "COCO-CF")):
        d = f"{ds}3_test_id"; info = {}; cells = collections.defaultdict(dict)
        for sfx, mk, ml in MODELS:
            runs = {"noop": (f"{ds}_base_frozen_0tok_none_seed0{sfx}", "flip_ll_noop_lead_p2_step_800"),
                    "instr": (f"{ds}_base_frozen_0tok_none_seed0{sfx}", "flip_ll_iv5_lead_p2_step_800"),
                    "word": (f"{ds}_neo_ctrl_1tok_ctm_cp_seed0{sfx}", "flip_ll_lead_p2_heldout_step_800"),
                    "prefix": (f"{ds}_pt_na_1tok_ctm_cp_seed0{sfx}", "flip_ll_lead_p2_heldout_step_800")}
            for meth, lab2 in LAD_METHODS:
                cell = {"label": lab2, "look": None, "read": None}
                srcs = ({rq: (f"{ds}_lora_r16_lr2e4_ad_ctm_cp_{rq}_seed0{sfx}", "flip_ll_lead_p2_heldout_step_800") for rq in ("look", "read")}
                        if meth == "lora" else {rq: runs[meth] for rq in ("look", "read")})
                for rq, (run, name) in srcs.items():
                    base = os.path.join(ROOT, "runs", run, "eval", d, name)
                    if not os.path.exists(base + ".json") or not os.path.exists(base + "_items.jsonl.gz"):
                        continue
                    j = json.load(open(base + ".json"))
                    summ = {k: j.get(f"{rq}_{k}") for k in ("acc", "flip", "stay_c")}
                    by = collections.defaultdict(dict)
                    for l in gzip.open(base + "_items.jsonl.gz", "rt"):
                        a_ = json.loads(l)
                        if "_meta" not in a_:
                            by[(int(a_["item"]), a_["request"])][a_["ctx"]] = a_
                    if not by:
                        continue
                    n = 1 + max(k[0] for k in by); ctxs = ("base", "imgswap", "capswap")
                    for k in range(n):
                        b_ = by.get((k, rq), {}).get("base")
                        if b_ and k not in info:
                            v = b_["values"] if isinstance(b_["values"], dict) else eval(b_["values"])
                            info[k] = {"caption": b_["caption"], "values": v}
                    def sc(r):
                        c = r.get("candidates"); c = c if isinstance(c, dict) else eval(c)
                        return [round(c[x]["sum"], 2) if x in c else None for x in "ABCD"]
                    cell[rq] = {"kind": "p2", "summary": summ,
                                "prompt": [by.get((k, rq), {}).get("base", {}).get("prompt") for k in range(n)],
                                "picks": [[by.get((k, rq), {}).get(c, {}).get("pick") for c in ctxs] for k in range(n)],
                                "scores": [[sc(by[(k, rq)][c]) if c in by.get((k, rq), {}) else None for c in ctxs] for k in range(n)]}
                cells[mk][meth] = cell
        json.dump({"dataset": lab, "protocol": "p2+lead", "items": [info.get(k) for k in range(max(info) + 1)] if info else [], "cells": cells},
                  open(os.path.join(out, f"ladder_{ds}.json"), "w"), separators=(",", ":"))
        res[ds] = len(info)
    return res


def export_generations(out):
    """v1.185: free-text outputs on Ground-CF. E2 open generation (one request per prompt; gen_step_800_gen_ho*.json):
    prompt, generated text (truncated to 700 characters), parse label, parsed value, forced-choice pick, gold. E5 one-prompt
    composition (compose_detect_step_800_cp_fc_x2*.json): prompt asking for both answers, raw text, parse label, the two
    parsed answers and the forced-choice prediction, per request order, original and agreeing context."""
    G = {"e2": {}, "e5": {}}
    e2runs = {"word": "clevr3_2s_1tok_ctm_cp_fc_seed0{s}", "prefix": "clevr3_2s_pt_1tok_ctm_cp_fc_seed0{s}",
              "lora_look": "clevr3_2s_lora_r16_ctm_cp_fc_look_seed0{s}", "lora_read": "clevr3_2s_lora_r16_ctm_cp_fc_read_seed0{s}"}
    for sfx, mk, ml in MODELS:
        G["e2"][mk] = {}; G["e5"][mk] = {}
        for meth, rf in list(e2runs.items()) + [("instr", GCF_HOST[sfx]), ("noop", GCF_HOST[sfx])]:
            run = rf.format(s=sfx) if "{s}" in rf else rf
            fn = "gen_step_800_gen_ho.json" if meth not in ("instr", "noop") else f"gen_step_800_gen_ho_fc_{meth}_hint.json"
            f = os.path.join(ROOT, "runs", run, "eval", "clevr3_test_id", fn)
            if os.path.exists(f):
                R = json.load(open(f)).get("records") or []
                G["e2"][mk][meth] = [{"id": r.get("id"), "skill": r.get("skill"), "req": r.get("source"), "ctx": r.get("ctx"),
                                      "prompt": r.get("prompt"), "text": (r.get("text") or "")[:700], "label": r.get("gen_label"),
                                      "parsed": r.get("gen_value"), "pick": r.get("pick_sum"), "gold": r.get("gold"),
                                      "values": r.get("values")} for r in R]
        for meth, rf, tag in (("word", "clevr3_2s_1tok_ctm_cp_fc_seed0{s}", "x2"), ("prefix", "clevr3_2s_pt_1tok_ctm_cp_fc_seed0{s}", "x2"),
                              ("prefix_start", "clevr3_2s_pt_1tok_ctm_cp_fc_seed0{s}", "x2_ptstart"), ("instr", GCF_HOST[sfx], "x2_instr"),
                              ("noop", GCF_HOST[sfx], "x2_noop")):
            run = rf.format(s=sfx) if "{s}" in rf else rf
            f = os.path.join(ROOT, "runs", run, "eval", "clevr3_test_id", f"compose_detect_step_800_cp_fc_{tag}.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f)); recs = []
            for r in d.get("records") or []:
                for ctx in ("conflict", "agree"):
                    c = r.get(ctx)
                    if c:
                        recs.append({"id": r.get("id"), "skill": r.get("skill"), "order": r.get("order"), "ctx": ctx,
                                     "prompt": c.get("prompt"), "text": (c.get("raw_generation") or "")[:700], "label": c.get("parse_label"),
                                     "parsed": c.get("parsed"), "pred": c.get("prediction"), "gold": c.get("gold"), "values": c.get("values")})
            G["e5"][mk][meth] = recs
    for mk in G["e2"]:   # one file per model so the page loads only what it shows
        json.dump({"e2": G["e2"][mk], "e5": G["e5"].get(mk, {})}, open(os.path.join(out, f"gen_{mk}.json"), "w"), separators=(",", ":"))
    return {k: {m: {x: len(v) for x, v in G[k][m].items()} for m in G[k]} for k in G}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--prompts_dir", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    audit, n_items, n_img = export_gcf(a.out, a.prompts_dir)
    lad = export_ladder(a.out)
    gen = export_generations(a.out)
    json.dump({"gcf_items": n_items, "gcf_images": n_img, "ladder_items": lad, "generations": gen, "prompt_audit": {f"{m}:{ok}": v for (m, ok), v in audit.items()}},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("[export] Ground-CF items", n_items, "images", n_img, "| ladder items", lad, "| prompt audit", dict(audit))


if __name__ == "__main__":
    main()
