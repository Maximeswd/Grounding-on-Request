#!/usr/bin/env python
"""Exact inputs and outputs, by dataset and method (2026-10-22, author request). For each benchmark we show the user
turn exactly as the model receives it ([IMAGE] marks where the image tokens are inserted), the image, the candidate
answers with their summed log-likelihoods, the forced-choice answer (the paper's measurement) and, where it was
recorded, the greedy generation. Every example is read from an evaluation file of the run named in its heading; nothing
is re-rendered here.
  Ground-CF  protocol v2, Gemma-3-4B, L=5, paired SFT + CTM (the runs with generation records); flip dumps + generation records
  Ladder     VCF / ColorSwap / COCO-CF, paper protocol, the runs behind the paper's grid cells (genconsistency --save_images)
  MVG        two-source MultiVENT-G retrain (v4), held-out videos; flip dumps (forced choice only)
Writes paper/iclr2027/examples_io/README.md (local image links; not part of the public release).
Usage: python scripts/build_examples_io.py"""
import gzip, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "iclr2027", "examples_io"); os.makedirs(OUT, exist_ok=True)
rel = lambda p: os.path.relpath(os.path.join(ROOT, p) if not os.path.isabs(p) else p, OUT)
REQ = {"look": "image request", "read": "caption request"}
L = ["# Inputs and outputs, by dataset and method", "", __doc__.split("\n\n")[0].strip(), "",
     "Scoring: each candidate answer is appended to the user turn as the assistant's reply and its tokens' log-probabilities",
     "are summed; the forced-choice answer is the candidate with the highest sum. Flip = correct on the original and the answer",
     "follows an edit to the requested source; Stay = correct on the original and unchanged when the competing source is edited.", ""]

def block(prompt):
    return ["```text", prompt.replace("```", "'''"), "```"]

def gz(p):
    return [json.loads(l) for l in gzip.open(p, "rt")]

def path_of(run):
    return f"runs/{run}" if os.path.isdir(os.path.join(ROOT, "runs", run)) else f"Unnecessary/runs/{run}"

# ------------------------------------------------------------------ Ground-CF
GCF = [("Neologism (word in the carrier)", "clevr3_v2_5tok_psftctm_neo_holdcs_seed0"),
       ("Prompt tuning (soft prefix before the text)", "clevr3_v2_5tok_psftctm_pt_holdcs_seed0"),
       ("Shared LoRA r16 (request in English)", "clevr3_v2_lora_r16_joint_psftctm_holdcs_seed0"),
       ("Separate LoRA r16, image adapter", "clevr3_v2_lora_r16_adapter_look_psftctmsm_holdcs_seed0"),
       ("Separate LoRA r16, caption adapter", "clevr3_v2_lora_r16_adapter_read_psftctmsm_holdcs_seed0")]
CTX_G = [("base", "original"), ("twin", "image edited (requested source for <look>)"), ("capedit", "caption edited (requested source for <read>)")]
L += ["## Ground-CF (rendered scenes; protocol v2)", ""]
dumps = {}
for name, run in GCF:
    p = os.path.join(ROOT, path_of(run), "eval", "clevr3_test_id", "flip_step_800_v2ref_items.jsonl.gz")
    if os.path.exists(p):
        dumps[run] = {(r["request"], r["id"], r["ctx"]): r for r in gz(p) if "_meta" not in r}
common = None
for run, d in dumps.items():
    if "adapter" in run: continue
    ids = {k[1] for k in d}
    common = ids if common is None else common & ids
pick_ids = []
for skill in ("color", "count"):
    c = sorted(i for i in (common or []) if i.endswith("_" + skill) and all(((rq, i, cx) in dumps[r]) for r in dumps if "adapter" not in r for rq in ("look", "read") for cx in ("base", "twin", "capedit")))
    if c: pick_ids.append(c[0])
for iid in pick_ids:
    L += [f"### Item `{iid}`", ""]
    for name, run in GCF:
        if run not in dumps:
            L += [f"#### {name}: `{run}`", "", "(evaluation not available yet)", ""]; continue
        gp = os.path.join(ROOT, path_of(run), "eval", "clevr3_test_id", "gen_step_800_v2gen.json")
        gens = {(r["source"], r["id"], r["ctx"]): r for r in json.load(open(gp)).get("records", [])} if os.path.exists(gp) else {}
        L += [f"#### {name}: `{run}`", ""]
        for rq in ("look", "read"):
            if "adapter_look" in run and rq != "look" or "adapter_read" in run and rq != "read": continue
            for cx, cname in CTX_G:
                r = dumps[run].get((rq, iid, cx))
                if r is None: continue
                cands = r.get("candidates", {})
                L += [f"**{REQ[rq]}, {cname}.** Values: image = `{r['values'].get('look')}`, caption = `{r['values'].get('read')}`; "
                      f"requested-source value = `{r['values'].get(rq)}`. Image: ![]({rel(r['image'])})", ""]
                L += block(r["prompt"])
                L += ["", "| candidate | log-likelihood (sum) |", "|---|---|"] + [f"| {c} | {v['sum']:.3f} |" for c, v in sorted(cands.items(), key=lambda x: -x[1]["sum"])]
                g = gens.get((rq, iid, cx))
                L += ["", f"Forced-choice answer: **{r['pick']}**" + (f"; generated answer: `{g['text'].strip()}`" if g else ""), ""]
# ------------------------------------------------------------------ ladder
LAD = {"VCF": ["vcf_cmpv2_5tok_ctm_seed0_v2", "vcf_pt_na_5tok_ctm_v2_seed0", "vcf_lora_r16_lr2e4_seed0", "vcf_lora_r16_lr2e4_ad_ctm_cp_look_seed0", "vcf_lora_r16_lr2e4_ad_ctm_cp_read_seed0"],
       "ColorSwap": ["color_cmpv2_5tok_ctm_seed0_v2", "color_pt_na_5tok_sft_seed0", "color_lora_r16_lr2e4_seed0", "color_lora_r16_lr2e4_ad_ctm_cp_look_seed0", "color_lora_r16_lr2e4_ad_ctm_cp_read_seed0"],
       "COCO-CF": ["coco_neo_ctrl_5tok_ctm_seed0", "coco_pt_na_5tok_sft_seed0", "coco_lora_r16_lr2e4_seed0", "coco_lora_r16_lr2e4_ad_ctm_cp_look_seed0", "coco_lora_r16_lr2e4_ad_ctm_cp_read_seed0"]}
LNAME = ["Neologism", "Prompt tuning", "Shared LoRA r16", "Separate LoRA r16, image adapter", "Separate LoRA r16, caption adapter"]
DSDIR = {"VCF": "vcf", "ColorSwap": "color", "COCO-CF": "coco"}
CTX_L = [("base", "original"), ("imgswap", "image swapped for its twin"), ("capswap", "caption swapped for its twin")]
for ds, runs in LAD.items():
    L += [f"## {ds} (paper protocol)", ""]
    for name, run in zip(LNAME, runs):
        p = os.path.join(ROOT, path_of(run), "eval", f"{DSDIR[ds]}_test_id", "genconsistency_step_800_examples.json")
        L += [f"### {name}: `{run}`", ""]
        if not os.path.exists(p):
            L += ["(examples not generated yet)", ""]; continue
        G = [g for g in json.load(open(p)).get("generations", []) if g.get("item") == 0]
        for g in G:
            if "image adapter" in name and g["op"] != "look" or "caption adapter" in name and g["op"] != "read": continue
            cname = dict(CTX_L).get(g["ctx"], g["ctx"])
            L += [f"**{REQ[g['op']]}, {cname}.** Image value `{g['v_image']}`, caption value `{g['v_caption']}`." +
                  (f" Image: ![]({rel(g['image_file'])})" if g.get("image_file") else ""), ""]
            L += block(g["prompt"])
            if g.get("loglik"):
                L += ["", "| candidate | log-likelihood (sum) |", "|---|---|"] + [f"| {g['candidates'][k]} ({k}) | {v:.3f} |" for k, v in g["loglik"].items()]
                fc = max(g["loglik"], key=g["loglik"].get)
                L += ["", f"Forced-choice answer: **{g['candidates'][fc]}** ({fc}); generated answer: `{g['text'].replace('<end_of_turn>', '').strip()}`", ""]
            else:
                L += ["", f"Generated answer: `{g['text'].strip()}`", ""]
# ------------------------------------------------------------------ MVG
MVG = [("Neologism", "mvgV4_5tok_ctm_neo_seed0"), ("Soft prompt in the word's slot (matched position)", "mvgV4_5tok_ctm_pt_seed0"), ("Shared LoRA r16", "mvgV4_lora_r16_ctm_seed0")]
pairs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "mvg_ctm", "pairs_clean_rb_heldout_videos.jsonl"))]
pkey = lambda p: (p["video_id"], p["role"], p["pair_type"], str(p["gold_a"]), str(p["gold_b"]))
pmap = {pkey(p): p for p in pairs}
CTX_M = [("own_a", "requested request, context A"), ("own_b", "requested request, context B (requested source edited)"),
         ("other_a", "other request, context A"), ("other_b", "other request, context B (its source unchanged)")]
L += ["## MultiVENT-G (news video frames; two-source retrain, held-out videos)", "",
      "Each pair edits one source: `vis_value_swap` changes the frame (the image request must follow, the description request must",
      "stay), `desc_swap` changes the description (the description request must follow, the image request must stay). Candidate",
      "continuations are `I used the requested source to answer the question. final answer: <value>`.", ""]
for name, run in MVG:
    p = os.path.join(ROOT, path_of(run), "eval", "mvg_flip", "flip_step_800_v4ho_items.jsonl.gz")
    L += [f"### {name}: `{run}`", ""]
    if not os.path.exists(p):
        L += ["(evaluation not available)", ""]; continue
    recs = [r for r in gz(p) if "_meta" not in r]
    for ptype in ("vis_value_swap", "desc_swap"):
        r = next((x for x in recs if x["pair_type"] == ptype), None)
        if r is None: continue
        pr = pmap.get((r["video_id"], r["role"], r["pair_type"], str(r["gold_a"]), str(r["gold_b"])))
        fa = (pr or {}).get("ctx_a", {}).get("frame"); fb = (pr or {}).get("ctx_b", {}).get("frame")
        L += [f"**Pair `{ptype}`**, video `{r['video_id']}`, role `{r['role']}`: value A = `{r['gold_a']}`, value B = `{r['gold_b']}`." +
              (f" Frame A: ![]({rel(fa)})" if fa and os.path.exists(os.path.join(ROOT, fa)) else "") +
              (f" Frame B: ![]({rel(fb)})" if fb and fb != fa and os.path.exists(os.path.join(ROOT, fb)) else ""), ""]
        for cx, cname in CTX_M:
            if cx not in r["prompts"]: continue
            L += [f"*{cname}*", ""] + block(r["prompts"][cx])
            sc = r["scores"][cx]
            L += ["", f"| candidate | log-likelihood (sum) |", "|---|---|", f"| A: {r['gold_a']} | {sc['a']['sum']:.3f} |", f"| B: {r['gold_b']} | {sc['b']['sum']:.3f} |",
                  "", f"Forced-choice answer: **{r['gold_a'] if r['pick_sum'][cx] == 'a' else r['gold_b']}**", ""]
open(os.path.join(OUT, "README.md"), "w").write("\n".join(L) + "\n")
print(f"[saved] {os.path.relpath(os.path.join(OUT, 'README.md'), ROOT)} ({len(L)} lines; Ground-CF items {pick_ids})")
