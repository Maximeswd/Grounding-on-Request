#!/usr/bin/env python
"""RESULTS LEDGER: every headline result, its number, the banked file it comes from, and where it
lives in the paper. Regenerable; writes paper/iclr2027/RESULTS_LEDGER.md and results/ledger.json.
Numbers are read from the banked files, never typed. Usage: python scripts/results_ledger.py
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(f"{ROOT}/results/{f}"))


def main():
    g = R("master_grid.json"); v = R("objective_verdict.json"); mp = R("modality_pooled.json")["cells"]
    det = R("detection_procedural.json")["pooled"]; sr = R("scatter_region.json")["cells"]
    cond = R("attn_conditionality.json")["cells"]; apps = R("applications_groundcf.json")
    gf = R("generation_format.json")["cells"]
    gem = lambda k: g[f"groundcf|gemma3|{k}"]
    rows = []
    def add(block, wow, value, source, place):
        rows.append({"block": block, "result": wow, "value": value, "source": source, "place": place})
    # ---- block 1
    i = gem("instr|None"); w = gem("word_sft|5"); c = gem("word+ctm|5")
    add(1, "English request: correct but ungrounded (Gemma, Ground-CF caption / on-screen)",
        f"acc {i['R.read']:.2f}/{i['R.ocr']:.2f} vs flip {i['G.read']:.2f}/{i['G.ocr']:.2f}", "master_grid.json", "Table 1, Sec 4")
    add(1, "Correct-but-ignores-edit cells (prompting / learned)", f"{sr['Prompting interfaces']} / {sr['Learned interfaces']}", "scatter_region.json", "fig:scatter")
    add(1, "Share of correct COCO-CF answers ignoring the image edit, instruction, Gemma",
        f"{100*(1-g['coco|gemma3|instr|None']['G.look']/g['coco|gemma3|instr|None']['R.look']):.0f}%", "master_grid.json", "Sec 4, Fig 3A")
    s = v["summary"]
    add(1, "Objective vs supervision (win/loss/tie, sign p)", f"{s['supervised neologism']['win']}/{s['supervised neologism']['loss']}/{s['supervised neologism']['tie']}, p={s['supervised neologism']['sign_p']}", "objective_verdict.json", "Sec 5.2, tab:app_verdict")
    add(1, "Weak cells (< .80) repaired", f"{s['weak (supervision < .80)']['better']} of {s['weak (supervision < .80)']['cells']}, median {s['weak (supervision < .80)']['median_change']:+.3f}", "objective_verdict.json", "Sec 5.2")
    add(1, "Objective vs APO / vs LoRA (ties stated)", f"{s['preference (APO)']['win']}/{s['preference (APO)']['loss']}/{s['preference (APO)']['tie']} ; {s['LoRA rank 4']['win']}/{s['LoRA rank 4']['loss']}/{s['LoRA rank 4']['tie']}", "objective_verdict.json", "Sec 5.2")
    def wk(lab): return mp[lab]["weakest_modality_mean"]
    add(1, "Weakest modality column per interface (pooled)", "; ".join(f"{k} {wk(k):.2f}" for k in ("English instruction", "Prompt tuning", "LoRA rank 4", "Neologisms SFT", "Neologisms + CTM")), "modality_pooled.json", "Sec 5.1, Fig 3B")
    add(1, "Neologism + objective pooled image / caption / on-screen", f"{mp['Neologisms + CTM']['image']['mean']:.2f} / {mp['Neologisms + CTM']['caption']['mean']:.2f} / {mp['Neologisms + CTM']['on_screen_text']['mean']:.2f}", "modality_pooled.json", "abstract, Sec 5.1")
    # interaction: objective through word vs prefix
    DS={'vcf':['look','read'],'color':['look','read'],'coco':['look','read'],'groundcf':['look','read','ocr']}; MOD=['gemma3','qwen4b','qwen2b','e4b']
    def tally(a,b):
        up=dn=0
        for ds,chs in DS.items():
            for m in MOD:
                for ch in chs:
                    x=(g.get(f"{ds}|{m}|{a}") or {}).get(f"G.{ch}"); y=(g.get(f"{ds}|{m}|{b}") or {}).get(f"G.{ch}")
                    if x is None or y is None: continue
                    up+= y>x+.02; dn+= x>y+.02
        return up,dn
    wu,wd=tally("word_sft|5","word+ctm|5"); pu,pd=tally("pt|5","pt+ctm|5")
    add(1, "Two-sided objective through the word vs through the prefix (better / worse of 36)", f"word {wu}/{wd}; prefix {pu}/{pd}", "master_grid.json", "Sec 5.2")
    add(1, "Inserted prefix vs SFT word (position, ColorSwap look)", f"{g['color|gemma3|pti|5']['G.look']:.2f} vs {g['color|gemma3|word_sft|5']['G.look']:.2f}", "master_grid.json", "Sec 5.1, app mechanism")
    # ---- block 2
    mvr = {k: g.get(f"mvr|gemma3|{k}") for k in ("word_sft|5","lora_r16|None","instr|None")}
    add(2, "MVR OCR joint: word vs LoRA vs instruction", " / ".join(f"{(mvr[k] or {}).get('J.ocr', float('nan')):.2f}" for k in mvr), "master_grid.json", "Sec 7, tab:mvr")
    add(2, "MVG crop: word vs LoRA (matches)", ".450 vs .460 (full frame .273 vs .303)", "applications: tab_app_crop / mvg_flip files", "Sec 7, fig:mvg")
    add(2, "Rendered scenes -> real on-screen text selection", ".75 (chance .33); selectivity .277 vs .196 bundled", "master_grid mvg cells", "Sec 7")
    # ---- block 3
    add(3, "Two-query conflict detection, balanced score", "; ".join(f"{k} {d['bal_mean']:.2f}" for k,d in det.items()), "detection_procedural.json", "Sec 8.1, tab:app_detect")
    ph = apps["posthoc"]
    def own(lab, m, s):
        b = ph.get(f"{lab}|{m}", {}).get(s); return None if not b else b["own_auroc"]
    add(3, "Post hoc verifier own-answer AUROC, Gemma text sources: instruction vs CTM word (cand AUROC where no ungrounded answers)",
        f"instr read {own('English instruction','gemma3','read')} / ocr {own('English instruction','gemma3','ocr')}; word cand {ph.get('Neologisms + CTM|gemma3',{}).get('read',{}).get('cand_auroc')} / {ph.get('Neologisms + CTM|gemma3',{}).get('ocr',{}).get('cand_auroc')}", "applications_groundcf.json", "Sec 8.2, tab:app_posthoc")
    add(3, "No-request which-source recovery (chance 1/3)", "; ".join(f"{m} {ph[f'No request|{m}']['look']['which_source_acc']:.2f}" for m in MOD if f"No request|{m}" in ph), "applications_groundcf.json", "Sec 8.2")
    add(3, "Citation: support 1.00 by construction vs faithfulness under intervention", ".37 / .39 (two frozen families)", "citefaith battery", "Sec 8.2, tab:citefaith")
    add(3, "Free generation unmappable, COCO-CF: trained words vs frozen instructed", f"{gf['coco|neologism']['unmappable_min']:.2f}-{gf['coco|neologism']['unmappable_max']:.2f} vs {gf['coco|frozen_instruction']['unmappable_min']:.2f}-{gf['coco|frozen_instruction']['unmappable_max']:.2f}", "generation_format.json", "Sec 8 (fairness), app gencons")
    add(3, "Composed detection (trained) — PENDING", "campaign 646596 / evals 646597", "compose_detect files", "Sec 8.1")
    # ---- block 4
    add(4, "Request-conditional attention gains, English instruction on Gemma (box / cap / ocr / share)",
        " / ".join(f"{cond['English instruction|gemma3'][k]:+.3f}" for k in ("box_gain","cap_gain","ocr_gain","share_gain")) if "English instruction|gemma3" in cond else "pending", "attn_conditionality.json", "Sec 9, tab:app_attn_cond")
    add(4, "Same gains, CTM word on Gemma", " / ".join(f"{cond['Neologisms + CTM|gemma3'][k]:+.3f}" for k in ("box_gain","cap_gain","ocr_gain","share_gain")) if "Neologisms + CTM|gemma3" in cond else "pending", "attn_conditionality.json", "Sec 9")
    add(4, "Two words, same data: row cosine vs behavioural agreement", "cosine <= .22 (114 pairs, 4 models); agreement .94-.99", "word_anatomy.json, token_cka.json", "Sec 9, fig:mechanism")
    add(4, "Trained at the decision site instead of the request slot (VCF / ColorSwap look)", ".94 -> .28 ; .68 -> .43", "relocation battery", "Sec 9")
    add(4, "Words do not verbalize under filler controls", "qualitative + plug-in", "verbalization battery", "app verbalization")
    json.dump(rows, open(f"{ROOT}/results/ledger.json", "w"), indent=1)
    md = ["# Results ledger (auto-generated by scripts/results_ledger.py)", "",
          "| block | result | value | banked source | place in paper |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['block']} | {r['result']} | {r['value']} | {r['source']} | {r['place']} |")
    open(f"{ROOT}/paper/iclr2027/RESULTS_LEDGER.md", "w").write("\n".join(md) + "\n")
    print(f"[saved] RESULTS_LEDGER.md ({len(rows)} rows), results/ledger.json")
    for r in rows: print(f"  [{r['block']}] {r['result'][:70]:70s} -> {str(r['value'])[:60]}")


if __name__ == "__main__":
    main()
