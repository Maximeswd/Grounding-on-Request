#!/usr/bin/env python
"""Harvest the three Ground-CF application batteries into one banked JSON and three
appendix tables:

  compose_detect_step_800*.json  -> zero-shot composed detection (one prompt, two requests)
  posthoc_ground_step_800*.json  -> post hoc grounding verifier (source ablation)
  attn_groundcf_step_800*.json   -> attention allocation (which / where / when)

Interfaces (rows): No request, English instruction, Prompt tuning, LoRA r=4, Neologisms SFT,
Neologisms + CTM; models (columns or blocks): Gemma 3 4B, Qwen 3.5 4B, Qwen 3.5 2B, Gemma 4 E4B.

Writes results/applications_groundcf.json (read by scripts/audit_numbers.py) and
paper/iclr2027/tables/tab_app_compose.tex, tab_app_posthoc.tex, tab_app_attn.tex.
Usage: python scripts/applications_groundcf.py
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "iclr2027", "tables")
MODELS = [("gemma3", "Gemma-3-4B", ""), ("qwen4b", "Qwen-3.5-4B", "_qwen35"),
          ("qwen2b", "Qwen-3.5-2B", "_qwen2b"), ("e4b", "Gemma-4-E4B", "_gemma4e4b")]
IF = [("No request", "clevr3_f_5tok_sft_seed0{s}", "_noop"),
      ("English instruction", "clevr3_f_5tok_sft_seed0{s}", "_instr_iv5"),
      ("Prompt tuning", "clevr3_f_pt_5tok_seed0{s}", ""),
      ("LoRA $r{=}4$", "clevr3_f_lora_r4_seed0{s}", ""),
      ("Neologisms, SFT", "clevr3_f_5tok_sft_seed0{s}", ""),
      ("Neologisms + CTM", "clevr3_f_5tok_ctm_seed0{s}", "")]
F = lambda v: "--" if v is None else (f"{v:.2f}".lstrip("0") if 0 < v < 1 else f"{v:.2f}")


def load(kind, run, tag, ds="clevr3"):
    p = f"{ROOT}/runs/{run}/eval/{ds}_test_id/{kind}_step_800{tag}.json"
    try:
        return json.load(open(p))
    except Exception:
        return None


# ladder run-name grammar (parity runs) per interface; {ds} in {vcf,color,coco}
IF_LADDER = [("No request", "{ds}_neo_ctrl_5tok_sft_seed0{s}", "_noop"),
             ("English instruction", "{ds}_neo_ctrl_5tok_sft_seed0{s}", "_instr_iv5"),
             ("Prompt tuning", "{ds}_pt_na_5tok_sft_seed0{s}", ""),
             ("LoRA $r{=}4$", "{ds}_lora_r4_lr2e4_seed0{s}", ""),
             ("Neologisms, SFT", "{ds}_neo_ctrl_5tok_sft_seed0{s}", ""),
             ("Neologisms + CTM", "{ds}_neo_ctrl_5tok_ctm_seed0{s}", "")]
# compose-TRAINED runs (both families) and the zero-shot references scored in the same format
IF_COMPOSE = {"clevr3": [("No request (zero-shot)", "clevr3_f_5tok_sft_seed0{s}", "_noop"),
                         ("English instruction (zero-shot)", "clevr3_f_5tok_sft_seed0{s}", "_instr_iv5"),
                         ("Neologisms + CTM (zero-shot)", "clevr3_f_5tok_ctm_seed0{s}", ""),
                         ("Prompt tuning, composed", "clevr3_f_pt_5tok_compose_seed0{s}", ""),
                         ("LoRA $r{=}4$, composed", "clevr3_f_lora_r4_compose_seed0{s}", ""),
                         ("Neologisms SFT, composed", "clevr3_f_5tok_sft_compose_seed0{s}", ""),
                         ("Neologisms + CTM, composed", "clevr3_f_5tok_ctm_compose_seed0{s}", "")],
              "ladder": [("No request (zero-shot)", "{ds}_neo_ctrl_5tok_sft_seed0{s}", "_noop"),
                         ("English instruction (zero-shot)", "{ds}_neo_ctrl_5tok_sft_seed0{s}", "_instr"),
                         ("Neologisms + CTM (zero-shot)", "{ds}_neo_ctrl_5tok_ctm_seed0{s}", "_zeroshot"),
                         ("Prompt tuning, composed", "{ds}_pt_na_5tok_sft_compose_seed0{s}", ""),
                         ("LoRA $r{=}4$, composed", "{ds}_lora_r4_lr2e4_compose_seed0{s}", ""),
                         ("Neologisms SFT, composed", "{ds}_neo_ctrl_5tok_sft_compose_seed0{s}", ""),
                         ("Neologisms + CTM, composed", "{ds}_neo_ctrl_5tok_ctm_compose_seed0{s}", "")]}
LADDER = [("vcf", "VCF"), ("color", "ColorSwap"), ("coco", "COCO-CF")]


def compose_table(fname, family, dss, pair):
    """Composed detection (trained and zero-shot) for one family; pair e.g. 'look|read'."""
    lines = head(["Interface", "Model", "Dataset", "detect", "attribute", "false alarm", "balanced",
                  "verdict bal.", "held-out template bal."], "lll" + "c" * 6,
                 f"source: compose_detect_step_800*.json, pair {pair}; held-out = --template heldout")
    bank = {}
    for lab, run, tag in IF_COMPOSE[family]:
        first = True
        for m, mlab, sfx in MODELS:
            for ds, dslab in dss:
                r = run.format(ds=ds, s=sfx)
                d = load("compose_detect", r, tag, ds="clevr3" if family == "clevr3" else ds)
                if not d or pair not in d["pairs"]:
                    continue
                p = d["pairs"][pair]
                h = load("compose_detect", r, tag + "_heldout" if tag else "_heldout",
                         ds="clevr3" if family == "clevr3" else ds)
                hb = h["pairs"][pair]["balanced"] if h and pair in h["pairs"] else None
                bank[f"{lab}|{m}|{ds}"] = {**p, "heldout_balanced": hb}
                lines.append(f"{lab if first else ''} & {mlab} & {dslab} & {F(p['detect'])} & {F(p['attrib'])} & "
                             f"{F(p['false'])} & {F(p['balanced'])} & {F(p['verdict_balanced'])} & {F(hb)} \\\\")
                first = False
        lines.append("\\addlinespace")
    open(f"{OUT}/{fname}", "w").write(tail(lines)); print(f"[saved] {fname}")
    return bank


def ladder_tables(bank):
    # post hoc verifier on the ladder
    lines = head(["Interface", "Model", "Dataset", "Source", "acc.", "own AUROC", "$n$ g/u", "cand AUROC",
                  "attrib.", "which source"], "llll" + "c" * 6, "source: posthoc_ground_step_800*.json (ladder)")
    for lab, run, tag in IF_LADDER:
        first = True
        for m, mlab, sfx in MODELS:
            for ds, dslab in LADDER:
                d = load("posthoc_ground", run.format(ds=ds, s=sfx), tag, ds=ds)
                if not d:
                    continue
                bank["posthoc_ladder"][f"{lab}|{m}|{ds}"] = d["by_source"]
                for s in ("look", "read"):
                    b = d["by_source"].get(s)
                    if not b:
                        continue
                    lines.append(f"{lab if first else ''} & {mlab if s == 'look' else ''} & {dslab if s == 'look' else ''} & "
                                 f"$\\langle${s}$\\rangle$ & {F(b['accuracy'])} & {F(b['own_auroc'])} & "
                                 f"{b['own_n_grounded']}/{b['own_n_ungrounded']} & {F(b['cand_auroc'])} & "
                                 f"{F(b['attrib_by_necessity'])} & {F(b['which_source_acc'])} \\\\")
                    first = False
        lines.append("\\addlinespace")
    open(f"{OUT}/tab_app_posthoc_ladder.tex", "w").write(tail(lines)); print("[saved] tab_app_posthoc_ladder.tex")
    # attention on the ladder
    lines = head(["Interface", "Model", "Dataset", "Request", "share img", "share cap", "prec. cap"],
                 "llll" + "c" * 3, "source: attn_ladder_step_800*.json")
    for lab, run, tag in IF_LADDER:
        first = True
        for m, mlab, sfx in MODELS:
            for ds, dslab in LADDER:
                d = load("attn_ladder", run.format(ds=ds, s=sfx), tag, ds=ds)
                if not d:
                    continue
                bank["attn_ladder"][f"{lab}|{m}|{ds}"] = {s: {k: v for k, v in b.items() if k != "layer_profile"}
                                                          for s, b in d["by_source"].items()}
                for s in ("look", "read"):
                    b = d["by_source"].get(s)
                    if not b:
                        continue
                    lines.append(f"{lab if first else ''} & {mlab if s == 'look' else ''} & {dslab if s == 'look' else ''} & "
                                 f"$\\langle${s}$\\rangle$ & {F(b['share']['img'])} & {F(b['share']['cap'])} & "
                                 f"{F(b['prec_cap'])} \\\\")
                    first = False
        lines.append("\\addlinespace")
    open(f"{OUT}/tab_app_attn_ladder.tex", "w").write(tail(lines)); print("[saved] tab_app_attn_ladder.tex")


def head(cols, spec, note):
    return ["% AUTO-GENERATED by scripts/applications_groundcf.py -- do not edit", f"% {note}",
            "\\begin{tabular}{" + spec + "}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]


def tail(lines):
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def main():
    bank = {"compose": {}, "posthoc": {}, "attn": {}, "posthoc_ladder": {}, "attn_ladder": {},
            "compose_trained_clevr3": {}, "compose_trained_ladder": {}}
    bank["compose_trained_clevr3"] = compose_table("tab_app_compose_trained.tex", "clevr3", [("clevr3", "Ground-CF")], "look|read")
    bank["compose_trained_ladder"] = compose_table("tab_app_compose_trained_ladder.tex", "ladder", LADDER, "look|read")
    ladder_tables(bank)
    # ---------------------------------------------------------------- compose
    lines = head(["Interface", "Model", "detect", "attribute", "false alarm", "balanced",
                  "detect (reversed order)"], "ll" + "c" * 5,
                 "source: compose_detect_step_800*.json, pair look|read (and read|look)")
    for lab, run, tag in IF:
        first = True
        for m, mlab, sfx in MODELS:
            d = load("compose_detect", run.format(s=sfx), tag)
            if not d:
                continue
            p = d["pairs"].get("look|read"); r = d["pairs"].get("read|look")
            bank["compose"][f"{lab}|{m}"] = {"pairs": d["pairs"], "n_items": d["meta"]["n_items"]}
            if p:
                lines.append(f"{lab if first else ''} & {mlab} & {F(p['detect'])} & {F(p['attrib'])} & "
                             f"{F(p['false'])} & {F(p['balanced'])} & {F(r['detect']) if r else '--'} \\\\")
                first = False
        lines.append("\\addlinespace")
    open(f"{OUT}/tab_app_compose.tex", "w").write(tail(lines)); print("[saved] tab_app_compose.tex")
    # ---------------------------------------------------------------- posthoc
    lines = head(["Interface", "Model", "Source", "acc.", "own AUROC", "$n$ g/u", "cand AUROC",
                  "attrib.", "which source"], "lll" + "c" * 6,
                 "source: posthoc_ground_step_800*.json")
    for lab, run, tag in IF:
        first = True
        for m, mlab, sfx in MODELS:
            d = load("posthoc_ground", run.format(s=sfx), tag)
            if not d:
                continue
            bank["posthoc"][f"{lab}|{m}"] = d["by_source"]
            for s in ("look", "read", "ocr"):
                b = d["by_source"].get(s)
                if not b:
                    continue
                # an AUROC with zero ungrounded answers is UNDEFINED, not missing: say so, or a
                # reader counts a perfectly grounded cell as an experiment we failed to run.
                au = ("\\textit{undef.}" if (b.get('own_n_ungrounded') or 0) == 0
                      else F(b['own_auroc']))
                lines.append(f"{lab if first else ''} & {mlab if s == 'look' else ''} & "
                             f"$\\langle${s}$\\rangle$ & {F(b['accuracy'])} & {au} & "
                             f"{b['own_n_grounded']}/{b['own_n_ungrounded']} & {F(b['cand_auroc'])} & "
                             f"{F(b['attrib_by_necessity'])} & {F(b['which_source_acc'])} \\\\")
                first = False
        lines.append("\\addlinespace")
    open(f"{OUT}/tab_app_posthoc.tex", "w").write(tail(lines)); print("[saved] tab_app_posthoc.tex")
    # ---------------------------------------------------------------- attention
    lines = head(["Interface", "Model", "Request", "share img", "share cap", "share ocr",
                  "prec. box", "prec. box (no sink)", "prec. cap", "prec. ocr"], "lll" + "c" * 7,
                 "source: attn_groundcf_step_800*.json; share = mass on that modality over the three; "
                 "prec. = share of the modality's mass on the evidence (box / value token); "
                 "no sink = the strongest non-evidence image cell removed from the image mass")
    for lab, run, tag in IF:
        first = True
        for m, mlab, sfx in MODELS:
            d = load("attn_groundcf", run.format(s=sfx), tag)
            if not d:
                continue
            bank["attn"][f"{lab}|{m}"] = {s: {k: v for k, v in b.items() if k != "layer_profile"}
                                         for s, b in d["by_source"].items()}
            bank["attn"][f"{lab}|{m}"]["layers"] = d["meta"]["layers"]
            for s in ("look", "read", "ocr"):
                b = d["by_source"].get(s)
                if not b:
                    continue
                lines.append(f"{lab if first else ''} & {mlab if s == 'look' else ''} & $\\langle${s}$\\rangle$ & "
                             f"{F(b['share']['img'])} & {F(b['share']['cap'])} & {F(b['share']['ocr'])} & "
                             f"{F(b['prec_img'])} & {F(b.get('prec_img_nosink'))} & {F(b['prec_cap'])} & {F(b['prec_ocr'])} \\\\")
                first = False
        lines.append("\\addlinespace")
    open(f"{OUT}/tab_app_attn.tex", "w").write(tail(lines)); print("[saved] tab_app_attn.tex")
    json.dump({"note": "Ground-CF application batteries: zero-shot composed detection, post hoc "
                       "grounding verifier, attention allocation; per interface and model",
               **bank}, open(f"{ROOT}/results/applications_groundcf.json", "w"), indent=1)
    print("[saved] results/applications_groundcf.json  "
          f"(compose {len(bank['compose'])}, posthoc {len(bank['posthoc'])}, attn {len(bank['attn'])} cells)")


if __name__ == "__main__":
    main()
