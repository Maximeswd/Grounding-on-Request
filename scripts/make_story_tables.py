#!/usr/bin/env python
"""Story tables for "correct is not grounded" and the weak-link claim (2026-09-13).

METRIC NOTE, and why the shading is what it is. On every dataset flip is scored as a CONJUNCTION:
the answer matches the requested modality's value in the original context AND follows the edit. So
flip is a subset of accuracy on the same items and can never exceed it; printing "flip below
accuracy" would therefore be structural, not a finding. The informative quantity is how much of the
correct mass survives the edit, the conditional flip = flip / accuracy; we shade its complement, the
share of correct answers that ignore the edit. STAY is NOT conditioned on correctness (it asks only
whether the answer is unchanged when a competing modality is edited), so a model can be stably wrong
and stay may legitimately exceed accuracy; stay is never shaded.
DENOMINATORS: on Ground-CF look_acc is scored over all items while look_flip is scored on the
image-twin subset, so the accuracy printed next to the Ground-CF image flip is R.looktwin.

  tab_story_all.tex        Table 1: four model blocks x the ways of asking a frozen model.
  tab_story_all_lora.tex   the same with LoRA r16 added, for the appendix.
  tab_story_<model>.tex    one block per model (appendix).
  tab_floor.tex            the weak link: pooled flip per word and the weakest cell, per model.
  results/story_tables.json   bank (ignore rates, shaded counts, floors, row wins).
Flags: --ood reads results/master_grid_ood.json and suffixes every output with _ood.
Usage: python scripts/make_story_tables.py [--ood]
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "iclr2027", "tables")
OOD = "--ood" in sys.argv
SFX = "_ood" if OOD else ""
IGN_LIGHT, IGN_DARK = .10, .25          # share of correct answers that ignore the edit
DS = [("vcf", "VCF"), ("color", "ColorSwap"), ("coco", "COCO-CF"), ("groundcf", "Ground-CF")]
# 2026-09-16: VCF leaves the MAIN table. Its three indicators coincide by construction (the twin is an
# agreement context and Flip|correct = .965), so it cannot carry the S4 argument that accuracy overstates
# control. It stays in the appendix grids and the transfer figure as the calibration control: it is what
# licenses reading COCO-CF's .88 as a property of photographs rather than of the evaluator.
DS_MAIN = [d for d in DS if d[0] != "vcf"]
MOD = [("gemma3", "Gemma-3-4B"), ("qwen4b", "Qwen-3.5-4B"), ("qwen2b", "Qwen-3.5-2B"), ("e4b", "Gemma-4-E4B")]
WORDS = [("look", "\\lk{}: the image"), ("read", "\\rdw{}: the caption")]
FROZEN = "No request"
# 2026-09-15 float cut: the main table keeps the two UNTRAINED rows plus the CTM word, which is what the
# S4 argument needs (correct answers that do not follow the edit, and that a trained word repairs them).
# Prompt tuning + CTM, LoRA and every other arm are in the appendix grids (tab_big_*).
# 2026-09-16: the SFT rows were added and then removed -- with six rows the CTM contrast stopped
# reading, and SFT vs CTM is the whole point of Fig. objheat, so it is made there instead. Table 1
# keeps the two untrained interfaces and the two learned ones AT A MATCHED OBJECTIVE.
# Both learned rows carry the SAME objective, so "+ CTM" twice is redundant ink; the caption states it
# once. FLOOR keeps explicit objective names because that table mixes them (author, 2026-09-16).
REP = [("noop|None", FROZEN), ("instr|None", "Instruction"),
       ("pt+ctm|5", "Prompt tuning"), ("word+ctm|5", "Neologism")]
REP_LORA = REP + [("lora_r16|None", "LoRA $r{=}16$")]
FLOOR = [("instr|None", "Instruction"), ("pt|5", "Prompt tuning, SFT"), ("pt+apo|5", "Prompt tuning + APO"),
         ("pt+ctm|5", "Prompt tuning + CTM"), ("lora_r16|None", "LoRA $r{=}16$"), ("word_sft|5", "Neologism, SFT"),
         ("word+apo|5", "Neologism + APO"), ("word+ctm|5", "Neologism + CTM")]
SHADE, SHADE_LIGHT = "\\cellcolor{gray!35}", "\\cellcolor{gray!13}"
SHORT = {"vcf": "VCF", "color": "CSw", "coco": "COCO", "groundcf": "GCF"}
F = lambda v: "--" if v is None else (f"{v:.2f}".lstrip("0") if 0 < v < 1 else f"{v:.2f}")


def cell(g, ds, m, key, k, w):
    """k in R/G/S. The accuracy paired with the Ground-CF image flip is the twin-subset one."""
    d = g.get(f"{ds}|{m}|{key}") or {}
    if k == "R" and ds == "groundcf" and w == "look" and d.get("R.looktwin") is not None:
        return d["R.looktwin"]
    return d.get(f"{k}.{w}")


def ignore_rate(a, f):
    return None if (a is None or f is None or a <= 0) else max(0.0, 1 - f / a)


def story_block(g, m, bank, rows, tag="", words=None, ds_list=None):
    words = words or WORDS
    DS = ds_list or globals()["DS"]
    lines = []
    best = {(w, ds, k): max([x for x in (cell(g, ds, m, key, k, w) for key, _ in rows) if x is not None] or [None])
            for w, _ in words for ds, _ in DS for k in ("R", "G", "Sc")}
    for key, lab in rows:
        cells, nsh = [], {"light": 0, "dark": 0}
        for w, _ in words:
            for ds, _ in DS:
                a, f, s_ = (cell(g, ds, m, key, k, w) for k in ("R", "G", "Sc"))
                ig = ignore_rate(a, f)
                bank["ignore"][f"{m}|{lab}|{ds}|{w}{tag}"] = None if ig is None else round(ig, 4)
                out = []
                for k, v in (("R", a), ("G", f), ("Sc", s_)):
                    # best in the column within the model block, i.e. per dataset x word x metric.
                    # RESTORED 2026-09-14 at the author's request. It marks the best value in each
                    # column and nothing more: with LoRA in the table the marks split by modality
                    # (LoRA takes the image columns, the neologism the caption columns), which is the
                    # asymmetry Sec. 7 quantifies. Do not read a count of these marks as a ranking.
                    t = F(v)
                    b = best.get((w, ds, k))
                    if v is not None and b is not None and v >= b - .005:
                        t = f"\\textbf{{{t}}}"
                    if k == "G" and ig is not None:
                        if ig >= IGN_DARK:
                            t = SHADE + t; nsh["dark"] += 1
                        elif ig >= IGN_LIGHT:
                            t = SHADE_LIGHT + t; nsh["light"] += 1
                    out.append(t)
                cells += out
        bank["shaded"][f"{m}|{lab}{tag}"] = nsh
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    return lines


def story_head(words=None, ds_list=None):
    words = words or WORDS
    DS = ds_list or globals()["DS"]
    n = len(words) * len(DS)
    # light rule between dataset groups so the three-metric blocks read apart
    _sep = "!{\\color{gray!40}\\vrule}"
    head = ["\\begin{tabular}{l" + _sep.join(["ccc"] * n) + "}", "\\toprule"]
    if len(words) > 1:
        w_span = 3 * len(DS)
        head += [" & " + " & ".join(f"\\multicolumn{{{w_span}}}{{c}}{{{wl}}}" for _, wl in words) + " \\\\",
                 "".join(f"\\cmidrule(lr){{{2 + i * w_span}-{1 + (i + 1) * w_span}}}" for i in range(len(words)))]
    head += [" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{dl}}}" for _ in words for _, dl in DS) + " \\\\",
             "Method & " + " & ".join(["A & F & S"] * n) + " \\\\", "\\midrule"]
    return head


def main():
    g = json.load(open(f"{ROOT}/results/master_grid{SFX}.json"))
    bank = {"note": (f"split={'test_ood' if OOD else 'test_id'}; L=5 parity. flip is a subset of accuracy by "
                     "construction (correct in the original context AND follows the edit), so the shading marks the "
                     f"IGNORE RATE 1-flip/acc: light >= {IGN_LIGHT}, dark >= {IGN_DARK}. stay is not conditioned on "
                     "correctness and is never shaded. Ground-CF image accuracy is the twin-subset accuracy "
                     "(R.looktwin), the same items the image flip is scored on."),
            "ignore_light": IGN_LIGHT, "ignore_dark": IGN_DARK,
            "ignore": {}, "shaded": {}, "floor": {}, "region_cells": {}}
    hdr = [f"% AUTO-GENERATED by scripts/make_story_tables.py from results/master_grid{SFX}.json -- do not edit",
           f"% shading = share of correct answers that ignore the edit (1-flip/acc): light >= {IGN_LIGHT}, dark >= {IGN_DARK}",
           "% bold = best value in the column within the model block (per dataset x word x metric);\n% Flip shading = share of correct answers that ignore the edit; stay is never shaded"]
    # Main Table 1 is the four reference interfaces; LoRA is compared across every rank in Sec. 7
    # and appears per cell in the appendix grids (author's call, 2026-09-14).
    for rows, name, tag, dsl in ((REP, "tab_story_all", "", DS_MAIN),
                                 (REP_LORA, "tab_story_all_lora", "|lora", DS)):
        stacked = []
        ncol = 1 + 3 * len(WORDS) * len(dsl)
        for m, ml in MOD:
            body = story_block(g, m, bank, rows, tag, ds_list=dsl)
            if name == "tab_story_all_lora":
                open(f"{OUT}/tab_story_{m}{SFX}.tex", "w").write(
                    "\n".join(hdr + story_head(ds_list=dsl) + body + ["\\bottomrule", "\\end{tabular}"]) + "\n")
            stacked += [f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{ml}}}}} \\\\"] + body + ["\\addlinespace[3pt]"]
        open(f"{OUT}/{name}{SFX}.tex", "w").write(
            "\n".join(hdr + story_head(ds_list=dsl) + stacked[:-1] + ["\\bottomrule", "\\end{tabular}"]) + "\n")
    # ONE WORD PER TABLE (12 numeric columns instead of 24) for the main text
    for w, wl in WORDS:
        stacked = []
        for m, ml in MOD:
            body = story_block(g, m, bank, REP, tag=f"|{w}", words=[(w, wl)], ds_list=DS_MAIN)
            stacked += [f"\\multicolumn{{{1 + 3 * len(DS_MAIN)}}}{{l}}{{\\textit{{{ml}}}}} \\\\"] + body + ["\\addlinespace[3pt]"]
        open(f"{OUT}/tab_story_{w}{SFX}.tex", "w").write(
            "\n".join(hdr + story_head(words=[(w, wl)], ds_list=DS_MAIN) + stacked[:-1] + ["\\bottomrule", "\\end{tabular}"]) + "\n")

    for key, lab in REP + REP_LORA + FLOOR:
        n = nc = nd = 0
        for m, _ in MOD:
            for ds, _ in DS:
                for w, _ in WORDS:
                    ig = ignore_rate(cell(g, ds, m, key, "R", w), cell(g, ds, m, key, "G", w))
                    if ig is None:
                        continue
                    nc += 1; n += ig >= IGN_LIGHT; nd += ig >= IGN_DARK
        bank["region_cells"][lab] = {"ignore_ge_light": n, "ignore_ge_dark": nd, "cells": nc}

    # floor table: pooled flip per word and the weakest of the eight cells, per model
    lines = hdr[:1] + ["\\begin{tabular}{l" + "ccc" * 4 + "}", "\\toprule",
                       " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{ml}}}" for _, ml in MOD) + " \\\\",
                       "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}",
                       "Method & " + " & ".join(["\\lk{} & \\rdw{} & weakest cell"] * 4) + " \\\\", "\\midrule"]
    vals = {}
    for key, lab in FLOOR:
        row = []
        for m, _ in MOD:
            pl = [cell(g, ds, m, key, "G", "look") for ds, _ in DS]
            pr = [cell(g, ds, m, key, "G", "read") for ds, _ in DS]
            allc = [(cell(g, ds, m, key, "G", w), f"{SHORT[ds]} {'img' if w == 'look' else 'cap'}")
                    for ds, _ in DS for w, _ in WORDS]
            ok = all(v is not None for v in pl + pr)
            L = sum(pl) / 4 if ok else None
            R = sum(pr) / 4 if ok else None
            mn = min(allc) if ok else (None, "")
            row.append([L, R, mn[0], mn[1]])
            bank["floor"][f"{m}|{lab}"] = {"look": L, "read": R, "min": mn[0], "min_cell": mn[1]}
        vals[lab] = row
    for mi in range(4):
        for ci in range(3):
            col = [vals[lab][mi][ci] for _, lab in FLOOR if vals[lab][mi][ci] is not None]
            b = max(col) if col else None
            for _, lab in FLOOR:
                v = vals[lab][mi][ci]
                t = F(v)
                if v is not None and b is not None and v >= b - .005:
                    t = f"\\textbf{{{t}}}"
                if ci == 2 and v is not None:
                    t += f" {{\\scriptsize {vals[lab][mi][3]}}}"
                vals[lab][mi][ci] = t
    for _, lab in FLOOR:
        if all(mb[0] == "--" for mb in vals[lab]):
            continue                      # no complete model block yet
        lines.append(f"{lab} & " + " & ".join(str(x) for mb in vals[lab] for x in mb[:3]) + " \\\\")
    open(f"{OUT}/tab_floor{SFX}.tex", "w").write("\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")

    # flip row wins among the main-text rows (ties within .005)
    wins = {lab: {"total": 0} for _, lab in REP}
    for m, _ in MOD:
        for _, lab in REP:
            wins[lab][m] = 0
        for ds, _ in DS:
            for w, _ in WORDS:
                v = [(cell(g, ds, m, key, "G", w), lab) for key, lab in REP]
                present = [x for x, _ in v if x is not None]
                if not present:
                    continue
                b = max(present)
                for x, lab in v:
                    if x is not None and x >= b - .005:
                        wins[lab][m] += 1; wins[lab]["total"] += 1
    bank["flip_row_wins_main"] = wins
    json.dump(bank, open(f"{ROOT}/results/story_tables{SFX}.json", "w"), indent=1)
    print(f"[saved] tab_story_look{SFX}/read{SFX} (main), tab_story_all{SFX}, tab_story_all_lora{SFX}, tab_story_<model>{SFX} x4, tab_floor{SFX}")
    for lab, r in bank["region_cells"].items():
        print(f"  {lab:24s} ignore >= {IGN_LIGHT}: {r['ignore_ge_light']:2d}, >= {IGN_DARK}: {r['ignore_ge_dark']:2d} of {r['cells']} cells")
    print("  flip row wins:", {lab: v["total"] for lab, v in wins.items()})


if __name__ == "__main__":
    main()
