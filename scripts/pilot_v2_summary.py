#!/usr/bin/env python
"""Protocol-v2 pilot summary (2026-10-02): one table over interface x objective x condition x split, with the
task-transfer split (trained colour/shape vs held-out count/spatial) from the per-skill records.

Runs: clevr3_v2_5tok_{psft,psftctm,psftapo,psftctmsm,psftctmce}_{neo,pt,ptslot}_holdcs_seed0{sfx}, clevr3_v2_lora_r16_joint_{psft,psftctm}_holdcs_seed0{sfx};
frozen baselines are evaluated through the CTM word run's dir with tags _v2_instructed / _v2_noop.
Conditions: v2ref (training question + training carrier), v2q (held-out question), v2c (held-out carrier), v2 (both).
Metrics: A = Accuracy, F = conjunctive Flip, S = conjunctive Stay (look_stay_c / read_stay_c); image request first, then caption.
Usage: python scripts/pilot_v2_summary.py [--sfx _qwen35] [--out results/pilot_v2_gemma_summary]
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [("Neologism", "psft", "clevr3_v2_5tok_psft_neo_holdcs_seed0{s}"), ("Prefix", "psft", "clevr3_v2_5tok_psft_pt_holdcs_seed0{s}"),
        ("Prefix in slot", "psft", "clevr3_v2_5tok_psft_ptslot_holdcs_seed0{s}"), ("Joint LoRA", "psft", "clevr3_v2_lora_r16_joint_psft_holdcs_seed0{s}"),
        ("Neologism", "psft+ctm", "clevr3_v2_5tok_psftctm_neo_holdcs_seed0{s}"), ("Prefix", "psft+ctm", "clevr3_v2_5tok_psftctm_pt_holdcs_seed0{s}"),
        ("Prefix in slot", "psft+ctm", "clevr3_v2_5tok_psftctm_ptslot_holdcs_seed0{s}"), ("Joint LoRA", "psft+ctm", "clevr3_v2_lora_r16_joint_psftctm_holdcs_seed0{s}"),
        ("Neologism", "psft+apo", "clevr3_v2_5tok_psftapo_neo_holdcs_seed0{s}"), ("Prefix", "psft+apo", "clevr3_v2_5tok_psftapo_pt_holdcs_seed0{s}"),
        ("Neologism", "psft+ctm+stay", "clevr3_v2_5tok_psftctmsm_neo_holdcs_seed0{s}"), ("Prefix", "psft+ctm+stay", "clevr3_v2_5tok_psftctmsm_pt_holdcs_seed0{s}"),
        ("Prefix in slot", "psft+ctm+stay", "clevr3_v2_5tok_psftctmsm_ptslot_holdcs_seed0{s}"), ("Joint LoRA", "psft+ctm+stay", "clevr3_v2_lora_r16_joint_psftctmsm_holdcs_seed0{s}"),
        ("Neologism", "psft+ctm+ce(ctx)", "clevr3_v2_5tok_psftctmce_neo_holdcs_seed0{s}"), ("Prefix", "psft+ctm+ce(ctx)", "clevr3_v2_5tok_psftctmce_pt_holdcs_seed0{s}")]
FROZEN = [("English", "frozen", "clevr3_v2_5tok_psftctm_neo_holdcs_seed0{s}", "_v2_instructed"), ("No request", "frozen", "clevr3_v2_5tok_psftctm_neo_holdcs_seed0{s}", "_v2_noop")]
CONDS = [("v2ref", "reference"), ("v2q", "question shift"), ("v2c", "carrier shift"), ("v2", "both shifts")]
TRAINED, HELD = ("color", "object"), ("count", "spatial")
f3 = lambda x: "--" if x is None else f"{x:.3f}"


def load(run, split, tag):
    p = os.path.join(ROOT, "runs", run, "eval", f"clevr3_{split}", f"flip_step_800{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def skill_mean(d, skills, key):
    vals = [d["per_skill"][s][key] for s in skills if s in d.get("per_skill", {}) and d["per_skill"][s].get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sfx", default=""); ap.add_argument("--out", default=None); ap.add_argument("--ntok", type=int, default=5); ap.add_argument("--data", default="holdcs", choices=["holdcs", "allattr"])
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "results", f"pilot_v2_gemma{a.sfx or ''}" + (f"_{a.ntok}tok" if a.ntok != 5 else "") + (f"_{a.data}" if a.data != "holdcs" else "") + "_summary")
    _sub = lambda r: r.replace("5tok", f"{a.ntok}tok").replace("holdcs", a.data)
    global RUNS, FROZEN
    RUNS = [(i, o, _sub(r)) for i, o, r in RUNS]; FROZEN = [(i, o, _sub(r), t) for i, o, r, t in FROZEN]
    rows, bank = [], {"runs": {}, "note": __doc__.split("\n")[0]}
    lines = ["# Protocol-v2 pilot summary" + (f" ({a.sfx})" if a.sfx else " (Gemma-3-4B)") + f" L={a.ntok}, data={a.data}", "",
             "Task holdout: count and spatial held out of training for both requests; 'trained' = colour+shape, 'held-out tasks' = count+spatial.",
             "A/F/S = Accuracy / conjunctive Flip / conjunctive Stay. Chance: 3 candidates (A .33, F .11).", ""]
    for split in ("test_id", "test_ood"):
        lines += [f"## {split}", "", "| interface | objective | condition | img A | img F | img S | cap A | cap F | cap S | img F trained | img F held-out tasks | cap F trained | cap F held-out tasks |", "|" + "---|" * 13]
        for (iface, obj, runt) in RUNS + [(n, o, r) for n, o, r, _ in FROZEN]:
            run = runt.format(s=a.sfx)
            if obj != "frozen":
                tags = CONDS
            else:
                _m = next(t for n_, o_, r_, t in FROZEN if n_ == iface)          # "_v2_instructed" / "_v2_noop"
                tags = [("v2ref" + _m[3:], "reference"), (_m[1:], "both shifts")]   # _v2ref_<mode> (2026-10-02), _v2_<mode>
            for tag, cname in tags:
                d = load(run, split, "_" + tag if obj != "frozen" else "_" + tag)
                if d is None:
                    rows.append({"split": split, "interface": iface, "objective": obj, "condition": cname, "missing": True}); continue
                r = {"split": split, "interface": iface, "objective": obj, "condition": cname, "n": d.get("n"),
                     "img_A": d.get("look_acc"), "img_F": d.get("look_flip"), "img_S": d.get("look_stay_c"),
                     "cap_A": d.get("read_acc"), "cap_F": d.get("read_flip"), "cap_S": d.get("read_stay_c"),
                     "img_F_trained": skill_mean(d, TRAINED, "look_flip"), "img_F_held": skill_mean(d, HELD, "look_flip"),
                     "cap_F_trained": skill_mean(d, TRAINED, "read_flip"), "cap_F_held": skill_mean(d, HELD, "read_flip"),
                     "img_S_trained": skill_mean(d, TRAINED, "look_stay_c"), "img_S_held": skill_mean(d, HELD, "look_stay_c"),
                     "file": os.path.relpath(os.path.join(ROOT, "runs", run, "eval", f"clevr3_{split}", f"flip_step_800_{tag}.json"), ROOT)}
                rows.append(r)
                lines.append(f"| {iface} | {obj} | {cname} | {f3(r['img_A'])} | {f3(r['img_F'])} | {f3(r['img_S'])} | {f3(r['cap_A'])} | {f3(r['cap_F'])} | {f3(r['cap_S'])} | {f3(r['img_F_trained'])} | {f3(r['img_F_held'])} | {f3(r['cap_F_trained'])} | {f3(r['cap_F_held'])} |")
        lines.append("")
    missing = [r for r in rows if r.get("missing")]
    lines.append(f"Missing result files: {len(missing)}" + (" -> " + "; ".join(f"{m['interface']}/{m['objective']}/{m['condition']}/{m['split']}" for m in missing[:12]) if missing else ""))
    bank["rows"] = rows
    json.dump(bank, open(out + ".json", "w"), indent=1); open(out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\n[saved] {out}.md / .json")


if __name__ == "__main__":
    main()
