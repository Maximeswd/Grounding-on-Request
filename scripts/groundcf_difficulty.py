#!/usr/bin/env python
"""What Ground-CF measures and where it is saturated: (a) <look> flip by twin type (colour recolour
vs shape redraw) per interface and model; (b) test_id vs test_ood flips for the interfaces evaluated
on both; (c) zero-shot routing accuracy on the held (source, skill) cells. Banks
results/groundcf_difficulty.json (read by the audit). Usage: python scripts/groundcf_difficulty.py
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [("No request", "clevr3_f_5tok_sft_seed0{s}", "_noop"), ("English instruction", "clevr3_f_5tok_sft_seed0{s}", "_instr_iv5"),
        ("Prompt tuning", "clevr3_f_pt_5tok_seed0{s}", ""), ("LoRA r=4", "clevr3_f_lora_r4_seed0{s}", ""),
        ("Neologism SFT", "clevr3_f_5tok_sft_seed0{s}", ""), ("Neologism + CTM", "clevr3_f_5tok_ctm_seed0{s}", "")]
SFX = {"gemma3": "", "qwen4b": "_qwen35", "qwen2b": "_qwen2b", "e4b": "_gemma4e4b"}
HELD = [("Neologism + CTM", "clevr3_f_5tok_ctm_seed0", "srconly_explicit_step_800.json"),
        ("Neologism SFT", "clevr3_f_5tok_sft_seed0", "srconly_explicit_step_800.json"),
        ("English instruction", "clevr3_f_5tok_sft_seed0", "instructed_explicit_step_800.json"),
        ("No request", "clevr3_f_5tok_sft_seed0", "noop_explicit_step_800.json")]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    out = {"twin_type": {}, "ood": {}, "held_cells": {}}
    for lab, run, tag in RUNS:
        for m, s in SFX.items():
            d = load(f"{ROOT}/runs/{run.format(s=s)}/eval/clevr3_test_id/flip_step_800{tag}.json")
            if d and d.get("per_skill"):
                out["twin_type"][f"{lab}|{m}"] = {"colour_look_flip": d["per_skill"].get("color", {}).get("look_flip"),
                                                  "shape_look_flip": d["per_skill"].get("object", {}).get("look_flip")}
            o = load(f"{ROOT}/runs/{run.format(s=s)}/eval/clevr3_test_ood/flip_step_800{tag}.json")
            if d and o:
                out["ood"][f"{lab}|{m}"] = {k: {"id": d[f"{k}_flip"], "ood": o[f"{k}_flip"]} for k in ("look", "read", "ocr")}
    for lab, run, f in HELD:
        for m, s in SFX.items():
            d = load(f"{ROOT}/runs/{run}{s}/eval/clevr3_test_id/{f}")
            if d and d.get("zero_shot"):
                out["held_cells"][f"{lab}|{m}"] = {"cells": d["zero_shot"], "mean": d.get("zero_shot_mean")}
    json.dump({"note": "Ground-CF difficulty facts: <look> flip by twin type; test_id vs test_ood; zero-shot routing accuracy "
                       "on the held (source, skill) cells look:spatial, read:count, ocr:object (3-way chance .33)", **out},
              open(f"{ROOT}/results/groundcf_difficulty.json", "w"), indent=1)
    print("twin type (colour / shape) look flip:")
    for k, v in out["twin_type"].items():
        print(f"  {k:28s} {v['colour_look_flip']} / {v['shape_look_flip']}")
    print("held cells (zero-shot routing accuracy):")
    for k, v in out["held_cells"].items():
        print(f"  {k:28s} {v['cells']}  mean {v['mean']}")
    print(f"[saved] results/groundcf_difficulty.json  (ood cells: {len(out['ood'])})")


if __name__ == "__main__":
    main()
