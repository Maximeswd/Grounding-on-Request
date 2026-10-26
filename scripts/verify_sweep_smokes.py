#!/usr/bin/env python
"""Gate for the capacity sweeps (2026-10-24): checks the eight smoke runs of configs/smoke_sweep_20261024_*.txt.
For every run: the training finished; the last progress line shows exactly the loss terms its objective requires
(word: ce, ce_twin, ctm, sm; prompt tuning: ce, sm and no ce_twin/ctm; LoRA: ce, ce_twin, sm and no ctm); per-request
adapters logged their source restriction; MVG standard prompt tuning recorded mvg_pt_prefix=1; the smoke evaluation
wrote its output. Prints PASS or FAIL per check and a final line 'GATE: PASS' only if all pass."""
import glob, json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQ = {"word": ({"ce", "ce_twin", "ctm", "sm"}, set()), "pt": ({"ce", "sm"}, {"ce_twin", "ctm"}), "lora": ({"ce", "ce_twin", "sm"}, {"ctm"}),
       "loraad": ({"ce", "ce_twin", "sm"}, {"ctm"})}
RUNS = {"smoke_sweep_vcf_word": "word", "smoke_sweep_vcf_pt": "pt", "smoke_sweep_vcf_lora": "lora", "smoke_sweep_vcf_loraad_look": "loraad",
        "smoke_sweep_mvg_word": "word", "smoke_sweep_mvg_pt": "pt", "smoke_sweep_mvg_lora": "lora", "smoke_sweep_mvg_loraad_vis": "loraad"}
def log_of(run):
    for p in sorted(glob.glob(os.path.join(ROOT, "logs", "sc-trA8-*.out")), key=os.path.getmtime, reverse=True):
        t = open(p, errors="ignore").read()
        if f"--run_name {run}" in t or t.rstrip().endswith(run): return t.replace("\r", "\n")
    return ""
ok_all = True
def check(name, cond):
    global ok_all
    ok_all &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}")
for run, kind in RUNS.items():
    t = log_of(run); d = os.path.join(ROOT, "runs", run)
    check(f"{run}: finished (ckpt/step_12)", os.path.isdir(os.path.join(d, "ckpt", "step_12")))
    prog = [l for l in t.split("\n") if re.search(r"\b12/12\b", l) and "ce=" in l]
    terms = set(re.findall(r"\b(ce|ce_twin|ctm|sm|ce_stay|apo)=", prog[-1])) if prog else set()
    need, forbid = REQ[kind]
    check(f"{run}: loss terms {sorted(terms)} include {sorted(need)}", need <= terms)
    check(f"{run}: no unexpected terms {sorted(forbid & terms)}", not (forbid & terms))
    if kind == "loraad":
        check(f"{run}: training restricted to one source", "training requests restricted to" in t)
        if "mvg" in run: check(f"{run}: MVG items restricted to the adapter's source", "[mvg] train_sources=" in t)
    if run == "smoke_sweep_mvg_pt":
        a = json.load(open(os.path.join(d, "config.json")))["args"] if os.path.exists(os.path.join(d, "config.json")) else {}
        check(f"{run}: config records mvg_pt_prefix=1", a.get("mvg_pt_prefix") == 1)
    if "vcf" in run:
        check(f"{run}: smoke evaluation written", bool(glob.glob(os.path.join(d, "eval", "vcf3_test_id", "flip_ll_step_12*.json"))))
    else:
        check(f"{run}: smoke evaluation written", os.path.exists(os.path.join(d, "eval", "mvg_flip", "flip_step_12_smoke.json")))
print("GATE: PASS" if ok_all else "GATE: FAIL")
