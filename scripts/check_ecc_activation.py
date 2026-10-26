#!/usr/bin/env python
"""ECC activation gate (2026-10-07): (1) lam_ecc reaches the objective dictionary of the ON run; (2) the training log of the
ON run shows ecc_n > 0 (records carried an ECC pair) and ecc values; (3) the trained rows of ON and OFF (same seed, same
data order, 16 steps) differ, i.e. the term produced gradients on the rows. Usage: python scripts/check_ecc_activation.py"""
import glob, json, os, re, sys, torch
on, off = "smoke_ecc_on", "smoke_ecc_off"; rc = 0
g_on = json.load(open(f"runs/{on}/config.json"))["obj_global"]; g_off = json.load(open(f"runs/{off}/config.json"))["obj_global"]
print(f"(1) obj_global.lam_ecc: ON={g_on.get('lam_ecc')} OFF={g_off.get('lam_ecc')}", "PASS" if g_on.get("lam_ecc", 0) > 0 and not g_off.get("lam_ecc", 0) else "FAIL")
rc |= 0 if g_on.get("lam_ecc", 0) > 0 else 1
logs = [f for f in glob.glob("logs/sc-train6L-*.out") + glob.glob("logs/sc-train6-*.out") if f"--run_name {on} " in open(f, errors="ignore").read() or open(f, errors="ignore").read().rstrip().endswith(on) or f"run_name {on}" in open(f, errors="ignore").read()]
txt = open(sorted(logs, key=os.path.getmtime)[-1], errors="ignore").read().replace("\r", "\n") if logs else ""
eccn = [float(x) for x in re.findall(r"ecc_n=([0-9.]+)", txt)]; ecc = [float(x) for x in re.findall(r"ecc=([0-9.]+)", txt)]
print(f"(2) log {logs[-1] if logs else 'NOT FOUND'}: ecc_n values {eccn[-5:]} ; ecc values {ecc[-5:]}", "PASS" if eccn and max(eccn) > 0 else "FAIL")
rc |= 0 if eccn and max(eccn) > 0 else 1
r_on = torch.load(f"runs/{on}/ckpt/step_16/neo.pt", map_location="cpu")["rows"].float(); r_off = torch.load(f"runs/{off}/ckpt/step_16/neo.pt", map_location="cpu")["rows"].float()
r0 = torch.load(f"runs/{on}/ckpt/step_0/neo.pt", map_location="cpu")["rows"].float()
d_onoff = (r_on - r_off).abs().max().item(); d_train = (r_on - r0).abs().max().item()
print(f"(3) max|rows_ON - rows_OFF| = {d_onoff:.3e} (movement from init: {d_train:.3e})", "PASS" if d_onoff > 1e-6 else "FAIL")
rc |= 0 if d_onoff > 1e-6 else 1
sys.exit(rc)
