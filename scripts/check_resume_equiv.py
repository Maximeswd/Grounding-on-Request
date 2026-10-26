#!/usr/bin/env python
"""Resume-fidelity gate (2026-10-07). Compares the step-20 parameters of: A (uninterrupted 20 steps), A2 (A repeated:
GPU noise floor), B (resumed from A's step_10 with full state), C (resumed with fresh AdamW = restarted continuation).
PASS when max|B-A| is within ~2x the noise floor max|A2-A| and clearly below max|C-A|.
Usage: python scripts/check_resume_equiv.py [--prefix smoke_resume_] [--step 20]"""
import argparse, glob, os, sys, torch
ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="smoke_resume_"); ap.add_argument("--step", type=int, default=20); a = ap.parse_args()
def load(run):
    d = f"runs/{run}/ckpt/step_{a.step}"
    if os.path.exists(f"{d}/neo.pt"):
        return torch.load(f"{d}/neo.pt", map_location="cpu")["rows"].float()
    from safetensors.torch import load_file
    sd = load_file(f"{d}/adapter_model.safetensors"); return torch.cat([v.float().flatten() for _, v in sorted(sd.items())])
rc = 0
for grp, names in (("word (neo rows)", ("A", "A2", "B", "C")), ("LoRA (adapter)", ("lA", "lA2", "lB", "lC"))):
    try:
        T = {n: load(a.prefix + n) for n in names}
    except FileNotFoundError as e:
        print(f"[{grp}] missing: {e}"); rc = 2; continue
    ref = T[names[0]]; print(f"[{grp}] |A| = {ref.norm():.4f}, {ref.numel()} values")
    res = {}
    for n in names[1:]:
        d = (T[n] - ref); res[n] = (d.abs().max().item(), (d.norm() / ref.norm()).item())
        print(f"  max|{n}-A| = {res[n][0]:.3e}   ||{n}-A||/||A|| = {res[n][1]:.3e}")
    floor, b, c = res[names[1]][0], res[names[2]][0], res[names[3]][0]
    ok = b <= max(2 * floor, 1e-6) and (c > b or c == 0 and b == 0)
    print(f"  -> {'PASS' if ok else 'FAIL'}: full-state resume within 2x the repeat-noise floor ({b:.2e} vs floor {floor:.2e}); restarted deviates {c:.2e}")
    rc = rc or (0 if ok else 1)
sys.exit(rc)
