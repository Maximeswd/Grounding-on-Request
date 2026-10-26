#!/usr/bin/env python
"""Assert that every pre-carrier-fix CTM run on a prefix or LoRA carrier is voided or marked _v2_.

Run after ANY edit to results/void_run_patterns.txt. The 2026-09-21 patterns missed three naming
variants and pre-fix runs were still backing Table 1 the next morning (STATUS v1.78), so this check
enumerates runs by CONFIG FLAG and CHECKPOINT DATE rather than trusting the name patterns.
Exit 2 if anything is uncaught."""
import re, glob, os, json, time, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.environ.get("SC_CARRIER_FIX", "2026-09-21 04:00")
fix_t = time.mktime(time.strptime(FIX, "%Y-%m-%d %H:%M"))
pats = [re.compile(l.strip()) for l in open(f"{ROOT}/results/void_run_patterns.txt")
        if l.strip() and not l.startswith("#")]
bad = []
for d in sorted(glob.glob(f"{ROOT}/runs/*/ckpt/step_800")):
    r = d.split("/runs/")[1].split("/")[0]
    if not re.match(r"^(color|coco|vcf)_(pt|pti|lora)", r):
        continue
    f = f"{ROOT}/runs/{r}/config.json"
    a = json.load(open(f))["args"] if os.path.exists(f) else {}
    if not (a.get("lam_ctm") or 0):
        continue
    if os.path.getmtime(d + "/meta.json") < fix_t and not any(p.search(r) for p in pats) and "_v2_" not in r:
        bad.append(r)
print(f"[void] fix date {FIX}; {len(pats)} patterns; uncaught pre-fix prefix/LoRA CTM runs: {len(bad)}")
for r in bad:
    print("   UNCAUGHT", r)
sys.exit(2 if bad else 0)
