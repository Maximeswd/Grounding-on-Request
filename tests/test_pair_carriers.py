#!/usr/bin/env python
"""Every pair builder must carry the METHOD'S OWN request (2026-09-21).

Found the hard way: build_ctm_pair rendered the neologism strings for every method, so a PT or LoRA
run's CTM pairs contained no trainable parameter, the margin had requires_grad False and _bw skipped it
silently. The run trained as plain SFT while its config said lam_ctm=1. This test builds one pair per
(method x builder x op) with encoding stubbed out and checks the user prompt text. CPU, ~20 s.
Run: python tests/test_pair_carriers.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scsrc.preference as P
from scsrc import data as D

OWN = {"neo": lambda op: f"<{op}_ctrl_0>", "pt": lambda op: f"<pt_{op}_0>", "lora": lambda op: P.LORA_INSTR[op]}


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "color", "train")
    if not os.path.isdir(root):
        print("SKIP: data/color/train not on disk"); return 0
    from datasets import load_from_disk
    P._seq = lambda processor, img, user, cont, device: {"user": user, "cont": cont}
    ds = load_from_disk(root)
    s = next(x for x in D.build_specs(ds, "train", conditions=("conflict",)) if x.cond == "conflict")
    fails = []
    for method, arm in (("neo", "ctrl"), ("pt", "na"), ("lora", "na")):
        for name, fn in (("APO", P.build_pref_pair), ("CTM", P.build_ctm_pair), ("SM", P.build_sm_pair)):
            for op in ("look", "read"):
                r = fn(None, ds, s, arm, op, 5, "cpu", method=method)
                if r is None:
                    fails.append(f"{method}/{name}/{op}: builder returned None"); continue
                users = [v["user"] for v in r.values() if isinstance(v, dict) and "user" in v]
                want = OWN[method](op)
                foreign = [OWN[m](op) for m in OWN if m != method]
                for u in users:
                    if want not in u:
                        fails.append(f"{method}/{name}/{op}: own carrier missing: {u[:90]!r}")
                    for bad in foreign:
                        if bad in u:
                            fails.append(f"{method}/{name}/{op}: foreign carrier present: {u[:90]!r}")
    if fails:
        print("PAIR CARRIER FAILURES:"); [print("  " + f) for f in fails]; return 1
    print("pair carriers OK: neo/pt/lora x APO/CTM/SM x look/read all carry their own request"); return 0


if __name__ == "__main__":
    sys.exit(main())
