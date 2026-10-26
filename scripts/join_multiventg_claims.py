#!/usr/bin/env python
"""Validate pasted LLM claims and join them with the gold manifest.
Paste raw LLM outputs (reasoning included, anything before 'CLAIMS:' is dropped)
into one or more files under data/multiventG/claims/ , then:

  python scripts/join_multiventg_claims.py \
      --claims_dir data/multiventG/claims --manifest data/multiventG/gold_manifest.jsonl

Writes claims.jsonl (validated, deduped, joined with role_question/certainty)
and claims_errors.txt (every rejected line with its reason).
"""
import argparse, glob, json, os
from collections import defaultdict

REQ = ("video_id", "channel", "role", "claim", "source_text")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims_dir", default="data/multiventG/claims")
    ap.add_argument("--manifest", default="data/multiventG/gold_manifest.jsonl")
    ap.add_argument("--out", default="data/multiventG/claims.jsonl")
    a = ap.parse_args()
    gold = defaultdict(list)
    for l in open(a.manifest):
        m = json.loads(l)
        gold[(m["video_id"], m["channel"], m["role"])].append(m)
    seen, ok, errs = set(), [], []
    for f in sorted(glob.glob(os.path.join(a.claims_dir, "*"))):
        txt = open(f).read()
        if "CLAIMS:" in txt:
            txt = txt.split("CLAIMS:", 1)[1]
        for ln, line in enumerate(txt.splitlines(), 1):
            line = line.strip().strip("`")
            if not line or line.startswith(("{\"skipped", "{'skipped")):
                continue
            try:
                c = json.loads(line)
            except Exception as e:
                errs.append(f"{f}:{ln} unparseable: {e} :: {line[:120]}")
                continue
            if any(k not in c for k in REQ):
                errs.append(f"{f}:{ln} missing fields {[k for k in REQ if k not in c]}")
                continue
            key = (c["video_id"], c["channel"], c["role"])
            cands = gold.get(key)
            if not cands:
                errs.append(f"{f}:{ln} no gold annotation for {key}")
                continue
            match = next((m for m in cands
                          if c["source_text"].strip().lower() in m["source_text"].strip().lower()
                          or m["source_text"].strip().lower() in c["source_text"].strip().lower()), None)
            if match is None:
                errs.append(f"{f}:{ln} source_text not in gold for {key}: {c['source_text'][:60]}")
                continue
            cid = c.get("claim_id") or f"{c['video_id']}#{c['channel']}#{len(ok)}"
            if cid in seen:
                continue
            seen.add(cid)
            ok.append({**c, "claim_id": cid, "certainty": match.get("certainty"),
                       "evidence": c.get("evidence") or match.get("evidence")})
    with open(a.out, "w") as f:
        for c in ok:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(os.path.dirname(a.out), "claims_errors.txt"), "w") as f:
        f.write("\n".join(errs))
    print(f"[join] accepted {len(ok)} claims, rejected {len(errs)} lines -> {a.out}")
    if errs:
        print("       first errors:")
        for e in errs[:5]:
            print("       ", e)

if __name__ == "__main__":
    main()
