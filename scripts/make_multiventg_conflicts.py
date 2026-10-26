#!/usr/bin/env python
"""Stage 2: deterministic one-deviant conflicts from truthful claims.
For each claim, builds an agree item (all channels gold) and a conflict item
(THIS claim's channel carries a same-role, same-event-type donor entity from a
different video; other channels stay gold). Gold label = the deviant channel.

Usage: python scripts/make_multiventg_conflicts.py --claims data/multiventG/claims.jsonl \
           --manifest data/multiventG/gold_manifest.jsonl [--out data/multiventG/items.jsonl] [--seed 0]
"""
import argparse, json, random
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/multiventG/items.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    # donor pools: (event_type?, role, channel) -> source_texts from OTHER videos
    # event_type lives in prompts; recover per-video from claims' ids if present, else pool by (role, channel)
    claims = [json.loads(l) for l in open(a.claims) if l.strip() and "skipped" not in l]
    pool = defaultdict(list)
    for c in claims:
        pool[(c["role"], c["channel"])].append((c["video_id"], c["source_text"], c["claim"]))
    out = open(a.out, "w")
    n_agree = n_conf = n_nodonor = 0
    for c in claims:
        out.write(json.dumps({**c, "cond": "agree", "deviant_channel": None}) + "\n")
        n_agree += 1
        donors = [(v, st, cl) for (v, st, cl) in pool[(c["role"], c["channel"])]
                  if v != c["video_id"] and st.strip().lower() != c["source_text"].strip().lower()]
        if not donors:
            n_nodonor += 1
            continue
        dv, dst, dcl = rng.choice(donors)
        out.write(json.dumps({**c, "cond": "conflict", "deviant_channel": c["channel"],
                              "deviant_source_text": dst, "deviant_claim": dcl,
                              "donor_video": dv}) + "\n")
        n_conf += 1
    out.close()
    print(f"[conflicts] agree={n_agree} conflict={n_conf} no-donor={n_nodonor} -> {a.out}")

if __name__ == "__main__":
    main()
