#!/usr/bin/env python
"""Split-local filter for MultiVENT-G training twins (2026-10-05, advisor audit): a desc_swap pair's replacement value
(gold_b) must be supported by a DESCRIPTION claim of a TRAINING video (same role) or of a video in no split (never evaluated); pairs whose
donor value is attested only by the EVALUATION videos (test_id + test_ood) are dropped. Visual pairs are within-video and unaffected.
Usage: python scripts/mvg_split_local_pairs.py --pairs data/mvg_ctm/pairs_clean_rb_trainvideos.jsonl \
          --train data/mvg_hard/train.jsonl --claims data/multiventG/claims.jsonl --out data/mvg_ctm/pairs_clean_rb_trainvideos_splitlocal.jsonl
"""
import argparse, collections, json
ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="data/mvg_ctm/pairs_clean_rb_trainvideos.jsonl"); ap.add_argument("--train", default="data/mvg_hard/train.jsonl")
ap.add_argument("--claims", default="data/multiventG/claims.jsonl"); ap.add_argument("--out", default="data/mvg_ctm/pairs_clean_rb_trainvideos_splitlocal.jsonl")
ap.add_argument("--heldout", default="data/mvg_hard/test_id.jsonl,data/mvg_hard/test_ood.jsonl",
                help="the EVALUATION splits: a donor is cross-split only if attested solely by these videos (videos in no split were never evaluated)")
a = ap.parse_args()
J = lambda p: [json.loads(l) for l in open(p)]
trv = {x["video_id"] for x in J(a.train)}
hov = {x["video_id"] for f in a.heldout.split(",") for x in J(f)}
support = collections.defaultdict(set)          # role -> values attested by DESC claims of NON-held-out videos
support_all = collections.defaultdict(set)
for c in J(a.claims):
    if c["channel"] != "DESC": continue
    v = str(c["answer"]).strip().lower(); support_all[c["role"]].add(v)
    if c["video_id"] not in hov: support[c["role"]].add(v)
pairs = J(a.pairs); keep, drop = [], collections.Counter()
for p in pairs:
    if p["pair_type"] != "desc_swap":
        keep.append(p); continue
    donor = str(p["gold_b"]).strip().lower()
    if donor in support[p["role"]]:
        keep.append(p)
    else:
        drop["held-out-only" if donor in support_all[p["role"]] else "unattested"] += 1
with open(a.out, "w") as f:
    for p in keep: f.write(json.dumps(p) + "\n")
n_desc = sum(1 for p in pairs if p["pair_type"] == "desc_swap"); k_desc = sum(1 for p in keep if p["pair_type"] == "desc_swap")
print(f"desc_swap pairs {n_desc} -> {k_desc} kept; dropped {dict(drop)}; total {len(pairs)} -> {len(keep)} written to {a.out}")
