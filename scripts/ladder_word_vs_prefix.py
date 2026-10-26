#!/usr/bin/env python
"""Paired word-vs-prefix comparison on the ladder from item logs (v1.148, 2026-09-24). CPU only.
Usage: python scripts/ladder_word_vs_prefix.py <vcf|color|coco> <model suffix, e.g. _gemma4e4b or "">
Image-request Acc / Flip / Stay per template (train vs held-out), exact McNemar on discordant items."""
import gzip, json, re, sys, collections
from math import comb
def load(run, ds, tpl):
    f = f"runs/{run}/eval/{ds}3_test_id/flip_ll{'_heldout' if tpl else ''}_step_800_items.jsonl.gz"
    x = [json.loads(l) for l in gzip.open(f, "rt")]; x = [a for a in x if "_meta" not in a]
    by = collections.defaultdict(dict)
    for a in x: by[(a["item"], a["request"])][a["ctx"]] = a
    out = {}
    for (i, rq), c in by.items():
        if rq != "look": continue
        b = c["base"]; acc = b["pick"] == "i"
        tp = re.sub(r'Caption: "[^"]*"\.', "", b["prompt"]).strip()
        mg = b["candidates"]["image_side"]["sum"] - b["candidates"]["caption_side"]["sum"]
        out[i] = dict(acc=acc, flip=acc and c["imgswap"]["pick"] == "i", stay=acc and c["capswap"]["pick"] == "i", tpl=tp, margin=mg)
    return out
def mcnemar(a, b):
    n01 = sum(1 for k in a if a[k] and not b[k]); n10 = sum(1 for k in a if b[k] and not a[k]); n = n01 + n10
    p = min(1.0, 2 * sum(comb(n, j) for j in range(0, min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p
ds, mdl = sys.argv[1], sys.argv[2]
W = f"{ds}_neo_ctrl_1tok_ctm_cp_seed0{mdl}"; P = f"{ds}_pt_na_1tok_ctm_cp_seed0{mdl}"
for tpl in (0, 1):
    try: w = load(W, ds, tpl); p = load(P, ds, tpl)
    except FileNotFoundError as e: print(ds, mdl, "tpl", tpl, "missing"); continue
    ks = sorted(set(w) & set(p))
    print(f"== {ds}{mdl} {'held-out' if tpl else 'train template'} n={len(ks)}")
    for m in ("acc", "flip", "stay"):
        a = {k: w[k][m] for k in ks}; b = {k: p[k][m] for k in ks}
        n01, n10, pv = mcnemar(a, b)
        print(f"   image {m:4s} word {sum(a.values())/len(ks):.3f} prefix {sum(b.values())/len(ks):.3f} | word-only {n01} prefix-only {n10} exact McNemar p={pv:.3f}")
    for name, d in (("word", w), ("prefix", p)):
        g = collections.defaultdict(list)
        for k in ks: g[d[k]["tpl"]].append(d[k])
        for t, v in g.items():
            print(f"   {name:6s} template {t[:55]!r:58s} n={len(v):3d} acc {sum(x['acc'] for x in v)/len(v):.3f} flip {sum(x['flip'] for x in v)/len(v):.3f} median margin {sorted(x['margin'] for x in v)[len(v)//2]:+.2f}")
