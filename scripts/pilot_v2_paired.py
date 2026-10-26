#!/usr/bin/env python
"""Paired exact McNemar tests between two Ground-CF runs on IDENTICAL items (2026-10-02, protocol-v2 pilot).

Reads runs/<run>/eval/clevr3_<split>/flip_step_800<tag>_items.jsonl.gz (one record per request x context x item;
contexts: base, twin = image edit, capedit = caption edit; each record's values[request] is the gold in that context).
  image Flip = correct(base) & correct(twin);     image Stay = correct(base) & correct(capedit)
  caption Flip = correct(base) & correct(capedit); caption Stay = correct(base) & correct(twin)
Pairs by (request, item id); exact two-sided binomial McNemar on the discordant pairs. Also recomputes each run's
aggregate from the items and checks it against the stored flip_step_800<tag>.json.
Usage: python scripts/pilot_v2_paired.py --a runs/A --b runs/B [--split test_id] [--tag _v2ref] [--skills count,spatial]
"""
import argparse, gzip, json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(run, split, tag):
    p = os.path.join(ROOT, run, "eval", f"clevr3_{split}", f"flip_step_800{tag}_items.jsonl.gz")
    recs = [json.loads(l) for l in gzip.open(p, "rt")]
    by = {}
    for r in recs:
        if "_meta" in r: continue
        by.setdefault((r["request"], r["id"]), {})[r["ctx"]] = r
    return by


def bits(by, skills=None):
    out = {}   # (request, metric) -> {id: bit}
    for (rq, iid), ctxs in by.items():
        if skills and ctxs["base"]["skill"] not in skills: continue
        ok = {c: (r["pick"] == r["values"][rq]) for c, r in ctxs.items()}
        flip_ctx, stay_ctx = ("twin", "capedit") if rq == "look" else ("capedit", "twin")
        if flip_ctx in ok: out.setdefault((rq, "Flip"), {})[iid] = int(ok["base"] and ok[flip_ctx])
        if stay_ctx in ok: out.setdefault((rq, "Stay"), {})[iid] = int(ok["base"] and ok[stay_ctx])
        out.setdefault((rq, "Acc"), {})[iid] = int(ok["base"])
    return out


def scene_ci(ids, xa, xb, B=2000, seed=0):
    """95% percentile interval of mean(B) - mean(A) under a SCENE-clustered bootstrap (2026-10-02, advisor note:
    items of one scene are not independent). Scene = the item id before the first underscore (id3_count -> id3)."""
    import random
    scenes = {}
    for i, x, y in zip(ids, xa, xb):
        scenes.setdefault(i.rsplit("_", 1)[0], []).append(y - x)      # id3_count -> id3; ood_count0_count -> ood_count0
    keys = list(scenes); rng = random.Random(seed); diffs = []
    for _ in range(B):
        tot = n = 0
        for k in rng.choices(keys, k=len(keys)):
            tot += sum(scenes[k]); n += len(scenes[k])
        diffs.append(tot / n)
    diffs.sort(); return diffs[int(.025 * B)], diffs[int(.975 * B) - 1]


def mcnemar_p(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--split", default="test_id"); ap.add_argument("--tag", default="_v2ref")
    ap.add_argument("--skills", default="", help="restrict to these skills (e.g. count,spatial = held-out tasks)")
    ap.add_argument("--json", default="", help="append the rows to this JSON file")
    ap.add_argument("--scenes", type=int, default=2000,
                    help="scene-clustered bootstrap replicates for the 95%% CI of B-A (scene = item id before '_'; 0 = off)")
    a = ap.parse_args()
    sk = set(a.skills.split(",")) if a.skills else None
    A, B = bits(load(a.a, a.split, a.tag), sk), bits(load(a.b, a.split, a.tag), sk)
    la, lb = os.path.basename(a.a.rstrip("/")), os.path.basename(a.b.rstrip("/"))
    print(f"# {a.split}{a.tag}{' skills=' + a.skills if a.skills else ''}: A = {la}, B = {lb}")
    print("| request | metric | n paired | A | B | B−A [scene-bootstrap 95% CI] | A✓B✗ | A✗B✓ | McNemar p |\n|---|---|---|---|---|---|---|---|---|")
    rows = []
    for key in sorted(A):
        if key not in B: continue
        ids = sorted(set(A[key]) & set(B[key]))
        xa, xb = [A[key][i] for i in ids], [B[key][i] for i in ids]
        b_ = sum(1 for x, y in zip(xa, xb) if x and not y); c_ = sum(1 for x, y in zip(xa, xb) if y and not x)
        ma, mb = sum(xa) / len(ids), sum(xb) / len(ids); p = mcnemar_p(b_, c_)
        ci = scene_ci(ids, xa, xb, a.scenes) if a.scenes else (None, None)
        rows.append({"split": a.split, "tag": a.tag, "skills": a.skills, "A": la, "B": lb, "request": key[0], "metric": key[1],
                     "n": len(ids), "n_scenes": len({i.rsplit("_", 1)[0] for i in ids}), "mean_A": ma, "mean_B": mb, "diff": mb - ma,
                     "A_only": b_, "B_only": c_, "p": p, "ci95_scene": ci})
        cis = f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci[0] is not None else ""
        print(f"| {key[0]} | {key[1]} | {len(ids)} | {ma:.3f} | {mb:.3f} | {mb - ma:+.3f}{cis} | {b_} | {c_} | {p:.3g}{' *' if p < .05 else ''} |")
    if not sk:   # sanity: recomputed aggregate vs the stored summary
        for run, X in ((a.a, A), (a.b, B)):
            s = json.load(open(os.path.join(ROOT, run, "eval", f"clevr3_{a.split}", f"flip_step_800{a.tag}.json")))
            chk = {"look_flip": ("look", "Flip"), "read_flip": ("read", "Flip"), "look_stay_c": ("look", "Stay"), "read_stay_c": ("read", "Stay")}
            bad = [k for k, kk in chk.items() if kk in X and abs(sum(X[kk].values()) / len(X[kk]) - s[k]) > 5e-4]
            print(f"  recomputed aggregates of {os.path.basename(run)} match the stored summary: {'YES' if not bad else 'NO ' + str(bad)}")
    if a.json:
        old = json.load(open(a.json)) if os.path.exists(a.json) else []
        json.dump(old + rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
