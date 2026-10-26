#!/usr/bin/env python
"""Offline re-scoring of item logs under different candidate scores (v1.143/v1.145, 2026-09-24). No GPU.

Ground-CF headline / probes (`flip_step_*_items.jsonl.gz`, eval_clevr3_flip.py --dump_picks 1, v1.142+): one record per
scored context (request x base / capedit / twin), candidates with the whole-turn sum and its split into lead-in /
answer / end-of-turn (scripts/tokscore.py). Scores:
  turn        sum over the whole assistant turn (banked rule) -- ranks exactly like answer + end-of-turn
  answer      answer tokens only (no lead-in, no end-of-turn)  -- the paper's m(y|C)
  mean        answer-token mean
MVG (`flip_step_*_rbg*_items.jsonl.gz`, eval_mvg_flip.py): two phrases per prompt, same three scores.

Reports per request (MVG: per pair type): Accuracy / Flip / conjunctive Stay under each score; the fraction of scored
contexts whose argmax differs between EVERY pair of scores (e.g. answer vs turn = the end-of-turn effect; MVG sum vs
mean = the length effect); and the mean per-item margin m(gold) - max_{y != gold} m(y) under each score.
--margins out.jsonl.gz also writes the per-item margins (template-shift and decisiveness analyses).
Usage: python scripts/score_variants.py [--margins out.jsonl.gz] <items.jsonl.gz> [...]
"""
import collections, gzip, itertools, json, sys

SC = {"turn": lambda c: c["sum"], "answer": lambda c: c["answer_sum"], "mean": lambda c: c["answer_mean"]}
EDIT = {"look": ("twin", "capedit"), "read": ("capedit", "twin")}      # (flip context, stay context)
m_ = lambda v: round(sum(v) / len(v), 4) if v else None


def _argmax(cands, f):
    return max(cands, key=lambda v: f(cands[v]))


def _margin(cands, f, gold):
    if gold not in cands or len(cands) < 2:
        return None
    return round(f(cands[gold]) - max(f(c) for v, c in cands.items() if v != gold), 4)


def analyse_gcf(recs, meta, margins_out):
    by = collections.defaultdict(dict)
    for r in recs:
        by[(r["id"], r["request"])][r.get("ctx")] = r
    out = {"kind": "groundcf", "run": meta.get("run"), "split": meta.get("split"), "tag": meta.get("tag"),
           "n_contexts": len(recs)}
    for req in ("look", "read"):
        o = {}
        for name, f in SC.items():
            acc, flip, stay, marg = [], [], [], []
            for (iid, rq), cx in by.items():
                if rq != req or "base" not in cx:
                    continue
                b = cx["base"]; g = lambda r: r["values"][req]
                a = int(_argmax(b["candidates"], f) == g(b)); acc.append(a)
                fc, sc_ = EDIT[req]
                if fc in cx:
                    flip.append(int(a and _argmax(cx[fc]["candidates"], f) == g(cx[fc])))
                if sc_ in cx:
                    stay.append(int(a and _argmax(cx[sc_]["candidates"], f) == _argmax(b["candidates"], f)))
                for ctx, r in cx.items():
                    mg = _margin(r["candidates"], f, g(r)); marg.append(mg) if mg is not None else None
                    if margins_out is not None:
                        margins_out.append({"run": meta.get("run"), "tag": meta.get("tag"), "split": meta.get("split"), "id": iid,
                                            "request": req, "ctx": ctx, "score": name, "gold": g(r), "margin": mg})
            o[name] = {"acc": m_(acc), "flip": m_(flip), "stay_c": m_(stay), "mean_margin": m_(marg), "n": len(acc)}
        ctxs = [r for (iid, rq), cx in by.items() if rq == req for r in cx.values()]
        o["argmax_differs"] = {f"{x}_vs_{y}": m_([int(_argmax(r["candidates"], SC[x]) != _argmax(r["candidates"], SC[y])) for r in ctxs])
                               for x, y in itertools.combinations(SC, 2)}
        out[req] = o
    return out


def analyse_mvg(recs, meta, margins_out):
    out = {"kind": "mvg", "run": meta.get("run", recs[0].get("run") if recs else None), "n_pairs": len(recs)}
    by = collections.defaultdict(list)
    for r in recs:
        by[r["pair_type"]].append(r)
    for pt, rs in by.items():
        o = {}
        for name, f in SC.items():
            pk = {k: [] for k in ("flip", "stay_other")}
            marg = []
            for r in rs:
                S = r["scores"]
                p = {k: _argmax(v, f) for k, v in S.items()}
                pk["flip"].append(int(p["own_a"] == "a" and p["own_b"] == "b"))
                pk["stay_other"].append(int(p["other_a"] == p["other_b"]))
                for k, gold in (("own_a", "a"), ("own_b", "b")):
                    mg = _margin(S[k], f, gold); marg.append(mg) if mg is not None else None
            o[name] = {"flip": m_(pk["flip"]), "stay_other": m_(pk["stay_other"]), "mean_margin": m_(marg), "n": len(rs)}
        allp = [v for r in rs for v in r["scores"].values()]
        o["argmax_differs"] = {f"{x}_vs_{y}": m_([int(_argmax(v, SC[x]) != _argmax(v, SC[y])) for v in allp])
                               for x, y in itertools.combinations(SC, 2)}
        out[pt] = o
    return out


def analyse(path, margins_out=None):
    recs = [json.loads(l) for l in gzip.open(path, "rt")]
    meta = recs[0].get("_meta", {}) if recs and "_meta" in recs[0] else {}
    recs = [r for r in recs if "_meta" not in r]
    if recs and "pair_type" in recs[0]:
        return analyse_mvg(recs, meta, margins_out)
    return analyse_gcf(recs, meta, margins_out)


if __name__ == "__main__":
    args = sys.argv[1:]; mout = None
    if "--margins" in args:
        i = args.index("--margins"); mout = args[i + 1]; args = args[:i] + args[i + 2:]
    M = [] if mout else None
    for p in args:
        r = analyse(p, M)
        print(f"{r.get('run')} [{r['kind']}] ({r.get('split', '')} {r.get('n_contexts', r.get('n_pairs'))})")
        for k, v in r.items():
            if not isinstance(v, dict):
                continue
            for name in SC:
                x = v[name]
                print(f"   {k:14s} {name:6s} " + " ".join(f"{kk} {vv}" for kk, vv in x.items() if kk != "n"))
            print(f"   {k:14s} argmax differs: {v['argmax_differs']}")
    if mout:
        with gzip.open(mout, "wt") as f:
            for r in M:
                f.write(json.dumps(r) + "\n")
        print("[saved]", mout, len(M), "margins")
