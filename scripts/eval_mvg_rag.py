#!/usr/bin/env python
"""RAG-stress ablation: channel routing under {oracle | top1-retrieved | distractor}.

For each mvg3 aligned item, rebuild the prompt with the CONDITION video's
channels (description, OCR strings, frame) while keeping the oracle item's
question and candidate values. Metrics per condition:
  selectivity      3-way routing matrix diag-off (as eval_mvg3) -- oracle/top1
  assert_gold      P(token still asserts the oracle answer) under DISTRACTOR
                   context = hallucination-under-retrieval-failure (lower=better)
  margin_drop      mean LL-margin loss of the gold answer vs oracle condition
Retrieval map from scripts/mvg_retrieval.py. The claim this licenses: the
grounding stage is retrieval-agnostic, degrades gracefully, and makes
retrieval failures VISIBLE (margins collapse / no gold assertion) rather than
silently absorbed.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from tqdm import tqdm
from scsrc.tokens import trigger
from scsrc.mvg import mvg_word, assert_words_registered
from evaluate import apply_checkpoint
from scripts.eval_mvg import user_content, loglik_mvg, cand

CH = {"DESC": "desc_ctrl", "OCR": "ocr_ctrl", "VIS": "vis_ctrl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--data_dir", default="data/mvg3")
    ap.add_argument("--map", default="data/multiventG/retrieval_map.jsonl")
    ap.add_argument("--mvg", default="data/multiventG/multivent_g.json")
    ap.add_argument("--claims", default="data/multiventG/claims.jsonl")
    ap.add_argument("--frames", default="data/multiventG/frames_manifest.jsonl")
    ap.add_argument("--max_n", type=int, default=150)
    a = ap.parse_args()

    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    mvg = json.load(open(a.mvg))
    rmap = {json.loads(l)["video_id"]: json.loads(l) for l in open(a.map)}
    ocr_by_vid = defaultdict(set)
    for l in open(a.claims):
        c = json.loads(l)
        if c["channel"] == "OCR":
            ocr_by_vid[c["video_id"]].add(c.get("source_text", ""))
    frames = defaultdict(list)
    for l in open(a.frames):
        r = json.loads(l); frames[r["video_id"]].append((r["t"], r["path"]))

    def channels_of(vid):
        return {"desc_text": mvg[vid]["description"][:1200],
                "ocr_strings": sorted(ocr_by_vid[vid])[:8],
                "vis_frame": sorted(frames[vid])[0][1] if frames[vid] else None}

    items = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))][:a.max_n]
    res = {c: defaultdict(list) for c in ("oracle", "top1", "distractor")}
    M = {c: defaultdict(lambda: defaultdict(int)) for c in ("oracle", "top1")}
    tot = {c: defaultdict(int) for c in ("oracle", "top1")}
    n_top1_correct = 0

    for it in tqdm(items, desc="rag-stress"):
        v = it["video_id"]
        if v not in rmap:
            continue
        conds = {"oracle": v, "top1": rmap[v]["top5"][0], "distractor": rmap[v]["distractor"]}
        n_top1_correct += int(conds["top1"] == v)
        present, vals = it["present"], it["channel_values"]
        for cname, cvid in conds.items():
            ch = channels_of(cvid)
            if ch["vis_frame"] is None:
                continue
            it2 = {**it, **ch}
            for qc in present:
                um = user_content(it2, mvg_word(CH[qc], n))
                scores = {c: loglik_mvg(model, proc, um, cand(vals[c]), "cuda")
                          for c in present}
                picked = max(scores, key=scores.get)
                gold_ll = scores[qc]
                if cname in ("oracle", "top1"):
                    M[cname][qc][picked] += 1; tot[cname][qc] += 1
                res[cname]["gold_ll"].append(gold_ll)
                res[cname]["assert_gold"].append(int(picked == qc))

    def sel(c):
        d = [M[c][q][q] / tot[c][q] for q in tot[c] if tot[c][q]]
        o = [M[c][q][p] / tot[c][q] for q in tot[c] for p in CH if p != q and tot[c][q]]
        return round(sum(d) / max(len(d), 1) - sum(o) / max(len(o), 1), 3)

    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    base = m(res["oracle"]["gold_ll"])
    out = {"run": os.path.basename(a.run_dir), "n_items": len(items),
           "top1_is_gold_rate": round(n_top1_correct / max(len(items), 1), 3),
           "selectivity": {"oracle": sel("oracle"), "top1": sel("top1")},
           "assert_gold_rate": {c: m(res[c]["assert_gold"]) for c in res},
           "gold_margin_drop_vs_oracle": {
               c: round(base - m(res[c]["gold_ll"]), 3) for c in ("top1", "distractor")}}
    print(json.dumps(out, indent=1))
    outd = os.path.join(a.run_dir, "eval", "rag_stress"); os.makedirs(outd, exist_ok=True)
    json.dump(out, open(os.path.join(outd, f"step_{a.step}.json"), "w"), indent=1)
    print(f"[saved] {outd}/step_{a.step}.json")


if __name__ == "__main__":
    main()
