#!/usr/bin/env python
"""Stratified sample of raw free generations for the LLM-judge pass (STATUS v1.85).

Reads every gen_step_<step>*.json the open-generation check produced, draws a fixed-seed sample per
(run, source, context, parser label) and writes one JSONL the judge labels blind: the judge sees the
question, the candidate values and the generated text, never the parser's label or the gold. After
labelling, --score compares judge labels with the parser labels and reports agreement per run.

  python scripts/label_gen_sample.py --out results/gen_judge_sample.jsonl --per_cell 6
  ... judge writes results/gen_judge_labels.jsonl with {"uid":..., "judge_value":..., "judge_label":...}
  python scripts/label_gen_sample.py --score results/gen_judge_labels.jsonl --sample results/gen_judge_sample.jsonl
"""
import argparse, glob, json, os, random, collections


def load_all(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("weight_dtype", "bfloat16") != "bfloat16":
            continue   # fp32 shadow/smoke files are never used (v1.70)
        tag = os.path.basename(f).replace(".json", "")
        for i, r in enumerate(d.get("records", [])):
            r = dict(r); r["run"] = d["run"]; r["file"] = tag; r["uid"] = f'{d["run"]}|{tag}|{i}'
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/clevr3_*/eval/clevr3_test_id/gen_step_800*.json")
    ap.add_argument("--out", default="results/gen_judge_sample.jsonl")
    ap.add_argument("--per_cell", type=int, default=6, help="records per (run, source, ctx, parser label)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", default=None, help="judge labels JSONL -> agreement with the parser")
    ap.add_argument("--sample", default=None)
    a = ap.parse_args()
    if a.score:
        samp = {json.loads(l)["uid"]: json.loads(l) for l in open(a.sample or a.out)}
        agree = collections.defaultdict(list); conf = collections.Counter()
        for l in open(a.score):
            j = json.loads(l); s = samp.get(j["uid"])
            if s is None:
                continue
            pv = s["_parser_value"]; jv = (j.get("judge_value") or "").strip().lower() or None
            pl = s["_parser_label"]; jl = j.get("judge_label")
            same = int((pv or "").lower() == (jv or "") if (pl == "cand" or jl == "cand") else pl == jl)
            agree[s["run"]].append(same); agree["ALL"].append(same); conf[(pl, jl)] += 1
        rep = {k: {"agree": round(sum(v) / len(v), 3), "n": len(v)} for k, v in sorted(agree.items())}
        print(json.dumps(rep, indent=1)); print("parser x judge:", dict(conf))
        json.dump({"agreement": rep, "confusion": {f"{k[0]}->{k[1]}": v for k, v in conf.items()}},
                  open(a.score.replace(".jsonl", "_agreement.json"), "w"), indent=1)
        return
    recs = load_all(a.glob)
    rng = random.Random(a.seed)
    cells = collections.defaultdict(list)
    for r in recs:
        cells[(r["run"], r["file"], r["source"], r["ctx"], r["gen_label"])].append(r)
    picked = []
    for k in sorted(cells):
        v = cells[k]; rng.shuffle(v); picked.extend(v[:a.per_cell])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        for r in picked:
            fh.write(json.dumps({"uid": r["uid"], "run": r["run"], "source": r["source"], "ctx": r["ctx"],
                                 "skill": r["skill"], "cands": r["cands"], "text": r["text"],
                                 "_parser_label": r["gen_label"], "_parser_value": r["gen_value"]}) + "\n")
    print(f"[sample] {len(picked)} records from {len(recs)} across {len({r['run'] for r in recs})} runs -> {a.out}")
    print("labels overall:", dict(collections.Counter(r["gen_label"] for r in recs)))


if __name__ == "__main__":
    main()
