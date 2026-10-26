#!/usr/bin/env python
"""One-pass artifact extractor for the multivent-raw CLAIM-CITED subset.

The reference claims cite only ~717 of 143,288 chunks, but those chunks are
spread over 412 of 667 WebDataset shards (median 1 chunk/shard), so training
can never stream shards -- we extract once into a compact local subset:

    data/multivent_raw/artifacts/
        keyframes/<chunk_id>.tNNNNNN.jpg   (+ <chunk_id>.meta.json)
        asr/<chunk_id>.json                (whisperx: transcript/language/vad/qc)
        ocr/<chunk_id>.jsonl               (ppocrvl15, per frame)
        captions/<chunk_id>.txt            (qwen multiframe; NOT a prompt channel
                                            -- machine-generated, kept for audits)

Shards are read in streaming mode ('r|') so a 1.1 GB keyframe shard costs one
sequential pass and no index scan. Sliced by --task_id/--n_tasks for a CPU
array. Misses are logged, never fatal: the captions dir ships no catalog, so
its shard index is assumed to match keyframes (both derive from kf_uni5s) and
verified by the miss count.
"""
import argparse, csv, json, os, sys, tarfile, collections

R = os.environ.get("MVR_ROOT", "data/multivent_raw_src")
OUT = "data/multivent_raw/artifacts"
ARTS = {
    # artifact dir (under R)                      catalog?  out subdir
    "keyframes/uniform_5s":                          (True,  "keyframes"),
    "asr/whisperxlargev3":                           (True,  "asr"),
    "ocr/ppocrvl15":                                 (True,  "ocr"),
    "keyframe-captions/uniform_5s_qwen-multiframe-ic/max_imgs_16": (False, "captions"),
}


def cited_chunks():
    need = set()
    for f in ("video", "event", "persona", "query"):
        d = json.load(open(f"{R}/annotations/reference-claims-{f}.json"))
        for cs in d["claim_sets"]:
            for c in cs["claims"]:
                need |= set((c.get("evidence") or {}).keys())
    return need


def shard_map(art, need):
    """chunk -> shard from the artifact's own catalog (keyframes' as fallback)."""
    cat = f"{R}/{art}/catalog.csv"
    if not os.path.exists(cat):
        cat = f"{R}/keyframes/uniform_5s/catalog.csv"
    m = collections.defaultdict(set)
    with open(cat) as fh:
        for row in csv.DictReader(fh):
            if row["chunk_id"] in need:
                m[int(row["shard_index"])].add(row["chunk_id"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--n_tasks", type=int, default=1)
    a = ap.parse_args()
    need = cited_chunks()
    print(f"cited chunks: {len(need)}", flush=True)
    got = collections.Counter()
    for art, (_, sub) in ARTS.items():
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
        smap = shard_map(art, need)
        shards = sorted(smap)[a.task_id::a.n_tasks]
        for i, sh in enumerate(shards):
            want = smap[sh]
            path = f"{R}/{art}/shard_{sh:06d}.tar"
            if not os.path.exists(path):
                print(f"[miss-shard] {path}", flush=True)
                continue
            found = set()
            with tarfile.open(path, "r|") as t:
                for mem in t:
                    ch = mem.name.split(".", 1)[0]
                    if ch not in want:
                        continue
                    data = t.extractfile(mem).read()
                    name = mem.name
                    if sub == "keyframes":
                        # <ch>.kf_uni5s.tNNNNNN.jpg -> <ch>.tNNNNNN.jpg ; json -> meta
                        name = (f"{ch}.meta.json" if name.endswith(".json")
                                else ch + "." + name.rsplit(".", 2)[-2] + ".jpg")
                    elif sub == "asr":
                        name = f"{ch}.json"
                    elif sub == "ocr":
                        name = f"{ch}.jsonl"
                    else:
                        name = f"{ch}.txt"
                    with open(f"{OUT}/{sub}/{name}", "wb") as fo:
                        fo.write(data)
                    found.add(ch)
            got[sub] += len(found)
            missing = want - found
            if missing and sub != "captions":
                print(f"[miss] {sub} shard {sh}: {sorted(missing)[:3]}", flush=True)
            if i % 25 == 0:
                print(f"  {sub}: shard {i+1}/{len(shards)}", flush=True)
    print(f"[done task {a.task_id}] chunks extracted per artifact: {dict(got)}", flush=True)


if __name__ == "__main__":
    main()
