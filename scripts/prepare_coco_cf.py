#!/usr/bin/env python3
"""COCO-Counterfactuals -> the open-vocab rung, as a REASONABLE SAMPLE.

17,042 clean single-noun pairs is overkill for embedding-only training and
makes the GPU oracle-gate expensive. We sample a pool, split by held-out NOUN
VOCABULARY (OOD = pairs whose changed noun was never seen in training), and
materialize the ColorSwap/UnifiedExample schema so the existing pipeline runs.

Phase 1 (this script, CPU): derive the single-noun diff (difflib, no
pre-labelled column exists), sample, noun-vocab split, materialize
data/coco/{train,test_id,test_ood}. Phase 2 (scripts/coco_oracle_gate.py, GPU)
then keeps only pairs the frozen model reads correctly on both images.

Conflict is built like ColorSwap: image_1=image_0 (shows n0), image_2=image_1
(shows n1), caption_1/2 = caption_0/1; build_specs cross-pairs into the look-vs
-read modality conflict. key = {"object": noun} (generic slot; the noun is the
answer). task tag "coco".
"""
import argparse, difflib, hashlib, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scsrc.tasks import UnifiedExample
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "prepare_task", os.path.join(os.path.dirname(__file__), "prepare_task.py"))
_pt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_pt)
to_hf = _pt.to_hf

STOP = {"a", "an", "the", "is", "are", "of", "on", "in", "with", "and", "to",
        "that", "this", "some", "their", "his", "her"}


def toks(s):
    return re.findall(r"[a-z]+", s.lower())


def changed(a, b):
    ta, tb = toks(a), toks(b)
    blocks = [op for op in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
              .get_opcodes() if op[0] != "equal"]
    if len(blocks) != 1 or blocks[0][0] != "replace":
        return None
    _, i1, i2, j1, j2 = blocks[0]
    sa, sb = ta[i1:i2], tb[j1:j2]
    if len(sa) != 1 or len(sb) != 1:      # keep 1-word swaps (2-word were artifacts)
        return None
    return sa[0], sb[0]


def synlike(x, y):
    return (x in y or y in x or x.rstrip("s") == y.rstrip("s")
            or (abs(len(x) - len(y)) <= 1 and sum(c1 != c2 for c1, c2 in zip(x, y)) <= 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="~/.cache/huggingface/hub/"
                    "datasets--Intel--COCO-Counterfactuals/snapshots/"
                    "958a3e0b32421b9f07c4c3f78aecbf98019a3402/data/examples.jsonl")
    ap.add_argument("--images", default="data/coco_cf/images")
    ap.add_argument("--out", default="data/coco")
    ap.add_argument("--pool", type=int, default=3500, help="pre-gate sample size")
    ap.add_argument("--ood_noun_frac", type=float, default=0.2)
    ap.add_argument("--id_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pairs = []
    for l in open(a.examples):
        r = json.loads(l)
        d = changed(r["caption_0"], r["caption_1"])
        if not d:
            continue
        n0, n1 = d
        if n0 in STOP or n1 in STOP or synlike(n0, n1):
            continue
        i0 = os.path.join(a.images, f"{r['id']}_img_0.jpg")
        i1 = os.path.join(a.images, f"{r['id']}_img_1.jpg")
        if not (os.path.exists(i0) and os.path.exists(i1)):
            continue
        pairs.append((r["id"], r["caption_0"], r["caption_1"], n0, n1, i0, i1))

    rng = random.Random(a.seed)
    rng.shuffle(pairs)
    pool = pairs[:a.pool]

    # held-out NOUN vocabulary -> OOD (pairs whose changed noun is held out)
    nouns = sorted({n for _, _, _, n0, n1, _, _ in pool for n in (n0, n1)})
    def h(s): return int(hashlib.sha1(("noun|" + s).encode()).hexdigest()[:8], 16)
    ood_nouns = {n for n in nouns if (h(n) % 100) < int(a.ood_noun_frac * 100)}

    from PIL import Image
    per = {"train": [], "test_id": [], "test_ood": []}
    for idx, (rid, c0, c1, n0, n1, i0, i1) in enumerate(pool):
        if n0 in ood_nouns or n1 in ood_nouns:
            split = "test_ood"
        else:
            split = "test_id" if (h(f"row|{rid}") % 100) < int(a.id_frac * 100) else "train"
        ex = UnifiedExample(
            image_1=Image.open(i0).convert("RGB"),   # shows n0
            image_2=Image.open(i1).convert("RGB"),   # shows n1
            caption_1=c0, caption_2=c1,
            key_1={"object": n0}, key_2={"object": n1},
            task="coco", klass=f"{n0},{n1}", id=idx, has_swap=True,
            caption_source="diffusion")
        per[split].append(ex)

    os.makedirs(a.out, exist_ok=True)
    counts = {}
    for s, exs in per.items():
        if not exs:
            counts[s] = 0; continue
        hf, nn = to_hf(exs)
        hf.save_to_disk(os.path.join(a.out, s))
        counts[s] = nn
    man = {"task": "coco", "source": "Intel/COCO-Counterfactuals",
           "axis": "look_vs_read (modality), open-vocab noun",
           "pool": len(pool), "n_nouns": len(nouns),
           "n_ood_nouns": len(ood_nouns), "counts": counts,
           "note": "PRE-oracle-gate; run scripts/coco_oracle_gate.py next"}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
