#!/usr/bin/env python
"""ABSTENTION TWINS ON GROUND-CF (2026-09-22, author's design, pre-registered STATUS v1.83).

Base rows are the `req_absent` rows of absence_<split>_hard_eval.jsonl: the requested source does not
contain the queried object, the competing source asserts a value for it (look: caption says B; read: the
image shows A while the caption is off-type silent). Gold = abstain. Two twins per row, both keeping the
sources in conflict:
  FLIP  the requested source gains the answer with a value the competitor does NOT state:
        look: re-render the base scene with the target present in colour/shape C, C not in {B, original}
              -> gold C, candidates {abstain, B (caption), C}
        read: caption := TEMPLATE(C), C != A (image value)              -> gold C, candidates {abstain, A, C}
  STAY  the competing source changes while the requested one stays silent:
        look: caption := TEMPLATE(D), D not in {B, C}                     -> gold abstain, candidates {abstain, D, distractor}
        read: image := target re-rendered in A' != A (still present)     -> gold abstain, candidates {abstain, A', distractor}
Scenes are recovered by replaying the generator (clevr3_twins_count_spatial.replay_scene_graphs); the
target is objs[0] of the base scene. Skills: color and object (shape), as in the absence files.
Writes data/clevr3/abstain_twins_<split>.jsonl and images to data/clevr3/images_abstain_twins/.
"""
import json, os, sys, random, importlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
pc = importlib.import_module("prepare_clevr3")
tw = importlib.import_module("clevr3_twins_count_spatial")
TEMPLATE = "The answer to the question is {}."
COLORS = list(pc.PALETTE.keys()); SHAPES = list(getattr(pc, "SHAPES_ID", ["circle", "square", "diamond", "cross"]))

def main():
    outimg = f"{ROOT}/data/clevr3/images_abstain_twins"; os.makedirs(outimg, exist_ok=True)
    captured, _ = tw.replay_scene_graphs()
    for split in ("test_id", "test_ood"):
        rows = [json.loads(l) for l in open(f"{ROOT}/data/clevr3/absence_{split}_hard_eval.jsonl")]
        out, rej = [], {"no_graph": 0, "skill": 0, "novalue": 0}
        for r in rows:
            if r["type"] != "req_absent" or r["skill"] not in ("color", "object"):
                if r["type"] == "req_absent": rej["skill"] += 1
                continue
            sid = r["base_id"].rsplit("_", 1)[0]
            objs = captured.get(sid)
            if objs is None:
                rej["no_graph"] += 1; continue
            t = objs[0]; attr = "color" if r["skill"] == "color" else "shape"
            vocab = COLORS if attr == "color" else SHAPES
            rng = random.Random(9000 + hash(r["id"]) % 100000)
            src = r["source"]
            if src == "look":
                B = str(r["leak"]); orig = str(r["req_value"])
                # exclude the competitor's value and the distractor only: with four ID shapes, also excluding the
                # original target shape left one candidate and rejected every ID object row (2026-09-22)
                cands = [v for v in vocab if v not in {B, str(r["distractor"])}]
                if len(cands) < 2: rej["novalue"] += 1; continue
                C = rng.choice(cands); D = rng.choice([v for v in cands if v != C])
                nt = dict(t); nt[attr] = C
                flip_img = os.path.join(outimg, f"{r['base_id']}_look_flip_{C}.png"); pc.render([nt] + objs[1:], flip_img)
                out.append({**{k: r[k] for k in ("id", "base_id", "source", "skill", "question", "image", "caption", "ocr_text", "bbox", "leak", "distractor", "req_value")},
                            "flip": {"image": os.path.relpath(flip_img, ROOT), "caption": r["caption"], "gold": C, "cands": {"abstain": None, "competitor": B, "value": C}},
                            "stay": {"image": r["image"], "caption": TEMPLATE.format(D), "gold": "abstain", "cands": {"abstain": None, "competitor": D, "distractor": str(r["distractor"])}}})
            else:
                A = str(r["leak"])   # the image's value (what a leaking read answer would copy)
                cands = [v for v in vocab if v not in {A, str(r["distractor"])}]
                if len(cands) < 2: rej["novalue"] += 1; continue
                C = rng.choice(cands); Ap = rng.choice([v for v in cands if v != C])
                nt = dict(t); nt[attr] = Ap
                stay_img = os.path.join(outimg, f"{r['base_id']}_read_stay_{Ap}.png"); pc.render([nt] + objs[1:], stay_img)
                out.append({**{k: r[k] for k in ("id", "base_id", "source", "skill", "question", "image", "caption", "ocr_text", "bbox", "leak", "distractor", "req_value")},
                            "flip": {"image": r["image"], "caption": TEMPLATE.format(C), "gold": C, "cands": {"abstain": None, "competitor": A, "value": C}},
                            "stay": {"image": os.path.relpath(stay_img, ROOT), "caption": r["caption"], "gold": "abstain", "cands": {"abstain": None, "competitor": Ap, "distractor": str(r["distractor"])}}})
        dst = f"{ROOT}/data/clevr3/abstain_twins_{split}.jsonl"
        with open(dst, "w") as f:
            for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
        miss = [o for o in out if not os.path.exists(o["flip"]["image"]) or not os.path.exists(o["stay"]["image"])]
        if miss: raise SystemExit(f"[STOP] {len(miss)} twin images missing, e.g. {miss[0]['flip']['image']}")
        import collections
        print(f"[abstain-twins] {split}: {len(out)} rows ({collections.Counter(o['source'] for o in out)}), rejected {rej} -> {dst}")

if __name__ == "__main__":
    main()
