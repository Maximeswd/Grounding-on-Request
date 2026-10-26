#!/usr/bin/env python
"""GROUND-CF ABSENCE ITEMS -- the "requested modality does not contain the answer" setting.

For every colour and shape item (the two skills with an exact target box) we build two absent
contexts that keep everything else byte-identical:
  absent IMAGE    the target object's box is erased to the background, so the question
                  ("What color is the diamond?" / "What shape is the yellow object?") has no
                  visual referent; the caption still states its (conflicting) value.
  absent CAPTION  a caption that names the object but not the queried attribute
                  ("The diamond is in the scene." / "The yellow object is in the scene."); the image
                  still shows the value.
The correct behaviour for a request whose source lacks the answer is to ABSTAIN with a designated
answer string (ABSTAIN below), which enters the forced choice as a candidate so nothing is generated
or judged.

Per split we write:
  data/clevr3/absence_<split>.jsonl      one TRAIN record per base item with both absent contexts
                                          (the dataset draws the type and the source at training time)
  data/clevr3/absence_<split>_eval.jsonl  typed EVAL rows: for each source in (look, read) three types
      req_absent    requested source lacks it, the other has it     -> gold = ABSTAIN[source]
      other_absent  requested source has it, the other lacks it     -> gold = requested value
      conflict      both have it and disagree (the original item)   -> gold = requested value
    each row carries `leak` (the other source's value) and `distractor` (a same-type value absent from
    the scene) so every row is scored as a forced choice over three candidates.
Usage: python scripts/clevr3_absence.py --split train ; --split test_id ; --split test_ood
"""
import argparse, json, os, sys
from PIL import Image, ImageDraw
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from prepare_clevr3 import PALETTE, SHAPES_ID, OOD_SHAPES, OOD_COLORS

BG = (245, 245, 245)
ABSTAIN = {"look": "not visible in the image", "read": "not stated in the caption"}


def object_noun(item):
    """The noun phrase the question uses for the target: 'diamond' (colour skill) or 'yellow object'."""
    q = item["question"]
    if item["skill"] == "color":            # What color is the {shape}?
        return q.split(" is the ", 1)[1].rstrip("?").strip()
    return q.split(" is the ", 1)[1].rstrip("?").strip()   # object: 'yellow object'


def distractor(item):
    vals = {str(v).lower() for v in item["channels"].values()}
    pool = (list(PALETTE) if item["skill"] == "color" else SHAPES_ID + OOD_SHAPES)
    ood = OOD_COLORS if item["skill"] == "color" else OOD_SHAPES
    gold = str(item["channels"]["look"])
    cand = [v for v in pool if v not in vals and ((v in ood) == (gold in ood))] or [v for v in pool if v not in vals]
    return sorted(cand)[hash(item["id"]) % len(cand)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--out_images", default="data/clevr3/images_absent")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(os.path.join(ROOT, a.data_dir, f"{a.split}.jsonl"))]
    outdir = os.path.join(ROOT, a.out_images); os.makedirs(outdir, exist_ok=True)
    train_rows, eval_rows, skipped = [], [], 0
    for it in items:
        if it.get("skill") not in ("color", "object") or not it.get("bbox"):
            continue
        ch = {k: str(v) for k, v in it["channels"].items()}
        if ch["look"] == ch["read"]:
            skipped += 1; continue          # no conflict to speak of
        src = os.path.join(ROOT, it["image"])
        im = Image.open(src).convert("RGB")
        x0, y0, x1, y1 = [int(round(v)) for v in it["bbox"]]
        ImageDraw.Draw(im).rectangle([max(0, x0 - 1), max(0, y0 - 1), min(im.width, x1 + 1), min(im.height, y1 + 1)], fill=BG)
        absent_img = os.path.join(a.out_images, f"{it['id']}_absent.png")
        im.save(os.path.join(ROOT, absent_img))
        absent_cap = f"The {object_noun(it)} is in the scene."
        dis = distractor(it)
        base = {k: it[k] for k in ("id", "image", "skill", "question", "caption", "ocr_text", "bbox", "channels")}
        train_rows.append({**base, "absent_image": absent_img, "absent_caption": absent_cap, "distractor": dis})
        for source in ("look", "read"):
            other = "read" if source == "look" else "look"
            for typ in ("req_absent", "other_absent", "conflict"):
                row = dict(base); row["channels"] = dict(ch)
                if typ == "req_absent":
                    if source == "look": row["image"] = absent_img
                    else: row["caption"] = absent_cap
                    row["channels"][source] = None
                    gold = ABSTAIN[source]
                elif typ == "other_absent":
                    if other == "look": row["image"] = absent_img
                    else: row["caption"] = absent_cap
                    row["channels"][other] = None
                    gold = ch[source]
                else:
                    gold = ch[source]
                row.update({"id": f"{it['id']}|{source}|{typ}", "base_id": it["id"], "source": source,
                            "type": typ, "gold": gold, "abstain": typ == "req_absent",
                            "leak": ch[other], "distractor": dis, "req_value": ch[source]})
                eval_rows.append(row)
    for name, rows in ((f"absence_{a.split}.jsonl", train_rows), (f"absence_{a.split}_eval.jsonl", eval_rows)):
        with open(os.path.join(ROOT, a.data_dir, name), "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    print(f"[{a.split}] {len(train_rows)} base items -> {len(eval_rows)} typed eval rows; skipped {skipped}; images in {a.out_images}")


if __name__ == "__main__":
    main()
