#!/usr/bin/env python
"""GROUND-CF IMAGE TWINS — the missing <look> counterfactual.

WHY THIS DID NOT EXIST, AND WHY THAT WAS A BUILD GAP RATHER THAN A DESIGN CHOICE.
Ground-CF's counterfactuals are channel EDITS: we rewrite the caption or the OCR
string and ask whether the answer follows. That gives <read> and <ocr> a proper
FOLLOWS/IGNORES pair and leaves <look> with nothing to follow -- the image never
changes, so there is no visual intervention. For a long time the docs said "no
image twin by design"; that was wrong. `prepare_clevr3.py` renders every scene
itself with PIL flat fills, so a second image differing in exactly one attribute
is constructible, and <look> is currently the ONLY channel in the whole project
with no FOLLOWS number anywhere.

WHAT AN EXACT TWIN NEEDS. The same property that makes VCF and ColorSwap work
(JOINT .85-1.00): the two images must differ ONLY in the probed value, so the pair
realises do(look := v') with everything else -- positions, other objects, palette,
lighting, resolution -- byte-identical. Anything less and the model can win on a
confound.

  colour  EXACT. The target's fill is one flat PALETTE colour inside a known bbox.
          Recolour those pixels and nothing else moves. This is the cleanest twin
          in the dataset and it is a true minimal pair.
  object  EXACT. Erase the target's bbox to the background and redraw the NEW
          shape at the same centre, same radius, same colour, with the renderer's
          own `draw()`. Only the silhouette changes.
  count   NOT ATTEMPTED. Adding or removing an object changes occlusion, spacing
          and the spatial relation between the target and the anchor, so the pair
          would differ in more than the probed value. It needs a re-render from
          the scene graph, which the item file does not store.
  spatial NOT ATTEMPTED, same reason: moving an object changes the whole layout.

So this covers 2 of 4 skills exactly and declines the other two rather than
shipping confounded pairs. `count` and `spatial` keep the channel-edit treatment
they already have.

Output: data/clevr3/images_twin/<id>_<skill>_<newvalue>.png plus a pairs file
whose schema matches the other twin sets, so eval_clevr3_flip.py can consume it.

Usage
  python scripts/clevr3_image_twins.py --split test_id
  python scripts/clevr3_image_twins.py --split test_ood
"""
import argparse, collections, json, os, sys
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
# reuse the GENERATOR's own palette and shape routine so the twin is drawn by the
# same code that drew the original -- no second implementation to drift.
from prepare_clevr3 import PALETTE, draw, SHAPES_ID, OOD_SHAPES, OOD_COLORS, W

BG = (245, 245, 245)


def nearest(px, table):
    """Which palette entry is this pixel? Flat fills, so exact match usually, but
    PNG round-trips can shift a channel by 1."""
    best, bd = None, 1e9
    for name, c in table.items():
        d = sum((a - b) ** 2 for a, b in zip(px[:3], c))
        if d < bd:
            best, bd = name, d
    return best, bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--out_images", default="data/clevr3/images_twin_clean")   # v1.191: the corrected set; never the old dir
    ap.add_argument("--out_pairs", default=None)
    ap.add_argument("--from_pairs", default="",
                    help="v1.191: regenerate the images of an EXISTING pairs file (its ids and gold_b values), so the "
                         "released twins are reproduced bit-exactly; without it the new value is chosen by crc32")
    a = ap.parse_args()
    import zlib
    forced = {}
    if a.from_pairs:
        for l in open(os.path.join(ROOT, a.from_pairs)):
            r = json.loads(l)
            if r.get("skill") in ("color", "object"):
                forced[r["id"]] = str(r["gold_b"])

    items = [json.loads(l) for l in
             open(os.path.join(ROOT, a.data_dir, f"{a.split}.jsonl"))]
    outdir = os.path.join(ROOT, a.out_images)
    os.makedirs(outdir, exist_ok=True)
    # OOD items may legitimately use held-out values; keep the alternative inside
    # the same value pool the item came from so the twin is not accidentally OOD
    # when the original is ID (or vice versa).
    pairs, rej = [], collections.Counter()

    for it in items:
        skill = it.get("skill")
        if forced and it["id"] not in forced and skill in ("color", "object"):
            rej["not in --from_pairs"] += 1
            continue
        if skill not in ("color", "object"):
            rej[f"{skill}: not an exact in-place edit (see docstring)"] += 1
            continue
        bb = it.get("bbox")
        if not bb:
            rej[f"{skill}: no bbox"] += 1
            continue
        src = os.path.join(ROOT, it["image"])
        if not os.path.exists(src):
            rej["image missing"] += 1
            continue
        gold = (it.get("channels") or {}).get("look")
        if not gold:
            rej["no look value"] += 1
            continue
        # CONFLICT PRESERVATION (2026-09-15): the replacement must differ from every value a TEXT
        # channel asserts. Otherwise the edited image agrees with the caption (or the on-screen text)
        # and the second half of Flip is free -- the failure mode the borrowed ladder benchmarks have
        # by construction. Measured before the fix: 86 of 374 test_id twins agreed with the caption.
        _txt = {str(v).strip().lower() for k, v in (it.get("channels") or {}).items()
                if k != "look" and v is not None}
        im = Image.open(src).convert("RGB")
        x0, y0, x1, y1 = [int(round(v)) for v in bb]
        x0, y0 = max(0, x0), max(0, y0)
        # v1.191 FIX: the renderer fills INCLUSIVE bounds [cx - r, cx + r] (PIL), so the bbox's last column and row
        # belong to the object; before this fix they were never erased / recoloured and every colour / object twin kept a
        # 1-px edge of the old object (results/twin_residue_audit.md). Clamp to the last pixel, iterate inclusively.
        x1, y1 = min(im.width - 1, x1), min(im.height - 1, y1)
        if x1 <= x0 or y1 <= y0:
            rej["degenerate bbox"] += 1
            continue

        if skill == "color":
            if gold not in PALETTE:
                rej["look colour not in palette"] += 1
                continue
            # EXCLUDE COLOURS ALREADY IN THE SCENE. make_scene() guarantees the
            # target has a UNIQUE colour; recolouring it to a colour another
            # object already wears silently breaks that invariant, which is what
            # makes the "What shape is the <colour> object?" question answerable.
            # Verified on id3_color: the naive pick was `cyan`, and two cyan
            # crosses were already in frame.
            present = set()
            pxa = im.load()
            for yy in range(0, im.height, 3):
                for xx in range(0, im.width, 3):
                    nm, dd = nearest(pxa[xx, yy], PALETTE)
                    if dd < 900:
                        present.add(nm)
            pool = [c for c in PALETTE if c != gold and c not in present and c not in _txt
                    and (c in OOD_COLORS) == (gold in OOD_COLORS)]
            if not pool:                       # relax scene uniqueness, never the text channels
                pool = [c for c in PALETTE if c != gold and c not in _txt
                        and (c in OOD_COLORS) == (gold in OOD_COLORS)]
                rej["colour reused from the scene (uniqueness relaxed)"] += 1
            if not pool:
                rej["no same-pool colour available that keeps the conflict"] += 1
                continue
            # v1.191: crc32, not hash() -- str hashes are salted per process, so the old choice was not reproducible
            new = forced.get(it["id"]) or sorted(pool)[zlib.crc32(it["id"].encode()) % len(pool)]
            tgt, rep = PALETTE[gold], PALETTE[new]
            px = im.load()
            n = 0
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    nm, d = nearest(px[x, y], PALETTE)
                    if nm == gold and d < 900:      # the target's own fill only
                        px[x, y] = rep
                        n += 1
            if n < 40:
                rej["too few target pixels recoloured"] += 1
                continue
            newval = new
        else:  # object == shape
            pool = [s for s in (SHAPES_ID + OOD_SHAPES) if s != gold and s not in _txt
                    and (s in OOD_SHAPES) == (gold in OOD_SHAPES)]
            if not pool:
                rej["no same-pool shape available that keeps the conflict"] += 1
                continue
            new = forced.get(it["id"]) or sorted(pool)[zlib.crc32(it["id"].encode()) % len(pool)]
            # the target's colour is whatever fills its bbox
            px = im.load()
            cnt = collections.Counter()
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    nm, d = nearest(px[x, y], PALETTE)
                    if d < 900:
                        cnt[nm] += 1
            if not cnt:
                rej["no palette fill inside bbox"] += 1
                continue
            colour = cnt.most_common(1)[0][0]
            d = ImageDraw.Draw(im)
            d.rectangle([x0, y0, x1, y1], fill=BG)   # erase INCLUSIVE (v1.191), keep the rest
            # centre and radius exactly as before the fix (end coordinates clamped to the image size, not size - 1)
            X1, Y1 = min(im.width, int(round(bb[2]))), min(im.height, int(round(bb[3])))
            cx, cy = (x0 + X1) // 2, (y0 + Y1) // 2
            r = max(6, min(X1 - x0, Y1 - y0) // 2)
            draw(d, new, cx, cy, r, PALETTE[colour])
            newval = new

        dst = os.path.join(outdir, f"{it['id']}_{newval}.png")
        im.save(dst)
        pairs.append({
            "pair_type": "look_image_swap", "channel": "look",
            "id": it["id"], "skill": skill, "question": it.get("question"),
            "gold_a": gold, "gold_b": newval,
            "ctx_a": {"image": it["image"]},
            "ctx_b": {"image": os.path.relpath(dst, ROOT)},
            # the TEXT channels are untouched, so they still support gold_a:
            # <read>/<ocr> must IGNORE this edit and <look> must FOLLOW it
            "read_value": (it.get("channels") or {}).get("read"),
            "ocr_value": (it.get("channels") or {}).get("ocr"),
            "bbox": bb, "split": a.split,
            "source": "in-place exact edit of our own render",
        })

    p = os.path.join(ROOT, a.out_pairs or
                     f"data/clevr3/pairs_look_{a.split.replace('test_','')}_clean.jsonl")   # v1.191: never the old file
    with open(p, "w") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {p}")
    print(f"  {len(pairs)} exact image twins  "
          f"{dict(collections.Counter(r['skill'] for r in pairs))}")
    print(f"  images in {a.out_images}")
    print(f"  declined/rejected: {dict(rej)}")


if __name__ == "__main__":
    main()
