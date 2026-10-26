#!/usr/bin/env python3
"""Procedural minimal-pair renderer for COUNTING and SPATIAL.

The old natural-image counting/spatial were dropped because their "twins" were
different scenes -> flip/stay confounded. Procedural rendering gives EXACT
minimal pairs (the only clean counterfactual for count/relation):

  counting: image_1 = N shapes at fixed positions; image_2 = the SAME N plus/
            minus ONE shape (rest identical). Only the count differs.
  spatial : image_1 = shape A left/above shape B; image_2 = positions SWAPPED
            (same two shapes, same sizes). Only the relation differs.

Emits the ColorSwap/UnifiedExample schema (image_1/2, caption_1/2, key_1/2) so
the existing pipeline (build_specs, eval_ladder, arms, k-sweep) runs unchanged.
Bounding boxes are recorded per shape (free grounding-metric material).

counting OOD = held-out high counts; spatial OOD = held-out object pairs.
"""
import argparse, hashlib, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw
from scsrc.tasks import UnifiedExample
import importlib.util
_s = importlib.util.spec_from_file_location(
    "prepare_task", os.path.join(os.path.dirname(__file__), "prepare_task.py"))
_pt = importlib.util.module_from_spec(_s); _s.loader.exec_module(_pt)
to_hf = _pt.to_hf

W = 336
PALETTE = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 170, 60),
           "yellow": (230, 200, 40), "purple": (150, 60, 200), "orange": (240, 140, 30)}
SHAPES = ["circle", "square", "triangle"]


def draw_shape(d, shape, cx, cy, r, color):
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    return [cx - r, cy - r, cx + r, cy + r]


def cells(n_cells=16):
    g = 4; step = W // g; pts = []
    for i in range(g):
        for j in range(g):
            pts.append((step * j + step // 2, step * i + step // 2))
    return pts


def render_counting(rng, shape, color, positions, count):
    im = Image.new("RGB", (W, W), (245, 245, 245)); d = ImageDraw.Draw(im)
    boxes = []
    for (cx, cy) in positions[:count]:
        jx, jy = rng.randint(-8, 8), rng.randint(-8, 8)
        boxes.append(draw_shape(d, shape, cx + jx, cy + jy, 22, PALETTE[color]))
    return im, boxes


def render_spatial(shape_a, ca, shape_b, cb, a_left):
    im = Image.new("RGB", (W, W), (245, 245, 245)); d = ImageDraw.Draw(im)
    y = W // 2
    xa, xb = (W // 4, 3 * W // 4) if a_left else (3 * W // 4, W // 4)
    ba = draw_shape(d, shape_a, xa, y, 34, PALETTE[ca])
    bb = draw_shape(d, shape_b, xb, y, 34, PALETTE[cb])
    return im, {"a": ba, "b": bb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["counting", "spatial"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--id_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    def h(s): return int(hashlib.sha1(str(s).encode()).hexdigest()[:8], 16)
    per = {"train": [], "test_id": [], "test_ood": []}
    grid = cells()

    for i in range(a.n):
        if a.task == "counting":
            shape = rng.choice(SHAPES); color = rng.choice(list(PALETTE))
            c1 = rng.randint(2, 9); delta = rng.choice([-1, 1, -2, 2])
            c2 = min(max(c1 + delta, 1), 10)
            if c2 == c1:
                continue
            hi = max(c1, c2)
            pos = rng.sample(grid, hi)                    # shared positions
            im1, b1 = render_counting(rng, shape, color, pos, c1)
            im2, b2 = render_counting(rng, shape, color, pos, c2)
            unit = shape if shape.endswith("s") else shape + "s"
            ex = UnifiedExample(
                image_1=im1, image_2=im2,
                caption_1=f"there are {c1} {unit}", caption_2=f"there are {c2} {unit}",
                key_1={shape: str(c1)}, key_2={shape: str(c2)},
                task="counting", klass=f"{c1},{c2}", id=i, has_swap=True,
                caption_source="synthetic")
            ood = (c1 >= 8 or c2 >= 8)                    # held-out high counts
        else:
            sa, sb = rng.sample(SHAPES, 2)
            ca, cb = rng.sample(list(PALETTE), 2)
            im1, _ = render_spatial(sa, ca, sb, cb, a_left=True)   # A left of B
            im2, _ = render_spatial(sa, ca, sb, cb, a_left=False)  # A right of B
            key = f"{ca} {sa}|{cb} {sb}"
            ex = UnifiedExample(
                image_1=im1, image_2=im2,
                caption_1=f"the {ca} {sa} is to the left of the {cb} {sb}",
                caption_2=f"the {ca} {sa} is to the right of the {cb} {sb}",
                key_1={key: "left"}, key_2={key: "right"},
                task="spatial", klass=f"{sa},{sb}", id=i, has_swap=True,
                caption_source="synthetic")
            ood = (sa == "triangle" or sb == "triangle")  # held-out object (triangle) pairs
        if ood:
            split = "test_ood"
        else:
            split = "test_id" if (h(f"{a.task}|{i}") % 100) < int(a.id_frac * 100) else "train"
        per[split].append(ex)

    os.makedirs(a.out, exist_ok=True)
    counts = {}
    for s, exs in per.items():
        if not exs:
            counts[s] = 0; continue
        hf, nn = to_hf(exs); hf.save_to_disk(os.path.join(a.out, s)); counts[s] = nn
    man = {"task": a.task, "source": "procedural", "counts": counts,
           "ood": "held-out high counts (>=8)" if a.task == "counting" else "held-out triangle-object pairs",
           "minimal_pair": "count +-1/2 same positions" if a.task == "counting" else "positions swapped"}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
