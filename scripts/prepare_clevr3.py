#!/usr/bin/env python3
"""CLEVR-3channel: the controlled testbed for the FACTORED source×operator algebra.

Total control over a scene of shapes → exact 3-way counterfactuals across three
channels, for multiple skills. Each channel independently states a value:
  look : the TRUE value from the rendered scene
  read : a fluent caption ("The answer to the question is X.")
  ocr  : extracted-text format ("COLOR=X") -- prompt strings, the MVG-twin
         provenance distinction (NOT burned into pixels; that's an appendix
         probe). Call it the extracted-text channel, never pixel-OCR.

IDENTIFIABILITY (audit 2026-07-22): every look question must be a FUNCTION OF
THE IMAGE. The v1 questions ("What color is one of the objects?", "Is the first
object left or right of the second?") were NOT -- 'one of'/'first' have no
visual referent, so the look gold was unlearnable noise. v2 scenes carry a
designated TARGET object (unique shape AND unique color) and an ANCHOR object
(second unique shape); questions reference objects by those unique attributes:
  count   : How many shapes are in the image?                 -> k
  color   : What color is the {target_shape}?                 -> target color
  object  : What shape is the {target_color} object?          -> target shape
  spatial : Is the {tgt} left of, right of, above, or below the {anchor}?
            -> dominant-axis relation, placement margin enforced.

STRICT VALUE HOLDOUT: OOD values (triangle / purple / counts 7-9 / 'above')
appear ONLY as the look gold of test_ood items -- never as read/ocr values,
never in train questions, never as train distractor-object attributes. ID
scenes are split train/test_id at the SCENE level (no shared images across
splits). OOD scenes are generated per axis and contribute ONLY their OOD-axis
item. Writes data/clevr3/{train,test_id,test_ood}.jsonl + images.
"""
import argparse, hashlib, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw

W = 336
PALETTE = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 170, 60),
           "yellow": (230, 200, 40), "purple": (150, 60, 200), "orange": (240, 140, 30),
           "pink": (245, 110, 170), "cyan": (30, 190, 210), "brown": (140, 90, 40)}
# >=2 held-out VALUES per axis wherever the inventory allows (a single held
# value can be idiosyncratic); counts already hold 3. spatial keeps 1 (only 4
# relations exist and the 4-way question needs >=3 trained ones).
OOD_COLORS = ["purple", "pink"]
OOD_SHAPES = ["triangle", "star"]
SHAPES_ID = ["circle", "square", "diamond", "cross"]
COLORS_ID = [c for c in PALETTE if c not in OOD_COLORS]
COUNTS_ID = list(range(2, 7)); OOD_COUNTS = [7, 8, 9]
RELS = ["left", "right", "above", "below"]; OOD_REL = "above"
RELS_ID = [r for r in RELS if r != OOD_REL]
MARGIN = 42   # dominant-axis margin (px) so the spatial relation is unambiguous


def draw(d, shape, cx, cy, r, color):
    import math
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "diamond":
        d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=color)
    elif shape == "cross":
        t = r // 3
        d.rectangle([cx - t, cy - r, cx + t, cy + r], fill=color)
        d.rectangle([cx - r, cy - t, cx + r, cy + t], fill=color)
    elif shape == "star":  # 5-pointed (OOD)
        pts = []
        for i in range(10):
            rad = r if i % 2 == 0 else r * 0.45
            ang = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        d.polygon(pts, fill=color)
    else:  # triangle (OOD)
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)


def cells():
    g, step = 4, W // 4
    return [(step * j + step // 2, step * i + step // 2) for i in range(g) for j in range(g)]


def render(objs, path):
    im = Image.new("RGB", (W, W), (245, 245, 245)); dd = ImageDraw.Draw(im)
    for o in objs:
        draw(dd, o["shape"], o["cx"], o["cy"], 24, PALETTE[o["color"]])
    im.save(path)


def rel_of(o0, o1):
    """Dominant-axis relation of o0 wrt o1; None if ambiguous (margin not met)."""
    dx, dy = o0["cx"] - o1["cx"], o0["cy"] - o1["cy"]
    if abs(abs(dx) - abs(dy)) < MARGIN // 2:
        return None
    if abs(dx) > abs(dy):
        return "left" if dx < 0 else "right"
    return "above" if dy < 0 else "below"


def make_scene(rng, k, tgt_shape, tgt_color, force_rel=None):
    """Target obj0: UNIQUE shape + UNIQUE color. Anchor obj1: second unique
    shape. Remaining k-2 objects: the third ID shape, non-target ID colors.
    Returns objs or None (rejection sampling on the spatial margin)."""
    id_shapes = [s for s in SHAPES_ID if s != tgt_shape]
    anchor_shape = rng.choice(id_shapes)
    rest_shape = rng.choice([s for s in id_shapes if s != anchor_shape])
    rest_colors = [c for c in COLORS_ID if c != tgt_color]
    for _ in range(60):
        pos = rng.sample(cells(), k)
        objs = []
        for i, (cx, cy) in enumerate(pos):
            shape = tgt_shape if i == 0 else (anchor_shape if i == 1 else rest_shape)
            color = tgt_color if i == 0 else rng.choice(rest_colors)
            objs.append({"shape": shape, "color": color,
                         "cx": cx + rng.randint(-8, 8), "cy": cy + rng.randint(-8, 8)})
        r = rel_of(objs[0], objs[1])
        if r is None:
            continue
        if force_rel is not None and r != force_rel:
            continue
        if force_rel is None and r == OOD_REL:
            continue          # ID scenes never have the held-out relation
        return objs, r
    return None, None


def q_and_true(objs, rel, k):
    t, a = objs[0], objs[1]
    return {
        "count": ("How many shapes are in the image?", str(k)),
        "color": (f"What color is the {t['shape']}?", t["color"]),
        "object": (f"What shape is the {t['color']} object?", t["shape"]),
        "spatial": (f"Is the {t['shape']} left of, right of, above, or below "
                    f"the {a['shape']}?", rel),
    }


def distractors(rng, skill, v_true, three_way):
    """read/ocr values: ALWAYS from the ID pools (strict value holdout), never
    equal to v_true; when three_way, ocr differs from read too."""
    pool = {"count": [str(x) for x in COUNTS_ID], "color": COLORS_ID,
            "object": SHAPES_ID, "spatial": RELS_ID}[skill]
    cs = [x for x in pool if x != v_true]
    v_cap = rng.choice(cs)
    if three_way:
        cs2 = [x for x in cs if x != v_cap]
        v_ocr = rng.choice(cs2) if cs2 else v_cap
    else:
        v_ocr = v_cap
    return v_cap, v_ocr


def caption_text(rng, skill, v, objs, diverse):
    """diverse=False: the fixed v3 template (kept for comparability). diverse=
    True: NATURAL per-skill paraphrases -- the fixed 'answer is X' template let
    <read> become positional extraction ('copy the word after *is*') rather
    than caption comprehension; the native-caption transfer drop (srconly read
    .68) is that shortcut failing. Natural rotating captions force reading."""
    if not diverse:
        return f"The answer to the question is {v}."
    t, a2 = objs[0], objs[1]
    forms = {
        "count": [f"There are {v} shapes in the scene.",
                  f"The scene contains {v} shapes.",
                  f"A total of {v} shapes are shown."],
        "color": [f"The {t['shape']} is {v}.",
                  f"There is a {v} {t['shape']} in the scene.",
                  f"The {t['shape']} in the picture is {v}."],
        "object": [f"The {t['color']} object is a {v}.",
                   f"There is a {t['color']} {v} in the scene.",
                   f"The {t['color']} shape shown is a {v}."],
        "spatial": [f"The {t['shape']} is {v} of the {a2['shape']}." if v in ("left", "right")
                    else f"The {t['shape']} is {v} the {a2['shape']}.",
                    f"Relative to the {a2['shape']}, the {t['shape']} is {v}.",
                    f"You can see the {t['shape']} {v} of the {a2['shape']}." if v in ("left", "right")
                    else f"You can see the {t['shape']} {v} the {a2['shape']}."],
    }
    return rng.choice(forms[skill])


def emit(rng, rows, split, sid, path, skill, q, v_true, k, ocr_frac, bbox=None,
         objs=None, diverse=False):
    three = (int(hashlib.sha1(f"{sid}_{skill}_ocr".encode()).hexdigest()[:6], 16)
             % 100) < int(ocr_frac * 100)
    v_cap, v_ocr = distractors(rng, skill, v_true, three)
    rows[split].append({
        "id": f"{sid}_{skill}", "image": path, "skill": skill, "question": q,
        "channels": {"look": v_true, "read": v_cap, "ocr": v_ocr},
        "caption": caption_text(rng, skill, v_cap, objs, diverse),
        # skill-BLIND field name: 'COUNT=3' would leak the skill into the
        # prompt, defeating the generic-question condition where the operator
        # token must be the only skill carrier
        "ocr_text": f"VALUE={v_ocr}",
        "bbox": bbox,   # target-object evidence box (AAT + causal-crop); None for count
        "n_distinct": len({v_true, v_cap, v_ocr}), "n_objects": k})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/clevr3")
    ap.add_argument("--n_scenes", type=int, default=1200, help="ID scenes")
    ap.add_argument("--n_ood_scenes", type=int, default=200, help="per OOD axis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--id_frac", type=float, default=0.15, help="test_id scene frac")
    ap.add_argument("--caption_diverse", type=int, default=0,
                    help="1: natural rotating per-skill captions instead of the "
                         "fixed 'answer is X' template (anti-shortcut variant)")
    ap.add_argument("--ocr_distinct_frac", type=float, default=0.5,
                    help="fraction of items where OCR states a THIRD value "
                         "(distinct from look and read) so the 3-way source "
                         "routing matrix is non-degenerate")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    imgdir = os.path.join(a.out, "images"); os.makedirs(imgdir, exist_ok=True)
    rows = {"train": [], "test_id": [], "test_ood": []}

    # ---- ID scenes: all values in-distribution; SCENE-level train/id split --
    for si in range(a.n_scenes):
        k = rng.choice(COUNTS_ID)
        objs, rel = make_scene(rng, k, rng.choice(SHAPES_ID), rng.choice(COLORS_ID))
        if objs is None:
            continue
        sid = f"id{si}"
        path = os.path.join(imgdir, f"{sid}.png"); render(objs, path)
        split = ("test_id" if (int(hashlib.sha1(sid.encode()).hexdigest()[:6], 16)
                               % 100) < int(a.id_frac * 100) else "train")
        t0 = objs[0]
        tb = [t0["cx"] - 24, t0["cy"] - 24, t0["cx"] + 24, t0["cy"] + 24]
        for skill, (q, v_true) in q_and_true(objs, rel, k).items():
            emit(rng, rows, split, sid, path, skill, q, v_true, k,
                 a.ocr_distinct_frac, bbox=(None if skill == "count" else tb),
                 objs=objs, diverse=bool(a.caption_diverse))

    # ---- OOD scenes: ONE axis takes its held-out value; only that axis's
    # item is kept, so held-out values never leak into other train items -----
    for axis in ("count", "color", "object", "spatial"):
        for si in range(a.n_ood_scenes):
            sid = f"ood_{axis}{si}"
            k = rng.choice(OOD_COUNTS if axis == "count" else COUNTS_ID)
            # alternate the held-out values within the axis
            tgt_shape = OOD_SHAPES[si % 2] if axis == "object" else rng.choice(SHAPES_ID)
            tgt_color = OOD_COLORS[si % 2] if axis == "color" else rng.choice(COLORS_ID)
            force = OOD_REL if axis == "spatial" else None
            objs, rel = make_scene(rng, k, tgt_shape, tgt_color, force_rel=force)
            if objs is None:
                continue
            path = os.path.join(imgdir, f"{sid}.png"); render(objs, path)
            q, v_true = q_and_true(objs, rel, k)[axis]
            t0 = objs[0]
            tb = [t0["cx"] - 24, t0["cy"] - 24, t0["cx"] + 24, t0["cy"] + 24]
            emit(rng, rows, "test_ood", sid, path, axis, q, v_true, k,
                 a.ocr_distinct_frac, bbox=(None if axis == "count" else tb),
                 objs=objs, diverse=bool(a.caption_diverse))

    os.makedirs(a.out, exist_ok=True)
    counts = {}
    for sp, r in rows.items():
        counts[sp] = len(r)
        with open(os.path.join(a.out, f"{sp}.jsonl"), "w") as f:
            for it in r:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    man = {"task": "clevr3", "n_scenes": a.n_scenes, "counts": counts,
           "skills": ["count", "color", "object", "spatial"],
           "sources": ["look", "read", "ocr"],
           "by_skill": dict(Counter(it["skill"] for r in rows.values() for it in r)),
           "n_distinct": dict(Counter(it["n_distinct"] for r in rows.values() for it in r)),
           "ood": "strict holdout: shapes {triangle,star} / colors {purple,pink} "
                  "/ counts {7,8,9} / relation 'above' occur ONLY as look-gold "
                  "of test_ood items (>=2 held values per axis where the "
                  "inventory allows; spatial structurally capped at 1)",
           "identifiability": "target = unique shape + unique color; anchor = "
                              "second unique shape; spatial = dominant axis w/ margin",
           "note": "3rd channel = extracted-text in PROMPT (MVG-twin), not pixel-OCR"}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
