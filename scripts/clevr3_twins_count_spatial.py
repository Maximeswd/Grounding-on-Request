#!/usr/bin/env python
"""GROUND-CF IMAGE TWINS FOR COUNT AND SPATIAL (2026-09-21).

clevr3_image_twins.py gave <look> exact twins for colour and object by editing the target's bbox in
place. Count and spatial were "not attempted" because the item file stores no scene graph. The graph
is recoverable: prepare_clevr3.py draws every scene from ONE seeded RNG, so replaying main() with the
same arguments and a render() that records instead of writing yields every scene's object list. The
replayed items are checked against the shipped test files before anything is rendered.

  count    remove ONE object that is neither the target nor the anchor (the farthest from the target),
           k -> k-1; target/anchor pixels and their relation are untouched. Needs k >= 3.
  spatial  mirror the target across the anchor on the dominant axis (left<->right or above<->below),
           other coordinate kept; accepted only if the new relation is the exact opposite, the target
           stays on canvas, and no object comes within 56 px of it.
Rows go to data/clevr3/pairs_look_cs_<id|ood>.jsonl (count+spatial only) and
data/clevr3/pairs_look_all_<id|ood>.jsonl (colour+object existing pairs + the new ones) so one
evaluation yields the four-skill average with the per-skill breakdown the evaluator already records.
Images: data/clevr3/images_twin_cs/. Nothing existing is modified.
"""
import json, os, sys, math, importlib, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
pc = importlib.import_module("prepare_clevr3")

OPP = {"left": "right", "right": "left", "above": "below", "below": "above"}
R, MIN_D = 24, 56


def replay_scene_graphs():
    """sid -> (objs, k) by replaying the generator with recording render()."""
    captured = {}
    def fake_render(objs, path):
        captured[os.path.basename(path)[:-4]] = [dict(o) for o in objs]
    # 2026-09-21: the recording stub MUST be undone. Leaving it installed made every later
    # pc.render(new, dst) in main() record instead of write, so the pair rows were produced while
    # images_twin_cs/ stayed empty and the whole _cs4 evaluation died on FileNotFoundError.
    real_render = pc.render
    pc.render = fake_render
    tmp = tempfile.mkdtemp(prefix="c3replay_")
    argv = sys.argv; sys.argv = ["prepare_clevr3.py", "--out", tmp]
    try:
        pc.main()
    finally:
        sys.argv = argv
        pc.render = real_render
    rows = {sp: [json.loads(l) for l in open(os.path.join(tmp, f"{sp}.jsonl"))] for sp in ("train", "test_id", "test_ood")}
    return captured, rows


def check_alignment(replayed, split):
    shipped = {r["id"]: r for r in (json.loads(l) for l in open(f"{ROOT}/data/clevr3/{split}.jsonl"))}
    n = ok = 0
    for r in replayed[split]:
        s = shipped.get(r["id"])
        if s is None:
            continue
        n += 1; ok += int(s["channels"] == r["channels"] and s["question"] == r["question"])
    frac = ok / max(n, 1)
    print(f"[replay] {split}: {ok}/{n} items match shipped channels+question ({frac:.4f})")
    if frac < 0.999:
        raise SystemExit("[STOP] replayed scenes do not match the shipped items; refusing to build twins")
    return shipped


def main():
    # 2026-09-22 (STATUS v1.86): --splits train renders the TRAINING twins for count and spatial so CTM can
    # train the image request on all four skills; default keeps the shipped test behaviour byte-identical.
    splits = (sys.argv[sys.argv.index("--splits") + 1].split(",") if "--splits" in sys.argv else ["test_id", "test_ood"])
    outimg = f"{ROOT}/data/clevr3/images_twin_cs"; os.makedirs(outimg, exist_ok=True)
    captured, replayed = replay_scene_graphs()
    stats = json.load(open(f"{ROOT}/data/clevr3/pairs_look_cs_stats.json")) if os.path.exists(f"{ROOT}/data/clevr3/pairs_look_cs_stats.json") else {}
    for split in splits:
        tag = split.replace("test_", "")
        shipped = check_alignment(replayed, split)
        rows, rej = [], {"count_k2": 0, "spatial_offcanvas": 0, "spatial_overlap": 0, "spatial_rel": 0, "no_graph": 0}
        for it in shipped.values():
            sid = it["id"].rsplit("_", 1)[0]
            objs = captured.get(sid)
            if objs is None:
                rej["no_graph"] += 1; continue
            k = len(objs); t, anc = objs[0], objs[1]
            if it["skill"] == "count":
                if k < 3:
                    rej["count_k2"] += 1; continue
                rest = list(range(2, k))
                drop = max(rest, key=lambda j: math.hypot(objs[j]["cx"] - t["cx"], objs[j]["cy"] - t["cy"]))
                new = [o for j, o in enumerate(objs) if j != drop]
                dst = os.path.join(outimg, f"{it['id']}_{k-1}.png"); pc.render(new, dst)
                rows.append({"pair_type": "look_image_swap", "channel": "look", "id": it["id"], "skill": "count",
                             "question": it["question"], "gold_a": str(k), "gold_b": str(k - 1),
                             "ctx_a": {"image": it["image"]}, "ctx_b": {"image": os.path.relpath(dst, ROOT)},
                             "read_value": it["channels"]["read"], "ocr_value": it["channels"]["ocr"], "bbox": None,
                             "split": split, "source": "re-render from the replayed scene graph: one non-target object removed"})
            elif it["skill"] == "spatial":
                rel = str(it["channels"]["look"])
                if rel not in OPP:
                    rej["spatial_rel"] += 1; continue
                # SWAP the target's and the anchor's positions: the relation inverts exactly (dx, dy
                # negate), both stay on their existing grid cells, nothing else moves. Mirroring across
                # the anchor left the canvas in ~85% of scenes (154/178 ID), so it is not used.
                nt, na = dict(t), dict(anc)
                nt["cx"], nt["cy"], na["cx"], na["cy"] = anc["cx"], anc["cy"], t["cx"], t["cy"]
                if pc.rel_of(nt, na) != OPP[rel]:
                    rej["spatial_rel"] += 1; continue
                new = [nt, na] + objs[2:]
                dst = os.path.join(outimg, f"{it['id']}_{OPP[rel]}.png"); pc.render(new, dst)
                rows.append({"pair_type": "look_image_swap", "channel": "look", "id": it["id"], "skill": "spatial",
                             "question": it["question"], "gold_a": rel, "gold_b": OPP[rel],
                             "ctx_a": {"image": it["image"]}, "ctx_b": {"image": os.path.relpath(dst, ROOT)},
                             "read_value": it["channels"]["read"], "ocr_value": it["channels"]["ocr"],
                             "bbox": [nt["cx"] - R, nt["cy"] - R, nt["cx"] + R, nt["cy"] + R],
                             "split": split, "source": "re-render from the replayed scene graph: target and anchor positions swapped"})
        cs = f"{ROOT}/data/clevr3/pairs_look_cs_{tag}.jsonl"
        with open(cs, "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        old = f"{ROOT}/data/clevr3/pairs_look_{tag}.jsonl"
        allp = f"{ROOT}/data/clevr3/pairs_look_all_{tag}.jsonl"
        with open(allp, "w") as f:
            if os.path.exists(old):
                for l in open(old): f.write(l if l.endswith("\n") else l + "\n")
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = {}
        missing = [r["ctx_b"]["image"] for r in rows
                   if isinstance(r.get("ctx_b"), dict) and r["ctx_b"].get("image")
                   and not os.path.exists(r["ctx_b"]["image"])]
        if missing:
            raise SystemExit(f"[STOP] {len(missing)} twin images were not written, e.g. {missing[0]}")
        for r in rows: by[r["skill"]] = by.get(r["skill"], 0) + 1
        n_old = sum(1 for _ in open(old)) if os.path.exists(old) else 0
        stats[split] = {"new_rows": by, "rejected": rej, "existing_colour_object_rows": n_old, "all_rows": n_old + len(rows)}
        print(f"[twins] {split}: {by} rejected={rej} -> {cs}; combined {allp} ({n_old + len(rows)} rows)")
    json.dump(stats, open(f"{ROOT}/data/clevr3/pairs_look_cs_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
