#!/usr/bin/env python
"""STEP 0 gate (frozen model). Measures whether source-collapse is real on
ColorSwap, and SAVES every model output (with the correct color, the distractor,
and exact-match) so you can inspect it yourself.

Modes:
  probe      (DEFAULT, the phenomenon, no separation scaffold): show the image +
             the CONFLICTING caption, ask a single naturalistic question
             ("what color is the X?"). Answer == image color -> grounded;
             == caption color -> LAUNDERED. Laundering among image-legible
             objects IS the phenomenon, undistorted by prompt scaffolding.
  structured (the model is TOLD to separate; a LOWER BOUND on collapse).
  natural    (free one-sentence description; weaker scoring).

Legibility is measured with the prose-tolerant parser and reported as
parse-vs-perception so a low pass-rate isn't silently a formatting artifact.
Writes runs/_diagnostic/{legible_<split>.json, legibility_<split>.jsonl,
records_<split>_<mode>.jsonl, diagnostic_<split>_<mode>.json}. Logs to wandb.
"""
import argparse, os, json
from collections import defaultdict
from tqdm import tqdm

from scsrc import paths
from scsrc.data import load_colorswap, legible_path, build_specs, get_image, load_task
from scsrc.colors import valid_swap
from scsrc import metrics as Me
from scsrc import prompts as P
from scsrc import model as M
from scsrc.infer import generate
from PIL import ImageFilter

PROBE_Q = ('Here is an image. A caption that is meant to describe it says: "{cap}". '
           'Looking at the IMAGE itself, what color is the {obj}? '
           'Answer with a single color word.')
NATURAL_Q = ('Here is an image and a caption describing it: "{cap}". '
             'Describe the scene in one short sentence.')


def _wandb_init(name):
    try:
        import wandb
        wandb.init(project="source-control", name=name, dir=paths.WANDB_DIR,
                   mode=os.environ.get("WANDB_MODE", "offline"))
        return wandb
    except Exception as e:
        print("[wandb] disabled:", e); return None


def probe_task(args, device):
    """Generic frozen-model baseline + laundering gate for any prepared task.

    Two questions, the right order:
      (1) CLEAN accuracy -- show the image + a neutral single-slot question (no
          conflicting caption). Can the model even do the task here? Laundering
          is only meaningful among the items it gets right when clean (the
          per-task analogue of ColorSwap legibility).
      (2) LAUNDERING under conflict -- show the image + the CONFLICTING caption.
          Answer == image value -> grounded; == text value -> LAUNDERED.
    Reports raw laundering AND laundering gated to clean-correct items. The
    gated number is the honest 'is source-collapse real on this task?' signal.
    """
    ddir = os.path.join(paths.RUNS, "_diagnostic"); os.makedirs(ddir, exist_ok=True)
    wb = _wandb_init(f"diagnostic_{args.task}_{args.split}_{args.framing}_probe")
    ds = load_task(args.data_dir, args.split)
    specs = build_specs(ds, args.split, conditions=("conflict",))
    model, processor = M.load(); model.eval()
    n = min(args.limit, len(specs))

    clean_correct = 0
    raw = {"grounded": 0, "laundered": 0, "other": 0}
    gated = {"grounded": 0, "laundered": 0, "other": 0}  # among clean-correct only
    records = []
    for i in tqdm(range(n), desc=f"probe[{args.task}:{args.split}]"):
        s = specs[i]
        obj = next(iter(s.image_key))
        img_v, txt_v = s.image_key[obj], s.text_key.get(obj)
        img = get_image(ds, s)
        # (1) clean perception (no conflicting caption)
        clean_ans = generate(model, processor, img, P.clean_q(args.task).format(obj=obj), device, 12)
        clean_v = Me.first_value(clean_ans, args.task)
        is_clean = (clean_v == img_v)
        clean_correct += int(is_clean)
        # (2) conflict
        conf_ans = generate(model, processor, img, P.probe_q(args.task, args.framing).format(obj=obj, cap=s.fed_caption), device, 12)
        conf_v = Me.first_value(conf_ans, args.task)
        label = Me.classify_source(conf_v, img_v, txt_v)
        raw[label] += 1
        if is_clean:
            gated[label] += 1
        records.append({"id": s.example_id, "object": obj, "image_value": img_v,
                        "text_value": txt_v, "clean_output": clean_ans, "clean_parsed": clean_v,
                        "clean_correct": is_clean, "conflict_output": conf_ans,
                        "conflict_parsed": conf_v, "label": label})
    tot = max(n, 1)
    ng = max(sum(gated.values()), 1)
    clean_acc = clean_correct / tot
    raw_rates = {k: raw[k] / tot for k in raw}
    gated_rates = {k: gated[k] / ng for k in gated}
    with open(os.path.join(ddir, f"records_{args.task}_{args.split}_probe.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(ddir, f"diagnostic_{args.task}_{args.split}_probe.json"), "w") as f:
        json.dump({"task": args.task, "framing": args.framing, "clean_accuracy": clean_acc, "n": tot,
                   "n_clean_correct": clean_correct, "raw": raw_rates,
                   "gated_on_clean_correct": gated_rates}, f, indent=2)
    if wb is not None:
        wb.log({"probe/clean_accuracy": clean_acc,
                **{f"probe/raw_{k}": v for k, v in raw_rates.items()},
                **{f"probe/gated_{k}": v for k, v in gated_rates.items()}})
    print(f"\n==== {args.task.upper()} BASELINE PROBE (frozen model, framing={args.framing}, n={tot}) ====")
    print(f"  clean accuracy (can the model do the task):     {clean_acc:.3f}  ({clean_correct}/{tot})")
    print(f"  raw   laundering (follows text, all items):     {raw_rates['laundered']:.3f}")
    print(f"  GATED laundering (follows text | clean-correct): {gated_rates['laundered']:.3f}  (n={ng})")
    print(f"  gated grounded={gated_rates['grounded']:.3f}  gated other/abstain={gated_rates['other']:.3f}")
    gl = gated_rates["laundered"]
    if args.framing == "text":
        print(f"  [framing=text] <read> ceiling: model follows the text value {gl:.3f} of the time "
              f"(higher = more steerable to text).")
    elif clean_acc < 0.15:
        print(f"  VERDICT: clean accuracy {clean_acc:.3f} is very low -> task too hard / GT noisy; "
              f"laundering is not interpretable. Pick another task or inspect the data.")
    elif gl >= 0.15:
        print(f"  VERDICT: gated laundering {gl:.3f} >> ColorSwap's 0.07 -> source-collapse IS present; worth training.")
    else:
        print(f"  VERDICT: gated laundering {gl:.3f} ~<= ColorSwap's 0.07 -> the model stays grounded when it "
              f"perceives; NO strong collapse here (same story as ColorSwap).")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--mode", default="probe", choices=["probe", "structured", "natural", "degrade"])
    ap.add_argument("--blur_levels", default="0,1,2,3,5", help="GaussianBlur radii for --mode degrade")
    ap.add_argument("--task", default="color", choices=["color", "counting", "spatial", "attr"])
    ap.add_argument("--data_dir", default=None, help="processed task dir; runs the generic task probe")
    ap.add_argument("--framing", default="leading", choices=["leading", "neutral", "text"],
                    help="probe caption framing: leading (biased to image), neutral (fair collapse test), text (<read> ceiling)")
    args = ap.parse_args()
    device = "cuda"
    if args.data_dir:
        return probe_task(args, device)
    ddir = os.path.join(paths.RUNS, "_diagnostic"); os.makedirs(ddir, exist_ok=True)
    wb = _wandb_init(f"diagnostic_{args.split}_{args.mode}")

    ds = load_colorswap(args.split, human_only=True)
    model, processor = M.load(); model.eval()
    n = min(args.limit, len(ds))

    # ---------- Phase 1: legibility (prose-tolerant) ----------
    leg_records, leg_reads = [], {}
    legible_ids, align1, align2 = [], 0, 0
    for idx in tqdm(range(n), desc=f"legibility[{args.split}]"):
        row = ds[idx]
        ok, k1, k2 = valid_swap(row["caption_1"], row["caption_2"])
        if not ok:
            continue
        eid = int(row.get("id", idx))
        img_raw = generate(model, processor, row["image_1"], P.IMG_ONLY, device, 48)
        txt_raw = generate(model, processor, None, P.TXT_ONLY.format(caption=row["caption_1"]), device, 48)
        img_read, txt_read = Me.parse_pred(img_raw), Me.parse_pred(txt_raw)
        leg_reads[eid] = (k1, k2, img_read)
        align1 += int(Me.key_match(img_read, k1)); align2 += int(Me.key_match(img_read, k2))
        conflicting = [o for o in k1 if o in k2 and k1[o] != k2[o]]
        img_ok = all(img_read.get(o) == k1[o] for o in conflicting)
        txt_ok = all(txt_read.get(o) == k1[o] for o in conflicting)
        if img_ok and txt_ok:
            legible_ids.append(eid)
        leg_records.append({"eid": eid, "conflicting": conflicting,
                            "img_read_raw": img_raw[:200], "img_read_parsed": img_read,
                            "img_legible": img_ok, "img_parsed_any": bool(img_read),
                            "txt_read_raw": txt_raw[:200], "txt_read_parsed": txt_read,
                            "txt_legible": txt_ok, "txt_parsed_any": bool(txt_read)})
        if len(leg_records) <= 5:
            print(f"  [legib eid {eid}] IMG='{img_raw[:80]}' -> {img_read} (ok={img_ok}) | "
                  f"TXT='{txt_raw[:60]}' -> {txt_read} (ok={txt_ok})")
    with open(os.path.join(ddir, f"legibility_{args.split}.jsonl"), "w") as f:
        for r in leg_records:
            f.write(json.dumps(r) + "\n")
    with open(legible_path(args.split), "w") as f:
        json.dump({"legible_ids": legible_ids, "split": args.split}, f)

    # parse-vs-perception: of examples that FAILED image legibility, how many
    # produced NO parseable color at all (=> likely format/parse, not perception)?
    img_fail = [r for r in leg_records if not r["img_legible"]]
    img_fail_no_parse = sum(1 for r in img_fail if not r["img_parsed_any"])
    print(f"\n[legibility] {len(legible_ids)}/{len(leg_records)} valid examples legible. "
          f"image-fail={len(img_fail)} of which {img_fail_no_parse} produced NO parseable color "
          f"(likely formatting, not blindness).")

    # ---------- DEGRADE: does laundering rise as the image gets blurry? ----------
    if args.mode == "degrade":
        levels = [float(x) for x in args.blur_levels.split(",")]
        records = []
        per_level = {b: {"grounded": [], "laundered": [], "other": []} for b in levels}
        for idx in tqdm(range(n), desc="degrade-sweep"):
            row = ds[idx]
            ok, k1, k2 = valid_swap(row["caption_1"], row["caption_2"])
            if not ok or int(row.get("id", idx)) not in leg_reads:
                continue
            eid = int(row.get("id", idx)); _, _, img_read = leg_reads[eid]
            conflicting = [o for o in k1 if o in k2 and k1[o] != k2[o]]
            legible = [o for o in conflicting if img_read.get(o) == k1[o]]   # seen when CLEAN
            if not legible:
                continue
            base = row["image_1"].convert("RGB"); cap2 = row["caption_2"]
            for b in levels:
                im = base if b == 0 else base.filter(ImageFilter.GaussianBlur(radius=b))
                for o in legible:
                    ans = generate(model, processor, im, PROBE_Q.format(cap=cap2, obj=o), device, 12)
                    col = Me.first_color(ans)
                    label = ("grounded" if col == k1[o] else "laundered" if col == k2[o] else "other")
                    for key in ("grounded", "laundered", "other"):
                        per_level[b][key].append(int(label == key))
                    records.append({"eid": eid, "object": o, "blur": b, "output": ans,
                                    "correct_image": k1[o], "distractor_caption": k2[o],
                                    "parsed": col, "label": label, "em": int(col == k1[o])})
        curve = {f"blur_{b}": {"laundered": (sum(per_level[b]["laundered"]) / len(per_level[b]["laundered"]) if per_level[b]["laundered"] else None),
                               "grounded": (sum(per_level[b]["grounded"]) / len(per_level[b]["grounded"]) if per_level[b]["grounded"] else None),
                               "other": (sum(per_level[b]["other"]) / len(per_level[b]["other"]) if per_level[b]["other"] else None),
                               "n": len(per_level[b]["laundered"])} for b in levels}
        with open(os.path.join(ddir, f"records_{args.split}_degrade.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(ddir, f"diagnostic_{args.split}_degrade.json"), "w") as f:
            json.dump({"curve": curve, "levels": levels, "n_legible": len(legible_ids)}, f, indent=2)
        if wb is not None:
            for b in levels:
                wb.log({"blur": b, "degrade/laundered": curve[f"blur_{b}"]["laundered"],
                        "degrade/grounded": curve[f"blur_{b}"]["grounded"],
                        "degrade/other": curve[f"blur_{b}"]["other"]})
        print("\n==== DEGRADATION CURVE (laundering vs blur radius) ====")
        for b in levels:
            c = curve[f"blur_{b}"]
            print(f"  blur={b}: laundered={c['laundered']:.3f}  grounded={c['grounded']:.3f}  "
                  f"other/abstain={c['other']:.3f}  (n={c['n']})")
        lo, hi = curve[f"blur_{levels[0]}"]["laundered"], curve[f"blur_{levels[-1]}"]["laundered"]
        print("\n==== VERDICT ====")
        if hi is not None and lo is not None and hi - lo >= 0.20:
            print(f"  GREEN: laundering rises {lo:.2f} -> {hi:.2f} with blur. Source-collapse is "
                  "REAL but GATED BY PERCEPTUAL UNCERTAINTY -> this is the regime to study "
                  "(and it is scale-robust: stronger language prior vs weak vision).")
        else:
            print(f"  laundering {lo:.2f} -> {hi:.2f}: blur mostly raises abstention, not laundering "
                  "(check the 'other/abstain' column). ColorSwap likely exhausted -> move to "
                  "retrieval / multi-claim conflicts (your pipeline).")
        return

    # ---------- Phase 2:
    records = []
    agg = defaultdict(list)
    rel_preds, rel_golds = [], []
    for idx in tqdm(range(n), desc=f"collapse[{args.mode}]"):
        row = ds[idx]
        ok, k1, k2 = valid_swap(row["caption_1"], row["caption_2"])
        if not ok:
            continue
        eid = int(row.get("id", idx))
        if eid not in leg_reads:
            continue
        _, _, img_read = leg_reads[eid]
        conflicting = [o for o in k1 if o in k2 and k1[o] != k2[o]]
        img_legible_objs = [o for o in conflicting if img_read.get(o) == k1[o]]
        img1, cap2 = row["image_1"], row["caption_2"]   # the conflict instance

        if args.mode == "probe":
            for o in img_legible_objs:                  # only objects the model CAN see
                ans = generate(model, processor, img1, PROBE_Q.format(cap=cap2, obj=o), device, 12)
                col = Me.first_color(ans)
                label = ("grounded" if col == k1[o] else
                         "laundered" if col == k2[o] else "other")
                agg["grounded"].append(int(label == "grounded"))
                agg["laundered"].append(int(label == "laundered"))
                rec = {"eid": eid, "mode": "probe", "object": o, "output": ans,
                       "correct_image": k1[o], "distractor_caption": k2[o],
                       "parsed": col, "label": label, "em": int(col == k1[o])}
                records.append(rec)
                if len(records) <= 8:
                    print(f"  [probe eid {eid} {o}] img={k1[o]} cap={k2[o]} -> '{ans.strip()[:50]}' [{label}]")

        elif args.mode == "structured":
            out = generate(model, processor, img1, P.CONFLICT_STRUCTURED.format(caption=cap2), device, 96)
            img_seg = out.split("IMAGE:", 1)[1].split("TEXT:", 1)[0] if ("IMAGE:" in out and "TEXT:" in out) else ""
            txt_seg = out.split("TEXT:", 1)[1].split("RELATION:", 1)[0] if "TEXT:" in out else ""
            pi, pt, prel = Me.parse_pred(img_seg), Me.parse_pred(txt_seg), Me.parse_relation(out)
            sep = Me.separation_correct(pi, pt, k1, k2)
            col = Me.is_collapse(pi, pt, prel, k1, k2)
            lr = Me.laundering_rate(pi, k1, k2)
            agg["separation"].append(int(sep)); agg["collapse"].append(int(col))
            if lr is not None:
                agg["laundering"].append(lr)
            rel_preds.append(prel); rel_golds.append("conflict")
            records.append({"eid": eid, "mode": "structured", "output": out[:300],
                            "correct_image": k1, "distractor_caption": k2,
                            "parsed_image": pi, "parsed_text": pt, "relation": prel,
                            "separation": int(sep), "collapse": int(col), "laundering": lr})
            if len(records) <= 5:
                print(f"  [struct eid {eid}] sep={sep} collapse={col} | out='{out[:90]}'")

        else:  # natural
            out = generate(model, processor, img1, NATURAL_Q.format(cap=cap2), device, 64)
            parsed = Me.parse_pred(out)
            for o in img_legible_objs:
                said = parsed.get(o)
                label = ("grounded" if said == k1[o] else "laundered" if said == k2[o]
                         else "omitted" if said is None else "other")
                agg["grounded"].append(int(label == "grounded"))
                agg["laundered"].append(int(label == "laundered"))
                records.append({"eid": eid, "mode": "natural", "object": o, "output": out[:200],
                                "correct_image": k1[o], "distractor_caption": k2[o],
                                "parsed": said, "label": label, "em": int(said == k1[o])})
            if len([r for r in records if r["eid"] == eid]) and eid % 23 == 0:
                print(f"  [natural eid {eid}] '{out[:90]}' -> {parsed}")

    # ---------- summarise ----------
    summary = {k: (sum(v) / len(v) if v else None) for k, v in agg.items()}
    if rel_preds:
        summary.update(Me.conflict_prf(rel_preds, rel_golds))
    summary.update({"mode": args.mode, "n_legible": len(legible_ids),
                    "n_valid": len(leg_records), "n_scored_items": len(records),
                    "align_match_caption1": align1, "align_match_caption2": align2,
                    "image_fail_no_parse": img_fail_no_parse, "image_fail_total": len(img_fail)})
    with open(os.path.join(ddir, f"records_{args.split}_{args.mode}.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(ddir, f"diagnostic_{args.split}_{args.mode}.json"), "w") as f:
        json.dump({"summary": summary, "qualitative": records[:20]}, f, indent=2)

    if wb is not None:
        wb.log({f"diag/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
        cols = sorted({k for r in records[:200] for k in r})
        tbl = wb.Table(columns=cols)
        for r in records[:200]:
            tbl.add_data(*[json.dumps(r.get(c)) if isinstance(r.get(c), (dict, list)) else r.get(c) for c in cols])
        wb.log({f"diag/records_{args.mode}": tbl})

    print("\n==== DIAGNOSTIC SUMMARY (mode=%s) ====" % args.mode)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if align2 > max(align1, 1) * 1.5:
        print("  [!] ALIGNMENT MAY BE FLIPPED (image_1 matched caption_2 more).")
    print("\n==== VERDICT ====")
    if args.mode in ("probe", "natural"):
        lr = summary.get("laundered")
        if lr is None:
            print("  no scored items (check legibility).")
        elif lr >= 0.20:
            print(f"  GREEN: laundering under conflict = {lr:.2f} on image-legible objects. "
                  "The phenomenon is present in naturalistic answering -> proceed.")
        else:
            print(f"  AMBER/RED: laundering = {lr:.2f}. On legible objects the model mostly stays "
                  "grounded; ColorSwap may not elicit collapse -> consider multi-claim / retrieval "
                  "conflicts (your pipeline) before training.")
    else:
        print("  structured mode is the LOWER BOUND (separation is scaffolded). Run --mode probe "
              "for the real phenomenon before judging.")


if __name__ == "__main__":
    main()
