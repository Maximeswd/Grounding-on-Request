#!/usr/bin/env python
"""Make the corrected-twin (residue-free) Ground-CF evaluations CANONICAL (v1.191, 2026-09-25, author: "the
corrected-twin rerun should replace, not coexist with, the affected main numbers"). Idempotent; run it as often as you like.

For every runs/*/eval/clevr3_<split>/flip_step_800<T>_ctw.json (the rerun on data/clevr3/pairs_look_all_*_clean.jsonl):
  flip_step_800<T>.json            -> renamed flip_step_800<T>_restwin.json   (demoted, never deleted; grid skips it)
  flip_step_800<T>_ctw.json        -> COPIED to flip_step_800<T>.json          (every consumer now reads the clean result)
  (the matching _items.jsonl[.gz] files the same way)
Core cells are promoted per (split, model) only when the model's whole core set has landed -- word, prefix, both LoRA
specialists, English and no request -- so paired within-model comparisons never mix twin sets; other files (transfer,
SFT only, CTM + Stay, early word, joint LoRA) are promoted as they land. Every promotion is logged with the before/after
headline metrics in results/clean_twin_promotion.json (the reproducibility note's source).
Usage: python scripts/promote_clean_twins.py [--dry_run 1]
"""
import argparse, glob, json, os, shutil, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "results", "clean_twin_promotion.json")
MODELS = ["", "_qwen35", "_qwen2b"]
CORE = [("clevr3_2s_1tok_ctm_cp_fc_seed0{s}", "_cp4_ho"), ("clevr3_2s_pt_1tok_ctm_cp_fc_seed0{s}", "_cp4_ho"),
        ("clevr3_2s_lora_r16_ctm_cp_fc_look_seed0{s}", "_cp4_ho"), ("clevr3_2s_lora_r16_ctm_cp_fc_read_seed0{s}", "_cp4_ho"),
        ("clevr3_2s_5tok_sft_seed0{s}", "_cp4_ho_fc_instr"), ("clevr3_2s_5tok_sft_seed0{s}", "_cp4_ho_fc_noop")]
KEYS = ("look_acc", "look_flip", "look_stay_c", "read_acc", "read_flip", "read_stay_c")


def core_files(split):
    return {s: [os.path.join(ROOT, "runs", r.format(s=s), "eval", f"clevr3_{split}", f"flip_step_800{t}_ctw.json") for r, t in CORE]
            for s in MODELS}


def promote(ctw, log, dry):
    canon = ctw[:-len("_ctw.json")] + ".json"
    key = os.path.relpath(canon, ROOT)
    if key in log["promoted"]:
        # already promoted: refresh if the clean file was rewritten after the promotion
        if os.path.getmtime(ctw) <= log["promoted"][key]["ctw_mtime"] + 1:
            return False
    new = json.load(open(ctw))
    old_p = canon[:-5] + "_restwin.json"
    old = None
    if os.path.exists(canon) and key not in log["promoted"] and not os.path.exists(old_p):
        old = json.load(open(canon))
        if not dry:
            os.rename(canon, old_p)
            for ext in ("_items.jsonl.gz", "_items.jsonl"):
                if os.path.exists(canon[:-5] + ext):
                    os.rename(canon[:-5] + ext, old_p[:-5] + ext)
    elif os.path.exists(old_p):
        old = json.load(open(old_p))
    if not dry:
        shutil.copy2(ctw, canon)
        for ext in ("_items.jsonl.gz", "_items.jsonl"):
            if os.path.exists(ctw[:-5] + ext):
                shutil.copy2(ctw[:-5] + ext, canon[:-5] + ext)
    log["promoted"][key] = {"ctw_mtime": os.path.getmtime(ctw), "at": time.strftime("%Y-%m-%d %H:%M"),
                            "residue_version": os.path.relpath(old_p, ROOT) if old else None,
                            "before": {k: (old or {}).get(k) for k in KEYS}, "after": {k: new.get(k) for k in KEYS}}
    print(("[dry] " if dry else "") + f"promoted {key}: look_flip {(old or {}).get('look_flip')} -> {new.get('look_flip')}, "
          f"read_stay_c {(old or {}).get('read_stay_c')} -> {new.get('read_stay_c')}")
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry_run", type=int, default=0)
    a = ap.parse_args()
    log = json.load(open(LOG)) if os.path.exists(LOG) else {"promoted": {}, "note": __doc__.split("\n")[0]}
    n = 0
    for split in ("test_id", "test_ood"):
        cf = core_files(split)
        core_all = {p for v in cf.values() for p in v}
        for s, files in cf.items():
            if all(os.path.exists(p) for p in files):
                for p in files:
                    n += promote(p, log, a.dry_run)
        for p in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "eval", f"clevr3_{split}", "flip_step_800*_ctw.json"))):
            if p in core_all:
                continue                                  # core cells wait for their model's full set (above)
            n += promote(p, log, a.dry_run)
    if not a.dry_run:
        json.dump(log, open(LOG, "w"), indent=1)
    print(f"[promote] {n} file(s) promoted this pass; {len(log['promoted'])} total")


if __name__ == "__main__":
    main()
