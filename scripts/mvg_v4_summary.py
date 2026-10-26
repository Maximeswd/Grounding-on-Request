#!/usr/bin/env python
"""Two-source MultiVENT-G retrain summary (2026-10-05): held-out-video pairs (173 visual from 22 videos, 300 description),
full frame and evidence crop, with VIDEO-clustered bootstrap intervals (the visual pool is dominated by three videos).
Reads runs/<run>/eval/mvg_flip/flip_step_800_<tag>[_crop0.25][_<style>].json and the matching _items.jsonl.gz.
Flip per pair = pick_sum correct in both contexts (summed-log-likelihood decision, as in the paper).
Usage: python scripts/mvg_v4_summary.py [--tag _v4ho] [--out results/mvg_v4_summary]
"""
import argparse, gzip, json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [("Gemma-3-4B", "", "_frozen_gemma"), ("Qwen-3.5-4B", "_qwen35", "_frozen_qwen35"), ("Qwen-3.5-2B", "_qwen2b", "_frozen_qwen2b")]
ARMS = [("No request", None, "noop"), ("English", None, "instr"),
        ("Word, CTM clean twins", "mvgV4_5tok_ctm_neo_seed0{s}", ""), ("Prefix (same slot), CTM", "mvgV4_5tok_ctm_pt_seed0{s}", ""), ("Joint LoRA, CTM", "mvgV4_lora_r16_ctm_seed0{s}", ""),
        ("Word, CTM+ECC [ECC INACTIVE = CTM]", "mvgV4_5tok_ctmecc_neo_seed0{s}", ""), ("Prefix, CTM+ECC [ECC INACTIVE = CTM]", "mvgV4_5tok_ctmecc_pt_seed0{s}", ""), ("Joint LoRA, CTM+ECC [ECC INACTIVE = CTM]", "mvgV4_lora_r16_ctmecc_seed0{s}", ""),
        ("Word, CTM+ECC+aug [ECC INACTIVE = CTM+aug]", "mvgV4_5tok_ctmeccaug_neo_seed0{s}", ""), ("Prefix, CTM+ECC+aug [ECC INACTIVE = CTM+aug]", "mvgV4_5tok_ctmeccaug_pt_seed0{s}", ""), ("Joint LoRA, CTM+ECC+aug [ECC INACTIVE = CTM+aug]", "mvgV4_lora_r16_ctmeccaug_seed0{s}", ""),
        ("Word, CTM+ECC (repaired 10-06)", "mvgV4_5tok_ctmeccv2_neo_seed0{s}", ""), ("Joint LoRA, CTM+ECC (repaired 10-06)", "mvgV4_lora_r16_ctmeccv2_seed0{s}", "")]


def load(run, tag, crop, style):
    base = f"flip_step_800{tag}" + (f"_{style}" if style else "") + ("_crop0.25" if crop else "") + (f"_{style}" if style else "")
    p = os.path.join(ROOT, "runs", run, "eval", "mvg_flip", base + "_items.jsonl.gz")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in gzip.open(p, "rt")]


def pick_pair(r):
    ps = r["pick_sum"]
    if isinstance(ps, dict):
        a = ps.get("a", ps.get("own_a")); b = ps.get("b", ps.get("own_b"))
    else:
        a, b = ps[0], ps[1]
    return str(a), str(b)


def stats(recs, ptype, B=2000, seed=0):
    rs = [r for r in recs if r["pair_type"] == ptype]
    if not rs:
        return None
    # picks are candidate LETTERS: 'a' = the value supported in context a (gold_a), 'b' = gold_b
    bits, vids = [], []
    for r in rs:
        a, b = pick_pair(r); bits.append(int(a == "a" and b == "b")); vids.append(r["video_id"])
    acc_a = sum(int(pick_pair(r)[0] == "a") for r in rs) / len(rs); acc_b = sum(int(pick_pair(r)[1] == "b") for r in rs) / len(rs)
    by = {}
    for v, x in zip(vids, bits): by.setdefault(v, []).append(x)
    keys = list(by); rng = random.Random(seed); boots = []
    for _ in range(B):
        tot = n = 0
        for k in rng.choices(keys, k=len(keys)): tot += sum(by[k]); n += len(by[k])
        boots.append(tot / n)
    boots.sort()
    return {"n": len(rs), "videos": len(keys), "flip": sum(bits) / len(bits), "ci": (boots[int(.025 * B)], boots[int(.975 * B) - 1]), "acc_a": acc_a, "acc_b": acc_b}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="_v4ho"); ap.add_argument("--out", default=os.path.join(ROOT, "results", "mvg_v4_summary"))
    a = ap.parse_args()
    f3 = lambda x: "--" if x is None else f"{x:.3f}"
    # cross-check against the evaluator's stored aggregate (flip of vis_value_swap / desc_swap) for every file read
    def check(run, tag, crop, style, vf, df):
        base = f"flip_step_800{tag}" + (f"_{style}" if style else "") + ("_crop0.25" if crop else "") + (f"_{style}" if style else "")
        p = os.path.join(ROOT, "runs", run, "eval", "mvg_flip", base + ".json")
        if not os.path.exists(p): return "no-json"
        d = json.load(open(p)); ok = True
        if vf: ok &= abs(d["vis_value_swap"]["flip"] - vf["flip"]) < 2e-3
        if df: ok &= abs(d["desc_swap"]["flip"] - df["flip"]) < 2e-3
        return "ok" if ok else f"MISMATCH stored vis {d['vis_value_swap']['flip']} desc {d['desc_swap']['flip']}"
    lines = ["# Two-source MultiVENT-G retrain, held-out videos (visual: 173 pairs / 22 videos; description: 300 pairs). Flip with video-clustered 95% CI; chance .25.", "",
             "| model | arm | image Flip full | image Flip crop | image per-frame acc full (a/b) | description Flip full |", "|---|---|---|---|---|---|"]
    bank = {}
    for ml, sfx, frz in MODELS:
        for lab, runt, style in ARMS:
            run = runt.format(s=sfx) if runt else frz
            full = load(run, a.tag, False, style); crop = load(run, a.tag, True, style)
            if full is None and crop is None:
                lines.append(f"| {ml} | {lab} | pending | pending | | |"); continue
            vf = stats(full, "vis_value_swap") if full else None; vc = stats(crop, "vis_value_swap") if crop else None; df = stats(full, "desc_swap") if full else None
            bank[f"{ml}|{lab}"] = {"run": run, "vis_full": vf, "vis_crop": vc, "desc_full": df, "check_full": check(run, a.tag, False, style, vf, df)}
            ci = lambda s: f"{s['flip']:.3f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]" if s else "pending"
            lines.append(f"| {ml} | {lab} | {ci(vf)} | {ci(vc)} | {f3(vf and vf['acc_a'])}/{f3(vf and vf['acc_b'])} | {ci(df)} |" + ("" if bank[f"{ml}|{lab}"]["check_full"] == "ok" else f" <!-- {bank[f'{ml}|{lab}']['check_full']} -->"))
    json.dump(bank, open(a.out + ".json", "w"), indent=1, default=str); open(a.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"[saved] {a.out}.md / .json")


if __name__ == "__main__":
    main()
