#!/usr/bin/env python3
"""Commitment-position attention table from attention_probe2 outputs.

Per (run, dataset): for each query set q (ctrl, lastp, first, preval, value,
ans) and band (mid, late), the RIGHT-SOURCE contrast:
  tok_{q}_img   = mass(q->img | look)  - mass(q->img | read)
  tok_{q}_cap   = mass(q->cap | read)  - mass(q->cap | look)
  tok_{q}_capval= mass(q->capval|read) - mass(q->capval|look)
and the INSTRUCTION analog at the same query sets (no ctrl row):
  instr_{q}_img / instr_{q}_cap / instr_{q}_capval
plus word-consultation (ans/value queries onto ctrlk keys) and sink shift.
Because preval/value/first are content-matched positions, token vs instruction
is apples-to-apples there. Gen runs: block-content and tag fidelity.
-> results/attn2_commitment.json + stdout table. CPU-only.
"""
import glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def merged(d, op):
    a = [np.load(os.path.join(d, f"{op}_{s}.npz"))["attn"]
         for s in ("correct", "incorrect")
         if os.path.exists(os.path.join(d, f"{op}_{s}.npz"))]
    return np.concatenate(a) if a else None


def band(v, L, which):
    sl = slice(L // 3, 2 * L // 3) if which == "mid" else slice(2 * L // 3, L)
    return float(v[sl].mean())


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "attention2", "*_test_id_step*"))):
        run = d.split("/runs/")[1].split("/")[0]
        dset = os.path.basename(d).split("_test_id")[0]
        summ = json.load(open(os.path.join(d, "summary.json")))
        CH, CHG = summ.get("channels", []), summ.get("channels_gen", [])

        gen = merged(d, "gen")
        if gen is not None:
            L = gen.shape[1]
            m = {nm: gen[..., i].mean((0, 2)) for i, nm in enumerate(CHG)}
            r = {"run": run, "dataset": dset, "kind": "gen", "n": int(gen.shape[0])}
            for b in ("mid", "late"):
                r[f"lookblk_img_fid_{b}"] = round(band(m["lookblk_img"], L, b) - band(m["lookblk_cap"], L, b), 4)
                r[f"readblk_cap_fid_{b}"] = round(band(m["readblk_cap"], L, b) - band(m["readblk_img"], L, b), 4)
                r[f"blk_img_contrast_{b}"] = round(band(m["lookblk_img"], L, b) - band(m["readblk_img"], L, b), 4)
                r[f"tag_img_contrast_{b}"] = round(band(m["looktag_img"], L, b) - band(m["readtag_img"], L, b), 4)
            rows.append(r)
            continue

        conds = {op: merged(d, op) for op in ("look", "read", "instr_look", "instr_read", "neutral")}
        if conds["look"] is None and conds["instr_look"] is None:
            continue
        L = next(v for v in conds.values() if v is not None).shape[1]
        r = {"run": run, "dataset": dset, "kind": "ctrl",
             "n": int(conds["look"].shape[0]) if conds["look"] is not None else 0}

        def curve(op, ch):
            A = conds[op]
            return None if (A is None or ch not in CH) else A[..., CH.index(ch)].mean((0, 2))

        for b in ("mid", "late"):
            for q in ("ctrl", "lastp", "first", "preval", "value", "ans"):
                pairs = (("tok", "look", "read"), ("instr", "instr_look", "instr_read"))
                for tag, oL, oR in pairs:
                    if tag == "instr" and q == "ctrl":
                        continue
                    li, ri = curve(oL, f"{q}_img"), curve(oR, f"{q}_img")
                    lc, rc = curve(oL, f"{q}_cap"), curve(oR, f"{q}_cap")
                    lv, rv = curve(oL, f"{q}_capval"), curve(oR, f"{q}_capval")
                    if li is not None and ri is not None:
                        r[f"{tag}_{q}_img_{b}"] = round(band(li, L, b) - band(ri, L, b), 4)
                        r[f"{tag}_{q}_cap_{b}"] = round(band(rc, L, b) - band(lc, L, b), 4)
                        r[f"{tag}_{q}_capval_{b}"] = round(band(rv, L, b) - band(lv, L, b), 4)
            # CITATION SITES (blocks arms): emitted <look>/<read> delimiter and
            # its span, vs the INSTRUCTION at the same commitment positions.
            for nm, ch in (("citetag", "citetag"), ("citeblk", "citeblk")):
                li_, ri_ = curve("look", f"{ch}_img"), curve("read", f"{ch}_img")
                lc_, rc_ = curve("look", f"{ch}_cap"), curve("read", f"{ch}_cap")
                if li_ is not None and ri_ is not None and max(abs(band(li_, L, b)), abs(band(ri_, L, b))) > 0:
                    r[f"{nm}_img_contrast_{b}"] = round(band(li_, L, b) - band(ri_, L, b), 4)
                    r[f"{nm}_cap_contrast_{b}"] = round(band(rc_, L, b) - band(lc_, L, b), 4)
                    r[f"{nm}_look_img_level_{b}"] = round(band(li_, L, b), 4)
            # word consultation + sink accounting (look condition)
            for q in ("value", "ans", "preval"):
                c = curve("look", f"{q}_ctrlk")
                if c is not None:
                    r[f"consult_{q}_{b}"] = round(band(c, L, b), 4)
            sb_l, sb_r = curve("look", "ans_bos"), curve("read", "ans_bos")
            if sb_l is not None and sb_r is not None:
                r[f"sink_shift_{b}"] = round(band(sb_l, L, b) - band(sb_r, L, b), 4)
        rows.append(r)

    out = {"rows": rows}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "attn2_commitment.json"), "w"), indent=1)
    hdr = ("run", "dataset", "tok_preval_img_mid", "instr_preval_img_mid",
           "tok_value_img_mid", "instr_value_img_mid", "tok_ctrl_img_mid",
           "tok_preval_capval_mid", "instr_preval_capval_mid", "consult_preval_mid")
    print(" | ".join(hdr))
    for r in rows:
        if r["kind"] == "ctrl":
            print(" | ".join(str(r.get(c, "")) for c in hdr))
    for r in rows:
        if r["kind"] == "gen":
            print(f"GEN {r['run']} {r['dataset']}: lookblk_img_fid={r.get('lookblk_img_fid_mid')} "
                  f"readblk_cap_fid={r.get('readblk_cap_fid_mid')} blk_img_contrast={r.get('blk_img_contrast_mid')} "
                  f"tag_img_contrast={r.get('tag_img_contrast_mid')}")
    print(f"[saved] results/attn2_commitment.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
