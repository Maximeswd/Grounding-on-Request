#!/usr/bin/env python
"""PROCEDURAL CONFLICT DETECTION on Ground-CF with two SEPARATE requests (the S9 procedure), scored
exactly like the ladder battery so the two can share a table.

Items: colour and shape questions whose image value and caption value differ. For each item
  conflict context   the original item
  agree context      the caption rewritten to the image's value (both sources attest the same value)
Both requests (<look>, <read>; or the run's prefixes; or the English names with --mode instructed; or a
neutral phrase with --mode noop) are scored as a forced choice between the SAME two candidates, the
image value and the caption value, in both contexts. Reported
  detect  share of conflict items on which the two answers differ
  attrib  ... and each is its own source's value
  false   share of agree items on which the two answers differ
  bal     (detect + 1 - false) / 2
plus per-item margins (log-odds image value minus caption value under each request, both contexts) so
the detector can be scored as an ROC (scripts/detect_roc.py convention).
Output: runs/<run>/eval/clevr3_<split>/detect_step_<step><tag>.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from scsrc.clevr3 import clevr3_user_content, _edited_item
from evaluate import apply_checkpoint
from scripts.eval_clevr3 import cand, loglik


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id"); ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=150)
    ap.add_argument("--mode", default=None, help="None (the run's own request) | instructed | noop")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode needs --tag so the trained arm's file is not overwritten")
    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    try:
        _cfg = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    except Exception:
        _cfg = {}
    meth = "neo" if a.mode in ("instructed", "noop") else _cfg.get("method", "neo")
    cfgm = a.mode or _cfg.get("factor_mode", "srconly")
    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]

    def margin(item, source, v_img, v_cap, seed):
        um = [{"role": "user", "content": clevr3_user_content(item, source, n, cfgm, seed, method=meth)}]
        return float(loglik(model, proc, um, cand(v_img), "cuda") - loglik(model, proc, um, cand(v_cap), "cuda"))

    R = {k: [] for k in ("detect", "attrib", "false", "look_margin", "read_margin",
                         "look_margin_agree", "read_margin_agree", "skill")}
    seen = 0
    for i, it in enumerate(tqdm(rows, desc="c3detect")):
        if seen >= a.max_n:
            break
        if it.get("skill") not in ("color", "object"):
            continue
        ch = {k: str(v) for k, v in it["channels"].items()}
        v_img, v_cap = ch["look"], ch["read"]
        if v_img == v_cap:
            continue
        seen += 1
        ml, mr = margin(it, "look", v_img, v_cap, i), margin(it, "read", v_img, v_cap, i)
        pl, pr = (v_img if ml > 0 else v_cap), (v_img if mr > 0 else v_cap)
        R["detect"].append(int(pl != pr)); R["attrib"].append(int(pl == v_img and pr == v_cap))
        agree = _edited_item(it, "read", v_img)          # caption now attests the image's value
        mla, mra = margin(agree, "look", v_img, v_cap, i), margin(agree, "read", v_img, v_cap, i)
        R["false"].append(int((mla > 0) != (mra > 0)))
        R["look_margin"].append(ml); R["read_margin"].append(mr)
        R["look_margin_agree"].append(mla); R["read_margin_agree"].append(mra); R["skill"].append(it["skill"])
    m = lambda v: round(sum(v) / len(v), 3) if v else None
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step, "mode": a.mode, "n": seen,
           "detect": m(R["detect"]), "attrib": m(R["attrib"]), "false": m(R["false"]),
           "bal": round((m(R["detect"]) + 1 - m(R["false"])) / 2, 3) if seen else None,
           "note": "conflict = original item; agree = caption rewritten to the image value; both requests "
                   "forced choice over (image value, caption value); margins = log-odds image minus caption",
           "records": {k: ([round(x, 4) for x in v] if k.endswith("margin") or k.endswith("agree") else v)
                       for k, v in R.items()}}
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=1))
    od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"detect_step_{a.step}{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
