#!/usr/bin/env python
"""COMPOSED CONFLICT DETECTION on the ladder (VCF / ColorSwap / COCO-CF): the two-source
version of scripts/eval_compose_detect.py, same prompt family and target
(scsrc.prompts.compose_*), same forced-choice scoring, same metrics.

Per conflict spec: one conflicting object (rotating, as in eval_flip_ll), image value v_i
and caption value v_t; the agree twin is the same image with its own caption.
Requests: (look, read) and (read, look).

Usage: python scripts/eval_compose_detect_ladder.py --run_dir runs/color_neo_ctrl_5tok_ctm_compose_seed0 \
           --task color --data_dir data/color --split test_id
       ... --mode instructed --tag _instr   /   --mode noop --tag _noop   /   --template heldout
Output: runs/<run>/eval/<ds>_<split>/compose_detect_step_<step><tag>.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc import prompts as P
from scsrc import score as SC
from scsrc.prompts import compose_template, compose_carrier, compose_request, compose_answers, compose_verdict
from evaluate import apply_checkpoint
from scripts.eval_flip_ll import norm, COLORS

ORDERS = [("look", "read"), ("read", "look")]


def ll(model, proc, image, user, cand):
    r = SC.loglik(model, proc, image, user, cand, "cuda")
    return r[0] if isinstance(r, (tuple, list)) else float(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", default="data/color")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=140)
    ap.add_argument("--mode", default=None, help="None (the run's own carrier) | instructed | noop")
    ap.add_argument("--template", default="train", choices=["train", "heldout"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode instructed/noop needs --tag")
    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
    n, method = meta["n_tokens"], meta["method"]
    meth = "neo" if a.mode in ("instructed", "noop") else method
    mode = a.mode or "srconly"
    heldout = a.template == "heldout"
    try:
        cfg = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    except Exception:
        cfg = {}
    trained_compose = cfg.get("ctrl_style") == "compose"
    print(f"[prompt] method={meth} mode={mode} L={n} compose-trained={trained_compose} template={a.template}")
    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",)) if s.cond == "conflict"]
    is_vcf = "vcf" in a.data_dir

    def user(cap, srcs, seed):
        op_str, prefix = compose_carrier(meth, mode, srcs, n)
        u = compose_request(op_str, cap, compose_template(seed, heldout=heldout), cap_first=True)
        return (prefix + " " + u) if prefix else u

    def pick_two(image, cap, srcs, cands, seed):
        u = user(cap, srcs, seed)
        sc = {(v1, v2): ll(model, proc, image, u, compose_answers(v1, v2)) for v1 in cands for v2 in cands}
        return max(sc, key=sc.get)

    def verdict(image, cap, srcs, p1, p2, seed):
        u = user(cap, srcs, seed); pre = compose_answers(p1, p2)
        sc = {r: ll(model, proc, image, u, pre + compose_verdict(r)) for r in ("agree", "conflict")}
        return max(sc, key=sc.get)

    R = {f"{s1}|{s2}": {"detect": [], "attrib": [], "false": [], "v_conflict": [], "v_agree": [], "consistent": []}
         for s1, s2 in ORDERS}
    recs, n_seen = [], 0
    for sp in tqdm(specs, desc="compose-ladder"):
        ik, tk = norm(sp.image_key), norm(sp.text_key)
        if not ik or set(ik) != set(tk):
            continue
        conf = [o for o in ik if ik[o] != tk[o] and not (is_vcf and any(c in o.split() for c in COLORS))]
        if not conf:
            continue
        if n_seen >= a.max_n:
            break
        n_seen += 1
        obj = conf[n_seen % len(conf)]
        ci = P.fmt_key_task({obj: ik[obj]}, a.task); ct = P.fmt_key_task({obj: tk[obj]}, a.task)
        img = get_image(ds, sp)
        capp = ds[sp.ex_index][f"caption_{sp.img_idx}"]          # agree twin: the image's own caption
        vals = {"look": ci, "read": ct}
        for s1, s2 in ORDERS:
            key = f"{s1}|{s2}"
            p1, p2 = pick_two(img, sp.fed_caption, (s1, s2), (ci, ct), n_seen)
            R[key]["detect"].append(int(p1 != p2))
            R[key]["attrib"].append(int(p1 == vals[s1] and p2 == vals[s2]))
            vc = verdict(img, sp.fed_caption, (s1, s2), p1, p2, n_seen)
            R[key]["v_conflict"].append(int(vc == "conflict"))
            R[key]["consistent"].append(int((vc == "conflict") == (p1 != p2)))
            q1, q2 = pick_two(img, capp, (s1, s2), (ci, ct), n_seen)
            R[key]["false"].append(int(q1 != q2))
            va = verdict(img, capp, (s1, s2), q1, q2, n_seen)
            R[key]["v_agree"].append(int(va == "agree"))
            recs.append({"ex": sp.ex_index, "pair": key, "p1": p1, "p2": p2, "gold": [vals[s1], vals[s2]],
                         "agree_p": [q1, q2], "verdict_conflict": vc, "verdict_agree": va})
    out = {"meta": {"run": a.run_dir, "step": a.step, "mode": mode, "method": meth, "n_items": n_seen,
                    "task": a.task, "data_dir": a.data_dir, "split": a.split, "template": a.template,
                    "compose_trained": trained_compose}, "pairs": {}}
    for key, d in R.items():
        if not d["detect"]:
            continue
        f = lambda k: sum(d[k]) / len(d[k])
        det, fa, vc, va = f("detect"), f("false"), f("v_conflict"), f("v_agree")
        out["pairs"][key] = {"detect": round(det, 4), "attrib": round(f("attrib"), 4), "false": round(fa, 4),
                             "balanced": round((det + 1 - fa) / 2, 4),
                             "verdict_conflict_recall": round(vc, 4), "verdict_agree_recall": round(va, 4),
                             "verdict_balanced": round((vc + va) / 2, 4),
                             "verdict_consistent": round(f("consistent"), 4), "n": len(d["detect"])}
        o = out["pairs"][key]
        print(f"  {key:10s} detect {det:.3f} attrib {o['attrib']:.3f} false {fa:.3f} bal {o['balanced']:.3f} | "
              f"verdict conflict {vc:.2f} agree {va:.2f} bal {o['verdict_balanced']:.2f} | n={o['n']}")
    out["records"] = recs
    dsname = os.path.basename(a.data_dir.rstrip("/"))
    od = os.path.join(a.run_dir, "eval", f"{dsname}_{a.split}")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, f"compose_detect_step_{a.step}{a.tag}.json")
    json.dump(out, open(fp, "w"), indent=1)
    print("[saved]", fp)


if __name__ == "__main__":
    main()
