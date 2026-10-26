#!/usr/bin/env python
"""Ladder eval core: control accuracy via the validated LL forced-choice.

Per conflict item, <look> should prefer the IMAGE value and <read> the CAPTION
value. Atomic answers (VCF) use forced_choice; multi-object (ColorSwap) uses
forced_choice_per_object (per-object routing, averaged) -- the fix for the
multi-object confound the anchor caught. VCF excludes objects whose NAME
contains a colour word (2.8%, confounds a colour task). Margin sign also gives
a cheap flip proxy: on a conflict item look's margin should favour image, read's
should favour caption.

Primary = LL forced-choice (no decode). Output per (run, split, step):
  ctrl_look  = P(look prefers image value)   [routing to vision, the hard axis]
  ctrl_read  = P(read prefers caption value)
  purity     = ctrl_look - (1 - ctrl_read_on_look)   ... reported as look/read sep
Usage: python scripts/eval_ladder.py --run_dir RUN --task color --data_dir data/X --split test_id --step 800
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger
from scsrc import prompts as P, score as SC
from evaluate import apply_checkpoint

COLORS = {"red","orange","yellow","green","blue","purple","pink","brown","black",
          "white","gray","grey","gold","silver","tan","beige","violet","cyan",
          "maroon","navy"}


def norm(k):
    return {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
            for o, v in (k or {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=120)
    ap.add_argument("--vcf_exclude_color_names", type=int, default=1)
    a = ap.parse_args()

    model, processor, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    try:
        _cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
        ctrl_site = _cfg.get("args", {}).get("ctrl_site", "user")
    except Exception:
        ctrl_site = "user"
    if ctrl_site == "decision":
        print("[eval] decision-site run: token-free user turn, trigger-prefixed "
              "candidates (atomic items only)")
    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",))
             if s.cond == "conflict"][:a.max_n]

    look_hits, read_hits, n_seen = [], [], 0
    is_vcf = "vcf" in a.data_dir
    for sp in tqdm(specs, desc=f"ctrl({os.path.basename(a.run_dir)})"):
        img = get_image(ds, sp)
        ik, tk = norm(sp.image_key), norm(sp.text_key)
        if not ik or set(ik) != set(tk):
            continue
        objs = list(ik)
        if is_vcf and a.vcf_exclude_color_names and \
                any(c in o.split() for o in objs for c in COLORS):
            continue
        n_seen += 1
        for op, want in (("look", ik), ("read", tk)):
            trig = trigger(f"{op}_ctrl", n)
            if ctrl_site == "decision":
                if len(objs) != 1:
                    continue        # decision-site scoring is atomic-only for now
                u = P.render_decision(sp.fed_caption)
            else:
                u = P.render_ctrl(P.CTRL_CANONICAL, trig, sp.fed_caption)
            if len(objs) == 1:
                o = objs[0]
                ci = P.fmt_key_task({o: ik[o]}, a.task)
                ct = P.fmt_key_task({o: tk[o]}, a.task)
                if ctrl_site == "decision":
                    # shared trigger prefix conditions both candidates; its own
                    # logprob is a common offset and cancels in the argmax
                    ci, ct = f"{trig} {ci}", f"{trig} {ct}"
                idx, _ = SC.forced_choice(model, processor, img, u, [ci, ct], "cuda")
                routed_img = (idx == 0)
                hit = routed_img if op == "look" else (not routed_img)
                (look_hits if op == "look" else read_hits).append(int(hit))
            else:
                # hold other objects at the OP's own target source, else the
                # held context biases every per-object choice (bug 2026-07-22)
                per = SC.forced_choice_per_object(model, processor, img, u, objs,
                                                  ik, tk, a.task, "cuda",
                                                  base_src=("img" if op == "look" else "txt"))
                want_img = (op == "look")
                frac = sum(1 for o in objs if (per[o][0] == "img") == want_img) / len(objs)
                (look_hits if op == "look" else read_hits).append(frac)

    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "n": n_seen, "n_tokens": n, "arm": meta.get("obj_tag", "?"),
           "ctrl_look": m(look_hits), "ctrl_read": m(read_hits)}
    print(json.dumps(out))
    outd = os.path.join(a.run_dir, "eval", f"{os.path.basename(a.data_dir)}_{a.split}")
    os.makedirs(outd, exist_ok=True)
    json.dump(out, open(os.path.join(outd, f"ctrl_ll_step_{a.step}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
