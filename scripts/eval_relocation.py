#!/usr/bin/env python
"""TEST-TIME RELOCATION: is the word's meaning in its VECTOR or its PLACE?

The functional-site hypothesis' cheapest falsification test (reframe plan,
2026-08-07). Take TRAINED rows, no retraining, and move them at eval time.
Design guard: moving the word out of the slot breaks the carrier sentence,
so every relocated condition keeps a FILLER word in the slot (the banked
filler control shows filler-in-slot ~= nothing). The only difference between
conditions is then WHERE the trained rows sit:

  slot      Caption: "...". Give me a <W> answer.            (trained position)
  filler    Caption: "...". Give me a <F> answer.            (floor control)
  prefix    <W> Caption: "...". Give me a <F> answer.
  postq     Caption: "...". Give me a <F> answer. <W>
  postcap   Caption: "...". <W> Give me a <F> answer.        (near image/caption boundary)

Vector hypothesis: C(w,p) roughly position-robust.
Functional-site hypothesis: C(w, trained slot) >> C(w, elsewhere).
Banked prior (train-time dsite arm): placement matters, look >> read.
Output: eval/{ds}_{split}/relocation_step_<STEP>.json with the full
routing/flip battery per position.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import token_strings
from scsrc import prompts as P, score as SC
from evaluate import apply_checkpoint

COLORS = {"red","orange","yellow","green","blue","purple","pink","brown","black",
          "white","gray","grey","gold","silver","tan","beige","violet","cyan",
          "maroon","navy"}
FILLER = "lantern"     # neutral existing word occupying the slot


def norm(k):
    return {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
            for o, v in (k or {}).items()}


def build_prompt(pos, trig, cap):
    clause = f'Caption: "{cap}".'
    if pos == "slot":
        return f"{clause} Give me a {trig} answer."
    body = f"{clause} Give me a {FILLER} answer."
    if pos == "filler":
        return body
    if pos == "prefix":
        return f"{trig} {body}"
    if pos == "postq":
        return f"{body} {trig}"
    if pos == "postcap":
        return f"{clause} {trig} Give me a {FILLER} answer."
    raise ValueError(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=80)
    a = ap.parse_args()

    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",))
             if s.cond == "conflict"]
    is_vcf = "vcf" in a.data_dir
    POS = ("slot", "filler", "prefix", "postq", "postcap")
    res = {p: {k: [] for k in ("look_acc", "read_acc", "look_flip", "read_flip")}
           for p in POS}
    n_seen = 0
    for sp in tqdm(specs, desc=f"reloc({os.path.basename(a.run_dir)})"):
        ik, tk = norm(sp.image_key), norm(sp.text_key)
        if not ik or set(ik) != set(tk):
            continue
        conf = [o for o in ik if ik[o] != tk[o]
                and not (is_vcf and any(c in o.split() for c in COLORS))]
        if not conf:
            continue
        if n_seen >= a.max_n:
            break
        n_seen += 1
        obj = conf[n_seen % len(conf)]
        v_i, v_t = ik[obj], tk[obj]
        img = get_image(ds, sp)
        imgp = ds[sp.ex_index][f"image_{3 - sp.img_idx}"]
        capp = ds[sp.ex_index][f"caption_{sp.img_idx}"]
        ci = P.fmt_key_task({obj: v_i}, a.task)
        ct = P.fmt_key_task({obj: v_t}, a.task)

        def pick(pos, op, image, cap):
            trig = "".join(token_strings(f"{op}_ctrl", n))
            u = build_prompt(pos, trig, cap)
            la, _, _ = SC.loglik(model, proc, image, u, ci, "cuda")
            lb, _, _ = SC.loglik(model, proc, image, u, ct, "cuda")
            return "i" if la > lb else "t"

        for pos in POS:
            li = pick(pos, "look", img, sp.fed_caption)
            lp = pick(pos, "look", imgp, sp.fed_caption)
            ri = pick(pos, "read", img, sp.fed_caption)
            rc = pick(pos, "read", img, capp)
            res[pos]["look_acc"].append(int(li == "i"))
            res[pos]["read_acc"].append(int(ri == "t"))
            res[pos]["look_flip"].append(int(li == "i" and lp == "t"))
            res[pos]["read_flip"].append(int(ri == "t" and rc == "i"))

    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "n": n_seen, "filler": FILLER,
           "positions": {p: {k: m(v) for k, v in d.items()}
                         for p, d in res.items()},
           "note": ("carrier sentence held constant via filler-in-slot; the "
                    "only difference across relocated conditions is where the "
                    "trained rows sit. 'filler' row = floor control (no "
                    "trained rows anywhere).")}
    print(json.dumps(out))
    outd = os.path.join(a.run_dir, "eval",
                        f"{os.path.basename(a.data_dir)}_{a.split}")
    os.makedirs(outd, exist_ok=True)
    json.dump(out, open(os.path.join(outd, f"relocation_step_{a.step}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
