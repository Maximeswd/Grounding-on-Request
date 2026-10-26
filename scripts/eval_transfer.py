#!/usr/bin/env python
"""Cross-dataset router transfer: CLEVR3-trained tokens evaluated on the ladder.

The pre-registered 3-router battery: Design A <src_look> (srconly, multi-skill)
vs Design B <src_look> (factored) vs single-task <src_look> (srconly-coloronly,
the 0.004 reproduction) -- all trained ONLY on CLEVR3 shapes, scored here on
real-photo ladder conflicts (VCF first). Transfer = does the routing survive
the domain shift (rendered shapes -> photos) and value shift (unseen colors,
real objects)?

Each ladder conflict spec is mapped into the CLEVR3 item schema and scored with
the SAME prompt builder + LL forced-choice the tokens were trained with:
  caption_style clevr3 : caption/ocr = trained format ("The answer to the
                         question is X.") -- isolates routing transfer.
  caption_style native : the dataset's own caption -- adds format shift (harder).
Usage: python scripts/eval_transfer.py --run_dir runs/clevr3_srconly_exp_seed0 \
         --data_dir data/vcf --split test_id --step 800
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.clevr3 import clevr3_user_content
from evaluate import apply_checkpoint
from scripts.eval_clevr3 import loglik, cand

COLORS = {"red","orange","yellow","green","blue","purple","pink","brown","black",
          "white","gray","grey","gold","silver","tan","beige","violet","cyan",
          "maroon","navy"}


Q_BY_TASK = {
    "color": "What color is the {obj}?",
    "noun": "What is the main object shown?",
    "count": "How many {obj} are there?",
}
# target task -> the clevr3 OPERATOR it instantiates (factored mode composes
# <src_x><op_y>; 'noun' is the object-identity skill, so it uses <op_object>)
OP_BY_TASK = {"color": "color", "noun": "object", "count": "count"}


def norm(k):
    return {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
            for o, v in (k or {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/vcf")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=120)
    ap.add_argument("--caption_style", default="clevr3", choices=["clevr3", "native"])
    ap.add_argument("--task", default="color", choices=["color", "noun", "count"],
                    help="question/answer type of the TARGET dataset: color "
                         "(vcf/colorswap), noun (coco-cf), count (counting) -- "
                         "the CROSS-TASK axis for single-task routers")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--question_style", default=None)
    ap.add_argument("--causal", type=int, default=0,
                    help="also run the FLIP/STAY battery: re-probe each item with "
                         "the twin image (image swap) and twin caption (caption "
                         "swap) so the clevr3->ladder transfer claim becomes CAUSAL "
                         "rather than routing-only. flip = correct in base AND "
                         "tracks the swap; chance .25")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    args = cfg.get("args", {})
    mode = a.mode or args.get("factor_mode", "factored")
    qstyle = a.question_style or args.get("question_style", "explicit")
    n_tokens = args.get("n_tokens", 5)

    model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",))
             if s.cond == "conflict"]

    look_hits, read_hits, n_seen = [], [], 0
    lf, ls, rf, rs = [], [], [], []          # causal: look flip/stay, read flip/stay
    is_vcf = "vcf" in a.data_dir
    for sp in tqdm(specs, desc=f"transfer({mode})"):
        ik, tk = norm(sp.image_key), norm(sp.text_key)
        if not ik or set(ik) != set(tk):
            continue
        objs = [o for o in ik if ik[o] != tk[o]]
        if not objs:
            continue
        if is_vcf and any(c in o.split() for o in objs for c in COLORS):
            continue                      # colour-in-name confound (eval_ladder rule)
        if n_seen >= a.max_n:
            break
        n_seen += 1
        # one probe per conflicting object (multi-object -> binding questions)
        obj = objs[n_seen % len(objs)]
        img = get_image(ds, sp)
        cap = (f"The answer to the question is {tk[obj]}." if a.caption_style == "clevr3"
               else sp.fed_caption)
        item = {"id": f"t{sp.ex_index}_{sp.img_idx}", "image": img,
                "skill": OP_BY_TASK[a.task],
                "question": Q_BY_TASK[a.task].format(obj=obj),
                "channels": {"look": ik[obj], "read": tk[obj], "ocr": tk[obj]},
                "caption": cap, "ocr_text": f"VALUE={tk[obj]}"}
        cands = sorted({ik[obj], tk[obj]})

        def pick(qs, it):
            um = [{"role": "user",
                   "content": clevr3_user_content(it, qs, n_tokens, mode, seed=0,
                                                  question_style=qstyle)}]
            sc = {v: loglik(model, proc, um, cand(v), "cuda") for v in cands}
            return max(sc, key=sc.get)

        for qs, want in (("look", ik[obj]), ("read", tk[obj])):
            (look_hits if qs == "look" else read_hits).append(int(pick(qs, item) == want))

        if a.causal:
            # exact twins: the OTHER image carries tk[obj]; the OTHER caption asserts ik[obj]
            imgp = ds[sp.ex_index][f"image_{3 - sp.img_idx}"]
            capp = (f"The answer to the question is {ik[obj]}." if a.caption_style == "clevr3"
                    else ds[sp.ex_index][f"caption_{sp.img_idx}"])
            it_iswap = {**item, "image": imgp}
            it_cswap = {**item, "caption": capp,
                        "channels": {**item["channels"], "read": ik[obj]},
                        "ocr_text": f"VALUE={ik[obj]}"}
            b_l = pick("look", item)
            lf.append(int(b_l == ik[obj] and pick("look", it_iswap) == tk[obj]))
            ls.append(int(b_l == pick("look", it_cswap)))
            b_r = pick("read", item)
            rf.append(int(b_r == tk[obj] and pick("read", it_cswap) == ik[obj]))
            rs.append(int(b_r == pick("read", it_iswap)))

    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    out = {"run": os.path.basename(a.run_dir), "mode": mode, "split": a.split,
           "step": a.step, "n": n_seen, "data": os.path.basename(a.data_dir),
           "caption_style": a.caption_style, "question_style": qstyle,
           "ctrl_look": m(look_hits), "ctrl_read": m(read_hits),
           **({"look_flip": m(lf), "look_stay": m(ls), "read_flip": m(rf),
               "read_stay": m(rs), "causal": 1} if a.causal else {})}
    print(json.dumps(out))
    outd = os.path.join(a.run_dir, "eval",
                        f"transfer_{os.path.basename(a.data_dir)}_{a.split}")
    os.makedirs(outd, exist_ok=True)
    fn = f"{a.task}_{a.caption_style}_step_{a.step}.json"
    out["task"] = a.task
    json.dump(out, open(os.path.join(outd, fn), "w"), indent=1)
    print(f"[saved] {outd}/{fn}")


if __name__ == "__main__":
    main()
