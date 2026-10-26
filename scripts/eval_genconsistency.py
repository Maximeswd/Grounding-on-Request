#!/usr/bin/env python
"""Generation-consistency audit: does free decoding agree with the LL argmax?

The headline flip/stay batteries score closed candidate sets by teacher-forced
log-likelihood (per-token mean; scsrc.score.loglik). That measures the model's
PREFERENCE structure under paired interventions -- deliberately not its free
production (paper SS2, 'Discrimination vs production'). This audit closes the
gap between the two layers with a measurement instead of an argument: on a
subsample of ladder conflict items, decode freely (greedy; emission of the new
word types masked exactly as at training time) and parse the emitted value,
then report

  agree_look/agree_read  P(parsed generated value == LL-argmax choice), on
                         mappable items -- THE number for the appendix table
  gen_look_acc/_flip     routing/flip recomputed on GENERATED answers
  gen_read_acc/_flip     (same twin contexts as eval_flip_ll)
  gen_look_stay/gen_read_stay
  unmappable             P(neither candidate value in the output)
  ambiguous              P(both values appear; last mention wins)

High agreement => the LL battery is a variance-reduced proxy for production
(one appendix table retires the objection). Low agreement anywhere is itself a
finding -- a preference/production dissociation of the compare-v1
'miscalibrated, not blind' kind. Either way, report it.

Usage:
  python scripts/eval_genconsistency.py --run_dir runs/X --data_dir data/color
  python scripts/eval_genconsistency.py --run_dir runs/X --data_dir data/vcf \
         --frozen_model_id base            # frozen + English instruction row
Writes eval/{ds}_{split}/genconsistency_step_{step}[ _frozen].json
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger, token_strings
from scsrc import prompts as P, score as SC
from scsrc.infer import generate
from evaluate import apply_checkpoint

COLORS = {"red","orange","yellow","green","blue","purple","pink","brown","black",
          "white","gray","grey","gold","silver","tan","beige","violet","cyan",
          "maroon","navy"}
INSTR = {"look": "Answer using ONLY what you can see in the image; ignore the caption.",
         "read": "Answer using ONLY what the caption states; ignore the image."}


def norm(k):
    return {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
            for o, v in (k or {}).items()}


def parse_val(text, v_i, v_t):
    """Map generated text to a candidate. Last mention wins on ambiguity
    (models restate the conflict, then conclude)."""
    for cut in ("<end_of_turn>", "<eos>", "<|im_end|>", "</s>"):
        text = text.split(cut)[0]
    t = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    def last(v):
        return t.rfind(" " + re.sub(r"[^a-z0-9 ]", " ", str(v).lower().strip()) + " ")
    pi, pt = last(v_i), last(v_t)
    if pi < 0 and pt < 0:
        return None, "unmappable"
    if pi >= 0 and pt >= 0:
        return ("i" if pi > pt else "t"), "ambiguous"
    return ("i" if pi >= 0 else "t"), "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=60)
    ap.add_argument("--tag", default="", help="v1.141: output suffix (e.g. _ho for SC_TEMPLATE=heldout)")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--dump_gens", type=int, default=1,
                    help="1: store the raw generations with their candidates and parse tag, so an "
                         "unmappable rate can be diagnosed instead of guessed (2026-09-15)")
    ap.add_argument("--save_images", default="",
                    help="2026-10-22: directory; saves the base and twin image of every dumped item as PNG and records the "
                         "file next to each generation (for the input/output examples document; local use only)")
    ap.add_argument("--frozen_model_id", default=None,
                    help="skip checkpoints: frozen model + English instruction "
                         "(--run_dir only names the output location; output "
                         "file gets a _frozen tag so nothing is clobbered)")
    a = ap.parse_args()

    forbid = None
    if a.frozen_model_id is not None:
        from scsrc import model as M
        model, proc = M.load(model_id=(a.frozen_model_id or None)
                             if a.frozen_model_id != "base" else None)
        model.eval()
        n, method = 0, "frozen"
    else:
        model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
        n, method = meta["n_tokens"], meta["method"]

    # v1.141 (author: exact matching everywhere): SC_TEMPLATE=heldout uses the SAME held-out request templates as the
    # headline ladder evaluator (eval_flip_ll: per-caption rotation over the eval-only templates), and a LoRA adapter
    # run gets the neutral prefix-free text it was trained on.
    from scripts.eval_flip_ll import _ctrl_tpl, _pt_tpl, _lora_instr
    _lora_ad = os.environ.get("SC_LORA_ADAPTER", "0") == "1"
    def prompt(op, cap):
        if method == "neo":
            toks = "".join(token_strings(f"{op}_ctrl", n))
            return P.render_ctrl(_ctrl_tpl(cap), toks, cap)
        if method == "pt":
            if os.environ.get("PT_INSERTED", "0") == "1":
                return P.render_ctrl(_ctrl_tpl(cap), trigger(f"pt_{op}", n), cap)
            return trigger(f"pt_{op}", n) + " " + _pt_tpl(cap).format(cap=cap)
        if method == "lora" and _lora_ad:
            return _pt_tpl(cap).format(cap=cap)
        return f'Caption: "{cap}". {_lora_instr(op, cap)}'

    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",))
             if s.cond == "conflict"]
    is_vcf = "vcf" in a.data_dir
    res = {k: [] for k in ("agree_look", "agree_read",
                           "gen_look_acc", "gen_read_acc",
                           "gen_look_flip", "gen_read_flip",
                           "gen_look_stay", "gen_read_stay",
                           "gens_look_acc", "gens_read_acc",
                           "gens_look_flip", "gens_read_flip",
                           "gens_look_stay", "gens_read_stay",
                           "ll_look_acc", "ll_read_acc",
                           "ll_look_flip", "ll_read_flip")}
    q = {"unmappable": 0, "ambiguous": 0, "contexts": 0}
    GENS = []
    n_seen = 0
    for sp in tqdm(specs, desc=f"gencons({os.path.basename(a.run_dir)})"):
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
        imgp = ds[sp.ex_index][f"image_{3 - sp.img_idx}"]     # twin image
        capp = ds[sp.ex_index][f"caption_{sp.img_idx}"]       # twin caption
        ci = P.fmt_key_task({obj: v_i}, a.task)
        ct = P.fmt_key_task({obj: v_t}, a.task)

        def ll_pick(op, image, cap):
            u = prompt(op, cap)
            la, _, _ = SC.loglik(model, proc, image, u, ci, "cuda")
            lb, _, _ = SC.loglik(model, proc, image, u, ct, "cuda")
            return "i" if la > lb else "t"

        def gen_pick(op, image, cap, ctx):
            u = prompt(op, cap)
            txt = generate(model, proc, image, u, "cuda",
                           max_new_tokens=a.max_new_tokens,
                           forbid_ids=forbid, temperature=0.0)
            v, tag = parse_val(txt, v_i, v_t)
            if a.dump_gens:
                # every field an offline re-scoring needs, so a scoring rule can change without a GPU
                rec = {"item": n_seen - 1, "op": op, "ctx": ctx, "v_image": v_i, "v_caption": v_t,
                       "tag": tag, "pick": v, "text": txt, "prompt": u}   # v1.141: full text + prompt
                if a.save_images:
                    # 2026-10-22: the scored candidates and the image file, so an example can be shown exactly
                    la, _, _ = SC.loglik(model, proc, image, u, ci, "cuda")
                    lb, _, _ = SC.loglik(model, proc, image, u, ct, "cuda")
                    rec["candidates"] = {"image value": ci, "caption value": ct}
                    rec["loglik"] = {"image value": float(la), "caption value": float(lb)}
                    _d = os.path.join(a.save_images, os.path.basename(a.run_dir.rstrip("/")))
                    os.makedirs(_d, exist_ok=True)
                    _f = os.path.join(_d, f"item{n_seen - 1}_{'twin' if ctx == 'imgswap' else 'base'}.png")
                    if not os.path.exists(_f):
                        image.save(_f)
                    rec["image_file"] = _f
                    rec["caption"] = cap
                GENS.append(rec)
            q["contexts"] += 1
            if tag == "unmappable":
                q["unmappable"] += 1
            elif tag == "ambiguous":
                q["ambiguous"] += 1
            return v

        # base contexts: agreement is measured here
        ll_li, ll_ri = ll_pick("look", img, sp.fed_caption), ll_pick("read", img, sp.fed_caption)
        g_li = gen_pick("look", img, sp.fed_caption, "base")
        g_ri = gen_pick("read", img, sp.fed_caption, "base")
        if g_li is not None:
            res["agree_look"].append(int(g_li == ll_li))
        if g_ri is not None:
            res["agree_read"].append(int(g_ri == ll_ri))
        # twin contexts: flip/stay on generations (and LL for the same subsample)
        g_lp = gen_pick("look", imgp, sp.fed_caption, "imgswap")   # image swap
        g_lc = gen_pick("look", img, capp, "capswap")              # caption swap
        g_rc = gen_pick("read", img, capp, "capswap")
        g_rp = gen_pick("read", imgp, sp.fed_caption, "imgswap")
        ll_lp = ll_pick("look", imgp, sp.fed_caption)
        ll_rc = ll_pick("read", img, capp)
        res["ll_look_acc"].append(int(ll_li == "i"))
        res["ll_read_acc"].append(int(ll_ri == "t"))
        res["ll_look_flip"].append(int(ll_li == "i" and ll_lp == "t"))
        res["ll_read_flip"].append(int(ll_ri == "t" and ll_rc == "i"))
        if g_li is not None:
            res["gen_look_acc"].append(int(g_li == "i"))
            if g_lp is not None:
                res["gen_look_flip"].append(int(g_li == "i" and g_lp == "t"))
            if g_lc is not None:
                res["gen_look_stay"].append(int(g_li == g_lc))
        if g_ri is not None:
            res["gen_read_acc"].append(int(g_ri == "t"))
            if g_rc is not None:
                res["gen_read_flip"].append(int(g_ri == "t" and g_rc == "i"))
            if g_rp is not None:
                res["gen_read_stay"].append(int(g_ri == g_rp))
        # STRICT scoring (2026-09-15): an answer that is neither candidate value does not follow the
        # requested source, so it counts as WRONG rather than being dropped. Complete denominator, no
        # judge, and conservative for every interface -- the trained words lose the most because their
        # answers are the least literal. Stay is the conjunctive form used everywhere else.
        res["gens_look_acc"].append(int(g_li == "i"))
        res["gens_read_acc"].append(int(g_ri == "t"))
        res["gens_look_flip"].append(int(g_li == "i" and g_lp == "t"))
        res["gens_read_flip"].append(int(g_ri == "t" and g_rc == "i"))
        res["gens_look_stay"].append(int(g_li == "i" and g_lc == "i"))
        res["gens_read_stay"].append(int(g_ri == "t" and g_rp == "t"))

    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "n": n_seen, "method": method,
           "unmappable_rate": round(q["unmappable"] / max(q["contexts"], 1), 3),
           **({"generations": GENS} if a.dump_gens else {}),
           "ambiguous_rate": round(q["ambiguous"] / max(q["contexts"], 1), 3),
           "n_contexts": q["contexts"],
           **{k: m(v) for k, v in res.items()},
           **{f"n_{k}": len(v) for k, v in res.items() if k.startswith(("agree", "gen"))}}
    print(json.dumps(out))
    outd = os.path.join(a.run_dir, "eval", f"{os.path.basename(a.data_dir)}_{a.split}")
    os.makedirs(outd, exist_ok=True)
    tag = ("_frozen" if a.frozen_model_id is not None else "") + a.tag
    json.dump(out, open(os.path.join(
        outd, f"genconsistency_step_{a.step}{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
