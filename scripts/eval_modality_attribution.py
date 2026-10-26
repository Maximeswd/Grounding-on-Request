#!/usr/bin/env python
"""MODALITY ATTRIBUTION: use the learned words as a READ-OUT on answers they did not produce.

The words are trained as a control interface, but they also define a cheap attributor. For any answer
string y in a context C that supplies several sources, score y once per request and take the largest:

    attribute(y) = argmax_s  log p(y | C, r_s)

The same string is scored under both requests, so the comparison is balanced by construction and the
attributor needs no threshold, no classifier and no access to whatever produced y. That is the useful
part: y can come from a different interface, a different prompt, or a different model entirely.

Two evaluations, both on the conflict items of a ladder benchmark:
  attested   y is one of the two attested values, so its provenance is known by construction. The
             attributor should send the image's value to <look> and the caption's value to <read>.
             Reported as accuracy over both answers of every item; chance .5.
  generated  y is FREE TEXT produced by another interface and banked by eval_genconsistency.py
             (--dump_gens). Its provenance is the candidate its text matched. The attributor never sees
             which interface produced it. Reported the same way, plus coverage.

Baselines use the identical procedure with the request carried differently: an English instruction, a
prompt tuning prefix, or nothing at all (the two "requests" are then the same string, so the margin is
identically zero and accuracy is chance -- a null that the battery should reproduce).

Usage:
  python scripts/eval_modality_attribution.py --run_dir runs/color_neo_ctrl_5tok_ctm_seed0 \
      --task color --data_dir data/color --split test_id [--gens runs/OTHER/eval/color_test_id/genconsistency_step_800.json]
Output: runs/<run>/eval/<ds>_<split>/attribution_step_<step><tag>.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger, token_strings
from scsrc import prompts as P, score as SC
from evaluate import apply_checkpoint

INSTR = {"look": "Answer using ONLY what you can see in the image; ignore the caption.",
         "read": "Answer using ONLY what the caption states; ignore the image."}


def norm(k):
    return {str(o).split("|")[0].strip().lower(): str(v).strip().lower() for o, v in (k or {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", required=True); ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800); ap.add_argument("--max_n", type=int, default=120)
    ap.add_argument("--frozen_model_id", default=None)
    ap.add_argument("--mode", default=None, help="None | instructed | noop -- how the request is carried")
    ap.add_argument("--gens", default="", help="a genconsistency_*.json with --dump_gens records, to "
                                               "attribute FREE TEXT produced by another interface")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if (a.mode or a.gens) and not a.tag:
        raise SystemExit("[STOP] --mode/--gens need --tag so the plain run's file is not overwritten")
    if a.frozen_model_id is not None:
        from scsrc import model as M
        model, proc = M.load(model_id=None if a.frozen_model_id == "base" else a.frozen_model_id)
        model.eval(); n, method = 0, "frozen"
    else:
        model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
        n, method = meta["n_tokens"], meta["method"]
    if a.mode in ("instructed", "noop"):
        method = "frozen" if a.mode == "instructed" else "noop"

    def user(op, cap):
        if method == "neo":
            return P.render_ctrl(P.CTRL_CANONICAL, "".join(token_strings(f"{op}_ctrl", n)), cap)
        if method == "pt":
            return trigger(f"pt_{op}", n) + " " + P.pt_template(0).format(cap=cap)
        if method == "noop":
            return f'Caption: "{cap}".'          # identical for both ops -> margin 0, accuracy chance
        return f'Caption: "{cap}". {INSTR[op]}'

    gens = {}
    if a.gens:
        for r in (json.load(open(a.gens)).get("generations") or []):
            if r.get("ctx") == "base" and r.get("pick") in ("i", "t"):
                gens.setdefault((r["item"], r["op"]), r)

    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",)) if s.cond == "conflict"]
    res = {"attested": [], "generated": []}
    seen = 0
    for sp in tqdm(specs, desc="attribution"):
        if seen >= a.max_n:
            break
        ik, tk = norm(sp.image_key), norm(sp.text_key)
        conf = [o for o in ik if o in tk and ik[o] != tk[o]]
        if not conf:
            continue
        obj = conf[seen % len(conf)]
        v_i, v_t = ik[obj], tk[obj]
        img, cap = get_image(ds, sp), sp.fed_caption

        def attribute(text):
            """The source whose request assigns this exact string the higher likelihood."""
            ll = {op: SC.loglik(model, proc, img, user(op, cap), text, "cuda")[0] for op in ("look", "read")}
            return ("look" if ll["look"] > ll["read"] else "read"), round(float(ll["look"] - ll["read"]), 4)

        for v, truth in ((v_i, "look"), (v_t, "read")):
            got, margin = attribute(P.fmt_key_task({obj: v}, a.task))
            res["attested"].append({"truth": truth, "got": got, "margin": margin})
        for op in ("look", "read"):
            g = gens.get((seen, op))
            if g and g.get("text"):
                truth = "look" if g["pick"] == "i" else "read"
                got, margin = attribute(g["text"][:200])
                res["generated"].append({"truth": truth, "got": got, "margin": margin, "producer_op": op})
        seen += 1

    acc = lambda rs: round(sum(r["got"] == r["truth"] for r in rs) / len(rs), 3) if rs else None
    out = {"run": os.path.basename(a.run_dir), "data_dir": a.data_dir, "split": a.split, "step": a.step,
           "method": method, "mode": a.mode, "n_items": seen, "chance": 0.5,
           "attested_acc": acc(res["attested"]), "n_attested": len(res["attested"]),
           "generated_acc": acc(res["generated"]), "n_generated": len(res["generated"]),
           "gens_source": os.path.basename(os.path.dirname(os.path.dirname(a.gens))) if a.gens else None,
           "note": __doc__.strip().split("\n\n")[1].replace("\n", " "),
           "records": res}
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=1))
    od = os.path.join(a.run_dir, "eval", f"{os.path.basename(a.data_dir)}_{a.split}")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"attribution_step_{a.step}{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
