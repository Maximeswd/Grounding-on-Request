#!/usr/bin/env python
"""CiteFlip/CiteStay: claim-level citation faithfulness, causal not correlational.

Existing attribution evaluation asks whether the cited source SUPPORTS the
claim. This battery asks the stronger causal question, per emitted claim:

  support    does the claim's value match the cited source's presented value?
  CiteFlip   substitute the CITED source's twin: does the claim track it?
  CiteStay   substitute an UNCITED source's twin: does the claim hold still?

Two attribution conditions are scored on the SAME generations:
  emitted    the label the model itself emitted (provenance words / blocks) --
             only defined for provenance-trained runs;
  posthoc    support-based assignment (the claim is attributed to whichever
             source's value it matches), i.e. the standard post-hoc rule.
For a frozen/instruction run only `posthoc` exists: that is the baseline the
words are compared against.

Every raw generation is stored in the output JSON, so the parser can be
re-run offline (--reparse) without touching a GPU.

Usage:
  python scripts/eval_citefaith.py --run_dir runs/X --data_dir data/color
  python scripts/eval_citefaith.py --run_dir runs/_frozen_g3 --data_dir \
         data/color --frozen_model_id base
  python scripts/eval_citefaith.py --reparse runs/X/eval/color_test_id/citefaith_step_800.json
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLORS = {"red","orange","yellow","green","blue","purple","pink","brown","black",
          "white","gray","grey","gold","silver","tan","beige","violet","cyan",
          "maroon","navy"}
INSTR = {"look": "Answer using ONLY what you can see in the image; ignore the caption.",
         "read": "Answer using ONLY what the caption states; ignore the image."}
OPS = ("look", "read")
# neologism markers decode either with row suffixes (<look_gen_0>) or, for the
# 1-token cite delimiters of the blocks runs, bare (<look>); match both, and
# only op-named tokens so <end_of_turn>/<eos> never count as markers.
NEO_RE = re.compile(r"<(?:look|read|ocr|asr|compare)(?:_[a-z0-9]+)*(?:_\d+)?>")


def norm(k):
    return {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
            for o, v in (k or {}).items()}


def cut_eos(text):
    for cut in ("<end_of_turn>", "<eos>", "<|im_end|>", "</s>"):
        text = text.split(cut)[0]
    return text


def find_val(text, v):
    t = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    return t.rfind(" " + re.sub(r"[^a-z0-9 ]", " ", str(v).lower().strip()) + " ")


def parse_val(text, v_i, v_t, obj=None):
    """i / t / None. Object-aware first: multi-object answers ('horse: brown,
    cat: black') bind each value to its object, so when the conflict object is
    known we read the value attached to IT ('obj: value' or 'obj is value').
    Falls back to last mention (models restate the conflict, then conclude)."""
    if obj:
        low = re.sub(r"[^a-z0-9 :]", " ", text.lower())
        hits = [(m.start(), m.group(1)) for m in re.finditer(
            re.escape(str(obj).lower().strip()) +
            r"\s*(?::|-|\bis\b|\bare\b|\blooks?\b)?\s+([a-z]+)", low)]
        vals = [v for _, v in hits if v in (v_i, v_t)]
        if vals:
            return "i" if vals[-1] == v_i else "t"
    pi, pt = find_val(text, v_i), find_val(text, v_t)
    if pi < 0 and pt < 0:
        return None
    if pi >= 0 and pt >= 0:
        return "i" if pi > pt else "t"
    return "i" if pi >= 0 else "t"


def attributed_span(text, op):
    """The span the model itself attributes to `op`: text following a
    neologism token whose name contains the op (cite/gen words decode as
    literal '<look_cite_0>'-style strings), up to the next neologism token.
    Falls back to None when the run emits no such marker."""
    spans, pos = [], 0
    toks = [(m.start(), m.end(), m.group(0)) for m in NEO_RE.finditer(text)]
    for i, (s, e, tok) in enumerate(toks):
        if op in tok:
            nxt = toks[i + 1][0] if i + 1 < len(toks) else len(text)
            spans.append(text[e:nxt])
    return " ".join(spans) if spans else None


def claim_of(text, op, v_i, v_t, obj=None):
    """(value, emitted?) for the claim attributed to `op`: the marked span if
    the model emitted one, else the whole output (posthoc-only parse)."""
    text = cut_eos(text)
    sp = attributed_span(text, op)
    if sp is not None:
        v = parse_val(sp, v_i, v_t, obj)
        if v is not None:
            return v, True
    return parse_val(text, v_i, v_t, obj), False


def score_records(records):
    """Aggregate per-item raw generations into the battery metrics."""
    agg = {c: {k: [] for k in
               ("support", "citeflip", "citestay", "faithful",
                "citeflip_sup", "citestay_sup", "faithful_sup")}
           for c in ("emitted", "posthoc")}
    n_marked = 0
    for r in records:
        v_i, v_t = r["v_i"], r["v_t"]
        gold = {"look": "i", "read": "t"}          # source -> its value id
        twin_of = {"look": "img_twin", "read": "cap_twin"}
        other = {"look": "read", "read": "look"}
        for op in OPS:
            base = r["gen"].get(f"{op}|orig")
            if base is None:
                continue
            v0, emitted = claim_of(base, op, v_i, v_t, r.get("obj"))
            if emitted:
                n_marked += 1
            if v0 is None:
                continue
            # regenerations under the two twins, same request
            t_cited = r["gen"].get(f"{op}|{twin_of[op]}")
            t_uncited = r["gen"].get(f"{op}|{twin_of[other[op]]}")
            conds = [("posthoc",
                      "look" if v0 == "i" else "read")]  # support-based label
            if emitted:
                conds.append(("emitted", op))
            for cond, label in conds:
                sup = int(v0 == gold[label])
                agg[cond]["support"].append(sup)
                # causal events are defined w.r.t. the LABEL's twin
                tc = r["gen"].get(f"{op}|{twin_of[label]}")
                tu = r["gen"].get(f"{op}|{twin_of[other[label]]}")
                fl = st = None
                if tc is not None:
                    vc, _ = claim_of(tc, op, v_i, v_t, r.get("obj"))
                    # the twin swaps the label source's value to the other id
                    fl = int(vc is not None and vc != v0)
                    agg[cond]["citeflip"].append(fl)
                    if sup:
                        agg[cond]["citeflip_sup"].append(fl)
                if tu is not None:
                    vu, _ = claim_of(tu, op, v_i, v_t, r.get("obj"))
                    st = int(vu == v0)
                    agg[cond]["citestay"].append(st)
                    if sup:
                        agg[cond]["citestay_sup"].append(st)
                if fl is not None and st is not None:
                    agg[cond]["faithful"].append(int(fl and st))
                    if sup:
                        agg[cond]["faithful_sup"].append(int(fl and st))
    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    out = {}
    for c, d in agg.items():
        for k, v in d.items():
            out[f"{c}_{k}"] = m(v)
            out[f"n_{c}_{k}"] = len(v)
    out["n_marked_claims"] = n_marked
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir")
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--frozen_model_id", default=None)
    ap.add_argument("--reparse", default=None,
                    help="path to an existing citefaith json: recompute the "
                         "metrics from its banked generations, no GPU")
    a = ap.parse_args()

    if a.reparse:
        d = json.load(open(a.reparse))
        d.update(score_records(d["records"]))
        json.dump(d, open(a.reparse, "w"), indent=1)
        print(json.dumps({k: v for k, v in d.items() if k != "records"}))
        return

    from tqdm import tqdm
    from scsrc.data import load_task, build_specs, get_image
    from scsrc.tokens import token_strings, trigger
    from scsrc import prompts as P
    from scsrc.infer import generate
    from evaluate import apply_checkpoint

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

    def prompt(op, cap):
        if method == "neo":
            toks = "".join(token_strings(f"{op}_ctrl", n))
            return P.render_ctrl(P.CTRL_CANONICAL, toks, cap)
        return f'Caption: "{cap}". {INSTR[op]}'

    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",))
             if s.cond == "conflict"]
    is_vcf = "vcf" in a.data_dir
    records, n_seen = [], 0
    for sp in tqdm(specs, desc=f"citefaith({os.path.basename(a.run_dir or 'frozen')})"):
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
        gen = {}
        for op in OPS:
            for ctx, image, cap in (("orig", img, sp.fed_caption),
                                    ("img_twin", imgp, sp.fed_caption),
                                    ("cap_twin", img, capp)):
                gen[f"{op}|{ctx}"] = generate(
                    model, proc, image, prompt(op, cap), "cuda",
                    max_new_tokens=a.max_new_tokens, forbid_ids=forbid,
                    temperature=0.0)
        records.append({"i": sp.ex_index, "obj": obj, "v_i": v_i, "v_t": v_t,
                        "gen": gen})

    out = {"run": os.path.basename(a.run_dir or f"frozen_{a.frozen_model_id}"),
           "split": a.split, "step": a.step, "n": n_seen, "method": method,
           "note": ("support = claim value matches the labelled source's "
                    "presented value; citeflip = claim tracks the labelled "
                    "source's twin; citestay = claim survives the other "
                    "source's twin; faithful = both. 'emitted' scores the "
                    "model's own attribution markers; 'posthoc' scores "
                    "support-based assignment on the same generations."),
           "records": records}
    out.update(score_records(records))
    print(json.dumps({k: v for k, v in out.items() if k != "records"}))
    outd = os.path.join(a.run_dir or ".", "eval",
                        f"{os.path.basename(a.data_dir)}_{a.split}")
    os.makedirs(outd, exist_ok=True)
    tag = "_frozen" if a.frozen_model_id is not None else ""
    json.dump(out, open(os.path.join(
        outd, f"citefaith_step_{a.step}{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
