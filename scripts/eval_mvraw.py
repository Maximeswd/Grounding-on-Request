#!/usr/bin/env python
"""multivent-raw channel-word eval: routing, leakage, flip/stay, absence.

  routing   Generate under the item's OWN word; parse lines; set-level P/R/F1
            against the chunk's own-channel claims. Matching is exact after
            normalization, plus a soft token-F1 >= phi fallback (annotation
            sentences are long; verbatim-only would under-credit).
  leakage   Fraction of generated lines that match a DIFFERENT channel's
            attested claims for the same chunk (cross-attested lines excluded
            -- those are legitimately shared). The direct "did the word stay
            in its lane" number.
  flip/stay LL forced-choice on the evidence-swap twin pairs (pairs.jsonl):
            score LL(gold_a) vs LL(gold_b) under ctx_a and under ctx_b (only
            this channel's content swapped). flip = preferred gold tracks the
            swap (chance .25 for a context-blind scorer, same convention as
            flip_ll). stay = a DIFFERENT channel's word keeps its own
            preference when this channel's content is swapped.
  absence   cond=absent items: does generation produce the absence string?
  EN/xx     every ASR metric also reported for EN vs non-EN transcripts
            (cross-lingual grounding is a headline breakdown, not a filter).

Output: runs/<run>/eval/mvraw_<split>/summary_step_<STEP>.json
"""
import argparse, collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from scsrc.mvraw import mvraw_user_content, _target, _override_for, ABSENT, _trig
from scsrc.mvg import _seq_mvg
from scsrc.tokens import trigger
from evaluate import apply_checkpoint


@torch.no_grad()
def gen_blocks(model, proc, content, device, max_new=200):
    """Free generation over prebuilt multi-block content (scsrc.infer.generate
    only takes a single image + text). enable_thinking=False for Qwen3.5."""
    umsg = [{"role": "user", "content": content}]
    try:
        enc = proc.apply_chat_template(umsg, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt",
                                       enable_thinking=False).to(device)
    except (TypeError, ValueError):
        enc = proc.apply_chat_template(umsg, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt").to(device)
    kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    out = model.generate(**kw, max_new_tokens=max_new, do_sample=False)
    return proc.decode(out[0, enc["input_ids"].shape[1]:],
                       skip_special_tokens=True)

CHS = ("VIS", "OCR", "ASR")


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.casefold())).strip()


def _tokf1(a, b):
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    ov = len(ta & tb)
    return 2 * ov / (len(ta) + len(tb))


def match(line, golds, phi):
    nl = _norm(line)
    for g in golds:
        if nl == _norm(g) or _tokf1(line, g) >= phi:
            return g
    return None


@torch.no_grad()
def ll_of(model, proc, item, cont, device, n, nf, override=None, style="neo"):
    ch = item["channel"].lower()
    ct = mvraw_user_content(item, _trig(ch, n, style), override, nf, style, ch)
    ex = _seq_mvg(proc, ct, cont, device)
    from scsrc.preference import seq_logprob
    return float(seq_logprob(model, ex, no_grad=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/multivent_raw")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--phi", type=float, default=0.6)
    ap.add_argument("--max_n", type=int, default=150)
    ap.add_argument("--max_pairs", type=int, default=60)
    ap.add_argument("--frozen_model_id", default=None,
                    help="skip the checkpoint entirely and evaluate the BASE model "
                         "('base' = config default id). Pair with --style instr for "
                         "the user-instruction baseline or --style noop for the "
                         "no-op floor.")
    ap.add_argument("--style", default=None,
                    choices=[None, "neo", "pt", "instr", "noop"],
                    help="carrier override; default follows the checkpoint's method "
                         "(neo->neo, pt->pt, lora->instr)")
    ap.add_argument("--coverage", default="all", choices=["all", "full", "sub"],
                    help="VIS items by frame coverage: full = chunk fits in "
                         "--n_frames (clean supervision), sub = chunk was "
                         "subsampled (evidence may fall between shown frames)")
    ap.add_argument("--ops", default=os.environ.get("SC_MVR_OPS", ""),
                    help="comma list of channels to score (STATUS v1.102: the paper requests VIS and ASR only); empty = every channel in the split")
    ap.add_argument("--tag", default="", help="suffix for the output filename so a restricted run never overwrites the full one")
    a = ap.parse_args()
    dev = "cuda"
    if a.frozen_model_id:
        from scsrc import model as M
        model, proc = M.load(model_id=None if a.frozen_model_id == "base"
                             else a.frozen_model_id)
        model.eval()
        n, method = 0, "frozen"
    else:
        model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, dev)
        n, method = meta["n_tokens"], meta["method"]
    style = a.style or {"neo": "neo", "pt": "pt", "lora": "instr",
                        "frozen": "instr"}[method]
    print(f"[eval] method={method} style={style} n_tokens={n} coverage={a.coverage}")

    items = [json.loads(l) for l in open(f"{a.data_dir}/{a.split}.jsonl")]
    if a.coverage != "all":
        want_full = a.coverage == "full"
        items = [it for it in items if it["channel"] != "VIS"
                 or ((len(it["frames"]) <= a.n_frames) == want_full)]
    if a.ops:
        keep = {o.strip().upper() for o in a.ops.split(",") if o.strip()}
        before = len(items); items = [it for it in items if str(it["channel"]).upper() in keep]
        print(f"[eval] channels restricted to {sorted(keep)}: {len(items)} of {before} items (STATUS v1.102)")
    ix = {(it["chunk_id"], it["channel"]): it for it in items}
    att = json.load(open(f"{a.data_dir}/attestation.json"))

    # ---------- routing + leakage + absence ----------
    R = collections.defaultdict(list)
    gen_items = [it for it in items if it["cond"] == "attested"][:a.max_n]
    abs_items = [it for it in items if it["cond"] == "absent"][:50]
    for it in tqdm(gen_items + abs_items, desc="routing"):
        ch_l = it["channel"].lower()
        ct = mvraw_user_content(it, _trig(ch_l, n, style), None, a.n_frames,
                                style, ch_l)
        txt = gen_blocks(model, proc, ct, dev, max_new=200)
        lines = [l.strip("-* \t") for l in txt.strip().split("\n") if l.strip()]
        ch = it["channel"]
        if it["cond"] == "absent":
            R["absence_honest"].append(int(_tokf1(txt, ABSENT) >= 0.6))
            continue
        golds = it["claims"]
        own_hits = {match(l, golds, a.phi) for l in lines} - {None}
        p = len([l for l in lines if match(l, golds, a.phi)]) / max(len(lines), 1)
        r = len(own_hits) / max(len(golds), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        R[f"{ch}_f1"].append(f1)
        # leakage: other channels' EXCLUSIVE claims for this chunk
        other = [t for t, ws in att.get(it["chunk_id"], {}).items()
                 if ch not in ws]
        if other:
            leak = len([l for l in lines if match(l, other, a.phi)]) / max(len(lines), 1)
            R[f"{ch}_leak"].append(leak)
        if it.get("asr_lang") and ch == "ASR":
            key = "ASR_f1_en" if it["asr_lang"] == "en" else "ASR_f1_xx"
            R[key].append(f1)

    # ---------- flip / stay on evidence-swap pairs ----------
    pf = f"{a.data_dir}/pairs.jsonl"
    pairs = [json.loads(l) for l in open(pf)] if os.path.exists(pf) else []
    pairs = [p for p in pairs
             if (p["chunk_a"], p["channel"]) in ix and (p["chunk_b"], p["channel"]) in ix]
    per_t = collections.Counter()
    for p in tqdm(pairs, desc="flip"):
        if per_t[p["channel"]] >= a.max_pairs:
            continue
        per_t[p["channel"]] += 1
        A, B = ix[(p["chunk_a"], p["channel"])], ix[(p["chunk_b"], p["channel"])]
        ga, gb = _target(A["claims"]), _target(B["claims"])
        ov_b = _override_for(p["channel"], B)
        la_a = ll_of(model, proc, A, ga, dev, n, a.n_frames, None, style)
        lb_a = ll_of(model, proc, A, gb, dev, n, a.n_frames, None, style)
        la_b = ll_of(model, proc, A, ga, dev, n, a.n_frames, ov_b, style)
        lb_b = ll_of(model, proc, A, gb, dev, n, a.n_frames, ov_b, style)
        flip = int(la_a > lb_a and lb_b > la_b)
        R[f"{p['channel']}_flip{'_hard' if p.get('hard') else '_easy'}"].append(flip)
        # stay: another channel's word, same swap, own preference unmoved
        for o in CHS:
            if o == p["channel"]:
                continue
            O = ix.get((p["chunk_a"], o))
            if O is None or O["cond"] != "attested":
                continue
            go = _target(O["claims"])
            foil = _target(B["claims"])
            s1 = ll_of(model, proc, O, go, dev, n, a.n_frames, None, style) > \
                 ll_of(model, proc, O, foil, dev, n, a.n_frames, None, style)
            s2 = ll_of(model, proc, O, go, dev, n, a.n_frames, ov_b, style) > \
                 ll_of(model, proc, O, foil, dev, n, a.n_frames, ov_b, style)
            R[f"{o}_stay_vs_{p['channel']}swap"].append(int(s1 and s2))
            break

    m = lambda v: round(sum(v) / len(v), 3) if v else None
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "method": method, "style": style, "coverage": a.coverage,
           "phi": a.phi, "n_gen": len(gen_items),
           **{k: m(v) for k, v in sorted(R.items())},
           **{f"n_{k}": len(v) for k, v in sorted(R.items())}}
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("n_")}, indent=1))
    od = os.path.join(a.run_dir, "eval", f"mvraw_{a.split}")
    os.makedirs(od, exist_ok=True)
    tag = f"summary_step_{a.step}"
    if a.frozen_model_id:
        tag = f"summary_{style}_frozen"
    if a.coverage != "all":
        tag += f"_{a.coverage}"
    out["channels"] = sorted({str(it["channel"]).upper() for it in items})
    _fn = f"{tag}{a.tag}"
    json.dump(out, open(os.path.join(od, f"{_fn}.json"), "w"), indent=1)
    print(f"[saved] {od}/{_fn}.json")


if __name__ == "__main__":
    main()
