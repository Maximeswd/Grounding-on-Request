#!/usr/bin/env python
"""MVR causal eval: do(channel ABSENT) — the flip/stay analogue for a corpus
with no conflicting values.

WHY THIS SHAPE. The ladder and MVG use do(source VALUE = v'): two contexts that
differ only in what a source says, so the answer must FLIP. MVR has no such
twins — its three channels describe the same real video, so they do not
disagree, and a sibling-chunk swap is not available either (only 8 of 477
videos have >1 extracted chunk, and 0% of those pairs have disjoint claim
sets). What MVR has instead is something the mined-twin datasets do NOT: a
HUMAN label saying which channels attest each claim. That licenses a different
intervention with an exactly known ground-truth effect:

    do(channel c absent)  ->  if c attests the claim, the claim must become
                              underivable;  if c does NOT attest it, the claim
                              must survive.

That is the same 2x2 logic as flip/stay, on presence instead of value:

    NECESSITY  (the flip analogue): drop an ATTESTING channel
                 -> score of the gold claim must DROP.
    SPECIFICITY(the stay analogue): drop a NON-attesting channel
                 -> score of the gold claim must NOT drop.
    JOINT = min(necessity, specificity), same anti-constant-policy rule as
                 elsewhere: a model that always says "score dropped" wins
                 necessity and loses specificity, and vice versa.

Crucially the exclusivity that MVG had to INFER from annotations (and got wrong
in 72% of pairs) is here DIRECTLY ANNOTATED, so this eval cannot degenerate the
same way.

Metrics (per channel and pooled):
  nec_drop_pos   P(logprob falls when an attesting channel is removed)
  nec_drop_nats  mean nat drop, attesting removal
  spec_stay_pos  P(logprob does NOT fall materially when a non-attesting
                 channel is removed)  [material = > --eps nats]
  spec_drop_nats mean nat drop, non-attesting removal (should be ~0)
  joint          min(nec_drop_pos, spec_stay_pos)
  margin         nec_drop_nats - spec_drop_nats  (the causal contrast; >0 means
                 the word's answer depends on the RIGHT channel specifically)

Output: runs/<run>/eval/mvraw_<split>/ablate<tag>_step_<STEP>.json
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from scsrc.mvraw import mvraw_user_content, _trig, _seq_mvg
from scsrc.preference import seq_logprob
from evaluate import apply_checkpoint

CH_KEY = {"ASR": "asr_text", "OCR": "ocr_strings", "VIS": "frames"}
EMPTY = {"asr_text": "", "ocr_strings": [], "frames": []}


def _claims(it):
    cs = it.get("claims") or []
    if isinstance(cs, str):
        try:
            cs = json.loads(cs.replace("'", '"'))
        except Exception:
            cs = [cs]
    return [str(c).strip() for c in cs if str(c).strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/multivent_raw")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--max_n", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.5,
                    help="nats: a drop smaller than this counts as 'unchanged' "
                         "on the specificity side")
    ap.add_argument("--style", default=None,
                    choices=[None, "neo", "pt", "instr", "noop"])
    ap.add_argument("--frozen_model_id", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if a.frozen_model_id is not None:
        from scsrc import model as M
        model, proc = M.load(model_id=(a.frozen_model_id or None)
                             if a.frozen_model_id != "base" else None)
        model.eval(); n = 0
        style = a.style or "instr"
    else:
        model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
        n = meta["n_tokens"]
        try:
            method = json.load(open(os.path.join(a.run_dir, "config.json"))
                               )["args"]["method"]
        except Exception:
            method = "neo"
        style = a.style or {"neo": "neo", "pt": "pt",
                            "lora": "instr", "frozen": "instr"}.get(method, "neo")

    rows = [json.loads(l) for l in
            open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    rows = [r for r in rows if r.get("cond") == "attested"][:a.max_n]

    R = collections.defaultdict(list)
    per_ch = collections.defaultdict(lambda: collections.defaultdict(list))

    def score(it, override):
        ch = it["channel"].lower()
        ct = mvraw_user_content(it, _trig(ch, n, style), override,
                                a.n_frames, style, ch)
        ex = _seq_mvg(proc, ct, cont, "cuda")
        # scored-span token count (identical across ablations of one item, so
        # per-token drops = summed/ntok -- the printed answer to the LL
        # length-bias question; JOINT stays defined on summed nats as banked)
        start = int(ex.get("score_start", ex["cont_start"]))
        ntok = max(int(ex["input_ids"].shape[1]) - 1 - start, 1)
        return float(seq_logprob(model, ex, no_grad=True)), ntok

    for it in tqdm(rows, desc="mvr-ablate"):
        cl = _claims(it)
        if not cl:
            continue
        cont = "\n".join(cl)
        att = {str(c).upper() for c in (it.get("attested_channels") or [])}
        if not att:
            continue
        # which channels are actually PRESENT in this item's input
        present = set()
        if (it.get("asr_text") or "").strip():
            present.add("ASR")
        if it.get("ocr_strings"):
            present.add("OCR")
        if it.get("frames") or it.get("vis_frame"):
            present.add("VIS")
        base, ntok = score(it, None)
        # NECESSITY: remove an attesting channel -> score must fall
        for c in sorted(att & present):
            d = base - score(it, {CH_KEY[c]: EMPTY[CH_KEY[c]]})[0]
            R["nec_drop_nats"].append(d)
            R["nec_drop_pertok"].append(d / ntok)
            R["nec_drop_pos"].append(int(d > a.eps))
            per_ch[c]["nec_nats"].append(d); per_ch[c]["nec_pos"].append(int(d > a.eps))
        # SPECIFICITY: remove a NON-attesting channel -> score must hold
        for c in sorted(present - att):
            d = base - score(it, {CH_KEY[c]: EMPTY[CH_KEY[c]]})[0]
            R["spec_drop_nats"].append(d)
            R["spec_drop_pertok"].append(d / ntok)
            R["spec_stay_pos"].append(int(d <= a.eps))
            per_ch[c]["spec_nats"].append(d); per_ch[c]["spec_pos"].append(int(d <= a.eps))

    m = lambda v: round(sum(v) / len(v), 4) if v else None
    nec, spec = m(R["nec_drop_pos"]), m(R["spec_stay_pos"])
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "style": style, "eps_nats": a.eps, "n_items": len(rows),
           "n_nec": len(R["nec_drop_pos"]), "n_spec": len(R["spec_stay_pos"]),
           "nec_drop_pos": nec, "nec_drop_nats": m(R["nec_drop_nats"]),
           "nec_drop_pertok": m(R["nec_drop_pertok"]),
           "spec_stay_pos": spec, "spec_drop_nats": m(R["spec_drop_nats"]),
           "spec_drop_pertok": m(R["spec_drop_pertok"]),
           "joint": (round(min(nec, spec), 4) if (nec is not None and spec is not None)
                     else None),
           "margin_nats": (round(m(R["nec_drop_nats"]) - m(R["spec_drop_nats"]), 4)
                           if R["nec_drop_nats"] and R["spec_drop_nats"] else None),
           "per_channel": {c: {"nec_pos": m(d["nec_pos"]),
                               "nec_nats": m(d["nec_nats"]),
                               "spec_pos": m(d["spec_pos"]),
                               "spec_nats": m(d["spec_nats"])}
                           for c, d in sorted(per_ch.items())},
           "note": ("do(channel absent) using the HUMAN attestation labels. "
                    "necessity = attesting-channel removal must lower the gold "
                    "score; specificity = non-attesting removal must not. "
                    "JOINT = min(.,.) blocks the always-drop / never-drop "
                    "constant policies. Chance for each side is NOT .25 -- "
                    "report against the frozen/instr baseline run with the same "
                    "flags, which is why --style/--frozen_model_id exist.")}
    print(json.dumps(out, indent=1))
    od = os.path.join(a.run_dir, "eval", f"mvraw_{a.split}")
    os.makedirs(od, exist_ok=True)
    stem = f"ablate{a.tag}" + ("" if style == "neo" else f"_{style}")
    json.dump(out, open(os.path.join(od, f"{stem}_step_{a.step}.json"), "w"), indent=1)
    print(f"[saved] {od}/{stem}_step_{a.step}.json")


if __name__ == "__main__":
    main()
