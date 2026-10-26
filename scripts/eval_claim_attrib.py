#!/usr/bin/env python
"""Claim -> channel ATTRIBUTION on real video (MultiVENT-Raw), against gold.

The citation question, run in the attribution direction: given a claim that is
already written (MVR's human-attested gold claims, each labelled with the
channel(s) that attest it), which source does it come from? The channel words
answer by intervention: V(claim, ch) = m(claim | C) - m(claim | C \\ ch) under
channel word ch; predict argmax over the channels present in the chunk.

This is the working cousin of the (chance-level) truth verifier: the words
cannot decide whether a claim is TRUE, but source addressability is exactly
the machinery for deciding WHERE a claim comes from. Metrics are computed on
the single-attested subset (clean gold) and on all claims (hit = prediction
in the attested set), with per-channel precision/recall/F1. All per-claim
records are banked; --reparse recomputes metrics offline.

Modes:
  words/instr  (GPU)  V under each channel word / instruction phrasing
  lexical      (CPU)  token-overlap baseline: claim vs ocr_strings/asr_text;
                      VIS when neither overlaps

Usage:
  python scripts/eval_claim_attrib.py --run_dir runs/X --split test_ood
  python scripts/eval_claim_attrib.py --frozen_model_id base --run_dir runs/F
  python scripts/eval_claim_attrib.py --mode lexical --run_dir out/lexical
"""
import argparse, collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CH_KEY = {"ASR": "asr_text", "OCR": "ocr_strings", "VIS": "frames"}
EMPTY = {"asr_text": "", "ocr_strings": [], "frames": []}
CHS = ("VIS", "OCR", "ASR")


def _claims(it):
    cs = it.get("claims") or []
    if isinstance(cs, str):
        try:
            cs = json.loads(cs.replace("'", '"'))
        except Exception:
            cs = [cs]
    return [str(c).strip() for c in cs if str(c).strip()]


def present(it):
    p = set()
    if (it.get("asr_text") or "").strip():
        p.add("ASR")
    if it.get("ocr_strings"):
        p.add("OCR")
    if it.get("frames") or it.get("vis_frame"):
        p.add("VIS")
    return p


def attested(it):
    return {str(c).upper() for c in
            (it.get("attested_channels") or [it.get("channel")]) if c}


def lexical_pred(it, claim):
    tok = lambda s: set(re.findall(r"[a-z0-9]+", str(s).lower()))
    ct = tok(claim)
    if not ct:
        return "VIS"
    ocr = tok(" ".join(map(str, it.get("ocr_strings") or [])))
    asr = tok(it.get("asr_text") or "")
    so, sa = len(ct & ocr) / len(ct), len(ct & asr) / len(ct)
    if max(so, sa) <= 0.15:
        return "VIS"
    return "OCR" if so >= sa else "ASR"


def score_records(records):
    """Aggregate banked per-claim records into attribution metrics."""
    hit_all, single = [], []
    conf = collections.defaultdict(lambda: collections.Counter())
    for r in records:
        att, pred = set(r["attested"]), r["pred"]
        if pred is None:
            continue
        hit_all.append(int(pred in att))
        if len(att) == 1:
            gold = next(iter(att))
            single.append((gold, pred))
            conf[gold][pred] += 1
    out = {"n_claims": len(hit_all),
           "hit_rate_all": round(sum(hit_all) / max(1, len(hit_all)), 3),
           "n_single": len(single),
           "acc_single": round(sum(g == p for g, p in single) /
                               max(1, len(single)), 3)}
    f1s = []
    for ch in CHS:
        tp = conf[ch][ch]
        fp = sum(conf[g][ch] for g in CHS if g != ch)
        fn = sum(v for p, v in conf[ch].items() if p != ch)
        pr = tp / max(1, tp + fp)
        rc = tp / max(1, tp + fn)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        out[f"{ch}_P"], out[f"{ch}_R"], out[f"{ch}_F1"] = \
            round(pr, 3), round(rc, 3), round(f1, 3)
        out[f"n_{ch}_gold"] = tp + fn
        if tp + fn:
            f1s.append(f1)
    out["macro_F1_single"] = round(sum(f1s) / max(1, len(f1s)), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="run to load (words) or output location (lexical/frozen)")
    ap.add_argument("--data_dir", default="data/multivent_raw_clean")
    ap.add_argument("--split", default="test_ood")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--n_frames", type=int, default=2)
    ap.add_argument("--max_claims", type=int, default=800)
    ap.add_argument("--claims_per_item", type=int, default=6)
    ap.add_argument("--mode", default="words", choices=("words", "lexical"))
    ap.add_argument("--style", default=None)
    ap.add_argument("--frozen_model_id", default=None)
    ap.add_argument("--reparse", default=None)
    a = ap.parse_args()

    if a.reparse:
        d = json.load(open(a.reparse))
        d.update(score_records(d["records"]))
        json.dump(d, open(a.reparse, "w"), indent=1)
        print(json.dumps({k: v for k, v in d.items() if k != "records"}))
        return

    rows = [json.loads(l)
            for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]

    records, n_done = [], 0
    if a.mode == "lexical":
        style = "lexical"
        for it in rows:
            pr = present(it)
            if len(pr) < 2:
                continue
            for claim in _claims(it)[:a.claims_per_item]:
                if n_done >= a.max_claims:
                    break
                n_done += 1
                records.append({"chunk_id": it["chunk_id"], "claim": claim,
                                "attested": sorted(attested(it)),
                                "present": sorted(pr), "V": None,
                                "pred": lexical_pred(it, claim)})
    else:
        from tqdm import tqdm
        from scsrc.mvraw import mvraw_user_content, _trig, _seq_mvg
        from scsrc.preference import seq_logprob
        from evaluate import apply_checkpoint
        if a.frozen_model_id is not None:
            from scsrc import model as M
            model, proc = M.load(model_id=(a.frozen_model_id or None)
                                 if a.frozen_model_id != "base" else None)
            model.eval(); n = 0
            style = a.style or "instr"
        else:
            model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
            n = meta["n_tokens"]
            style = a.style or "neo"

        def score(it, claim, ch, override):
            ct = mvraw_user_content(it, _trig(ch.lower(), n, style), override,
                                    a.n_frames, style, ch.lower())
            ex = _seq_mvg(proc, ct, claim, "cuda")
            return float(seq_logprob(model, ex, no_grad=True))

        def V(it, claim, ch):
            return (score(it, claim, ch, None) -
                    score(it, claim, ch, {CH_KEY[ch]: EMPTY[CH_KEY[ch]]}))

        for it in tqdm(rows, desc=f"attrib({os.path.basename(a.run_dir)})"):
            pr = present(it)
            if len(pr) < 2:
                continue
            for claim in _claims(it)[:a.claims_per_item]:
                if n_done >= a.max_claims:
                    break
                n_done += 1
                vs = {ch: V(it, claim, ch) for ch in sorted(pr)}
                records.append({"chunk_id": it["chunk_id"], "claim": claim,
                                "attested": sorted(attested(it)),
                                "present": sorted(pr),
                                "V": {k: round(v, 4) for k, v in vs.items()},
                                "pred": max(vs, key=vs.get)})

    out = {"run": os.path.basename(a.run_dir), "split": a.split,
           "step": a.step, "mode": a.mode, "style": style,
           "note": ("gold = attested_channels; prediction = argmax over "
                    "present channels of V(claim,ch)=m(claim|C)-m(claim|C-ch) "
                    "under channel word ch (words/instr) or token overlap "
                    "(lexical). Items with <2 present channels skipped."),
           "records": records}
    out.update(score_records(records))
    print(json.dumps({k: v for k, v in out.items() if k != "records"}))
    outd = os.path.join(a.run_dir, "eval", f"mvraw_{a.split}")
    os.makedirs(outd, exist_ok=True)
    tag = {"words": "", "lexical": "_lexical"}[a.mode]
    if a.frozen_model_id is not None:
        tag = "_frozen"
    json.dump(out, open(os.path.join(
        outd, f"attrib_step_{a.step}{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
