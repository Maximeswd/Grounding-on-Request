#!/usr/bin/env python
"""CLAIM VERIFIER: the words as a grounding auditor for ARBITRARY claims.

Downstream framing (the report-audit loop): a generated report is a list of
claims; the verifier scores each claim's grounding in each channel with NO
gold labels needed at use time:

    V(claim, ch) = m(claim | C, word_ch) - m(claim | C \\ ch, word_ch)

i.e. the per-claim necessity margin under the channel word -- the same
quantity the JOINT battery certifies in aggregate, applied as a classifier.
Both terms share the claim string and the request, so claim length, fluency,
language and register cancel; only the channel's presence varies.

Battery (scored against MVR's human attestation labels):
 1. GROUNDED vs NOT -- positives = the chunk's own attested claims;
    negatives = the topic/channel/language-matched MATE chunk's claims
    (pairs_topic.jsonl: the swap battery's honest negatives, so topic and
    register cannot leak). Reports AUROC and accuracy at eps = 0.5 nats.
 2. CHANNEL ATTRIBUTION -- for attested claims, does argmax_ch V(claim, ch)
    recover the attested channel (among channels present)?
 3. --audit_file chunk_id<TAB>CH<TAB>claim -- verify arbitrary claims
    (e.g. parsed from a generated report), write per-claim verdicts.

Usage:
  python scripts/eval_claim_verifier.py --run_dir runs/mvr_g_5tok_sft_seed0c2h2 --split test_ood
  python scripts/eval_claim_verifier.py --run_dir runs/X --frozen_model_id base --style instr ...
Output: runs/<run>/eval/mvraw_<split>/verifier_step_<STEP>.json
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from scsrc.mvraw import mvraw_user_content, _trig, _seq_mvg
from scsrc.preference import seq_logprob
from evaluate import apply_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def auroc(pos, neg):
    """Rank-based AUROC, no sklearn."""
    if not pos or not neg:
        return None
    ranked = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r, s = 0.0, 0
    for i, (v, lab) in enumerate(ranked, 1):
        if lab:
            r += i
            s += 1
    return round((r - s * (s + 1) / 2) / (s * (len(ranked) - s)), 4)


def present(it):
    p = set()
    if (it.get("asr_text") or "").strip():
        p.add("ASR")
    if it.get("ocr_strings"):
        p.add("OCR")
    if it.get("frames") or it.get("vis_frame"):
        p.add("VIS")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/multivent_raw_clean")
    ap.add_argument("--pairs", default="data/multivent_raw/pairs_topic.jsonl")
    ap.add_argument("--split", default="test_ood")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--n_frames", type=int, default=2)
    ap.add_argument("--max_n", type=int, default=60,
                    help="max (chunk, mate) pairs for the battery")
    ap.add_argument("--claims_per_side", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--style", default=None)
    ap.add_argument("--frozen_model_id", default=None)
    ap.add_argument("--audit_file", default=None,
                    help="TSV chunk_id<TAB>CH<TAB>claim: verify arbitrary "
                         "claims (e.g. a generated report) instead of the battery")
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

    rows = {json.loads(l)["chunk_id"]: json.loads(l)
            for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))}

    def score(it, claim, ch, override):
        ct = mvraw_user_content(it, _trig(ch.lower(), n, style), override,
                                a.n_frames, style, ch.lower())
        ex = _seq_mvg(proc, ct, claim, "cuda")
        return float(seq_logprob(model, ex, no_grad=True))

    def V(it, claim, ch):
        base = score(it, claim, ch, None)
        abl = score(it, claim, ch, {CH_KEY[ch]: EMPTY[CH_KEY[ch]]})
        return base - abl

    outd = os.path.join(a.run_dir, "eval", f"mvraw_{a.split}")
    os.makedirs(outd, exist_ok=True)

    # ---------------- audit mode: verify arbitrary claims ----------------
    if a.audit_file:
        verdicts = []
        for line in open(a.audit_file):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            cid, ch, claim = parts
            it = rows.get(cid)
            if not it or ch.upper() not in present(it):
                verdicts.append({"chunk_id": cid, "channel": ch, "claim": claim,
                                 "v_nats": None, "grounded": None,
                                 "note": "chunk/channel unavailable"})
                continue
            v = V(it, claim, ch.upper())
            verdicts.append({"chunk_id": cid, "channel": ch.upper(),
                             "claim": claim, "v_nats": round(v, 3),
                             "grounded": bool(v > a.eps)})
        out = {"run": os.path.basename(a.run_dir), "mode": "audit",
               "eps": a.eps, "n": len(verdicts),
               "grounded_frac": round(sum(1 for v in verdicts
                                          if v["grounded"]) /
                                      max(1, sum(1 for v in verdicts
                                                 if v["grounded"] is not None)), 3),
               "verdicts": verdicts}
        p = os.path.join(outd, f"verifier_audit_step_{a.step}.json")
        json.dump(out, open(p, "w"), indent=1)
        print(json.dumps({k: v for k, v in out.items() if k != "verdicts"}))
        print(f"[saved] {p}")
        return

    # ---------------- battery mode ----------------
    pairs = []
    for l in open(a.pairs):
        p = json.loads(l)
        A, B = p.get("a") or p.get("chunk_a"), p.get("b") or p.get("chunk_b")
        if A in rows and B in rows and _claims(rows[A]) and _claims(rows[B]):
            pairs.append((A, B))
    pairs = pairs[:a.max_n]

    pos, neg = [], []                      # V under the chunk's own channel word
    attr_ok, attr_n = 0, 0
    per_ch = collections.defaultdict(lambda: {"pos": [], "neg": []})
    for A, B in tqdm(pairs, desc="verifier"):
        it, mate = rows[A], rows[B]
        ch = str(it["channel"]).upper()
        if ch not in present(it):
            continue
        for claim in _claims(it)[:a.claims_per_side]:
            v = V(it, claim, ch)
            pos.append(v); per_ch[ch]["pos"].append(v)
            # attribution among present channels (single-attested items only)
            att = {str(c).upper() for c in (it.get("attested_channels") or [ch])}
            if len(att & present(it)) == 1:
                vs = {c: (v if c == ch else V(it, claim, c))
                      for c in present(it)}
                attr_n += 1
                attr_ok += int(max(vs, key=vs.get) in att)
        for claim in _claims(mate)[:a.claims_per_side]:
            v = V(it, claim, ch)           # mate's claim against THIS chunk
            neg.append(v); per_ch[ch]["neg"].append(v)

    m = lambda v: round(sum(v) / len(v), 3) if v else None
    acc = (round((sum(1 for v in pos if v > a.eps) +
                  sum(1 for v in neg if v <= a.eps)) /
                 max(1, len(pos) + len(neg)), 3) if pos or neg else None)
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "style": style, "mode": "battery", "eps": a.eps,
           "n_pos": len(pos), "n_neg": len(neg),
           "auroc": auroc(pos, neg), "acc_at_eps": acc,
           "pos_mean_nats": m(pos), "neg_mean_nats": m(neg),
           "attr_acc": (round(attr_ok / attr_n, 3) if attr_n else None),
           "n_attr": attr_n,
           "per_channel": {c: {"auroc": auroc(d["pos"], d["neg"]),
                               "n_pos": len(d["pos"]), "n_neg": len(d["neg"])}
                           for c, d in sorted(per_ch.items())},
           "note": ("V = m(claim|C) - m(claim|C minus ch) under the channel "
                    "word; negatives are topic/channel/language-matched mate "
                    "claims scored in the SAME context, so claim length and "
                    "register cancel in the comparison.")}
    p = os.path.join(outd, f"verifier_step_{a.step}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out))
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
