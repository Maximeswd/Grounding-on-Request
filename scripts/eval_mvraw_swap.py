#!/usr/bin/env python
"""MultiVENT-Raw EVIDENCE SWAP — do(channel := another chunk's content).

WHY THIS AND NOT THE ABLATION. `eval_mvraw_ablate.py` REMOVES a channel, which
changes the prompt LENGTH as well as its content: deleting the visual channel
deletes ~256 image tokens. The measured consequence is unmistakable -- frozen
no-op shows spec_drop of 34.5 nats on ⟨vis⟩ (id) and 40.3 (ood), and instruction
54.7. No evidence effect is 50 nats; that is a length effect, and it is why
⟨vis⟩ specificity sits at .09-.27 for EVERY method including the frozen ones.

A SWAP keeps the token count identical -- same number of frames, same number of
OCR strings, a transcript of comparable length -- so content is isolated from
length. Mates come from pairs_topic.jsonl, matched on (topic, channel, language),
so the model also cannot win by noticing a topic or script change.

TWO METRICS, and the second is new for MVR:

  DROP (necessity / specificity), same shape as the ablation so the two are
  comparable:
      necessity   : channel ATTESTS the claim -> logprob should FALL on swap
      specificity : channel does NOT attest   -> logprob should HOLD
      JOINT = min(nec, spec)

  FLIP (a genuine counterfactual, which MVR has never had): a forced choice
  between chunk A's claim and chunk B's claim, scored under both contexts.
      flip = (A's claim wins under ctx_A) AND (B's claim wins under ctx_B)
  Chance is .25 for the conjunction, .5 per side -- stated, not assumed. This is
  the only MVR number that is comparable in KIND to the Ground-CF/MVG flip axis,
  because it is a change of evidence rather than its removal.

Usage
  python scripts/eval_mvraw_swap.py --run_dir runs/mvr_g_5tok_sft_seed0c2h2 \
      --split test_id --step 800 --n_frames 2
  python scripts/eval_mvraw_swap.py --run_dir runs/_frozen_gemma --style instr \
      --frozen_model_id base --split test_id
"""
import argparse, collections, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/multivent_raw_clean")
    ap.add_argument("--pairs", default="data/multivent_raw/pairs_topic.jsonl")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--n_frames", type=int, default=2)
    ap.add_argument("--max_n", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.5,
                    help="nats a drop must exceed to count (same as the ablate eval)")
    ap.add_argument("--style", default=None, choices=[None, "neo", "pt", "instr", "noop"])
    ap.add_argument("--frozen_model_id", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    from scsrc.mvraw import mvraw_user_content, _target, _trig, _seq_mvg, _override_for
    from scsrc.preference import seq_logprob

    items = {}
    for r in (json.loads(l) for l in
              open(os.path.join(ROOT, a.data_dir, f"{a.split}.jsonl"))):
        items[(r["chunk_id"], r["channel"])] = r
    # a mate is usable only if BOTH sides are in this split and both carry claims
    pairs = []
    for p in (json.loads(l) for l in open(os.path.join(ROOT, a.pairs))):
        A = items.get((p["chunk_a"], p["channel"]))
        B = items.get((p["chunk_b"], p["channel"]))
        if A and B and (A.get("claims") and B.get("claims")):
            pairs.append((A, B, p["channel"]))
    pairs = pairs[:a.max_n]
    if not pairs:
        sys.exit(f"[swap] no usable topic-matched pairs in {a.split}")

    if a.frozen_model_id is not None:
        from scsrc import model as M
        mid = None if a.frozen_model_id == "base" else a.frozen_model_id
        model, proc = M.load(model_id=mid)
        model.eval()
        n_tok, style = 0, (a.style or "instr")
    else:
        from evaluate import apply_checkpoint
        model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
        n_tok = meta.get("n_tokens", 1)
        style = a.style or ("pt" if meta.get("method") == "pt"
                            else "instr" if meta.get("method") == "lora" else "neo")

    def att(r):
        v = r.get("attested_channels")
        if isinstance(v, str):
            try:
                import ast
                v = ast.literal_eval(v)
            except Exception:
                v = [v]
        return {str(x).upper() for x in (v or [])}

    R = collections.defaultdict(list)
    PC = collections.defaultdict(lambda: collections.defaultdict(list))
    from tqdm import tqdm
    for A, B, ch in tqdm(pairs, desc="mvr-swap"):
        trig = _trig(ch.lower(), n_tok, style)
        gA, gB = _target(A["claims"]), _target(B["claims"])
        ctA = mvraw_user_content(A, trig, None, a.n_frames, style, ch)
        # ctx_B = A's context with ONLY this channel's content replaced by B's.
        # Same length by construction -- that is the entire point of the swap.
        ctB = mvraw_user_content(A, trig, _override_for(ch, B), a.n_frames, style, ch)
        lp = lambda ct, g: float(seq_logprob(model, _seq_mvg(proc, ct, g, "cuda"),
                                             no_grad=True))
        aa, ab = lp(ctA, gA), lp(ctB, gA)          # A's claim under both contexts
        d = aa - ab                                 # >0 means the swap hurt A's claim
        if ch.upper() in att(A):
            R["nec_pos"].append(int(d > a.eps)); R["nec_nats"].append(d)
            PC[ch]["nec_pos"].append(int(d > a.eps)); PC[ch]["nec_nats"].append(d)
        else:
            R["spec_pos"].append(int(d <= a.eps)); R["spec_nats"].append(d)
            PC[ch]["spec_pos"].append(int(d <= a.eps)); PC[ch]["spec_nats"].append(d)
        # FLIP: forced choice between the two claims under the two contexts.
        ba, bb = lp(ctA, gB), lp(ctB, gB)
        f = int(aa > ba and bb > ab)
        R["flip"].append(f); PC[ch]["flip"].append(f)
        R["acc_a"].append(int(aa > ba))

    m = lambda v: round(sum(v) / len(v), 4) if v else None
    out = {"run": os.path.basename(a.run_dir.rstrip("/")), "split": a.split,
           "step": a.step, "style": style, "n_frames": a.n_frames,
           "n_pairs": len(pairs), "eps_nats": a.eps,
           "nec_drop_pos": m(R["nec_pos"]), "nec_drop_nats": m(R["nec_nats"]),
           "spec_stay_pos": m(R["spec_pos"]), "spec_drop_nats": m(R["spec_nats"]),
           "n_nec": len(R["nec_pos"]), "n_spec": len(R["spec_pos"]),
           "flip": m(R["flip"]), "acc_a": m(R["acc_a"]), "n_flip": len(R["flip"]),
           "per_channel": {c: {k: m(v) for k, v in d.items()} for c, d in PC.items()},
           "note": ("do(channel := another chunk's content), mates matched on "
                    "(topic, channel, language). Prompt LENGTH is preserved, unlike "
                    "the ablation, whose 34-55 nat vis 'specificity' drops are a "
                    "length artifact. flip = A's claim wins under ctx_A AND B's wins "
                    "under ctx_B; chance .25 for the conjunction, .5 per side.")}
    if out["nec_drop_pos"] is not None and out["spec_stay_pos"] is not None:
        out["joint"] = round(min(out["nec_drop_pos"], out["spec_stay_pos"]), 4)
    d = os.path.join(a.run_dir, "eval", f"mvraw_{a.split}")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"swap{a.tag}_step_{a.step}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
