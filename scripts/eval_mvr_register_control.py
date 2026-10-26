#!/usr/bin/env python
"""THE REGISTER CONTROL for MultiVENT-Raw — is the token reading evidence or style?

THE CONFOUND. In the cross-channel contrastive (build_mvraw_apo_pair) the chosen
claim comes from the requested channel and the rejected claim from another channel
of the SAME chunk. Measured on the real claims, a bag-of-words classifier predicts
which channel a claim came from **from the claim text alone** at 67% balanced
accuracy against 33% chance -- 2x chance. The markers are pure register:

    OCR  livestream, sign, reads, video      (screen vocabulary)
    ASR  says, heard, man, woman             (speech-report vocabulary)
    VIS  are, seen, large, water             (scene-description vocabulary)

So a model can prefer the right claim by matching REGISTER to the requested token,
having read no evidence at all. And the claim distribution is VIS 66% / OCR 22% /
ASR 13%, so the prior is strongest for exactly the channel we care most about.

THE CONTROL. Score the same forced choice twice:

    MATCHED   chunk X's claim pair with chunk X's OWN evidence
    SHUFFLED  chunk X's claim pair with chunk Y's evidence  (Y != X)

Under SHUFFLED nothing in the context supports either claim, so any preference that
survives is register, not grounding. The quantity that matters is the GAP:

    gap = matched - shuffled

    gap ~ 0        the token is reading register. The headline number is void.
    gap large      the token is reading evidence, and `shuffled` is the floor that
                   should be subtracted from -- or at least reported beside -- every
                   contrastive score.

This is the MultiVENT-Raw analogue of the cross-video EASY control on MultiVENT-G,
and it exists for the same reason: a pair that looks like an intervention and is not
produces a number that looks like grounding and is not.

Usage
  python scripts/eval_mvr_register_control.py --run_dir runs/mvr_f_5tok_apo_seed0
  python scripts/eval_mvr_register_control.py --run_dir runs/frozen_mvraw_gemma \
      --frozen_model_id base --style instr
"""
import argparse, collections, json, os, random, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/multivent_raw")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--n_frames", type=int, default=4)
    ap.add_argument("--max_n", type=int, default=150)
    ap.add_argument("--style", default=None, choices=[None, "neo", "pt", "instr", "noop"])
    ap.add_argument("--frozen_model_id", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from scsrc.mvraw import mvraw_user_content, _target, _trig, _seq_mvg
    from scsrc.preference import seq_logprob

    items = [json.loads(l) for l in
             open(os.path.join(ROOT, a.data_dir, f"{a.split}.jsonl"))]
    # only items that HAVE a cross-channel negative -- that is what the contrastive
    # objective trains on, so it is what the control must probe
    items = [x for x in items if x.get("neg_claims") and x.get("claims")]
    rng = random.Random(a.seed)
    rng.shuffle(items)
    items = items[:a.max_n]
    if not items:
        sys.exit("[control] no items with neg_claims")

    if a.frozen_model_id is not None:
        from scsrc import model as M
        mid = None if a.frozen_model_id == "base" else a.frozen_model_id
        model, proc = M.load(model_id=mid)
        n_tok, style = 0, (a.style or "instr")
    else:
        from evaluate import apply_checkpoint
        model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
        n_tok = meta.get("n_tokens", 1)
        style = a.style or ("pt" if meta.get("method") == "pt"
                            else "instr" if meta.get("method") == "lora" else "neo")

    res = collections.defaultdict(lambda: collections.Counter())
    for i, it in enumerate(items):
        ch = it["channel"]
        trig = _trig(ch.lower(), n_tok, style)
        pos, neg = _target(it["claims"]), _target(it["neg_claims"])
        # MATCHED: its own evidence.  SHUFFLED: another chunk's evidence, same
        # claim pair -- so nothing in context supports either side.
        other = items[(i + 1 + rng.randrange(len(items) - 1)) % len(items)]
        for cond, host in (("matched", it), ("shuffled", other)):
            ct = mvraw_user_content(host, trig, None, a.n_frames, style, ch)
            sp = float(seq_logprob(model, _seq_mvg(proc, ct, pos, "cuda"), no_grad=True))
            sn = float(seq_logprob(model, _seq_mvg(proc, ct, neg, "cuda"), no_grad=True))
            res[(ch, cond)]["n"] += 1
            res[(ch, cond)]["win"] += int(sp > sn)
            res[(ch, cond)]["margin_sum"] += (sp - sn)

    out = {"run": os.path.basename(a.run_dir.rstrip("/")), "split": a.split,
           "style": style, "n_items": len(items), "per_channel": {}}
    tot = collections.Counter()
    for ch in sorted({k[0] for k in res}):
        m, s = res[(ch, "matched")], res[(ch, "shuffled")]
        if not m["n"]:
            continue
        am, asf = m["win"] / m["n"], (s["win"] / s["n"] if s["n"] else None)
        out["per_channel"][ch] = {
            "n": m["n"], "matched": round(am, 4),
            "shuffled": round(asf, 4) if asf is not None else None,
            "gap": round(am - asf, 4) if asf is not None else None,
            "matched_margin": round(m["margin_sum"] / m["n"], 3),
            "shuffled_margin": round(s["margin_sum"] / s["n"], 3) if s["n"] else None}
        tot["m"] += m["win"]; tot["mn"] += m["n"]
        tot["s"] += s["win"]; tot["sn"] += s["n"]
    if tot["mn"] and tot["sn"]:
        out["overall"] = {"matched": round(tot["m"] / tot["mn"], 4),
                          "shuffled": round(tot["s"] / tot["sn"], 4),
                          "gap": round(tot["m"] / tot["mn"] - tot["s"] / tot["sn"], 4)}
    d = os.path.join(a.run_dir, "eval", f"mvraw_{a.split}")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"register_control{a.tag}_step_{a.step}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"[saved] {p}")
    if out.get("overall", {}).get("gap") is not None:
        g = out["overall"]["gap"]
        print(f"\n  GAP = {g:+.3f}   " + ("<- near zero: the preference is REGISTER, "
              "not evidence. Do not report the matched score alone."
              if abs(g) < 0.08 else "<- the token is reading evidence; quote "
              "`shuffled` as the floor."))


if __name__ == "__main__":
    main()
