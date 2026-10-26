#!/usr/bin/env python
"""VisualSwap self-interrupt probe: does <recheck> restore grounding where
natural-language reflection fails?

Protocol (faithful to VisualSwap): current image = B (shows v_true); the
assistant is PREFILLED with stale reasoning asserting v_stale (the twin's
value) + a trigger; the model continues. flip-to-correct = the continuation
gives v_true (re-grounded); persist = it gives v_stale (textual inertia).

Trigger conditions = the missing rows of their Appendix H:
  recheck   trained <recheck_ctrl>            (ours)
  nl        "Wait, let me look at the image again."   (their reflection phrase)
  filler    "(blicket)"                        (clean nonce, verified non-value)
  highppl   a high-perplexity random string    (their degenerate-string control)
  none      no trigger                          (bare inertia baseline)

Inertia axis (their Sec 5.4): stale reasoning at increasing lengths; a
quotable curve is <recheck> flip-to-correct staying flat while nl decays.

Usage: python scripts/recheck_probe.py --run_dir runs/recheck_neo_ctrl_1tok_seed0 \
         --data_dir data/vcf --split test_id --step 400
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger
from scsrc import metrics as Me
from scsrc.infer import generate
from scsrc.recheck import QUESTION
from evaluate import apply_checkpoint
from scripts.swap_probe import norm_key

NL = "Wait, let me look at the image again."
HIGHPPL = "qux zzt vorplex nimb"           # high-perplexity nonword string
FILLER = "(blicket)"
# stale reasoning at 3 retained-context lengths (VisualSwap Sec 5.4 axis)
PAD = ("I examined the overall scene, the lighting, the background, and the "
       "main object carefully before reaching that conclusion. ")


def stale_text(obj, v, length):
    base = f"Looking at the earlier image, I concluded the {obj} is {v}."
    if length == "short":
        return base
    if length == "med":
        return base + " " + PAD
    return base + " " + PAD * 3   # long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data/vcf")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=400)
    ap.add_argument("--max_n", type=int, default=60)
    ap.add_argument("--task", default="color")
    a = ap.parse_args()

    model, processor, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    triggers = {"recheck": trigger("recheck_ctrl", n), "nl": NL,
                "filler": FILLER, "highppl": HIGHPPL, "none": ""}
    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",))
             if s.cond == "conflict"][:a.max_n]

    # metrics[length][trigger] = [flip_to_correct, persist_stale]
    res = {L: {t: {"flip": [], "persist": []} for t in triggers}
           for L in ("short", "med", "long")}
    rec = []
    for sp in tqdm(specs, desc="recheck"):
        img = get_image(ds, sp)                    # current image B (shows v_true)
        ik, tk = norm_key(sp.image_key), norm_key(sp.text_key)
        if not ik or set(ik) != set(tk):
            continue
        obj = next(iter(ik))
        v_true, v_stale = ik[obj], tk[obj]
        if v_true == v_stale:
            continue
        q = QUESTION.format(obj=obj)
        for L in ("short", "med", "long"):
            stale = stale_text(obj, v_stale, L)
            for tname, tstr in triggers.items():
                prefill = f"{stale} {tstr}".strip() + " " if tstr else stale + " "
                o = generate(model, processor, img, q, "cuda", 64,
                             assistant_prefill=prefill, forbid_ids=forbid,
                             stop_anchor="final answer:")
                pred = norm_key(Me.parse_final_answer(o, a.task))
                res[L][tname]["flip"].append(
                    int(Me.object_accuracy(pred, {obj: v_true}) == 1.0))
                res[L][tname]["persist"].append(
                    int(Me.object_accuracy(pred, {obj: v_stale}) == 1.0))
                if L == "short":
                    rec.append({"idx": sp.ex_index, "trigger": tname,
                                "v_true": v_true, "v_stale": v_stale,
                                "out": o[:160]})

    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    table = {"run": a.run_dir, "step": a.step, "n": len(specs)}
    for L in ("short", "med", "long"):
        table[L] = {t: {"flip_to_correct": m(res[L][t]["flip"]),
                        "persist_stale": m(res[L][t]["persist"])}
                    for t in triggers}
    print(f"\n== recheck self-interrupt  {a.run_dir}  step {a.step} ==")
    print(f"  {'len':6s} " + " ".join(f"{t:>9s}" for t in triggers))
    for L in ("short", "med", "long"):
        print(f"  {L:6s} " + " ".join(
            f"{table[L][t]['flip_to_correct']:>9.3f}" for t in triggers))
    print("  (flip-to-correct: continuation re-grounds to the CURRENT image)")
    rc = table["short"]["recheck"]["flip_to_correct"]
    nlc = table["short"]["nl"]["flip_to_correct"]
    print(f"  headline: recheck {rc:.3f} vs NL {nlc:.3f}  (Δ {rc-nlc:+.3f})")

    outd = os.path.join(a.run_dir, "eval", f"recheck_{a.split}")
    os.makedirs(outd, exist_ok=True)
    json.dump(table, open(os.path.join(outd, f"recheck_step_{a.step}.json"), "w"),
              indent=2)
    with open(os.path.join(outd, f"recheck_records_step_{a.step}.jsonl"), "w") as f:
        for x in rec:
            f.write(json.dumps(x) + "\n")
    print(f"[saved] {outd}/recheck_step_{a.step}.json")


if __name__ == "__main__":
    main()
