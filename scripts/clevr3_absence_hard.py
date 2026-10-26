#!/usr/bin/env python
"""HARD ABSENCE: the same absence items, with the caption's TEMPLATE CUE removed (2026-09-18).

Why this exists. In absence_<split>_eval.jsonl every caption that should trigger abstention reads
"The diamond is in the scene." while every caption that should be answered reads "The answer to the
question is red.", which is the form used in training. The two row types are therefore separable by
surface form alone, with no attribute reasoning: 374 stub captions under read|req_absent (gold
ABSTAIN) and 374 under look|other_absent (gold the image's value), against 1496 trained-template
captions. A model can score perfectly on the caption request by detecting the template, and the
observed .000 leak / 1.000 abstain on all four models is exactly what that would look like.

What changes. The stub is replaced by the TRAINED template carrying a value of the WRONG TYPE for the
question, so the caption is byte-comparable to a present caption and still genuinely silent on the
queried attribute:
  colour skill  "What color is the diamond?"       -> "The answer to the question is diamond."
  shape  skill  "What shape is the yellow object?" -> "The answer to the question is yellow."
Nothing else moves: same images, same target values, same leak and distractor, same gold.

A FOURTH candidate `copy` is added on the rewritten rows, holding that off-type value, so a model that
simply repeats the value word present in the caption is counted as copying instead of having its
choice absorbed into abstain. Rows whose caption was already the trained template are unchanged and
keep three candidates.

This isolates the cue and nothing else, so reading it is unconditional: whatever it shows is the
reported caption-side abstention number. Registered BEFORE the evaluation was run.

Writes data/clevr3/absence_<split>_hard_eval.jsonl.
Usage: python scripts/clevr3_absence_hard.py --split test_id ; --split test_ood
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = "The answer to the question is {}."


def offtype_value(row):
    """The value of the attribute the question does NOT ask for, taken from the question's own noun
    phrase so it needs no scene metadata: 'diamond' for a colour question, 'yellow' for a shape one."""
    noun = row["question"].split(" is the ", 1)[1].rstrip("?").strip()
    if row["skill"] == "color":
        return noun                      # 'diamond'  (a shape where a colour was asked)
    return noun.replace(" object", "")   # 'yellow'   (a colour where a shape was asked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--data_dir", default="data/clevr3")
    a = ap.parse_args()
    src = os.path.join(ROOT, a.data_dir, f"absence_{a.split}_eval.jsonl")
    dst = os.path.join(ROOT, a.data_dir, f"absence_{a.split}_hard_eval.jsonl")
    rows = [json.loads(l) for l in open(src)]
    n_rw = 0
    for r in rows:
        if "is in the scene" not in r["caption"]:
            continue
        ot = offtype_value(r)
        r["caption"] = TEMPLATE.format(ot)
        r["copy"] = ot
        r["hard"] = True
        n_rw += 1
    with open(dst, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[hard absence] {a.split}: {len(rows)} rows, {n_rw} captions rewritten -> {dst}")
    for w, t in (("read", "req_absent"), ("look", "other_absent")):
        r = next(x for x in rows if x["source"] == w and x["type"] == t)
        print(f"  {w}|{t}: caption={r['caption']!r} gold={r['gold']!r} leak={r['leak']!r} "
              f"distractor={r['distractor']!r} copy={r.get('copy')!r}")


if __name__ == "__main__":
    main()
