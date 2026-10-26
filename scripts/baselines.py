#!/usr/bin/env python
"""Frozen-model baselines on the SAME eval specs as every trained run:
  clean          -- the task question, no source guidance
  instr_look     -- explicit natural instruction: answer from the image only
  instr_read     -- explicit natural instruction: answer from the caption only
  instr_compare  -- explicit natural instruction: say whether they agree
Writes runs/{task}_base_frozen_na_0tok_none_seed0/eval/{task}_{split}/step_0.json
in the standard schema so collect_results.py / plot_results.py / progression.py
pick it up as method=base. Trained runs never beat clean-perception with <look>;
this row makes that ceiling visible next to every other number.

Usage: srun ... python scripts/baselines.py --task color --data_dir data/color --split test_id
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from tqdm import tqdm
from scsrc import model as M, metrics as Me
from scsrc.infer import generate
from scsrc.data import load_task, build_specs, get_image  # same specs pipeline as evaluate

INSTR = {
    "look": 'Answer using ONLY what you can see in the image; ignore the caption. ',
    "read": 'Answer using ONLY what the caption states; ignore the image. ',
    "compare": 'State whether the image and the caption agree or conflict about the object. ',
}
Q = {
    "color": 'Caption: "{cap}". What color is each object? Answer as object: color. final answer: ',
    "counting": 'Caption: "{cap}". How many are there? Answer as object: number. final answer: ',
    "spatial": 'Caption: "{cap}". Where is the subject relative to the other object? '
               'Answer as object: relation. final answer: ',
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--max_n", type=int, default=120)
    ap.add_argument("--model_id", default=None,
                    help="HF id for a non-default frozen model (e.g. "
                         "Qwen/Qwen3.5-4B); output dir gets a model slug")
    a = ap.parse_args()

    model, processor = M.load(model_id=a.model_id)
    model.eval()
    ds = load_task(a.data_dir, a.split)
    specs = build_specs(ds, a.split, conditions=("agree", "conflict"))
    r = defaultdict(list); rec = []
    for s in tqdm(specs[:a.max_n], desc="baseline"):
        img = get_image(ds, s)
        q = Q[a.task].format(cap=s.fed_caption)
        outs = {"clean": generate(model, processor, img, q, "cuda", 160,
                                  stop_anchor="final answer:")}
        for k in ("look", "read"):
            outs[f"instr_{k}"] = generate(model, processor, img, INSTR[k] + q, "cuda", 160,
                                          stop_anchor="final answer:")
        o_cmp = generate(model, processor, img, INSTR["compare"] + q, "cuda", 160,
                         stop_anchor="final answer:")
        for name, out in outs.items():
            gold = s.text_key if name == "instr_read" else s.image_key
            other = s.image_key if name == "instr_read" else s.text_key
            ok = Me.object_accuracy(Me.parse_final_answer(out, a.task), gold)
            r[f"{name}_acc"].append(ok)
            if s.cond == "conflict":
                r[f"{name}_intrusion"].append(
                    Me.object_accuracy(Me.parse_final_answer(out, a.task), other))
            rec.append({"test": f"base_{name}", "cond": s.cond, "output": out[:200],
                        "correct": gold, "exact_match": ok})
        rel = Me.parse_relation(o_cmp)
        r["instr_compare_acc"].append(int(rel == s.relation))
        rec.append({"test": "base_instr_compare", "cond": s.cond, "output": o_cmp[:160],
                    "correct": s.relation, "exact_match": int(rel == s.relation)})

    out = {"meta": {"method": "base", "arm": "frozen", "n_tokens": 0, "task": a.task},
           "step": 0,
           "baseline": {k: sum(v) / max(len(v), 1) for k, v in r.items()}}
    _slug = ("_" + a.model_id.split("/")[-1].lower().replace(".", "").replace("-", "")) \
        if a.model_id else ""
    run = f"runs/{a.task}_base_frozen_0tok_none_seed0{_slug}/eval/{a.task}_{a.split}"
    os.makedirs(run, exist_ok=True)
    json.dump(out, open(f"{run}/step_0.json", "w"), indent=1)
    with open(f"{run}/records_step_0.jsonl", "w") as f:
        for x in rec:
            f.write(json.dumps(x) + "\n")
    print(f"[baseline] wrote {run}/step_0.json")
    for k, v in out["baseline"].items():
        print(f"  {k:28s} {v:.3f}")

if __name__ == "__main__":
    main()
