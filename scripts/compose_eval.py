#!/usr/bin/env python
"""Zero-shot COMPOSITION eval: chains of operators the model never saw together.
The boids test: were look/read/compare learned as LOCAL rules that compose into
a faithful multi-claim report, untrained?

Design encodes the dual-phase lessons:
  - explicit per-line slots (bare juxtaposition collapses to one operator)
  - CONFLICT items only (agree items give channel binding away for free)
  - attribution-resolved scoring: line i is scored against ITS op's own gold
    (image key / caption key / verdict) -- presence never counts.

Usage: srun ... python scripts/compose_eval.py --run_dir runs/color_neo_ctrl_1tok_sft_seed0 \
             --task color --data_dir data/color --split test_id --step 400
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger
from scsrc import metrics as Me
from scsrc.infer import generate
from evaluate import apply_checkpoint

CHAINS = [("look",), ("read",),                      # single-op controls, same format
          ("look", "read"), ("read", "look"),        # 2-rule compositions, both orders
          ("look", "read", "compare"),               # 3-rule report
          ("read", "look", "compare")]

def chain_prompt(chain, caption, n):
    lines = [f'The caption for this image is: "{caption}"',
             f"Answer in exactly {len(chain)} numbered lines, one line per instruction, "
             f"each ending with its own 'final answer:'."]
    for i, op in enumerate(chain, 1):
        lines.append(f"{i}. Give me a {trigger(f'{op}_ctrl', n)} answer.")
    return "\n".join(lines)

def split_lines(out, k):
    """Best-effort: map numbered lines back to claims."""
    parts = {}
    cur = None
    for raw in out.split("\n"):
        t = raw.strip()
        if len(t) > 1 and t[0].isdigit() and t[1] in ".):":
            cur = int(t[0]); parts[cur] = t[2:]
        elif cur is not None and t:
            parts[cur] = parts.get(cur, "") + " " + t
    return [parts.get(i + 1, "") for i in range(k)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=400)
    ap.add_argument("--max_n", type=int, default=40)
    a = ap.parse_args()

    model, processor, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("agree", "conflict"))
             if s.cond == "conflict"][:a.max_n]

    res, rec = defaultdict(lambda: defaultdict(list)), []
    for s in tqdm(specs, desc="compose"):
        img = get_image(ds, s)
        for chain in CHAINS:
            u = chain_prompt(chain, s.fed_caption, n)
            o = generate(model, processor, img, u, "cuda", 90 * len(chain), forbid_ids=forbid)
            lines = split_lines(o, len(chain))
            oks = []
            for op, line in zip(chain, lines):
                if op == "compare":
                    ok = float(Me.parse_relation(line) == s.relation)
                else:
                    gold = s.image_key if op == "look" else s.text_key
                    ok = Me.object_accuracy(Me.parse_final_answer(line, a.task), gold)
                oks.append(ok)
                res["|".join(chain)][f"claim_{op}"].append(ok)
            name = "|".join(chain)
            res[name]["all_claims"].append(float(all(o_ == 1.0 for o_ in oks)))
            res[name]["parsed_all_lines"].append(float(all(bool(l.strip()) for l in lines)))
            for j in range(1, len(oks)):           # cascade: later failure given earlier failure
                if oks[j - 1] < 1.0:
                    res[name]["later_ok_given_earlier_fail"].append(float(oks[j] == 1.0))
            rec.append({"eid": s.ex_index, "chain": name, "output": o[:400],
                        "per_claim": oks})

    tag = f"{a.task}_{a.split}"
    outd = os.path.join(a.run_dir, "eval", tag); os.makedirs(outd, exist_ok=True)
    print(f"\n== composition  {a.run_dir}  step {a.step}  (conflict n={len(specs)}) ==")
    table = {}
    for name in ["|".join(c) for c in CHAINS]:
        m = {k: round(sum(v) / len(v), 3) for k, v in res[name].items() if v}
        table[name] = m
        print(f"  {name:24s} " + "  ".join(f"{k}={v}" for k, v in sorted(m.items())))
    with open(os.path.join(outd, f"compose_step_{a.step}.json"), "w") as f:
        json.dump(table, f, indent=2)
    with open(os.path.join(outd, f"compose_records_step_{a.step}.jsonl"), "w") as f:
        for x in rec:
            f.write(json.dumps(x) + "\n")
    print(f"[saved] {outd}/compose_step_{a.step}.json (+records)")

if __name__ == "__main__":
    main()
