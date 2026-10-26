#!/usr/bin/env python
"""The filler-control experiment: on CONFLICT items only, fill the operator slot
with (a) the real token, (b) its own description, (c) the READ token's
description, (d) generic fillers, (e) a random real word -- scoring every output
against BOTH the image gold and the caption gold. Separates three hypotheses:
  slot effect      -> all fillers route alike
  content routing  -> own-desc beats fillers; read-desc flips toward caption
  token-specific   -> only the real token routes
Usage: srun ... python scripts/plugin_control.py --run_dir runs/color_neo_ctrl_1tok_sft_seed0 \
              --task color --data_dir data/color --split test_id --step 400
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger
from scsrc import prompts as P, metrics as Me
from scsrc.infer import generate
from evaluate import apply_checkpoint

# provably NOT a candidate answer in any rung (checked vs VCF/ColorSwap colours
# + COCO change-nouns 2026-07-21): banana/tuesday/kettle COLLIDE with COCO nouns,
# "give" collides with the template. Three lexical classes: generic function
# word, random real content word, pure nonce.
FILLERS = ["please", "trombone", "blicket"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=400)
    ap.add_argument("--max_n", type=int, default=60)
    a = ap.parse_args()

    model, processor, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("agree", "conflict"))
             if s.cond == "conflict"][:a.max_n]

    tag = f"{a.task}_{a.split}"
    ev = json.load(open(os.path.join(a.run_dir, "eval", tag, f"step_{a.step}.json")))
    vt = ev.get("verbalization_text", {})
    desc_look = (vt.get("look_ctrl") or "").strip().rstrip(".")
    desc_read = (vt.get("read_ctrl") or "").strip().rstrip(".")

    slots = {"token": trigger("look_ctrl", n),
             "own_desc": f"({desc_look})",
             "read_desc": f"({desc_read})"}
    for f in FILLERS:
        slots[f"filler:{f}"] = f"({f})"

    res, rec = defaultdict(lambda: defaultdict(list)), []
    for s in tqdm(specs, desc="control"):
        img = get_image(ds, s)
        for name, fill in slots.items():
            u = P.render_ctrl(P.CTRL_CANONICAL, fill, s.fed_caption)
            o = generate(model, processor, img, u, "cuda", 160,
                         forbid_ids=forbid, stop_anchor="final answer:")
            p = Me.parse_final_answer(o, a.task)
            res[name]["img"].append(Me.object_accuracy(p, s.image_key))
            res[name]["txt"].append(Me.object_accuracy(p, s.text_key))
            rec.append({"slot": name, "output": o[:200],
                        "img_gold": s.image_key, "txt_gold": s.text_key})

    outd = os.path.join(a.run_dir, "eval", tag)
    print(f"\n== filler control  {a.run_dir}  step {a.step}  (conflict n={len(specs)}) ==")
    print(f"{'slot':22s} {'IMG-acc':>8s} {'TXT-acc':>8s}   interpretation")
    table = {}
    for name in slots:
        i = sum(res[name]["img"]) / max(len(res[name]["img"]), 1)
        t = sum(res[name]["txt"]) / max(len(res[name]["txt"]), 1)
        table[name] = {"img_acc": round(i, 3), "txt_acc": round(t, 3)}
        note = "image-routed" if i > t + 0.15 else ("caption-routed" if t > i + 0.15 else "mixed/none")
        print(f"{name:22s} {i:8.3f} {t:8.3f}   {note}")
    fillers_img = [table[k]["img_acc"] for k in table if k.startswith("filler:")]
    f_mean = sum(fillers_img) / max(len(fillers_img), 1)
    tok_i, own_i = table["token"]["img_acc"], table["own_desc"]["img_acc"]
    print(f"\n  filler mean IMG-acc      {f_mean:.3f}")
    print(f"  VF_adjusted (own-filler) {own_i - f_mean:+.3f}   token-filler {tok_i - f_mean:+.3f}")
    print("  -> description carries meaning only if own-filler is clearly positive")
    table["_summary"] = {"filler_mean_img": round(f_mean, 3),
                         "vf_adjusted": round(own_i - f_mean, 3),
                         "token_minus_filler": round(tok_i - f_mean, 3)}
    with open(os.path.join(outd, f"plugin_control_step_{a.step}.json"), "w") as f:
        json.dump({"table": table, "desc_look": desc_look[:200], "desc_read": desc_read[:200]}, f, indent=2)
    with open(os.path.join(outd, f"plugin_control_records_step_{a.step}.jsonl"), "w") as f:
        for x in rec:
            f.write(json.dumps(x) + "\n")
    print(f"[saved] {outd}/plugin_control_step_{a.step}.json (+records)")

if __name__ == "__main__":
    main()
