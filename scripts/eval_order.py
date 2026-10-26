#!/usr/bin/env python
"""ORDER FIDELITY for the ordered-composition arm (ctrl_style=blockscmp).

The arm trains: word order in the PROMPT (<look><read><compare> vs
<read><look><compare> vs compare-first) must be mirrored by BLOCK order in the
ANSWER, with the verdict block referring to sources as FIRST/SECOND. Without
this eval the arm's defining property is untested.

Per item, four request variants are scored by FREE GENERATION:
    lr_lrc  <look><read><compare>   -> expect blocks look,read,compare
    rl_lrc  <read><look><compare>   -> expect blocks read,look,compare
    lr_clr  <compare><look><read>   -> expect blocks compare,look,read
    rl_clr  <compare><read><look>   -> expect blocks compare,read,look

Metrics
  order_exact      emitted block sequence == requested word sequence
  order_pairwise   look-vs-read relative order matches the request (isolates
                   the 50/50-trained bit from the 2:1-trained verdict slot)
  block_content    look block contains the image value & read block the text
                   value (grounding inside the right block, by string match)
  verdict_binding  the compare block's "first ... second" values match the
                   REQUESTED order (the order-relative receipt actually tracks)
  final_ok         final answer = abstain on conflict / value on agree
  chance           order_exact expected under always-emit-training-mode ~ the
                   empirical frequency of the majority target (reported).

Output: runs/<run>/eval/<ds>_<split>/order_step_<STEP>.json
"""
import argparse, collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger
from scsrc import prompts as P
from evaluate import apply_checkpoint
from scripts.eval_mvraw import gen_blocks

TAGS = {"look": "<look>", "read": "<read>", "compare": "<compare>"}


def block_seq(txt):
    """Sequence of block types by OPENING tag occurrence (delimiters are the
    same word at both ends, so occurrences come in pairs; order = first hits)."""
    hits = []
    for op, tag in TAGS.items():
        for m in re.finditer(re.escape(tag), txt):
            hits.append((m.start(), op))
    hits.sort()
    seq, open_ = [], set()
    for _, op in hits:
        if op in open_:
            open_.discard(op)      # closing tag
        else:
            open_.add(op)
            seq.append(op)
    return seq


def block_text(txt, op):
    tag = re.escape(TAGS[op])
    m = re.search(f"{tag}(.*?){tag}", txt, re.S)
    return m.group(1).strip() if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=60)
    ap.add_argument("--single", type=int, default=0,
                    help="1 = single-word prompts; measures block-type-follows-"
                         "word (chance .5) instead of multi-word order fidelity")
    a = ap.parse_args()
    dev = "cuda"
    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, dev)
    n = meta["n_tokens"]

    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("agree", "conflict"))]
    R = collections.defaultdict(list)
    seen = 0
    for sp in tqdm(specs, desc="order"):
        if seen >= a.max_n:
            break
        ik = {str(k).split("|")[0].strip().lower(): str(v).strip().lower()
              for k, v in (sp.image_key or {}).items()}
        tk = {str(k).split("|")[0].strip().lower(): str(v).strip().lower()
              for k, v in (sp.text_key or {}).items()}
        if not ik or set(ik) != set(tk):
            continue
        seen += 1
        obj = sorted(ik)[seen % len(ik)]
        v_i, v_t = ik[obj], tk[obj]
        img = get_image(ds, sp)
        wl, wr = trigger("look_ctrl", n), trigger("read_ctrl", n)
        wc = trigger("compare_ctrl", n)
        if a.single:
            # SINGLE-WORD counterfactual: one control word -> does the emitted
            # BLOCK TYPE follow the word? (the generation-side word->block test
            # for link/blocks runs; chance .5)
            variants = {"look_only": ([wl], ["look"]),
                        "read_only": ([wr], ["read"])}
        else:
            variants = {"lr_lrc": ([wl, wr, wc], ["look", "read", "compare"]),
                    "rl_lrc": ([wr, wl, wc], ["read", "look", "compare"]),
                    "lr_clr": ([wc, wl, wr], ["compare", "look", "read"]),
                    "rl_clr": ([wc, wr, wl], ["compare", "read", "look"])}
        if a.single:
            pass
        for name, (words, want) in variants.items():
            u = P.render_ctrl(P.CTRL_CANONICAL, " ".join(words), sp.fed_caption)
            content = [{"type": "image", "image": img},
                       {"type": "text", "text": u}]
            txt = gen_blocks(model, proc, content, dev, max_new=180)
            got = block_seq(txt)
            if a.single:
                R["block_follows_word"].append(int(got[:1] == want))
                R[f"block_follows_{name}"].append(int(got[:1] == want))
                continue
            R[f"order_exact"].append(int(got == want))
            R[f"order_exact_{name}"].append(int(got == want))
            if "look" in got and "read" in got:
                lr_got = got.index("look") < got.index("read")
                lr_want = want.index("look") < want.index("read")
                R["order_pairwise"].append(int(lr_got == lr_want))
            bl, br = block_text(txt, "look"), block_text(txt, "read")
            if bl or br:
                R["block_content"].append(int(v_i in bl and v_t in br))
            bc = block_text(txt, "compare")
            if bc:
                first_v = v_t if want.index("read") < want.index("look") else v_i
                m = re.search(r"first[^.]*?being ([^.]+)\.", bc)
                if m:
                    R["verdict_binding"].append(int(first_v in m.group(1)))
            fa = txt.split("final answer:")[-1].strip().lower() if "final answer:" in txt else ""
            if sp.cond == "conflict":
                R["final_ok"].append(int("cannot" in fa or "determined" in fa))
            else:
                R["final_ok"].append(int(v_i in fa))

    m = lambda v: round(sum(v) / len(v), 3) if v else None
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "n_items": seen,
           **{k: m(v) for k, v in sorted(R.items())},
           **{f"n_{k}": len(v) for k, v in sorted(R.items())},
           "note": ("order_exact chance if the model always emits ONE fixed "
                    "order = max variant frequency = 0.25 here (4 variants "
                    "equally probed); order_pairwise chance .5")}
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("n_")}, indent=1))
    od = os.path.join(a.run_dir, "eval", f"{os.path.basename(a.data_dir)}_{a.split}")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"order_step_{a.step}.json"), "w"), indent=1)
    print(f"[saved] {od}/order_step_{a.step}.json")


if __name__ == "__main__":
    main()
