#!/usr/bin/env python
"""Render-audit: for every (task x arm x style x override) training cell,
build REAL examples through scsrc.dataset/collate on CPU and assert the
input/output invariants that cell promises. String-level (the tokenizer
neologism ids are added at train time by add_neologisms; here tokens are
audited as strings, which is what defines the prompt/target contract).

Run:  python scripts/audit_render.py [--n 4]
Exit nonzero on any failed invariant; prints a PASS/FAIL table.
"""
import argparse, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from scsrc.data import build_specs, load_task, get_image  # noqa: E402
from scsrc.collate import build_train_example  # noqa: E402

FAILS = []


def check(cell, name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append((cell, name, detail))
        print(f"  {tag}  {cell:<34} {name}  {detail[:110]}")


def decode_target(processor, out):
    ids = out["input_ids"][0]
    lab = out["labels"][0]
    tgt_ids = ids[lab != -100]
    return processor.tokenizer.decode(tgt_ids, skip_special_tokens=False)


def decode_prompt(processor, out):
    ids = out["input_ids"][0]
    lab = out["labels"][0]
    pl = int((lab == -100).sum())
    return processor.tokenizer.decode(ids[:pl], skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--model_id", default="google/gemma-3-4b-it")
    args = ap.parse_args()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_id)

    for task in ("color", "counting", "spatial"):
        ds = load_task(f"data/{task}", "train")
        specs = build_specs(ds, "train",
                            conditions=("agree", "conflict"))[: args.n * 2]
        for i, s in enumerate(specs[: args.n]):
            img = get_image(ds, s)
            base = f"{task}/ex{i}"
            for op in ("look", "read", "compare"):
                s.op = op
                # ---- ctrl neutral ----
                o = build_train_example(processor, s, img, "neo", "ctrl", 1,
                                        "cpu", task=task,
                                        cap_first=(i % 2 == 0),
                                        ctrl_style="neutral")
                p, t = decode_prompt(processor, o), decode_target(processor, o)
                cell = f"{base}/ctrl-neutral/{op}"
                check(cell, "token-in-slot", f"<{op}_ctrl>" in p)
                check(cell, "caption-present", s.fed_caption[:20] in p)
                check(cell, "final-answer-anchor", "final answer:" in t, t)
                check(cell, "no-token-in-target", "_ctrl>" not in t, t)
                if op != "compare":
                    check(cell, "neutral-no-channel-nouns",
                          not re.search(r"\b(image|picture|text|caption)\b",
                                        t, re.I), t)
                if task == "spatial":
                    check(cell, "no-pipe", "|" not in p and "|" not in t)
                check(cell, "labels-masked",
                      int((o["labels"] != -100).sum()) > 0
                      and bool((o["labels"][0][:5] == -100).all()))
                check(cell, "image-attached", "pixel_values" in o)
                # ---- ctrl sourced ----
                if op != "compare":
                    o2 = build_train_example(processor, s, img, "neo", "ctrl",
                                             1, "cpu", task=task,
                                             ctrl_style="sourced")
                    t2 = decode_target(processor, o2)
                    cell = f"{base}/ctrl-sourced/{op}"
                    want = r"\b(image|photo|picture)\b" if op == "look" \
                        else r"\b(text|caption)\b"
                    check(cell, "sourced-names-channel",
                          re.search(want, t2, re.I), t2)
                    check(cell, "final-answer-anchor", "final answer:" in t2, t2)
                # ---- gen ----
                og = build_train_example(processor, s, img, "neo", "gen", 1,
                                         "cpu", task=task)
                tg = decode_target(processor, og)
                cell = f"{base}/gen/{op}"
                for gt in ("<look_gen>", "<read_gen>", "<compare_gen>"):
                    check(cell, f"tag-{gt}", tg.count(gt) >= 2, tg[:120])
                # ---- link ----
                ol = build_train_example(processor, s, img, "neo", "link", 1,
                                         "cpu", task=task)
                pl_, tl = decode_prompt(processor, ol), decode_target(processor, ol)
                cell = f"{base}/link/{op}"
                check(cell, "ctrl-token-in-prompt", f"<{op}_ctrl>" in pl_)
                check(cell, "gen-block-target",
                      tl.count(f"<{op}_gen>") == 2, tl[:120])
            # ---- resolve (image-wins arbitration word) ----
            s.op = "resolve"
            orz = build_train_example(processor, s, img, "neo", "ctrl", 1,
                                      "cpu", task=task)
            prz, trz = decode_prompt(processor, orz), decode_target(processor, orz)
            cell = f"{base}/ctrl-resolve"
            check(cell, "token-in-slot", "<resolve_ctrl>" in prz, prz[-140:])
            check(cell, "final-answer-anchor", "final answer:" in trz, trz)
            if s.relation == "conflict":
                check(cell, "policy-stated", "image takes precedence" in trz, trz)
            else:
                check(cell, "agree-form", "both sources give" in trz, trz)
            if task == "spatial":
                check(cell, "no-pipe", "|" not in trz, trz)
            # ---- arbiter (argument-taking arbitration word) ----
            s.op = "arbiter"
            oab = build_train_example(processor, s, img, "neo", "ctrl", 1,
                                      "cpu", task=task,
                                      cap_first=(i % 2 == 0))
            pab, tab = decode_prompt(processor, oab), decode_target(processor, oab)
            cell = f"{base}/ctrl-arbiter"
            check(cell, "op-token-in-slot", "<arbiter_ctrl>" in pab, pab[-160:])
            check(cell, "arg-token-in-slot",
                  ("<look_ctrl>" in pab) != ("<read_ctrl>" in pab), pab[-160:])
            check(cell, "final-answer-anchor", "final answer:" in tab, tab)
            if s.relation == "conflict":
                check(cell, "favored-source-form", "favored source" in tab, tab)
            else:
                check(cell, "agree-form", "both sources give" in tab, tab)
            if task == "spatial":
                check(cell, "no-pipe", "|" not in tab, tab)
            # ---- v3 slot-negative (op-independent) ----
            s.op = "look"
            ov = build_train_example(processor, s, img, "neo", "ctrl", 1,
                                     "cpu", task=task, slot_negative=True)
            pv, tv = decode_prompt(processor, ov), decode_target(processor, ov)
            cell = f"{base}/ctrl-v3-slotneg"
            check(cell, "no-ctrl-token", "_ctrl>" not in pv, pv[-140:])
            check(cell, "unknown-operator-gold",
                  "unknown operator" in tv and "final answer:" in tv, tv)
            # ---- absence: image ----
            oa = build_train_example(processor, s, None, "neo", "ctrl", 1,
                                     "cpu", task=task, absence="image")
            ta = decode_target(processor, oa)
            cell = f"{base}/ctrl-absence-image"
            check(cell, "unavailable-gold",
                  "not available" in ta and "unavailable" in ta, ta)
            check(cell, "no-pixels", "pixel_values" not in oa)
            # ---- pt ----
            op_ = build_train_example(processor, s, img, "pt", "na", 1,
                                      "cpu", task=task)
            pp, tp = decode_prompt(processor, op_), decode_target(processor, op_)
            cell = f"{base}/pt"
            check(cell, "no-inline-ctrl-token", "_ctrl>" not in pp)
            check(cell, "final-answer-anchor", "final answer:" in tp, tp)
            # ---- lora ----
            olr = build_train_example(processor, s, img, "lora", "na", 1,
                                      "cpu", task=task)
            tlr = decode_target(processor, olr)
            cell = f"{base}/lora"
            check(cell, "structured-format",
                  "IMAGE:" in tlr and "TEXT:" in tlr and "RELATION:" in tlr,
                  tlr[:120])
            if task == "spatial":
                check(cell, "no-pipe", "|" not in tlr, tlr[:120])

    print(f"\n{'='*60}")
    if FAILS:
        print(f"AUDIT FAILED: {len(FAILS)} invariant violations")
        sys.exit(1)
    print("AUDIT PASSED: all rendered cells honor their contracts")


if __name__ == "__main__":
    main()
