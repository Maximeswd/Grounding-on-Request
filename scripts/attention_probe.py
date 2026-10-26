#!/usr/bin/env python
"""Attention analysis for source-control neologisms.

WHY THIS IS MEANINGFUL DESPITE FROZEN ATTENTION WEIGHTS: the trained W_Q/W_K/W_V
are frozen, but attention PATTERNS are input-dependent -- inserting a learned
token changes (a) the queries/keys/values computed FROM that token's embedding,
(b) what every later position computes when attending TO it, and (c) via the
residual stream, the queries of downstream answer tokens. So the token can and
should reshape where attention mass goes, with zero attention parameters trained.
What frozen weights DO mean: any pattern change is attributable to the embedding
alone -- which is exactly the mechanistic claim worth corroborating. (Caveat we
inherit from CRG: attention mass is not causal contribution; this probe
CORROBORATES the causal swap tests, it does not replace them.)

Measured, per example, from a single teacher-forced forward with
output_attentions=True over [prompt + generated answer]:
  A) answer->image  : mean attention from answer-token queries to image-token keys
  B) answer->caption: ... to the fed-caption span
  C) ctrl->image / ctrl->caption : from the control-token position(s)  [ctrl runs]
  D) gen-token rows: attention FROM emitted <look_gen>/... positions    [gen runs]
Aggregated per layer (line plots) and per layer x head (heatmaps), grouped by
op (look/read/noop) x correctness (correct/incorrect), saved as npz + png + json.

Run (one checkpoint, ~15-25 min for 60 examples):
  srun --gres=gpu:l40s:1 --time=01:30:00 python scripts/attention_probe.py \
      --run_dir $SC/runs/color_neo_ctrl_1tok_aposftvgm_seed0 --step last \
      --task color --data_dir $SC/data/color --split test_id --max_n 60
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from tqdm import tqdm

from scsrc import metrics as Me, prompts as P
from scsrc.data import load_task, build_specs, get_image
from scsrc.infer import generate
from scsrc.tokens import trigger


def _find_sub(hay, needle):
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i, i + len(needle)
    return None


def spans(input_ids, tokenizer, cap_text, new_ids):
    ids = input_ids[0].tolist()
    # image span: contiguous run of the image placeholder id(s)
    img_tok_ids = set()
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(getattr(tokenizer, "_model_config", None) or object(), attr, None)
        if v is not None:
            img_tok_ids.add(int(v))
    for tk in ("image_token", "boi_token"):
        t = getattr(tokenizer, tk, None)
        if t:
            try:
                img_tok_ids.add(tokenizer.convert_tokens_to_ids(t))
            except Exception:
                pass
    img_pos = [i for i, t in enumerate(ids) if t in img_tok_ids]
    cap_ids = tokenizer(cap_text, add_special_tokens=False)["input_ids"]
    cap_span = _find_sub(ids, cap_ids)
    ctrl_pos = [i for i, t in enumerate(ids) if t in set(new_ids or [])]
    return img_pos, (list(range(*cap_span)) if cap_span else []), ctrl_pos


def mass(attn_layer, q_pos, k_pos):
    """attn_layer: (heads, Q, K). Mean over q_pos queries of the SUM over k_pos
    keys -> per-head mass in [0,1]."""
    if not q_pos or not k_pos:
        return None
    a = attn_layer[:, q_pos][:, :, k_pos]     # (H, |q|, |k|)
    return a.sum(-1).mean(-1)                  # (H,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--step", default="last")
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--max_n", type=int, default=60)
    args = ap.parse_args()
    device = "cuda"

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import evaluate as EV
    steps = sorted(int(os.path.basename(p).split("_")[1]) for p in
                   glob.glob(os.path.join(args.run_dir, "ckpt", "step_*")))
    step = steps[-1] if args.step == "last" else int(args.step)
    model, processor, meta, forbid = EV.apply_checkpoint(args.run_dir, step, device)
    arm, n, method = meta["arm"], meta["n_tokens"], meta["method"]
    tokenizer = processor.tokenizer

    ds = load_task(args.data_dir, args.split)
    specs = [s for s in build_specs(ds, args.split, conditions=("conflict",))][:args.max_n]

    if method == "pt":
        ops = ("look", "read", "neutral", "noop", "instr_look", "instr_read")
    elif method == "lora":
        ops = ("instr_look", "instr_read", "neutral", "noop")
    elif arm == "ctrl":
        ops = ("look", "read", "neutral", "noop", "instr_look", "instr_read")
    else:
        ops = ("gen",)
    INSTR = {"instr_look": 'Answer using ONLY what you can see in the image; ignore the caption.',
             "instr_read": 'Answer using ONLY what the caption states; ignore the image.'}
    groups = {}   # (op, "correct"/"incorrect") -> list of (L,H,2) arrays [img, cap]
    recs = []
    for s in tqdm(specs, desc="attention"):
        img = get_image(ds, s)
        for op in ops:
            if arm == "ctrl" or method in ("pt", "lora"):
                if op == "neutral":
                    # BARE question -- no source directive; measures the
                    # model's DEFAULT channel preference (the motivating
                    # phenomenon) and serves as the clean attention baseline
                    # (ctrl_noop says "from the image" and is confounded)
                    from scsrc.prompts import slot_word as _sw
                    user = (f'Caption: "{s.fed_caption}". '
                            f"What is each object's {_sw(args.task)}?")
                elif op == "noop":
                    user = P.ctrl_noop(args.task).format(cap=s.fed_caption)
                elif op.startswith("instr_"):
                    user = f'Caption: "{s.fed_caption}". ' + INSTR[op]
                elif method == "pt":
                    user = trigger(f"pt_{op}", n) + " " + P.pt_template(0).format(cap=s.fed_caption)
                else:
                    user = P.render_ctrl(P.CTRL_CANONICAL, trigger(f"{op}_ctrl", n), s.fed_caption)
                out = generate(model, processor, img, user, device, 32, forbid_ids=forbid)
                pred = Me.parse_pred_task(out, args.task)
                gold = s.image_key if op in ("look", "noop", "neutral", "instr_look") else s.text_key
                ok = Me.object_accuracy(pred, gold) >= 1.0
            else:
                user = P.gen_template(args.task, 0).format(cap=s.fed_caption)
                out = generate(model, processor, img, user, device, 96)
                ok = trigger("look_gen", n) in out
            # teacher-forced forward over prompt+answer with attentions
            msgs = [{"role": "user", "content": [{"type": "image"},
                     {"type": "text", "text": user}]}]
            prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            enc = processor(text=prompt + out, images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                o = model(**enc, output_attentions=True)
            plen = processor(text=prompt, images=img, return_tensors="pt")["input_ids"].shape[1]
            T = enc["input_ids"].shape[1]
            ans_pos = list(range(plen, T))
            img_pos, cap_pos, ctrl_pos = spans(enc["input_ids"], tokenizer, s.fed_caption,
                                               getattr(model, "_neo_all_ids", None))
            L = len(o.attentions); H = o.attentions[0].shape[1]
            arr = np.zeros((L, H, 4), dtype=np.float32)   # ans->img, ans->cap, tok->img, tok->cap
            src = ctrl_pos if (op in ("look", "read") and method in ("neo", "pt")) else \
                  [i for i in ans_pos if enc["input_ids"][0, i].item() in set(getattr(model, "_neo_all_ids", []) or [])]
            for li, A in enumerate(o.attentions):
                a = A[0].float().cpu().numpy()
                for gi, (q, k) in enumerate([(ans_pos, img_pos), (ans_pos, cap_pos),
                                             (src, img_pos), (src, cap_pos)]):
                    m = mass(a, q, k)
                    if m is not None:
                        arr[li, :, gi] = m
            del o; torch.cuda.empty_cache()
            key = (op, "correct" if ok else "incorrect")
            groups.setdefault(key, []).append(arr)
            recs.append({"eid": s.example_id, "op": op, "ok": bool(ok), "out": out[:150]})

    out_dir = os.path.join(args.run_dir, "attention", f"{args.task}_{args.split}_step{step}")
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for (op, okk), arrs in groups.items():
        A = np.stack(arrs)                      # (N, L, H, 4)
        np.savez(os.path.join(out_dir, f"{op}_{okk}.npz"), attn=A)
        summary[f"{op}_{okk}"] = {"n": int(A.shape[0]),
                                  "ans_to_img_by_layer": A[..., 0].mean((0, 2)).round(4).tolist(),
                                  "ans_to_cap_by_layer": A[..., 1].mean((0, 2)).round(4).tolist(),
                                  "tok_to_img_by_layer": A[..., 2].mean((0, 2)).round(4).tolist(),
                                  "tok_to_cap_by_layer": A[..., 3].mean((0, 2)).round(4).tolist()}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    # plots (best effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, v in summary.items():
            ax.plot(v["ans_to_img_by_layer"], label=f"{k} ans->img")
        ax.set_xlabel("layer"); ax.set_ylabel("attention mass"); ax.legend(fontsize=7)
        ax.set_title("Answer-token attention to IMAGE span, by layer")
        fig.savefig(os.path.join(out_dir, "ans_to_img_by_layer.png"), dpi=150, bbox_inches="tight")
        for (op, okk), arrs in groups.items():
            A = np.stack(arrs).mean(0)          # (L, H, 4)
            fig, ax = plt.subplots(figsize=(9, 5))
            im = ax.imshow(A[..., 0].T, aspect="auto", cmap="viridis")
            ax.set_xlabel("layer"); ax.set_ylabel("head")
            ax.set_title(f"{op}/{okk}: answer->image attention (head x layer)")
            fig.colorbar(im); fig.savefig(os.path.join(out_dir, f"heat_{op}_{okk}.png"),
                                          dpi=150, bbox_inches="tight")
    except Exception as e:
        print(f"[plots skipped: {type(e).__name__}: {e}]")
    print(f"[attention] wrote {out_dir}/ (npz per group + summary.json + records + plots)")


if __name__ == "__main__":
    main()
