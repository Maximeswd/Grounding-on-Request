#!/usr/bin/env python
"""Attention probe v2: positional decomposition at the MOMENT OF COMMITMENT,
captured over a full query x key grid so no re-run is needed for new analyses.

v1 averaged attention over ALL generated answer tokens -- that conflates the
decision with the narration of the answer (instructed look/read answers differ
3x in length, inflating cross-method comparisons). v2 records, per layer x head,
attention mass from SIX query sets onto FIVE key spans:

  queries                              keys
  ctrl  : <op>_ctrl trigger positions  img    : image token span
  lastp : final prompt token (emits    cap    : fed-caption span
          the first answer token)      capval : the caption's asserted VALUE
  first : first generated token                token(s) inside the caption
  preval: token emitting the value     ctrlk  : the ctrl positions as KEYS
          (position value_start - 1)            (do decision tokens consult
  value : parsed answer-value span               the word?)
  ans   : all generated tokens         bos    : position 0 (attention sink)

'preval' is the mechanistically causal query: next-token logits for the value
come from the position BEFORE it. 'ctrlk' measures word-as-retrieval-anchor.
'bos' tracks sink mass so source shifts can be traced (conservation).
Span sizes go to records.jsonl for per-token renormalization post-hoc.

Correctness: v1 used all-objects-exact generation parsing (degenerate on
multi-object items -- 1/59 'correct'). v2 stores object_accuracy as a FLOAT
in records and groups files by acc >= 0.5, so item-level attention-vs-
correctness analyses are possible.

Conditions per ctrl/pt run: look, read, neutral, instr_look, instr_read --
token-vs-instruction comparisons at MATCHED query positions.
Gen runs: tagged-block CONTENT and TAG attention onto img/cap.

Output: runs/<run>/attention2/<ds>_<split>_step<STEP>/. npz attn shape
(N, L, H, 30) for ctrl runs, channel names in summary.json; gen: (N, L, H, 8).
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

QUERIES = ("ctrl", "lastp", "first", "preval", "value", "ans")
KEYS = ("img", "cap", "capval", "ctrlk", "bos", "carrier", "prevans")
CH = [f"{q}_{k}" for q in QUERIES for k in KEYS] + [
    # ctrl_style=blocks only: the EMITTED <look>/<read> citation mark and the
    # span it delimits. This is the site the user's question targets -- does the
    # emitted citation word pull attention to its cited source more than a user
    # instruction does, and is the whole block's pattern different?
    "citetag_img", "citetag_cap", "citeblk_img", "citeblk_cap"]   # 34
CH_GEN = ["lookblk_img", "lookblk_cap", "readblk_img", "readblk_cap",
          "looktag_img", "looktag_cap", "readtag_img", "readtag_cap"]


def _find_sub(hay, needle, start=0):
    for i in range(start, len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i, i + len(needle)
    return None


def _tok_spans_in(ids, tokenizer, text, lo, hi):
    """all token spans of `text` variants inside ids[lo:hi]."""
    out = []
    for cand in (text, " " + text, text.capitalize(), " " + text.capitalize()):
        t = tokenizer(cand, add_special_tokens=False)["input_ids"]
        j = lo
        while True:
            sp = _find_sub(ids, t, start=j)
            if not sp or sp[0] >= hi:
                break
            out.extend(range(*sp)); j = sp[1]
    return sorted(set(out))


def mass(attn_layer, q_pos, k_pos):
    if not q_pos or not k_pos:
        return None
    a = attn_layer[:, q_pos][:, :, k_pos]
    return a.sum(-1).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--step", default="last")
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--max_n", type=int, default=40)
    args = ap.parse_args()
    device = "cuda"

    import evaluate as EV
    steps = sorted(int(os.path.basename(p).split("_")[1]) for p in
                   glob.glob(os.path.join(args.run_dir, "ckpt", "step_*")))
    step = steps[-1] if args.step == "last" else int(args.step)
    model, processor, meta, forbid = EV.apply_checkpoint(args.run_dir, step, device)
    arm, n, method = meta["arm"], meta["n_tokens"], meta["method"]
    tokenizer = processor.tokenizer

    ds = load_task(args.data_dir, args.split)
    specs = [s for s in build_specs(ds, args.split, conditions=("conflict",))][:args.max_n]

    INSTR = {"instr_look": 'Answer using ONLY what you can see in the image; ignore the caption.',
             "instr_read": 'Answer using ONLY what the caption states; ignore the image.'}
    if method == "lora":
        ops = ("instr_look", "instr_read", "neutral")
    elif arm == "gen":
        ops = ("gen",)
    else:
        ops = ("look", "read", "neutral", "instr_look", "instr_read")

    img_tok_ids = set()
    for tk in ("image_token", "boi_token"):
        t = getattr(tokenizer, tk, None)
        if t:
            try:
                img_tok_ids.add(tokenizer.convert_tokens_to_ids(t))
            except Exception:
                pass

    groups, gtraj, recs = {}, {}, []
    for s in tqdm(specs, desc="attn2"):
        img = get_image(ds, s)
        for op in ops:
            if op == "gen":
                user = P.gen_template(args.task, 0).format(cap=s.fed_caption)
                out = generate(model, processor, img, user, device, 96)
                acc = float(trigger("look_gen", n) in out)
            else:
                if op == "neutral":
                    from scsrc.prompts import slot_word as _sw
                    user = (f'Caption: "{s.fed_caption}". '
                            f"What is each object's {_sw(args.task)}?")
                elif op.startswith("instr_"):
                    user = f'Caption: "{s.fed_caption}". ' + INSTR[op]
                elif method == "pt":
                    user = trigger(f"pt_{op}", n) + " " + P.pt_template(0).format(cap=s.fed_caption)
                else:
                    user = P.render_ctrl(P.CTRL_CANONICAL, trigger(f"{op}_ctrl", n), s.fed_caption)
                out = generate(model, processor, img, user, device, 32, forbid_ids=forbid)
                pred = Me.parse_pred_task(out, args.task)
                gold = s.image_key if op in ("look", "neutral", "instr_look") else s.text_key
                acc = float(Me.object_accuracy(pred, gold))   # FLOAT, per-object fraction

            msgs = [{"role": "user", "content": [{"type": "image"},
                     {"type": "text", "text": user}]}]
            prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            enc = processor(text=prompt + out, images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                o = model(**enc, output_attentions=True)
            plen = processor(text=prompt, images=img, return_tensors="pt")["input_ids"].shape[1]
            T = enc["input_ids"].shape[1]
            ids = enc["input_ids"][0].tolist()
            img_pos = [i for i, t in enumerate(ids) if t in img_tok_ids]
            cap_ids = tokenizer(s.fed_caption, add_special_tokens=False)["input_ids"]
            csp = _find_sub(ids, cap_ids)
            cap_pos = list(range(*csp)) if csp else []
            ctrl_pos = [i for i, t in enumerate(ids)
                        if t in set(getattr(model, "_neo_all_ids", []) or [])]
            L = len(o.attentions); H = o.attentions[0].shape[1]

            if op == "gen":
                tags = {}
                for gop in ("look", "read"):
                    t_ids = tokenizer(trigger(f"{gop}_gen", n),
                                      add_special_tokens=False)["input_ids"]
                    sp = _find_sub(ids, t_ids, start=plen)
                    if sp:
                        tags[gop] = sp
                bounds = sorted(sp[0] for sp in tags.values()) + [T]
                blk = {g_: list(range(sp[1], min([b for b in bounds if b > sp[1]] or [T])))
                       for g_, sp in tags.items()}
                arr = np.zeros((L, H, len(CH_GEN)), dtype=np.float32)
                qk = [(blk.get("look", []), img_pos), (blk.get("look", []), cap_pos),
                      (blk.get("read", []), img_pos), (blk.get("read", []), cap_pos),
                      (list(range(*tags["look"])) if "look" in tags else [], img_pos),
                      (list(range(*tags["look"])) if "look" in tags else [], cap_pos),
                      (list(range(*tags["read"])) if "read" in tags else [], img_pos),
                      (list(range(*tags["read"])) if "read" in tags else [], cap_pos)]
                sizes = {}
            else:
                pv = list((pred or {}).values())
                val = str(pv[0]).strip() if pv else ""
                vpos = _tok_spans_in(ids, tokenizer, val, plen, T) if val else []
                # caption's ASSERTED value tokens (what <read> should key on)
                capval = []
                if cap_pos:
                    for tv in {str(v).strip().lower() for v in (s.text_key or {}).values()}:
                        capval += _tok_spans_in(ids, tokenizer, tv, cap_pos[0], cap_pos[-1] + 1)
                    capval = sorted(set(capval))
                # emitted citation delimiter + delimited span (blocks arms)
                cite_pos, blk_pos = [], []
                if op in ("look", "read"):
                    c_ids = tokenizer(f"<{op}>", add_special_tokens=False)["input_ids"]
                    hits = []
                    jj = plen
                    while True:
                        sp = _find_sub(ids, c_ids, start=jj)
                        if not sp:
                            break
                        hits.append(sp); jj = sp[1]
                    if hits:
                        cite_pos = [i for sp in hits for i in range(*sp)]
                        # content = between the first pair of delimiters
                        if len(hits) >= 2:
                            blk_pos = list(range(hits[0][1], hits[1][0]))
                        else:
                            blk_pos = list(range(hits[0][1], T))
                # carrier = prompt positions not in img/cap/ctrl/bos: the
                # template + question + (in instr conditions) the ENGLISH
                # INSTRUCTION -- the span the digest comparison was missing.
                # prevans = already-generated answer tokens before the query
                # position (self-conditioning channel).
                _known = set(img_pos) | set(cap_pos) | set(ctrl_pos) | {0}
                carrier_pos = [i for i in range(1, plen) if i not in _known]
                prevans_pos = list(range(plen, max(plen, (vpos[0] - 1)
                                                   if vpos else plen)))
                q_sets = {"ctrl": ctrl_pos, "lastp": [plen - 1],
                          "first": [plen] if plen < T else [],
                          "preval": [vpos[0] - 1] if vpos and vpos[0] - 1 >= 0 else [],
                          "value": vpos, "ans": list(range(plen, T))}
                k_sets = {"img": img_pos, "cap": cap_pos, "capval": capval,
                          "ctrlk": ctrl_pos, "bos": [0],
                          "carrier": carrier_pos, "prevans": prevans_pos}
                arr = np.zeros((L, H, len(CH)), dtype=np.float32)
                qk = [(q_sets[q], k_sets[k]) for q in QUERIES for k in KEYS]
                qk += [(cite_pos, img_pos), (cite_pos, cap_pos),
                       (blk_pos, img_pos), (blk_pos, cap_pos)]
                sizes = {"n_cite": len(cite_pos), "n_blk": len(blk_pos),
                         "n_img": len(img_pos), "n_cap": len(cap_pos),
                         "n_capval": len(capval), "n_ctrl": len(ctrl_pos),
                         "n_carrier": len(carrier_pos), "n_prevans": len(prevans_pos),
                         "n_val": len(vpos), "plen": plen, "T": T}
            # per-DECODING-POSITION trajectory (sustained vs transient gaze):
            # (band mid/late, answer position <=32, key img/cap), nan-padded
            traj = np.full((2, 32, 2), np.nan, dtype=np.float32)
            tcnt = np.zeros(2, dtype=np.float32)
            ans_pos_t = list(range(plen, T))
            bands = (range(L // 3, 2 * L // 3), range(2 * L // 3, L))
            for li, A in enumerate(o.attentions):
                a = A[0].float().cpu().numpy()
                for gi, (q, k) in enumerate(qk):
                    m = mass(a, q, k)
                    if m is not None:
                        arr[li, :, gi] = m
                if ans_pos_t and img_pos:
                    for bi, rng in enumerate(bands):
                        if li in rng:
                            kk = min(len(ans_pos_t), 32)
                            ai = a[:, ans_pos_t[:kk]][:, :, img_pos].sum(-1).mean(0)
                            ac = (a[:, ans_pos_t[:kk]][:, :, cap_pos].sum(-1).mean(0)
                                  if cap_pos else np.zeros(kk, dtype=np.float32))
                            if tcnt[bi] == 0:
                                traj[bi, :kk, :] = 0.0
                            traj[bi, :kk, 0] += ai
                            traj[bi, :kk, 1] += ac
                            tcnt[bi] += 1
            for bi in range(2):
                if tcnt[bi] > 0:
                    traj[bi] /= tcnt[bi]
            del o; torch.cuda.empty_cache()
            key = (op, "correct" if acc >= 0.5 else "incorrect")
            groups.setdefault(key, []).append(arr)
            gtraj.setdefault(key, []).append(traj)
            recs.append({"eid": s.example_id, "op": op, "acc": round(acc, 3),
                         **sizes, "out": out[:120]})

    out_dir = os.path.join(args.run_dir, "attention2",
                           f"{os.path.basename(args.data_dir)}_{args.split}_step{step}")
    os.makedirs(out_dir, exist_ok=True)
    summary = {"channels": CH, "channels_gen": CH_GEN}
    for (op, okk), arrs in groups.items():
        A = np.stack(arrs)
        Tr = np.stack(gtraj[(op, okk)]) if (op, okk) in gtraj else None
        if Tr is not None:
            np.savez(os.path.join(out_dir, f"{op}_{okk}.npz"), attn=A, traj=Tr)
        else:
            np.savez(os.path.join(out_dir, f"{op}_{okk}.npz"), attn=A)
        names = CH_GEN if op == "gen" else CH
        summary[f"{op}_{okk}"] = {"n": int(A.shape[0]),
                                  **{f"{nm}_by_layer": A[..., i].mean((0, 2)).round(4).tolist()
                                     for i, nm in enumerate(names)}}
        if Tr is not None:
            for bi, bn in enumerate(("mid", "late")):
                summary[f"{op}_{okk}"][f"traj_img_by_pos_{bn}"] = \
                    np.round(np.nanmean(Tr[:, bi, :, 0], 0), 4).tolist()
                summary[f"{op}_{okk}"][f"traj_cap_by_pos_{bn}"] = \
                    np.round(np.nanmean(Tr[:, bi, :, 1], 0), 4).tolist()
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"[attn2] wrote {out_dir} ({len(recs)} records)")


if __name__ == "__main__":
    main()
