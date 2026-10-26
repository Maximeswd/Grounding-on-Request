#!/usr/bin/env python
"""First MVG number: channel-routing of the stage-1 tokens via LL forced-choice.

Each MVG item is one (video, role, channel) claim; on conflict items the item's
OWN channel was source-perturbed so gold = the (current) channel value and
answer_raw = the original (distractor no longer present in any source). We score
gold vs answer_raw under the item's MATCHED token (<desc>/<ocr>/<vis>) and under
a MISMATCHED token. A channel-selective token routes to gold when matched and
away when mismatched.

  matched_acc[c]   = P(<c> prefers the channel's gold)  -- routing to channel c
  mismatched_acc[c]= P(non-c token prefers gold)        -- lower = selective
  selectivity[c]   = matched - mismatched
Multi-block MVG prompt (3 sources + frame, perm6 order) scored with a local
loglik (score.loglik assumes single image+text). Primary = LL forced-choice.
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from tqdm import tqdm
from scsrc.collate import _encode, _prompt_len, _user_content
from scsrc.tokens import trigger
from scsrc.mvg import mvg_word, assert_words_registered
from evaluate import apply_checkpoint

_PERMS = list(itertools.permutations(("desc", "ocr", "vis")))


def blocks(item):
    return {"desc": f'Description: "{item["desc_text"]}"',
            "ocr": ("On-screen text: " + " | ".join(f'"{s}"' for s in item["ocr_strings"])
                    if item["ocr_strings"] else "On-screen text: (none)"),
            "vis": None}


def user_content(item, op_trig):
    b = blocks(item); order = _PERMS[item["perm6"] % 6]
    img = Image.open(item["vis_frame"]).convert("RGB")
    content, buf = [], []
    for x in order:
        if x == "vis":
            if buf:
                content.append({"type": "text", "text": "\n".join(buf)}); buf = []
            content.append({"type": "image", "image": img})
        else:
            buf.append(b[x])
    q = (f'{item["role_question"]} Give me a {op_trig} answer.'
         if item["role_question"] else f"Give me a {op_trig} answer.")
    buf.append(q)
    content.append({"type": "text", "text": "\n".join(buf)})
    return [{"role": "user", "content": content}]


@torch.no_grad()
def loglik_mvg(model, proc, umsg, candidate, device, with_n=False, tokens=False):
    fmsg = umsg + [{"role": "assistant", "content": [{"type": "text", "text": candidate}]}]
    enc = _encode(proc, fmsg, device); plen = _prompt_len(proc, umsg, device)
    ids = enc["input_ids"]; n = ids.shape[1] - plen
    if n <= 0:
        return -1e9
    kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    lg = model(**kw, use_cache=False).logits.float()
    tgt = ids[0, plen:]
    lp = torch.log_softmax(lg[0, plen - 1:ids.shape[1] - 1, :], -1)
    tl = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    total = float(tl.sum())
    if tokens:   # v1.142: ids + per-token log-probs of the whole assistant turn
        return total, int(n), tgt.tolist(), [round(float(x), 4) for x in tl]
    return (total, int(n)) if with_n else total   # v1.141: token count for the per-token mean


def cand(v):
    return f"I used the requested source to answer the question. final answer: {v}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--data_dir", default="data/mvg")
    ap.add_argument("--max_n", type=int, default=200)
    a = ap.parse_args()

    model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    items = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    items = [it for it in items if it["cond"] == "conflict"
             and str(it["gold"]).strip().lower() != str(it["answer_raw"]).strip().lower()][:a.max_n]
    ch = {"DESC": "desc_ctrl", "OCR": "ocr_ctrl", "VIS": "vis_ctrl"}
    from collections import defaultdict
    matched = defaultdict(list); mism = defaultdict(list)
    for it in tqdm(items, desc="mvg"):
        gold = str(it["gold"]).strip().lower(); dist = str(it["answer_raw"]).strip().lower()
        c = it["channel"]
        # matched token
        um = user_content(it, mvg_word(ch[c], n))
        lg = loglik_mvg(model, proc, um, cand(gold), "cuda")
        ld = loglik_mvg(model, proc, um, cand(dist), "cuda")
        matched[c].append(int(lg > ld))
        # mismatched token (a different channel)
        other = "VIS" if c != "VIS" else "DESC"
        uo = user_content(it, mvg_word(ch[other], n))
        og = loglik_mvg(model, proc, uo, cand(gold), "cuda")
        od = loglik_mvg(model, proc, uo, cand(dist), "cuda")
        mism[c].append(int(og > od))

    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "n_tokens": n, "n": len(items)}
    for c in ("DESC", "OCR", "VIS"):
        out[f"matched_{c}"] = m(matched[c]); out[f"mismatched_{c}"] = m(mism[c])
        out[f"selectivity_{c}"] = round(m(matched[c]) - m(mism[c]), 3)
        out[f"n_{c}"] = len(matched[c])
    out["matched_overall"] = m([x for c in matched for x in matched[c]])
    print(json.dumps(out, indent=1))
    outd = os.path.join(a.run_dir, "eval", f"mvg_{a.split}")
    os.makedirs(outd, exist_ok=True)
    json.dump(out, open(os.path.join(outd, f"routing_step_{a.step}.json"), "w"), indent=1)
    print(f"[saved] {outd}/routing_step_{a.step}.json")


if __name__ == "__main__":
    main()
