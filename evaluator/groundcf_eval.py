#!/usr/bin/env python
"""groundcf-eval: measure a model's SOURCE DEPENDENCE on Ground-CF.

Training-code-free. Three subcommands:

  requests   materialize the evaluation contexts as a JSONL of requests
             (prompt text + image path + candidate answers). Run YOUR model
             (any API, any framework) over them, one output per request.
  score      score your answers.jsonl against gold: routing (acc), FLIP,
             STAY per channel, with Wilson 95% intervals and failure
             categories (unmappable / ambiguous).
  hf         run both steps directly for a HuggingFace model id using
             teacher-forced likelihood over the candidate answers
             (no generation, no parsing noise).

Construction (mirrors the paper's battery exactly):
  read / ocr : channel-edit twins -- the caption or on-screen-text block
               asserts a different value attested by another channel.
               FLIP = answer tracks the edit to the requested channel;
               STAY = answer unchanged when the OTHER text channel is edited.
  look       : image-swap pairs (pairs_look_*.jsonl) with exact twin renders.
               FLIP = answer tracks the image swap.
Chance for an evidence-blind chooser: .25 per flip; routing chance 1/3.
Report FLIP and STAY separately; JOINT = min(FLIP, STAY) is the aggregate.

Examples:
  python groundcf_eval.py requests --data ../groundcf --split test_id -o req.jsonl
  python groundcf_eval.py score    --requests req.jsonl --answers ans.jsonl
  python groundcf_eval.py hf       --data ../groundcf --split test_id \
                                   --model google/gemma-3-4b-it
"""
import argparse, json, math, os, re, sys

INSTR = {"look": "Answer using only the image.",
         "read": "Answer using only the caption.",
         "ocr": "Answer using only the on-screen text."}


def wilson(p, n, z=1.96):
    if not n:
        return None
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def blocks_text(caption, ocr_text):
    return f'Caption: "{caption}"\nOn-screen text: "{ocr_text}"'


def prompt_of(item, source):
    return (f"{blocks_text(item['caption'], item['ocr_text'])}\n"
            f"{item['question']} {INSTR[source]}")


def load_items(data, split):
    return [json.loads(l) for l in open(os.path.join(data, f"{split}.jsonl"))]


def alt_of(ch, source):
    gold = str(ch[source])
    return next((str(v) for k, v in ch.items() if k != source and str(v) != gold), None)


def edited(item, channel, new_value):
    it = dict(item)
    old = str(item["channels"][channel])
    if channel == "read":
        it["caption"] = item["caption"].replace(old, str(new_value))
    else:
        it["ocr_text"] = item["ocr_text"].replace(old, str(new_value))
    return it


def build_requests(a):
    items = load_items(a.data, a.split)
    reqs = []

    def add(rid, item, source, cands, image=None):
        reqs.append({"request_id": rid, "channel": source,
                     "image": image or item["image"],
                     "prompt": prompt_of(item, source),
                     "candidates": list(cands)})
    n = 0
    for it in items:
        if a.max_n and n >= a.max_n:
            break
        ch = {k: str(v) for k, v in it["channels"].items()}
        used = False
        for c in ("read", "ocr"):
            alt = alt_of(ch, c)
            if alt is None:
                continue
            used = True
            cands = (ch[c], alt)
            other = "ocr" if c == "read" else "read"
            add(f"{it['id']}|{c}|orig", it, c, cands)
            add(f"{it['id']}|{c}|edit_self", edited(it, c, alt), c, cands)
            if alt_of(ch, other) is not None:
                add(f"{it['id']}|{c}|edit_other",
                    edited(it, other, alt_of(ch, other)), c, cands)
        if used:
            n += 1
    # look: image-swap pairs
    pfile = os.path.join(a.data, f"pairs_look_{'id' if 'id' in a.split else 'ood'}.jsonl")
    if os.path.exists(pfile):
        n_pairs = 0
        for p in (json.loads(l) for l in open(pfile)):
            if a.max_n and n_pairs >= a.max_n:
                break
            n_pairs += 1
            cands = (str(p["gold_a"]), str(p["gold_b"]))
            base = {"caption": f'The answer to the question is {p["read_value"]}.',
                    "ocr_text": f'VALUE={p["ocr_value"]}',
                    "question": p["question"], "image": None, "channels": {}}
            add(f"{p['id']}|look|orig", base, "look", cands,
                image=p["ctx_a"]["image"])
            add(f"{p['id']}|look|edit_self", base, "look", cands,
                image=p["ctx_b"]["image"])
    out = a.out or "requests.jsonl"
    with open(out, "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    print(f"[requests] {len(reqs)} requests -> {out}")


def parse_answer(text, cands):
    t = " " + re.sub(r"[^a-z0-9 ]", " ", str(text).lower()) + " "
    pos = {c: t.rfind(" " + re.sub(r"[^a-z0-9 ]", " ", c.lower()) + " ")
           for c in cands}
    hit = [c for c, p in pos.items() if p >= 0]
    if not hit:
        return None, "unmappable"
    if len(hit) == 2:
        return max(hit, key=lambda c: pos[c]), "ambiguous"
    return hit[0], "clean"


def score(a):
    reqs = {r["request_id"]: r
            for r in (json.loads(l) for l in open(a.requests))}
    ans = {r["request_id"]: r["output"]
           for r in (json.loads(l) for l in open(a.answers))}
    picks, quality = {}, {"unmappable": 0, "ambiguous": 0, "n": 0}
    for rid, r in reqs.items():
        if rid not in ans:
            continue
        v, tag = parse_answer(ans[rid], r["candidates"])
        quality["n"] += 1
        if tag != "clean":
            quality[tag] += 1
        picks[rid] = v
    agg = {}
    for ch in ("look", "read", "ocr"):
        ids = sorted({rid.split("|")[0] for rid in picks
                      if rid.split("|")[1] == ch})
        acc, flip, stay = [], [], []
        for i in ids:
            p0 = picks.get(f"{i}|{ch}|orig")
            p1 = picks.get(f"{i}|{ch}|edit_self")
            p2 = picks.get(f"{i}|{ch}|edit_other")
            g = reqs.get(f"{i}|{ch}|orig", {}).get("candidates", [None])[0]
            if p0 is not None:
                acc.append(int(p0 == g))
                if p1 is not None:
                    flip.append(int(p0 == g and p1 != p0))
                if p2 is not None:
                    stay.append(int(p2 == p0))
        m = lambda v: round(sum(v) / len(v), 3) if v else None
        agg[ch] = {"routing": m(acc), "flip": m(flip), "stay": m(stay),
                   "joint": (round(min(m(flip), m(stay)), 3)
                             if flip and stay else None),
                   "n": len(acc),
                   "flip_ci": wilson(m(flip), len(flip)) if flip else None,
                   "stay_ci": wilson(m(stay), len(stay)) if stay else None}
    out = {"channels": agg,
           "unmappable_rate": round(quality["unmappable"] / max(1, quality["n"]), 3),
           "ambiguous_rate": round(quality["ambiguous"] / max(1, quality["n"]), 3),
           "note": ("flip chance for an evidence-blind chooser = .25; "
                    "report flip and stay separately; joint = min")}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def hf(a):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).to("cuda").eval()

    tmp_req = a.out + ".requests.jsonl" if a.out else "hf_requests.jsonl"
    a.out2, a.out = a.out, tmp_req
    ar = argparse.Namespace(data=a.data, split=a.split, out=tmp_req,
                            max_n=a.max_n)
    build_requests(ar)

    def loglik(image, prompt, answer):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": Image.open(image).convert("RGB")},
            {"type": "text", "text": prompt}]}]
        full = proc.apply_chat_template(
            msgs + [{"role": "assistant", "content": [
                {"type": "text", "text": answer}]}],
            tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        pre = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        n_pre = pre["input_ids"].shape[1]
        with torch.no_grad():
            logits = model(**full).logits[0]
        ids = full["input_ids"][0]
        lp = torch.log_softmax(logits[:-1].float(), -1)
        tail = range(n_pre - 1, len(ids) - 1)
        return sum(lp[t, ids[t + 1]].item() for t in tail) / max(1, len(ids) - n_pre)

    answers = []
    for line in open(tmp_req):
        r = json.loads(line)
        sc = {c: loglik(r["image"], r["prompt"], c) for c in r["candidates"]}
        answers.append({"request_id": r["request_id"],
                        "output": max(sc, key=sc.get)})
    ans_path = (a.out2 or "hf") + ".answers.jsonl"
    with open(ans_path, "w") as f:
        for x in answers:
            f.write(json.dumps(x) + "\n")
    score(argparse.Namespace(requests=tmp_req, answers=ans_path, out=a.out2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("requests")
    r.add_argument("--data", default="../groundcf")
    r.add_argument("--split", default="test_id")
    r.add_argument("--max_n", type=int, default=0)
    r.add_argument("-o", "--out", default="requests.jsonl")
    s = sub.add_parser("score")
    s.add_argument("--requests", required=True)
    s.add_argument("--answers", required=True)
    s.add_argument("-o", "--out", default=None)
    h = sub.add_parser("hf")
    h.add_argument("--model", required=True)
    h.add_argument("--data", default="../groundcf")
    h.add_argument("--split", default="test_id")
    h.add_argument("--max_n", type=int, default=0)
    h.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    {"requests": build_requests, "score": score, "hf": hf}[a.cmd](a)


if __name__ == "__main__":
    main()
