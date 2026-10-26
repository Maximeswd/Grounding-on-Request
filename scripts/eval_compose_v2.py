#!/usr/bin/env python
"""ONE-PROMPT SOURCE SWITCHING under protocol v2 (2026-10-02, advisor note item 6). One Ground-CF item, both evidence
blocks once (balanced order), the question once, then TWO numbered requests on the same question and a two-line answer:

    Question:
    <q>

    1. <request sentence with control 1>
    2. <request sentence with control 2>

    Return:
    1: <answer>
    2: <answer>

Controls (the SAME carrier sentence for every interface, "Answer according to <control>."):
  word          the neologism tokens;              pt / ptslot    the soft tokens in the same slot (the standard prefix is
                                                                  thereby RELOCATED: a position-shift condition, declared)
  lora (joint)  "Answer using only the image|caption."           instructed (--mode) the same English on the frozen model
  noop (--mode) "Answer the question." twice (no control: the floor)
Both control orders. Contexts: CONFLICT (original item; the two answers must differ) and AGREE (caption edited to the image
value; both answers = the image value). Candidates: the item's conflict-preserving set Y = {image value, caption value, third
value}; the 9 ordered pairs are scored by the summed log-likelihood of the full two-line answer "1: a\\n2: b". Greedy
generation of the same prompt is parsed with a frozen rule (line "1: x" / "2: y"; anything else = failure).
Reported per context: first / second / both correct, copy rate (2nd answer = 1st when the golds differ), derived verdict
(answers differ <-> conflict; balanced accuracy over the two contexts), parse rate; plus a DIRECT verdict prompt
("Do the image and the caption agree ...? A. agree B. conflict") scored over the two options.
Usage: python scripts/eval_compose_v2.py --run_dir runs/<run> --split test_id [--mode instructed|noop --tag _x] [--n_items 200]
       python scripts/eval_compose_v2.py --dry 1 --method neo        (CPU: print the rendered prompts of two items)
Output: runs/<run>/eval/clevr3_<split>/compose_v2_step_<step><tag>.json (+ _items.jsonl.gz)
"""
import argparse, collections, gzip, itertools, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scsrc.clevr3 import (_blocks, _block_order, _edited_item, eval_design, cp_third_value, set_c3_vocab, clevr3_token)
from scsrc.tokens import trigger

CARRIER = "Answer according to {c}."
ENGLISH = {"look": "Answer using only the image.", "read": "Answer using only the caption."}
NOOP = "Answer the question."
RETURN = "Return:\n1: <answer>\n2: <answer>"
ATTR = {"color": "the colour of the queried object", "object": "the shape of the queried object", "count": "the number of shapes",
        "spatial": "the position of the queried object"}
_NUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def control_sentences(method, n, item):
    if method == "neo":
        return {s: CARRIER.format(c=clevr3_token(item, s, n, "srconly")) for s in ("look", "read")}
    if method == "pt":
        return {s: CARRIER.format(c=trigger(f"pt_{s}", n)) for s in ("look", "read")}
    if method in ("lora", "instructed"):
        return dict(ENGLISH)
    return {"look": NOOP, "read": NOOP}


def render(item, order, ctl, seed, verdict=False):
    """User content: evidence blocks (balanced order from eval_design), question, two numbered requests, return format."""
    from PIL import Image
    s2, cap_first = eval_design(item, seed, "explicit")
    blocks = _blocks(item); border = _block_order(s2, cap_first)
    content, buf = [], []
    img = Image.open(item["image"]).convert("RGB") if isinstance(item["image"], str) else item["image"]
    for b in border:
        if b == "look":
            if os.environ.get("SC_GCF_FORMAT") == "v2":
                buf.append("Image:")
            if buf:
                content.append({"type": "text", "text": "\n".join(buf)}); buf = []
            content.append({"type": "image", "image": img})
        elif b in blocks and blocks[b]:
            buf.append(blocks[b])
    if verdict:
        q = (f"Question:\nDo the image and the caption agree about {ATTR.get(item['skill'], 'the queried value')}?\n\n"
             f"A. agree\nB. conflict\n\nReturn only A or B.")
    else:
        q = f"Question:\n{item['question']}\n\n1. {ctl[order[0]]}\n2. {ctl[order[1]]}\n\n{RETURN}"
    buf.append(q); content.append({"type": "text", "text": "\n".join(buf)})
    return content


def parse_pair(text, cands):
    """Frozen rule: the first '1: x' and '2: y' lines; x/y must be candidates (case-insensitive, number words mapped)."""
    out = {}
    for k in ("1", "2"):
        m = re.search(rf"(?m)^\s*{k}\s*[:.)-]\s*(.+?)\s*$", text or "")
        if not m:
            return None, "missing_line"
        w = re.sub(r"[^a-z0-9 ]+", " ", m.group(1).lower()).split()
        w = [_NUM.get(x, x) for x in w]
        hit = [c for c in cands if c.lower() in w]
        if len(hit) != 1:
            return None, ("ambiguous" if len(hit) > 1 else "not_a_candidate")
        out[k] = hit[0]
    return (out["1"], out["2"]), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir"); ap.add_argument("--data_dir", default="data/clevr3"); ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800); ap.add_argument("--mode", default=None, help="instructed | noop (frozen baselines; need --tag)")
    ap.add_argument("--n_items", type=int, default=200, help="conflict items, balanced over the four skills")
    ap.add_argument("--max_new", type=int, default=48); ap.add_argument("--tag", default="")
    ap.add_argument("--dry", type=int, default=0, help="1 = render prompts for two items without a model (CPU)")
    ap.add_argument("--method", default=None, help="dry runs only: neo | pt | lora | instructed | noop")
    a = ap.parse_args()
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode needs --tag")
    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    set_c3_vocab(rows)
    conf = [r for r in rows if str(r["channels"]["look"]) != str(r["channels"]["read"])]
    per = collections.Counter(); items = []
    for r in conf:
        if per[r["skill"]] < max(a.n_items // 4, 1):
            per[r["skill"]] += 1; items.append(r)
    if a.dry:
        os.environ.setdefault("SC_GCF_FORMAT", "v2"); os.environ.setdefault("SC_C3_SOURCES", "look,read"); os.environ.setdefault("SC_CAPTION_STYLE", "fact")
        method = a.method or "neo"; n = 5
        for it in items[:2]:
            ctl = control_sentences(method, n, it); Y = [str(it["channels"]["look"]), str(it["channels"]["read"])]
            w = cp_third_value(it, exclude=set(Y)); Y = Y + ([w] if w else [])
            for order in (("read", "look"), ("look", "read")):
                for cname, item in (("conflict", it), ("agree", _edited_item(it, "read", str(it["channels"]["look"])))):
                    c = render(item, order, ctl, 0)
                    txt = " ".join("[IMAGE]" if x.get("type") == "image" else x.get("text", "") for x in c)
                    print(f"--- {method} {it['id']} order={order} ctx={cname} Y={Y}\n{txt}\n")
            print("--- verdict prompt:\n" + " ".join("[IMAGE]" if x.get("type") == "image" else x.get("text", "") for x in render(it, None, ctl, 0, verdict=True)))
        for t, want in (("1: yellow\n2: blue", ("yellow", "blue")), ("1: Blue.\n2: blue", ("blue", "blue")), ("The answers are 1: red 2: red", None),
                        ("1: six\n2: 4", ("6", "4")), ("1: red\n2: green or red", None)):
            got, lab = parse_pair(t, ("yellow", "blue", "red", "green", "6", "4"))
            assert got == want, (t, got, lab)
        print("parse_pair self-test passed"); return

    import torch
    from evaluate import apply_checkpoint
    from scripts.eval_clevr3 import loglik
    from scripts.eval_clevr3_generate import generate_free
    model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    if forbid is None:
        forbid = getattr(model, "_neo_all_ids", None)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    method = a.mode or cfg.get("method", "neo"); n = meta["n_tokens"]
    print(f"[compose] method={method} n_tokens={n} format={os.environ.get('SC_GCF_FORMAT')} cont={os.environ.get('SC_GCF_CONT')} items={len(items)} {dict(per)}", flush=True)
    recs = []; R = collections.defaultdict(list)
    from tqdm import tqdm
    for k, it in enumerate(tqdm(items, desc="compose")):
        ctl = control_sentences(method, n, it)
        Y = [str(it["channels"]["look"]), str(it["channels"]["read"])]
        w = cp_third_value(it, exclude=set(Y)); Y = tuple(Y + ([w] if w else []))
        pairs = list(itertools.product(Y, Y))
        for cname, item in (("conflict", it), ("agree", _edited_item(it, "read", str(it["channels"]["look"])))):
            gold = {"look": str(item["channels"]["look"]), "read": str(item["channels"]["read"])}
            for order in (("read", "look"), ("look", "read")):
                um = [{"role": "user", "content": render(item, order, ctl, k)}]
                sc = {p: float(loglik(model, proc, um, f"1: {p[0]}\n2: {p[1]}", "cuda")) for p in pairs}
                pick = max(sc, key=sc.get)
                txt, nnew = generate_free(model, proc, um[0]["content"], "cuda", a.max_new, forbid, 8)
                gp, lab = parse_pair(txt, Y)
                g = (gold[order[0]], gold[order[1]])
                rec = {"id": it["id"], "skill": it["skill"], "ctx": cname, "order": list(order), "cands": list(Y), "gold": list(g),
                       "pick": list(pick), "scores": {f"{p[0]}|{p[1]}": round(v, 4) for p, v in sc.items()},
                       "gen_text": txt, "gen_pair": (list(gp) if gp else None), "gen_label": lab, "gen_ntok": nnew,
                       "prompt": " ".join("[IMAGE]" if x.get("type") == "image" else x.get("text", "") for x in um[0]["content"])}
                recs.append(rec)
                for src_, pk in (("fc", pick), ("gen", gp)):
                    ok1 = int(pk is not None and pk[0] == g[0]); ok2 = int(pk is not None and pk[1] == g[1])
                    R[f"{cname}_{src_}_first"].append(ok1); R[f"{cname}_{src_}_second"].append(ok2); R[f"{cname}_{src_}_both"].append(ok1 & ok2)
                    if cname == "conflict":
                        R[f"conflict_{src_}_copy"].append(int(pk is not None and pk[0] == pk[1]))
                    R[f"{cname}_{src_}_derived_correct"].append(int(pk is not None and ((pk[0] != pk[1]) == (cname == "conflict"))))
                R[f"{cname}_gen_parsed"].append(int(gp is not None))
            # direct verdict (one prompt per context, control order irrelevant)
            umv = [{"role": "user", "content": render(item, None, ctl, k, verdict=True)}]
            fmt = os.environ.get("SC_GCF_CONT", "leadin")
            cont = (lambda c: f"Final answer: {c}") if fmt == "final" else (lambda c: c)
            sv = {o: float(loglik(model, proc, umv, cont(o), "cuda")) for o in ("A", "B")}
            vp = max(sv, key=sv.get); R[f"{cname}_verdict_correct"].append(int(vp == ("B" if cname == "conflict" else "A")))
            recs.append({"id": it["id"], "skill": it["skill"], "ctx": cname, "verdict": True, "pick": vp, "scores": sv})
    m = lambda v: round(sum(v) / len(v), 4) if v else None
    S = {k: m(v) for k, v in R.items()}
    for src_ in ("fc", "gen"):
        S[f"derived_balanced_{src_}"] = m([S.get(f"conflict_{src_}_derived_correct") or 0, S.get(f"agree_{src_}_derived_correct") or 0])
    S["verdict_balanced"] = m([S.get("conflict_verdict_correct") or 0, S.get("agree_verdict_correct") or 0])
    od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}"); os.makedirs(od, exist_ok=True)
    out = {"run": os.path.basename(a.run_dir), "method": method, "mode": a.mode, "step": a.step, "split": a.split, "n_items": len(items),
           "per_skill_n": dict(per), "carrier": CARRIER, "english": ENGLISH, "format": os.environ.get("SC_GCF_FORMAT"),
           "cont": os.environ.get("SC_GCF_CONT"), "summary": S, "chance_pair": round(1 / 9, 3)}
    json.dump(out, open(os.path.join(od, f"compose_v2_step_{a.step}{a.tag}.json"), "w"), indent=1)
    with gzip.open(os.path.join(od, f"compose_v2_step_{a.step}{a.tag}_items.jsonl.gz"), "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(S, indent=1)); print("[saved]", os.path.join(od, f"compose_v2_step_{a.step}{a.tag}.json"))


if __name__ == "__main__":
    main()
