#!/usr/bin/env python
"""E5 v2 -- ONE-PROMPT COMPOSITION with the learned requests themselves (v1.188, 2026-09-25, author). No modality is
named, except by English, which names it by nature. One Ground-CF item: its image and caption once (balanced order,
eval_design), then the SAME held-out question twice, each in the word's TRAINING carrier (NEO_TEMPLATES; the same
carrier for both halves, rotated by item), once with the image request and once with the caption request; both orders.
  word      "<carrier with <look> and q>\n\n<carrier with <read> and q>"
  prefix    the source's soft tokens immediately before each question (local placement)
  English   "<q> According to the image (and only the image), what is the answer?" -- the English row's sentence
            (INSTR_VARIANT=5); a joint LoRA gets the same sentence with its adapter loaded
  none      (--mode noop) the question twice, no request: the baseline
  a specialist LoRA (one adapter per request) is not applicable: adapter selection is global state.
E5a ANSWER PAIR: the 9 pairs over {image value, caption value, a third value} scored (teacher-forced turn log-lik) as the
  familiar training continuation twice, "<C3_CONT v1>\n<C3_CONT v2>". composed = both answers right; on CONFLICT
  contexts a correct pair differs, so the derived conflict call is "the two answers differ"; on AGREE contexts (the
  caption edited to the image value) the gold pair is (image value, image value). Chance for a correct pair: 1/9.
E5b VERDICT: the same prompt + "Do the answers agree?\nA. agree\nB. conflict"; "A. agree" vs "B. conflict" scored in the
  training continuation ("... final answer: B. conflict") and bare. Conflict recall (conflict contexts), agree recall
  (agree contexts), balanced = their mean (.5 for any constant policy).
--gen_n N also greedy-generates the E5a and E5b prompts of the first N items (raw text kept for the outputs page).
Usage: python scripts/eval_compose_pair.py --run_dir runs/clevr3_2s_1tok_ctm_cp_fc_seed0 --split test_id --tag _v2
Output: runs/<run>/eval/clevr3_<split>/compose_pair_step_<step><tag>.json (+ _items.jsonl)
"""
import argparse, collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from tqdm import tqdm
from scsrc.clevr3 import (_blocks, _edited_item, _eval_rank, clevr3_token, heldout_question, cp_third_value, set_c3_vocab,
                          eval_design, NEO_TEMPLATES, C3_CONT)
from scsrc.tokens import trigger

VERDICT_Q = "Do the answers agree?\nA. agree\nB. conflict"
VERDICTS = {"agree": "A. agree", "conflict": "B. conflict"}


def requests_text(q, item, order, control, carrier):
    """The two requests on the same question, in `order` (a pair of sources)."""
    parts = []
    for src in order:
        kind, ctl = control(item, src)
        if kind == "word":
            parts.append(carrier.format(q=q, neo=ctl))
        elif kind == "prefix":
            parts.append(f"{ctl} {q}")
        elif kind == "english":
            parts.append(f"{q} {ctl}")
        else:
            parts.append(q)
    return "\n\n".join(parts)


def content_for(item, text, cap_first):
    img = Image.open(item["image"]).convert("RGB")
    cap = _blocks(item).get("read")
    if cap_first:
        return [{"type": "text", "text": cap}, {"type": "image", "image": img}, {"type": "text", "text": text}]
    return [{"type": "image", "image": img}, {"type": "text", "text": f"{cap}\n{text}"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id"); ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--per_skill", type=int, default=50, help="items per attribute (lowest eval ranks: balanced order)")
    ap.add_argument("--mode", default=None, help="None (the run's own request) | instructed | noop")
    ap.add_argument("--tag", default="")
    ap.add_argument("--gen_n", type=int, default=30); ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--dry_run", type=int, default=0, help="CPU: print the prompts of the first N items, load no model")
    a = ap.parse_args()
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit("[STOP] --mode needs --tag")
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    od = os.path.join(a.run_dir, "eval", f"clevr3_{a.split}"); os.makedirs(od, exist_ok=True)
    out_json = os.path.join(od, f"compose_pair_step_{a.step}{a.tag}.json")
    method = cfg.get("method", "neo")
    joint_lora = method == "lora" and cfg.get("factor_mode") == "instructed" and not cfg.get("train_sources")
    if a.mode is None and method == "lora" and not joint_lora:
        json.dump({"run": os.path.basename(a.run_dir), "applicable": False,
                   "note": "LoRA with one adapter per request: adapter selection is global model state, so the two "
                           "requests of one prompt cannot be given different adapters."}, open(out_json, "w"), indent=1)
        print(f"[saved] {out_json} (not applicable for a per-request LoRA adapter)"); return
    if a.dry_run:
        from scsrc.clevr3 import c3_sources  # noqa: F401  (env restored below as apply_checkpoint would)
        for k, v in (("SC_NEO_TEMPLATE", cfg.get("neo_template") or "bare"), ("SC_CAPTION_STYLE", cfg.get("caption_style") or "template"),
                     ("SC_C3_SOURCES", cfg.get("c3_sources") or "look,read"), ("SC_CP_PAIRS", "1")):
            os.environ.setdefault(k, v)
        if os.environ.get("SC_CAPTION_STYLE_OVERRIDE"):
            os.environ["SC_CAPTION_STYLE"] = os.environ["SC_CAPTION_STYLE_OVERRIDE"]
        model = proc = forbid = None; n = cfg.get("n_tokens", 1)
    else:
        from evaluate import apply_checkpoint
        from scripts.eval_clevr3_generate import loglik_both, forbid_ids_processor
        from transformers import LogitsProcessorList
        import torch
        model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
        if forbid is None:
            forbid = getattr(model, "_neo_all_ids", None)
        n = meta["n_tokens"]
    mode = a.mode or cfg.get("factor_mode", "srconly")
    kind = ("none" if a.mode == "noop" else "english" if (a.mode == "instructed" or joint_lora)
            else "prefix" if method == "pt" else "word")

    def control(item, src):
        if kind == "none":
            return "none", ""
        if kind == "english":
            return "english", clevr3_token(item, src, n, "instructed")
        if kind == "prefix":
            return "prefix", trigger(f"pt_{src}", n)
        return "word", clevr3_token(item, src, n, mode)

    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    set_c3_vocab(rows)
    items = sorted([r for r in rows if str(r["channels"]["look"]) != str(r["channels"]["read"]) and _eval_rank(r) < a.per_skill],
                   key=lambda r: (r["skill"], _eval_rank(r)))
    orders = [("look", "read"), ("read", "look")]

    def score(um, cont):
        return loglik_both(model, proc, um, cont, "cuda")[0]

    def gen(content):
        enc = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        plen = enc["input_ids"].shape[1]
        extra = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask", "token_type_ids")}
        lp = LogitsProcessorList([forbid_ids_processor(forbid)]) if forbid else None
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=a.max_new,
                                 do_sample=False, logits_processor=lp, **extra)
        return proc.tokenizer.decode(out[0, plen:], skip_special_tokens=True).strip()

    R = collections.defaultdict(list)
    fout = None if a.dry_run else open(out_json.replace(".json", "_items.jsonl"), "w")
    for i, item in enumerate(tqdm(items[: a.dry_run] if a.dry_run else items, desc="compose_pair")):
        s2, cap_first = eval_design(item, 0, "heldout")
        q = heldout_question(item, s2)
        carrier = NEO_TEMPLATES[(_eval_rank(item) // 2) % len(NEO_TEMPLATES)]
        A, B = str(item["channels"]["look"]), str(item["channels"]["read"])
        W = cp_third_value(item, exclude={A, B}, salt="switch") or A
        cands = [A, B] + ([W] if W not in (A, B) else [])
        ctxs = {"conflict": item, "agree": _edited_item(item, "read", A)}
        for cname, it in ctxs.items():
            for order in orders:
                req = requests_text(q, it, order, control, carrier)
                c_a = content_for(it, req, cap_first)
                c_b = content_for(it, f"{req}\n\n{VERDICT_Q}", cap_first)
                gold = tuple(str(it["channels"][s]) for s in order)
                if a.dry_run:
                    print(f"\n==== {item['id']} {item['skill']} {cname} order={order} cap_first={cap_first} gold={gold} cands={cands}")
                    print("[E5a user]", " | ".join("<IMAGE>" if c["type"] == "image" else c["text"] for c in c_a))
                    print("[E5a cont]", "\n".join(C3_CONT.format(v=v) for v in gold))
                    print("[E5b tail]", VERDICT_Q.replace("\n", " / "))
                    continue
                um_a = [{"role": "user", "content": c_a}]; um_b = [{"role": "user", "content": c_b}]
                sc = {f"{v1}|{v2}": score(um_a, f"{C3_CONT.format(v=v1)}\n{C3_CONT.format(v=v2)}") for v1 in cands for v2 in cands}
                best = max(sc, key=sc.get); p = tuple(best.split("|"))
                vs_lead = {k: score(um_b, C3_CONT.format(v=t)) for k, t in VERDICTS.items()}
                vs_bare = {k: score(um_b, t) for k, t in VERDICTS.items()}
                v_lead = max(vs_lead, key=vs_lead.get); v_bare = max(vs_bare, key=vs_bare.get)
                okp = int(p == gold); differ = int(p[0] != p[1])
                key = f"{cname}|{'>'.join(order)}"
                R[key + "|pair"].append(okp); R[key + "|q1"].append(int(p[0] == gold[0])); R[key + "|q2"].append(int(p[1] == gold[1]))
                R[key + "|differ"].append(differ)
                R[key + "|v_lead_ok"].append(int(v_lead == cname)); R[key + "|v_bare_ok"].append(int(v_bare == cname))
                R[f"{key}|{item['skill']}|pair"].append(okp)
                g = {}
                if i < a.gen_n:
                    g = {"gen_E5a": gen(c_a), "gen_E5b": gen(c_b)}
                    fa = re.findall(r"final answer:\s*([^\n.;]+)", g["gen_E5a"], flags=re.I)
                    g["gen_E5a_answers"] = [x.strip() for x in fa[:2]]
                fout.write(json.dumps({
                    "run": os.path.basename(a.run_dir), "model": cfg.get("model_id"), "dataset": "Ground-CF", "split": a.split,
                    "method": kind if kind != "english" or not joint_lora else "joint_lora", "eval_type": "compose_pair",
                    "id": item["id"], "skill": item["skill"], "context": cname, "order": list(order), "cap_first": cap_first,
                    "prompt_text": " | ".join("<IMAGE>" if c["type"] == "image" else c["text"] for c in c_a),
                    "verdict_question": VERDICT_Q, "candidates": cands, "gold_pair": list(gold), "pair_scores": {k: round(v, 4) for k, v in sc.items()},
                    "pred_pair": list(p), "pair_correct": okp, "answers_differ": differ,
                    "verdict_scores_lead": {k: round(v, 4) for k, v in vs_lead.items()},
                    "verdict_scores_bare": {k: round(v, 4) for k, v in vs_bare.items()},
                    "verdict_lead": v_lead, "verdict_bare": v_bare, "verdict_gold": cname, **g}) + "\n")
    if a.dry_run:
        return
    fout.close()
    m = lambda k: round(sum(R[k]) / max(len(R[k]), 1), 3)
    mo = lambda c, f: round(sum(m(f"{c}|{'>'.join(o)}|{f}") for o in orders) / 2, 3)
    summ = {"per_order": {k: m(k) for k in sorted(R) if k.count("|") == 2},
            "per_skill_pair": {k: m(k) for k in sorted(R) if k.count("|") == 3}}
    e5a = {"composed_conflict": mo("conflict", "pair"), "composed_agree": mo("agree", "pair"),
           "conflict_recall_derived": mo("conflict", "differ"), "agree_recall_derived": round(1 - mo("agree", "differ"), 3)}
    e5a["balanced_derived"] = round((e5a["conflict_recall_derived"] + e5a["agree_recall_derived"]) / 2, 3)
    e5b = {}
    for f, nm in (("v_lead_ok", "lead"), ("v_bare_ok", "bare")):
        cr, ar = mo("conflict", f), mo("agree", f)
        e5b[nm] = {"conflict_recall": cr, "agree_recall": ar, "balanced": round((cr + ar) / 2, 3)}
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step, "mode": a.mode, "method": method, "request": kind,
           "joint_lora": joint_lora, "applicable": True, "n_items": len(items), "n_per_cell": len(R["conflict|look>read|pair"]),
           "chance": {"pair": round(1 / 9, 3), "balanced": 0.5}, "E5a": e5a, "E5b": e5b, "detail": summ,
           "caption_style": os.environ.get("SC_CAPTION_STYLE", "template"),
           "note": "one Ground-CF item per prompt; the same held-out question twice, each in the word's training carrier with "
                   "its request (prefix: local soft tokens; English: the English row's sentence; none: no request); both orders; "
                   "conflict and agree contexts; E5a candidate-scored answer pair (9 pairs, familiar continuation twice), E5b "
                   "candidate-scored agree/conflict verdict"}
    json.dump(out, open(out_json, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("run", "split", "request", "n_items", "E5a", "E5b")}, indent=1)); print("[saved]", out_json)


if __name__ == "__main__":
    main()
