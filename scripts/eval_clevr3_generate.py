#!/usr/bin/env python
"""Ground-CF OPEN-GENERATION check (2026-09-22, STATUS v1.85).

The paper's Flip/Stay numbers come from a forced choice: summed sequence log-likelihood of
"... final answer: <v>" over the candidate values of the sources. Reviewers ask whether a model
that PICKS the right candidate also SAYS it when it is left to generate freely. This script
answers that on the same items, same prompts and same contexts as scripts/eval_clevr3_flip.py:

  per twin pair (four-skill file pairs_look_all_<split>.jsonl) and per requested source
    base  : the original context
    flip  : the requested source edited      (look: exact image twin; read: caption edit)
    stay  : the competing source edited      (look: caption edit;     read: OCR edit)
  in every context we record BOTH
    the forced-choice pick (sum log-lik, plus the mean-normalised pick for the record) and
    the greedy free generation (stop shortly after "final answer:", new-token rows forbidden
    exactly as in training-time sampling), parsed with a closed-vocabulary parser.

Reported: Accuracy/Flip/Stay under generation next to the same under the forced choice, the
item-level agreement of the two (per context and on the Flip bit, with Cohen's kappa), the
parse-label histogram (candidate / other / abstain / ambiguous / empty) and the share of
candidate sets with unequal token counts (length correction question). Raw generations are
kept in `records` for the LLM-judge sample (scripts/label_gen_sample.py).

Ground-CF values are 1 token (colour, shape, spatial) or 2 tokens (digits; the space is its own
token) and every candidate set is within one skill, so sum and mean pick the same candidate;
the script still records both.

Output: <run>/eval/clevr3_<split>/gen_step_<step><tag>[_<dtype>].json  (or --out_dir)
"""
import argparse, json, os, re, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from tqdm import tqdm
from transformers import StoppingCriteriaList, LogitsProcessorList
from scsrc.clevr3 import clevr3_user_content, _edited_item, set_c3_vocab, cp_text_edit_value, cp_fix_look_collision, eval_design
from scsrc.tokens import c3_sources
from scsrc.collate import _encode, _prompt_len
from scsrc.infer import StopAfterAnchor
from scsrc.model import forbid_ids_processor
from evaluate import apply_checkpoint
from scripts.eval_clevr3 import cand, loglik, loglik_batch

NUMWORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
            "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ABSTAIN_MARKERS = ("not visible", "not stated", "not shown", "not mentioned", "cannot", "can't",
                   "unable", "unknown", "no information", "not possible", "does not say",
                   "doesn't say", "not specified", "impossible to")
ANCHOR = "final answer:"


def parse_answer(text, cands):
    """Closed-vocabulary parse. Returns (label, value).
    label: 'cand' (value = one candidate), 'ambiguous' (several candidates, none leading),
           'abstain', 'other' (text without any candidate), 'empty'."""
    if not text or not text.strip():
        return "empty", None
    low = text.lower()
    seg = low.rsplit(ANCHOR, 1)[1] if ANCHOR in low else low
    seg = re.sub(r"[^a-z0-9 ]+", " ", seg)
    words = [NUMWORDS.get(w, w) for w in seg.split()]
    if not words:
        return "empty", None
    cl = [c.lower() for c in cands]
    present = [c for c in cl if c in words]
    if len(present) == 1:
        return "cand", cands[cl.index(present[0])]
    if len(present) > 1:
        # a structured tail leads with the value ("final answer: red"); take the first word if
        # it is a candidate, otherwise the answer names several candidates and is ambiguous
        if words[0] in cl:
            return "cand", cands[cl.index(words[0])]
        return "ambiguous", None
    if any(m in low for m in ABSTAIN_MARKERS):
        return "abstain", None
    return "other", None


VERDICT_WORDS = {"agree": "agree", "agrees": "agree", "match": "agree", "matches": "agree", "same": "agree",
                 "consistent": "agree", "conflict": "conflict", "conflicts": "conflict", "disagree": "conflict",
                 "disagrees": "conflict", "differ": "conflict", "different": "conflict", "mismatch": "conflict"}
FORMAT_HINT = ' End your reply with "final answer: <answer>".'
FORMAT_HINT_COMPOSE = ' End your reply with "final answer: agree" or "final answer: conflict".'
# v1.131 (author): MULTIPLE CHOICE. The item's candidate set, lettered in a per-item order fixed across contexts and
# requests (so a Flip is a change of letter), appended after the question; the forced choice scores the letters.
MC_INSTR = "\nOptions:\n{opts}\nAnswer with the letter of the correct option."
FORMAT_HINT_MC = ' End your reply with "final answer: <letter>".'


def mc_options(item_id, cands):
    import random as _r, zlib as _z
    order = list(cands); _r.Random(_z.crc32(str(item_id).encode())).shuffle(order)
    return dict(zip("ABCDEFG", order))


def parse_mc(text, opts):
    """(label, value, route). route: 'anchor' (letter right after the last "final answer:"), 'letter' (a lone option
    letter such as "B", "(B)", "B." at the start or after "answer"/"option"), 'value' (no letter; the option's value
    named instead, closed-vocabulary parse), else label 'other'/'empty'."""
    if not text or not text.strip():
        return "empty", None, None
    L = "".join(opts)
    low = text.lower()
    if ANCHOR in low:
        seg = text[low.rfind(ANCHOR) + len(ANCHOR):].strip()
        m = re.match(rf"^\(?([{L}])\b", seg)
        if m:
            return "cand", opts[m.group(1)], "anchor"
    m = (re.match(rf"^\s*\(?([{L}])(?:[\.\):]|\s*$)", text)            # "A." / "(A)" / "A" -- not the article "A red"
         or re.search(rf"\b(?:answer|option)(?:\s+is)?\s*:?\s*\(?([{L}])\b", text, flags=re.I))
    if m:
        return "cand", opts[m.group(1).upper()], "letter"
    lab, v = parse_answer(text, tuple(opts.values()))
    return lab, v, ("value" if lab == "cand" else None)


from scripts.detect_two_prompts import detect_two_prompts   # noqa: E402  (v1.131; torch-free module)


def _first_in_order(text, cands, k):
    """The first k DISTINCT candidates in order of appearance (word-level, number words mapped)."""
    low = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    words = [NUMWORDS.get(w, w) for w in low.split()]
    cl = [c.lower() for c in cands]; out = []
    for w in words:
        if w in cl and cands[cl.index(w)] not in out:
            out.append(cands[cl.index(w)])
        if len(out) == k:
            break
    return out


def parse_pair(text, c1, c2):
    """Two answers "a1; a2" (within-prompt switching). Returns (label, v1, v2); label 'pair' when both parse.
    Split on the first ';' of the tail after the anchor (if any); otherwise the first candidate of each set in order."""
    if not text or not text.strip():
        return "empty", None, None
    low = text.lower()
    seg = text[low.rfind(ANCHOR) + len(ANCHOR):] if ANCHOR in low else text
    if ";" in seg:
        a, b = seg.split(";", 1)
        la, va = parse_answer(a, c1); lb, vb = parse_answer(b, c2)
        if la == "cand" and lb == "cand":
            return "pair", va, vb
    words = [NUMWORDS.get(w, w) for w in re.sub(r"[^a-z0-9 ]+", " ", seg.lower()).split()]
    l1, l2 = [c.lower() for c in c1], [c.lower() for c in c2]
    i1 = next((i for i, w in enumerate(words) if w in l1), None)          # first answer: first Q1 candidate
    i2 = (next((i for i, w in enumerate(words) if i > i1 and w in l2), None) if i1 is not None
          else next((i for i, w in enumerate(words) if w in l2), None))  # second: first Q2 candidate after it
    got1 = [c1[l1.index(words[i1])]] if i1 is not None else []
    got2 = [c2[l2.index(words[i2])]] if i2 is not None else []
    if got1 and got2:
        return "pair", got1[0], got2[0]
    return ("partial" if (got1 or got2) else "other"), (got1[0] if got1 else None), (got2[0] if got2 else None)


def parse_compose(text, cands):
    """Composed answer: "From the first requested source, the answer is v1. From the second ..., v2. The requested
    sources agree|conflict. final answer: agree|conflict". Returns (label, v1, v2, verdict). The two answers come from
    the text between the 'first'/'second' markers when present, else the first two distinct candidates in order; the
    verdict is exact-matched after the LAST 'final answer:' (synonyms mapped), else None."""
    if not text or not text.strip():
        return "empty", None, None, None
    low = text.lower()
    verdict = None
    if ANCHOR in low:
        tail = re.sub(r"[^a-z ]+", " ", low.rsplit(ANCHOR, 1)[1]).split()
        verdict = next((VERDICT_WORDS[w] for w in tail if w in VERDICT_WORDS), None)
    body = low.rsplit(ANCHOR, 1)[0] if ANCHOR in low else low
    v1 = v2 = None
    if "first" in body and "second" in body:
        i1, i2 = body.find("first"), body.find("second")
        if i1 < i2:
            s1 = body[i1:i2]; s2 = body[i2:]
            s2 = s2.split("the requested sources")[0]
            l1, v1 = parse_answer(s1, cands); l2, v2 = parse_answer(s2, cands)
            v1 = v1 if l1 == "cand" else None; v2 = v2 if l2 == "cand" else None
    if v1 is None or v2 is None:
        got = _first_in_order(body, cands, 2)
        if len(got) == 2 and v1 is None and v2 is None:
            v1, v2 = got
    lab = "full" if (v1 and v2 and verdict) else "answers" if (v1 and v2) else "verdict_only" if verdict else "other"
    return lab, v1, v2, verdict


def verdict_loose(text):
    """The LAST verdict word anywhere in the text (no anchor required) -- a looser reading kept beside the strict one."""
    w = [x for x in re.sub(r"[^a-z ]+", " ", (text or "").lower()).split() if x in VERDICT_WORDS]
    return VERDICT_WORDS[w[-1]] if w else None


def _selftest():
    C = ("red", "blue", "green")
    assert parse_answer("I used the requested source to answer the question. final answer: red", C) == ("cand", "red")
    assert parse_answer("The diamond is blue.", C) == ("cand", "blue")
    assert parse_answer("final answer: Blue.", C) == ("cand", "blue")
    assert parse_answer("The caption says red but the image shows blue. final answer: blue", C) == ("cand", "blue")
    assert parse_answer("It is red in the caption and blue in the image.", C) == ("ambiguous", None)
    assert parse_answer("There are six shapes. final answer: six", ("4", "6")) == ("cand", "6")
    assert parse_answer("final answer: 4", ("4", "6")) == ("cand", "4")
    assert parse_answer("The color is not visible in the image.", C) == ("abstain", None)
    assert parse_answer("The object is purple.", C) == ("other", None)
    assert parse_answer("", C) == ("empty", None)
    assert parse_answer("final answer: left of the cube", ("left", "right")) == ("cand", "left")
    assert parse_answer("final answer: unknown", ("red", "blue", "unknown")) == ("cand", "unknown")
    P = ("yellow", "red", "blue"); Q = ("3", "4", "5")
    assert parse_pair("yellow; 4", P, Q) == ("pair", "yellow", "4")
    assert parse_pair("final answer: red;four", P, Q) == ("pair", "red", "4")
    assert parse_pair("The first is blue and the second is 3.", P, Q) == ("pair", "blue", "3")
    assert parse_pair("yellow", P, Q)[0] == "partial"
    _o = {"A": "red", "B": "blue", "C": "green", "D": "cyan"}
    for t, want in [("B", "blue"), ("(C) green", "green"), ("final answer: D", "cyan"), ("The answer is B.", "blue"),
                    ("A red diamond. final answer: B", "blue"), ("A red diamond.", "red"), ("It is blue.", "blue"), ("", None)]:
        got = parse_mc(t, _o)[1]
        assert got == want, (t, got, want)
    T = "From the first requested source, the answer is yellow. From the second requested source, the answer is red. The requested sources conflict. final answer: conflict"
    assert parse_compose(T, P) == ("full", "yellow", "red", "conflict")
    assert parse_compose("yellow, red. final answer: they disagree", P) == ("full", "yellow", "red", "conflict")
    assert parse_compose("Both say red. final answer: agree", P)[3] == "agree"
    assert parse_compose("", P)[0] == "empty"
    print("[selftest] parser ok (answer, pair, compose, mc)")


@torch.no_grad()
def generate_free(model, proc, content, device, max_new, forbid, grace):
    um = [{"role": "user", "content": content}]
    # SAME template call as scsrc.collate._prompt_len (training and forced-choice scoring), so the
    # generation prompt is token-identical to the scored prompt
    enc = proc.apply_chat_template(um, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to(device)
    plen = enc["input_ids"].shape[1]
    extra = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask", "token_type_ids")}
    stops = StoppingCriteriaList([StopAfterAnchor(proc.tokenizer, plen, anchor=ANCHOR, grace=grace)])
    lp = LogitsProcessorList([forbid_ids_processor(forbid)]) if forbid else None
    out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         max_new_tokens=max_new, do_sample=False, logits_processor=lp,
                         stopping_criteria=stops, **extra)
    new = out[0, plen:]
    return proc.tokenizer.decode(new, skip_special_tokens=True).strip(), int(new.shape[0])


@torch.no_grad()
def loglik_both(model, proc, umsg, candidate, device, tokens=False):
    """(sum log-lik, mean log-lik, n_tokens) of the candidate continuation -- one forward pass. Every assistant-turn token
    is scored: the lead-in, the answer and the end-of-turn marker(s). tokens=True (v1.142) also returns the token ids
    and per-token log-probs, so the answer span can be separated (scripts/tokscore.py)."""
    fmsg = umsg + [{"role": "assistant", "content": [{"type": "text", "text": candidate}]}]
    enc = _encode(proc, fmsg, device); plen = _prompt_len(proc, umsg, device)
    ids = enc["input_ids"]; n = ids.shape[1] - plen
    if n <= 0:
        return -1e9, -1e9, 0
    kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    lg = model(**kw, use_cache=False).logits.float()
    tgt = ids[0, plen:]
    lp = torch.log_softmax(lg[0, plen - 1:ids.shape[1] - 1, :], -1)
    tl = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    s = float(tl.sum())
    if tokens:
        return s, s / n, int(n), tgt.tolist(), [round(float(x), 4) for x in tl]
    return s, s / n, int(n)


def kappa(a, b):
    """Cohen's kappa between two binary lists."""
    n = len(a)
    if n == 0:
        return None
    po = sum(int(x == y) for x, y in zip(a, b)) / n
    pa1 = sum(a) / n; pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, help="required unless --selftest")
    ap.add_argument("--data_dir", default="data/clevr3")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--pairs_file", default=None, help="default: data_dir/pairs_look_all_<id|ood>.jsonl")
    ap.add_argument("--per_skill", type=int, default=40, help="pairs per skill, first in file order (60 -> 40 on 2026-09-22 to fit the 3 h walltime with the larger token budget)")
    ap.add_argument("--max_n", type=int, default=10**9)
    ap.add_argument("--sources", default="look,read")
    ap.add_argument("--mode", default=None, help="None = the run's own mode; instructed | noop = frozen backbone baselines")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_new", type=int, default=192, help="Qwen-3.5 free answers open with a reasoning preamble and passed 64 tokens before the anchor in the smoke (STATUS v1.90)")
    ap.add_argument("--grace", type=int, default=16, help="tokens generated after the anchor before stopping")
    ap.add_argument("--out_dir", default=None, help="override the run's eval dir (smoke tests)")
    ap.add_argument("--cp", type=int, default=1, help="conflict-preserving edits (STATUS v1.86); 0 = the pre-fix rule")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mc", type=int, default=0, help="v1.131: lettered multiple choice over the item's candidates (needs --tag)")
    ap.add_argument("--agree_ctx", type=int, default=0,
                    help="v1.131: add the agree context (caption edited to the image value) for both requests, so conflict "
                         "detection from two separate prompts can be computed (detect_two_prompts)")
    ap.add_argument("--format_hint", type=int, default=0,
                    help="v1.130: append ' End your reply with \"final answer: <answer>\".' to the prompt for BOTH the forced "
                         "choice and the generation in this file (frozen baselines never learned the format). Needs --tag.")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    if not a.run_dir:
        ap.error("--run_dir is required")
    if a.mode in ("instructed", "noop") and not a.tag:
        raise SystemExit(f"[STOP] --mode {a.mode} needs --tag (would overwrite the trained arm's file)")

    t0 = time.time()
    model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    if forbid is None:
        forbid = getattr(model, "_neo_all_ids", None)
    n = meta["n_tokens"]
    try:
        cfg = json.load(open(os.path.join(a.run_dir, "config.json")))["args"]
    except Exception:
        cfg = {}
    meth = "neo" if a.mode in ("instructed", "noop") else cfg.get("method", "neo")
    cfgm = a.mode or cfg.get("factor_mode", "srconly")
    print(f"[prompt] method={meth} mode={cfgm} forbid={len(forbid) if forbid else 0} ids", flush=True)

    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    by_id = {r["id"]: r for r in rows}
    if a.cp:
        print(f"[cp] vocabulary {set_c3_vocab(rows)}")
    twin = a.pairs_file or os.path.join(a.data_dir, f"pairs_look_all_{a.split.replace('test_', '')}.jsonl")
    prs_all = [json.loads(l) for l in open(twin)]
    per = collections.Counter(); prs = []
    for pr in prs_all:
        sk = pr.get("skill", "?")
        if per[sk] < a.per_skill and pr["id"] in by_id:
            per[sk] += 1; prs.append(pr)
    prs = prs[:a.max_n]
    print(f"[items] {len(prs)} pairs from {twin} per skill {dict(per)}", flush=True)
    sources = [s for s in a.sources.split(",") if s]

    _QS = os.environ.get("SC_QSTYLE", "explicit")      # v1.130: held-out templates and interface probes
    if (_QS != "explicit" or a.format_hint or a.mc or a.agree_ctx) and not a.tag:
        raise SystemExit("[STOP] SC_QSTYLE / --format_hint need --tag so the default file is not overwritten")

    def content(item, source, seed):
        c = clevr3_user_content(item, source, n, cfgm, seed, method=meth, question_style=_QS)
        if a.mc:
            c = list(c) + [{"type": "text", "text": MC_INSTR.format(
                opts="\n".join(f"{k}. {v}" for k, v in _MC["opts"].items()))}]
        if a.format_hint:
            c = list(c) + [{"type": "text", "text": FORMAT_HINT_MC if a.mc else FORMAT_HINT}]
        return c
    _MC = {"opts": None}

    def pick(item, source, cands, seed):
        um = [{"role": "user", "content": content(item, source, seed)}]
        # 2026-09-23 (STATUS v1.104): sequential scoring only -- the batched pass is not numerically equivalent
        if a.mc:                                        # the letters are scored; each maps back to its value
            lt = {v: k for k, v in _MC["opts"].items()}
            sc = {v: loglik_both(model, proc, um, cand(lt[v]), "cuda", tokens=True) for v in cands}
        else:
            sc = {v: loglik_both(model, proc, um, cand(v), "cuda", tokens=True) for v in cands}
        from scripts.tokscore import split_spans as _ss
        sp = _ss({v: (sc[v][3], sc[v][4]) for v in cands})     # v1.144: lead-in / answer / end + per-token log-probs
        _dec = (lambda v: sp[v]["answer_sum"]) if os.environ.get("SC_SCORE", "turn") == "answer" else (lambda v: sc[v][0])
        p_sum = max(sc, key=_dec); p_mean = max(sc, key=lambda v: sc[v][1])
        pick.last = {v: {"sum": round(sc[v][0], 4), "mean": round(sc[v][1], 4), "ntok": sc[v][2], **sp[v],
                         **({"letter": lt[v]} if a.mc else {})} for v in cands}
        return p_sum, p_mean, {v: sc[v][2] for v in cands}

    recs = []
    for j, pr in enumerate(tqdm(prs, desc="c3gen")):
        base = by_id[pr["id"]]
        if a.cp and pr["gold_b"] in (pr.get("read_value"), pr.get("ocr_value")):
            base, _mv = cp_fix_look_collision(base, pr["gold_b"])
            if base is None:
                continue
        ch = {k: str(v) for k, v in base["channels"].items()}
        if a.cp:
            alt = {c: (cp_text_edit_value(base, c)[0] if c in c3_sources() else None) for c in ("read", "ocr")}
            if alt["read"] is None:
                continue
            # STATUS v1.94: ONE candidate set per item for every context and both requests
            Y = tuple(dict.fromkeys([ch["look"], ch["read"], alt["read"], pr["gold_b"]]))
        else:
            alt = {c: next((v for k, v in ch.items() if k != c and v != ch[c]), None) for c in ("read", "ocr")}
        for src in sources:
            if src == "look":
                cands = Y if a.cp else tuple(dict.fromkeys([pr["gold_a"], pr["gold_b"]] + [ch["read"], ch["ocr"]]))
                ia = dict(base); ia["image"] = pr["ctx_a"]["image"]
                ib = dict(base); ib["image"] = pr["ctx_b"]["image"]
                ctxs = {"base": (ia, pr["gold_a"]), "flip": (ib, pr["gold_b"])}
                if alt["read"] is not None:
                    ctxs["stay"] = (_edited_item(ia, "read", alt["read"]), pr["gold_a"])
            elif src == "read":
                if alt["read"] is None:
                    continue
                cands = Y if a.cp else (ch["read"], alt["read"])
                ctxs = {"base": (base, ch["read"]), "flip": (_edited_item(base, "read", alt["read"]), alt["read"])}
                if a.cp:
                    # v1.86: the caption's competitor is the image -> exact image twin, text untouched
                    if pr["gold_b"] != ch["read"]:
                        ib_ = dict(base); ib_["image"] = pr["ctx_b"]["image"]
                        ctxs["stay"] = (ib_, ch["read"])
                elif alt["ocr"] is not None:
                    ctxs["stay"] = (_edited_item(base, "ocr", alt["ocr"]), ch["read"])
            else:
                raise SystemExit(f"unsupported source {src}")
            if a.agree_ctx and src in ("look", "read") and ch["look"] != ch["read"]:
                ag = dict(base); ag["image"] = pr["ctx_a"]["image"] if src == "look" else base["image"]
                ctxs["agree"] = (_edited_item(ag, "read", ch["look"]), ch["look"])
            _MC["opts"] = mc_options(pr["id"], cands) if a.mc else None
            for cname, (item, gold) in ctxs.items():
                p_sum, p_mean, ntok = pick(item, src, cands, j)
                cc = content(item, src, j)
                txt, nnew = generate_free(model, proc, cc, "cuda", a.max_new, forbid, a.grace)
                if a.mc:
                    lab, val, route = parse_mc(txt, _MC["opts"])
                else:
                    (lab, val), route = parse_answer(txt, cands), None
                ptxt = " ".join("[IMAGE]" if c.get("type") == "image" else c.get("text", "") for c in cc)
                recs.append({"id": pr["id"], "skill": pr.get("skill", "?"), "source": src, "ctx": cname,
                             "gold": gold, "cands": list(cands), "ntok": ntok,
                             "pick_sum": p_sum, "pick_mean": p_mean,
                             "gen_label": lab, "gen_value": val, "gen_ntok": nnew, "text": txt,   # v1.130: full text
                             "cand_scores": pick.last, "question_style": _QS, "format_hint": int(bool(a.format_hint)),
                             "prompt": ptxt,                                    # v1.131: prompt kept for re-annotation
                             # what each source states in THIS context (the image twin keeps the base channels dict)
                             "values": {"image": (str(pr["gold_b"]) if item["image"] == pr["ctx_b"]["image"]
                                                  else str(item["channels"]["look"])),
                                        "caption": str(item["channels"]["read"])},
                             "lp_gold": pick.last.get(gold, {}).get("sum"),
                             "cap_first": eval_design(item, j, _QS)[1],             # v1.133: block order
                             **({"mc_options": dict(_MC["opts"]), "parse_route": route} if a.mc else {})})

    # ---------- aggregate
    def m(v):
        return round(sum(v) / len(v), 3) if v else None
    R = collections.defaultdict(lambda: collections.defaultdict(list))
    SK = collections.defaultdict(lambda: collections.defaultdict(list))
    labels = collections.defaultdict(collections.Counter)
    byitem = collections.defaultdict(dict)
    for r in recs:
        byitem[(r["id"], r["source"])][r["ctx"]] = r
        labels[f'{r["source"]}/{r["ctx"]}'][r["gen_label"]] += 1
        R[r["source"]]["agree_" + r["ctx"]].append(int(r["gen_label"] == "cand" and r["gen_value"] == r["pick_sum"]))
        R[r["source"]]["summean_disagree"].append(int(r["pick_sum"] != r["pick_mean"]))
        R[r["source"]]["parse_cand"].append(int(r["gen_label"] == "cand"))
        R[r["source"]]["unequal_ntok"].append(int(len(set(r["ntok"].values())) > 1))
        R[r["source"]]["gen_ntok"].append(r["gen_ntok"])
    for (iid, src), cx in byitem.items():
        b = cx.get("base"); f = cx.get("flip"); s = cx.get("stay")
        if b is None:
            continue
        sk = b["skill"]
        gv = lambda r: r["gen_value"] if r["gen_label"] == "cand" else None
        ll_acc = int(b["pick_sum"] == b["gold"]); g_acc = int(gv(b) == b["gold"])
        R[src]["ll_acc"].append(ll_acc); R[src]["gen_acc"].append(g_acc)
        SK[sk][f"{src}_ll_acc"].append(ll_acc); SK[sk][f"{src}_gen_acc"].append(g_acc)
        if f is not None:
            ll_flip = int(ll_acc and f["pick_sum"] == f["gold"])
            g_flip = int(g_acc and gv(f) == f["gold"])
            R[src]["ll_flip"].append(ll_flip); R[src]["gen_flip"].append(g_flip)
            SK[sk][f"{src}_ll_flip"].append(ll_flip); SK[sk][f"{src}_gen_flip"].append(g_flip)
        if s is not None:
            ll_stay = int(s["pick_sum"] == b["pick_sum"]); g_stay = int(gv(s) is not None and gv(s) == gv(b))
            R[src]["ll_stay"].append(ll_stay); R[src]["gen_stay"].append(g_stay)
            R[src]["ll_stay_c"].append(int(ll_acc and ll_stay)); R[src]["gen_stay_c"].append(int(g_acc and g_stay))
            SK[sk][f"{src}_ll_stay_c"].append(int(ll_acc and ll_stay)); SK[sk][f"{src}_gen_stay_c"].append(int(g_acc and g_stay))
    summary = {}
    for src, d in R.items():
        o = {k: m(v) for k, v in d.items() if k != "gen_ntok"}
        o["parse_cand_rate"] = o.pop("parse_cand", None)
        o["n_items"] = len(d["ll_acc"]); o["n_flip"] = len(d.get("ll_flip", [])); o["n_stay"] = len(d.get("ll_stay", []))
        o["gen_ntok_mean"] = round(sum(d["gen_ntok"]) / max(len(d["gen_ntok"]), 1), 1)
        if d.get("ll_flip"):
            o["flip_bit_agreement"] = m([int(x == y) for x, y in zip(d["ll_flip"], d["gen_flip"])])
            o["flip_kappa"] = kappa(d["ll_flip"], d["gen_flip"])
            lf = [g for l, g in zip(d["ll_flip"], d["gen_flip"]) if l]; gf = [l for l, g in zip(d["ll_flip"], d["gen_flip"]) if g]
            o["p_genflip_given_llflip"] = m(lf); o["p_llflip_given_genflip"] = m(gf)
        if d.get("ll_stay_c"):
            o["stay_bit_agreement"] = m([int(x == y) for x, y in zip(d["ll_stay_c"], d["gen_stay_c"])])
            o["stay_kappa"] = kappa(d["ll_stay_c"], d["gen_stay_c"])
        summary[src] = o
    _dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step, "mode": cfgm, "method": meth, "cp": int(a.cp),
           "question_style": _QS, "format_hint": int(bool(a.format_hint)),
           "caption_style": os.environ.get("SC_CAPTION_STYLE", "template"), "mc": int(bool(a.mc)), "agree_ctx": int(bool(a.agree_ctx)),
           "two_prompt_detection": (detect_two_prompts([r for r in recs if r["source"] == "look"],
                                                       [r for r in recs if r["source"] == "read"])
                                    if a.agree_ctx and {"look", "read"} <= set(sources) else None),
           "pairs_file": twin, "per_skill": a.per_skill, "n_pairs": len(prs), "sources": sources,
           "max_new": a.max_new, "grace": a.grace, "decoding": "greedy", "weight_dtype": _dtype,
           "summary": summary, "labels": {k: dict(v) for k, v in labels.items()},
           "per_skill": {k: {kk: m(vv) for kk, vv in v.items()} for k, v in SK.items()},
           "per_skill_n": {k: {kk: len(vv) for kk, vv in v.items()} for k, v in SK.items()},
           "elapsed_s": round(time.time() - t0, 1),
           "note": ("Forced choice and free generation scored on the same items, prompts and contexts. "
                    "look: base = twin image A, flip = twin image B, stay = caption edit; read: base, "
                    "flip = caption edit, stay = OCR edit. gen_* use the closed-vocabulary parser "
                    "(label 'cand' only); agree_<ctx> = parsed value equals the sum-log-lik pick."),
           "records": recs}
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=1), flush=True)
    od = a.out_dir or os.path.join(a.run_dir, "eval", f"clevr3_{a.split}")
    os.makedirs(od, exist_ok=True)
    _dt = "" if _dtype == "bfloat16" else f"_{_dtype}"
    fn = os.path.join(od, f"gen_step_{a.step}{a.tag}{_dt}.json")
    json.dump(out, open(fn, "w"), indent=1)
    print(f"[saved] {fn}")


if __name__ == "__main__":
    main()
