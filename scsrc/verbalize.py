"""Self-verbalization + plug-in test, per checkpoint. Uses each token's trained
usage: ctrl is probed as an input instruction, gen as an output symbol."""
from . import prompts as P
from . import metrics as Me
from .tokens import trigger
from .infer import generate
from .data import get_image


def verbalize_operator(model, processor, name, n, device, kind, forbid_ids=None):
    """Ask what a token means. The ANSWER must be in ordinary words, so ALL
    added-token ids are forbidden at decode time -- without this, tied output
    embeddings let the model "define" a token by emitting the token (the
    degenerate <tok><tok><tok> loops)."""
    op = trigger(name, n)
    if kind == "ctrl":
        txt, pre = P.VERBALIZE_CTRL.format(op=op), P.VERBALIZE_CTRL_PREFILL.format(op=op)
    else:
        txt, pre = P.VERBALIZE_GEN.format(op=op), P.VERBALIZE_GEN_PREFILL.format(op=op)
    return generate(model, processor, None, txt, device, 96, assistant_prefill=pre,
                    forbid_ids=forbid_ids)


def synonyms_operator(model, processor, name, n, device, forbid_ids=None):
    """Elicit five ordinary-word replacements for a ctrl token; parse to a list."""
    op = trigger(name, n)
    txt = P.SYNONYMS_CTRL.format(op=op)
    pre = P.SYNONYMS_CTRL_PREFILL.format(op=op)
    out = generate(model, processor, None, txt, device, 96, assistant_prefill=pre,
                   forbid_ids=forbid_ids)
    if out.count(",") < 2 and "\n" not in out:   # degenerate list -> one plainer retry
        out2 = generate(model, processor, None,
                        f"List five short English phrases with the same meaning as {op}, separated by commas.",
                        device, 96, forbid_ids=forbid_ids)
        if out2.count(",") >= 2:
            out = out2
    cands, seen = [], set()
    for part in out.replace("\n", ",").split(","):
        w = part.strip().strip(".").strip("'\"").lstrip("0123456789. )-")
        wl = w.lower()
        if 0 < len(w.split()) <= 4 and wl not in seen and "<" not in w and wl not in ("", "and", "or"):
            seen.add(wl); cands.append(w)
        if len(cands) == 5:
            break
    return cands, out


def eval_synonyms(model, processor, specs, ds, name, synonyms, device, n,
                  source_key_attr, forbid_ids=None, task="color", rec=None, max_n=40):
    """Plug each elicited synonym into the token's instruction slot; accuracy per
    synonym. Comparable against the token itself, its description, and the
    instructed baseline."""
    per = {}
    for syn in synonyms:
        accs = []
        for i, s in enumerate(specs[:max_n]):
            img = get_image(ds, s)
            gold = getattr(s, source_key_attr)
            u = P.render_ctrl(P.CTRL_CANONICAL, syn, s.fed_caption)
            o = generate(model, processor, img, u, device, 160,
                         forbid_ids=forbid_ids, stop_anchor="final answer:")
            a = Me.object_accuracy(Me.parse_final_answer(o, task), gold)
            accs.append(a)
            if rec is not None:
                rec.append({"eid": i, "cond": s.cond, "test": "plugin_synonym",
                            "phrase": syn, "output": o[:200], "correct": gold,
                            "exact_match": a})
        per[syn] = sum(accs) / max(len(accs), 1)
    vals = list(per.values())
    return {"per_synonym": per,
            "syn_best": max(vals) if vals else 0.0,
            "syn_mean": (sum(vals) / len(vals)) if vals else 0.0}


def plugin_test(model, processor, specs, ds, name, verbalization, device, n,
                source_key_attr, forbid_ids=None, task="color", rec=None):
    """Replace the {op} token in the trained instruction with its own
    verbalization phrase; check the source binding reproduces. Per-example
    outputs (token-prompted AND verbalization-prompted, plus the phrase used)
    are appended to `rec` so the substitution is auditable record by record."""
    op = trigger(name, n)
    phrase = verbalization.strip().rstrip(".")
    acc_tok, acc_verb = [], []
    for i, s in enumerate(specs):
        img = get_image(ds, s)
        gold = getattr(s, source_key_attr)
        u_tok = P.render_ctrl(P.CTRL_CANONICAL, op, s.fed_caption)
        u_verb = P.render_ctrl(P.CTRL_CANONICAL, f"({phrase})", s.fed_caption)
        o_tok = generate(model, processor, img, u_tok, device, 160,
                         forbid_ids=forbid_ids, stop_anchor="final answer:")
        o_verb = generate(model, processor, img, u_verb, device, 160,
                          forbid_ids=forbid_ids, stop_anchor="final answer:")
        a_t = Me.object_accuracy(Me.parse_final_answer(o_tok, task), gold)
        a_v = Me.object_accuracy(Me.parse_final_answer(o_verb, task), gold)
        acc_tok.append(a_t)
        acc_verb.append(a_v)
        if rec is not None:
            rec.append({"eid": i, "cond": s.cond, "test": "plugin_token",
                        "output": o_tok[:200], "correct": gold, "exact_match": a_t})
            rec.append({"eid": i, "cond": s.cond, "test": "plugin_verbalization",
                        "phrase": phrase, "output": o_verb[:200],
                        "correct": gold, "exact_match": a_v})
    a_t = sum(acc_tok) / max(len(acc_tok), 1)
    a_v = sum(acc_verb) / max(len(acc_verb), 1)
    return {"acc_token": a_t, "acc_verbalization": a_v,
            "plugin_fidelity": (a_v / a_t) if a_t > 0 else 0.0,
            "phrase": phrase}
