"""Teacher-forced log-likelihood forced-choice scoring.

The primary eval for the causal-ladder program: instead of generating and
parsing (brittle) or judging (banned), we score each candidate answer's
length-normalized log-likelihood under teacher forcing and let the model
"pick" the argmax. Judge-free, parse-free, and identical in form to the
training objective. ~10-50x faster than autoregressive generation because
there is no decode loop -- one forward per candidate over a short suffix.

  loglik(model, processor, image, user_text, candidate) -> (sum_lp, n_tok)
  forced_choice(...candidates...) -> (argmax_index, [norm_lp per candidate])
  margin = lp[requested] - max lp[others]   (continuous, low-noise signal)

Encoding mirrors infer.generate / collate exactly (same chat template, same
assistant-turn construction), so a candidate scored here is the same string
the model would be trained/evaluated to produce.
"""
import torch

from .collate import _user_content, _encode, _prompt_len


@torch.no_grad()
def loglik(model, processor, image, user_text, candidate, device="cuda", tokens=False):
    """Length-normalized log P(candidate | prompt) under teacher forcing.
    Returns (mean_token_logprob, n_scored_tokens)."""
    umsg = [{"role": "user", "content": _user_content(image, user_text)}]
    fmsg = umsg + [{"role": "assistant",
                    "content": [{"type": "text", "text": candidate}]}]
    enc = _encode(processor, fmsg, device)
    plen = _prompt_len(processor, umsg, device)
    ids = enc["input_ids"]
    n_ans = ids.shape[1] - plen
    if n_ans <= 0:
        return float("-inf"), 0
    kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    logits = model(**kw, use_cache=False).logits.float()
    # predict token t from position t-1; score the answer span [plen, end)
    tgt = ids[0, plen:]
    lp = torch.log_softmax(logits[0, plen - 1:ids.shape[1] - 1, :], dim=-1)
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    total = float(tok_lp.sum())
    if tokens:   # v1.142: ids + per-token log-probs of the whole assistant turn (lead-in, answer, end of turn)
        return total, total / n_ans, int(n_ans), tgt.tolist(), [round(float(x), 4) for x in tok_lp]
    return total, total / n_ans, int(n_ans)


@torch.no_grad()
def forced_choice(model, processor, image, user_text, candidates, device="cuda"):
    """Proper forced choice: each candidate is the FULL in-format assistant
    target (rationale + "final answer: {value}") with only the source value
    differing. Argmax by TOTAL log-prob selects the highest-scoring CANDIDATE
    sequence; the shared prefix cancels, so the decision is the routed value in
    its true context. This is NOT greedy generation: a free generation can leave
    the candidate set, so generation is measured separately
    (scripts/eval_clevr3_generate.py). Returns (argmax_idx, [total_lp,...]).

    Do NOT pass bare values here -- scoring "cat: white" as the whole turn is
    out-of-distribution and diverges from generation (anchor fail 2026-07-21)."""
    tots = [loglik(model, processor, image, user_text, c, device)[0]
            for c in candidates]
    best = max(range(len(tots)), key=lambda i: tots[i])
    return best, tots


@torch.no_grad()
def forced_choice_per_object(model, processor, image, user_text, obj_keys,
                            img_key, txt_key, task, device="cuda", base_src="img"):
    """Multi-object forced choice (ColorSwap fix). A conjunction answer like
    'butterfly: blue, spots: orange' routes PER OBJECT, so a single pure-vs-pure
    choice is confounded (anchor 2026-07-21). For each object, score two
    full-answer candidates that differ ONLY in that object's value (image vs
    caption); the per-object routing is the argmax. Returns {obj: ('img'|'txt', margin)}.

    base_src = which source the OTHER objects are held at. This MUST match the
    token being tested, or the held context biases every per-object choice
    toward that source (bug 2026-07-22: base always 'img' crushed <read> to
    0.03 on ColorSwap). So the caller passes base_src='img' when testing <look>
    and 'txt' when testing <read>: the held context matches the op's target,
    the shared prefix cancels, and only object o's value is under test.
    Atomic answers (VCF/COCO, one object) reduce to the ordinary forced choice."""
    from .prompts import fmt_key_task
    out = {}
    base = dict(img_key if base_src == "img" else txt_key)
    for o in obj_keys:
        cand_i = dict(base); cand_i[o] = img_key[o]
        cand_t = dict(base); cand_t[o] = txt_key[o]
        si = loglik(model, processor, image, user_text,
                    fmt_key_task(cand_i, task), device)[0]
        st = loglik(model, processor, image, user_text,
                    fmt_key_task(cand_t, task), device)[0]
        out[o] = ("img" if si >= st else "txt", si - st)
    return out


def margin(scores, requested_idx):
    """logP(requested full target) - max_{j != requested} logP(j). >0 iff the
    model prefers the requested source's value. Sign = forced-choice decision;
    magnitude = confidence -- both feed flip/stay and IIA as continuous signals."""
    others = [s for j, s in enumerate(scores) if j != requested_idx]
    return scores[requested_idx] - (max(others) if others else float("-inf"))
