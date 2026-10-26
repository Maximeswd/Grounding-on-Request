#!/usr/bin/env python
"""Split candidate continuations into lead-in / answer / end-of-turn tokens (v1.142, 2026-09-24).

Every forced-choice score in this repository is the teacher-forced SUM of log p over the whole assistant turn: a
lead-in identical for all candidates ("I used the requested source to answer the question. final answer:", plus
"</think>\\n\\n" for Qwen-3.5-4B), the answer tokens, and the end-of-turn marker(s). The lead-in precedes the answer and
is the same for every candidate, so it cancels in the argmax; the end-of-turn term follows the answer and does not.
Given all candidates of one prompt, the lead-in is their longest common token prefix and the end-of-turn span their
longest common token suffix (never overlapping the answer). Torch-free.
"""


def split_spans(cands):
    """cands: {name: (ids, lps)} for ONE prompt. Returns {name: {...}} with lead/answer/end sums and counts."""
    seqs = list(cands.values())
    if not seqs:
        return {}
    ids0 = [s[0] for s in seqs]
    L = min(len(x) for x in ids0)
    pre = 0
    while pre < L and all(x[pre] == ids0[0][pre] for x in ids0):
        pre += 1
    suf = 0
    while suf < L - pre and all(x[len(x) - 1 - suf] == ids0[0][len(ids0[0]) - 1 - suf] for x in ids0):
        suf += 1
    out = {}
    for k, (ids, lps) in cands.items():
        a0, a1 = pre, len(ids) - suf
        if len(cands) == 1:           # nothing to compare against: whole turn is the "answer"
            a0, a1 = 0, len(ids)
        ans = lps[a0:a1]
        out[k] = {"lead_sum": round(sum(lps[:a0]), 4), "lead_ntok": a0,
                  "answer_sum": round(sum(ans), 4), "answer_ntok": len(ans),
                  "answer_mean": round(sum(ans) / max(len(ans), 1), 4),
                  "end_sum": round(sum(lps[a1:]), 4), "end_ntok": len(ids) - a1,
                  "token_ids": ids, "token_lps": lps}
    return out


if __name__ == "__main__":
    a = ([1, 2, 3, 10, 11, 90, 91], [-.1, -.2, -.3, -1.0, -2.0, -.5, -.01])
    b = ([1, 2, 3, 12, 90, 91], [-.1, -.2, -.3, -.7, -.9, -.02])
    r = split_spans({"a": a, "b": b})
    assert r["a"]["lead_ntok"] == 3 and r["a"]["answer_ntok"] == 2 and r["a"]["end_ntok"] == 2, r["a"]
    assert r["b"]["answer_ntok"] == 1 and abs(r["b"]["answer_mean"] + .7) < 1e-9, r["b"]
    print("[selftest] tokscore ok")
