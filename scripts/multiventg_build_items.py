#!/usr/bin/env python
"""Stage 3: assemble SFT items from claims.jsonl with (a) per-(role,channel) CAP,
(b) refusal/mismatch items from gold absence, (c) uniform deviant-channel conflicts.
Emits train items keyed by operator word. Frames must already be extracted; items
for VIS/OCR whose frames are missing are written with need_frames=True and skipped
by the trainer's collate if absent.

Usage: python scripts/multiventg_build_items.py --claims data/multiventG/claims.jsonl \
   --json data/multiventG/multivent_g.json [--cap 70] [--mismatch_frac 0.25] [--seed 0]
"""
import argparse, json, random
from collections import defaultdict

CH_NAME = {"DESC": "the description", "OCR": "the on-screen text", "VIS": "the video"}
CH_WORD = {"DESC": "read", "OCR": "read_ocr", "VIS": "look"}

def tgt(channel, body, answer):
    w = CH_WORD[channel]
    return f"<{w}> Based on {CH_NAME[channel]}, {body} final answer: {answer} <{w}>"

def refusal(channel):
    w = CH_WORD[channel]
    return (f"<{w}> {CH_NAME[channel].capitalize()} does not address this question. "
            f"final answer: not shown in {CH_NAME[channel]} <{w}>")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--cap", type=int, default=70)
    ap.add_argument("--mismatch_frac", type=float, default=0.25)
    ap.add_argument("--out", default="data/multiventG/items.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    claims = [json.loads(l) for l in open(a.claims) if l.strip()]
    data = json.load(open(a.json))

    # (a) cap per (role, channel), preferring higher certainty then shorter answers
    by_rc = defaultdict(list)
    for c in claims:
        by_rc[(c["role"], c["channel"])].append(c)
    kept = []
    for (role, ch), lst in by_rc.items():
        lst.sort(key=lambda c: (-(c.get("certainty") or 0), len(c.get("answer", ""))))
        kept.extend(lst[:a.cap])
    have = defaultdict(set)
    donors = defaultdict(list)
    for c in claims:
        have[(c["video_id"], c["role"])].add(c["channel"])
        donors[(c["role"], c["channel"])].append(c)

    items = []
    # positive + conflict items from kept claims
    for c in kept:
        vid, role, ch = c["video_id"], c["role"], c["channel"]
        body = c["claim"] if c["claim"].endswith(".") else c["claim"] + "."
        items.append({"video_id": vid, "role": role, "channel": ch, "cond": "agree",
                      "word": CH_WORD[ch], "target": tgt(ch, body, c["answer"]),
                      "evidence": c.get("evidence"), "need_frames": ch in ("VIS", "OCR")})
        pool = [d for d in donors[(role, ch)]
                if d["video_id"] != vid
                and d["answer"].strip().lower() != c["answer"].strip().lower()]
        if pool:
            d = rng.choice(pool)
            dbody = d["claim"] if d["claim"].endswith(".") else d["claim"] + "."
            items.append({"video_id": vid, "role": role, "channel": ch, "cond": "conflict",
                          "word": CH_WORD[ch], "target": tgt(ch, dbody, d["answer"]),
                          "deviant_answer": d["answer"], "donor_video": d["video_id"],
                          "evidence": c.get("evidence"), "need_frames": ch in ("VIS", "OCR")})

    # global (role, channel) support: fraction of that role's videos where the channel answers
    role_videos = defaultdict(set); role_ch_videos = defaultdict(set)
    for c in claims:
        role_videos[c["role"]].add(c["video_id"])
        role_ch_videos[(c["role"], c["channel"])].add(c["video_id"])
    def structural_zero(role, ch):
        rv = len(role_videos.get(role, ()))
        return rv >= 10 and len(role_ch_videos.get((role, ch), ())) / rv < 0.02

    # (b) refusal items. Two valid sources of "channel does not answer this":
    #   (i)  per-video: role answered in ANOTHER channel of this video, absent here;
    #   (ii) structural: role's channel-support is ~0 across the corpus (e.g. when x VIS).
    elig = []
    for vid, rec in data.items():
        for role in rec.get("template", {}):
            chans = have.get((vid, role), set())
            for ch in ("DESC", "OCR", "VIS"):
                if ch in chans:
                    continue
                if chans or structural_zero(role, ch):   # answered elsewhere here, OR globally absent
                    elig.append((vid, role, ch))
    n_ref = int(a.mismatch_frac * len(items))
    rng.shuffle(elig)
    for vid, role, ch in elig[:n_ref]:
        items.append({"video_id": vid, "role": role, "channel": ch, "cond": "refusal",
                      "word": CH_WORD[ch], "target": refusal(ch),
                      "need_frames": ch in ("VIS", "OCR")})
    rng.shuffle(items)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    from collections import Counter
    cc = Counter((it["word"], it["cond"]) for it in items)
    print(f"[items] {len(items)} total -> {a.out}")
    for k in sorted(cc):
        print(f"   {k[0]:<10}{k[1]:<10}{cc[k]}")
    print(f"[items] refusal items: {min(n_ref, len(elig))} from {len(elig)} eligible")

if __name__ == "__main__":
    main()
