#!/usr/bin/env python
"""Materialize multivent-raw channel-word items from the video-stage reference
claims + the extracted artifact subset (scripts/mvraw_extract.py).

Writes data/multivent_raw/{train,test_id,test_ood}.jsonl + pairs.jsonl +
attestation.json + manifest.json.

Design decisions (each one is a lesson carried over from MVG):
  - CLAIM SOURCE: reference-claims-video.json ONLY. The event/persona/query
    stages rewrite these same facts; mixing stages would leak paraphrases
    across splits. Query-stage claims stay untouched for LEADERBOARD eval.
  - ITEM = (chunk, channel) CELL, target = ALL that channel's claims for the
    chunk, one per line (median 1-2, up to 6+): multi-claim report supervision.
  - CHANNELS: VIS<-video-non-text, OCR<-video-text, ASR<-audio-speech.
    audio-non-speech is excluded (no input rendering shipped); its claims are
    neither targets nor negatives.
  - CROSS-ATTESTED claims (>1 modality) are targets for EACH attesting channel
    and NEVER negatives for any of them (the MVG cross-attestation fix).
  - BALANCED NEGATIVES: each item gets a rotation-assigned neg_channel cycling
    uniformly over the other two channels (per-channel round-robin), falling
    back to the other-other channel when the assigned one has no claims in
    this chunk. Cross-attested lines are removed from neg_claims.
  - ABSENCE items only where the channel's RAW CONTENT is empty (no speech
    detected / no OCR text) -- content-present-but-unannotated cells are
    SKIPPED, because claim-centric annotation means missing != absent.
  - SPLITS BY TOPIC: 20% of topics -> test_ood (unseen events); 15% of chunks
    within remaining topics -> test_id. A chunk cited in topics that landed in
    different splits is forced to the eval split (no leakage into train).
  - PAIRS: evidence-swap twins per channel, same-topic where possible
    (hard=true) else cross-topic (hard=false, reported separately -- the
    mvg_hard lesson); claim sets must be disjoint after casefolding.
  - LANGUAGE: kept multilingual (only ~20% of cited chunks have EN speech);
    asr_lang is stored so evals can report EN vs non-EN grounding separately.
"""
import argparse, collections, glob, json, os, random, sys

R = os.environ.get("MVR_ROOT", "data/multivent_raw_src")
ART = "data/multivent_raw/artifacts"
OUT = "data/multivent_raw"
TAG2CH = {"video-non-text": "VIS", "video-text": "OCR", "audio-speech": "ASR"}
CHS = ("VIS", "OCR", "ASR")


def asr_text_of(chunk):
    p = f"{ART}/asr/{chunk}.json"
    if not os.path.exists(p):
        return "", ""
    d = json.load(open(p))
    t = d.get("transcript")
    # WhisperX ships transcript = {"segments": [{"start","end","text","speaker_id"}],
    # "words": [...], "timestamp_resolution": ...}. Joining segment texts is the
    # only correct read -- t.get("text") does NOT exist and silently yields "".
    if isinstance(t, dict):
        segs = t.get("segments") or []
        txt = " ".join(str(s.get("text", "")).strip() for s in segs
                       if isinstance(s, dict)).strip()
        if not txt and t.get("words"):
            txt = " ".join(str(w.get("word", "")).strip()
                           for w in t["words"] if isinstance(w, dict)).strip()
    elif isinstance(t, list):
        txt = " ".join(s.get("text", "").strip() if isinstance(s, dict) else str(s)
                       for s in t).strip()
    else:
        txt = str(t or "").strip()
    txt = " ".join(txt.split())[:MAX_ASR_CHARS]
    lang = ((d.get("language") or {}).get("detected") or "") if isinstance(
        d.get("language"), dict) else str(d.get("language") or "")
    return txt, lang


MAX_OCR_STR = 220      # one frame's worth of on-screen text
MAX_OCR_TOTAL = 1600   # all frames combined
MAX_ASR_CHARS = 2400


def ocr_strings_of(chunk):
    """Deduped per-frame OCR, LENGTH-CAPPED. Without caps a single dense frame
    contributed 20k characters (~5.7k tokens) -- it inflated the sequence past
    what a 4-forward CTM step fits on a 44G card AND buried the answer span in a
    wall of noisy OCR. Caps are per-string and cumulative, longest-last so short
    salient captions survive truncation."""
    p = f"{ART}/ocr/{chunk}.jsonl"
    if not os.path.exists(p):
        return []
    seen, out = set(), []
    for l in open(p):
        r = json.loads(l)
        s = " ".join((r.get("cleaned") or r.get("txt") or r.get("raw") or "").split())
        if not s or s.casefold() in seen:
            continue
        seen.add(s.casefold())
        out.append(s[:MAX_OCR_STR])
    out.sort(key=len)                      # keep the short, salient ones
    kept, total = [], 0
    for s in out:
        if total + len(s) > MAX_OCR_TOTAL:
            break
        kept.append(s)
        total += len(s)
    return kept[:12]


def frames_of(chunk):
    return sorted(glob.glob(f"{ART}/keyframes/{chunk}.t*.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ood_frac", type=float, default=0.20)
    ap.add_argument("--id_frac", type=float, default=0.15)
    ap.add_argument("--absence_cap_frac", type=float, default=0.15)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    d = json.load(open(f"{R}/annotations/reference-claims-video.json"))
    # ---- claim pass: (chunk, channel) -> [claim texts]; chunk -> topics
    cell = collections.defaultdict(list)
    chunk_topics = collections.defaultdict(set)
    attested = collections.defaultdict(lambda: collections.defaultdict(set))
    n_drop_untagged = n_ans = 0
    for cs in d["claim_sets"]:
        t = cs["topic"]
        tid = t["topic_id"] if isinstance(t, dict) else str(t)
        for c in cs["claims"]:
            ev = c.get("evidence") or {}
            if not ev or all(not m for m in ev.values()):
                n_drop_untagged += 1
                continue
            if (c.get("meta") or {}).get("is_negative_assertion"):
                continue
            txt = " ".join(str(c["text"]).split())
            for ch, mods in ev.items():
                chunk_topics[ch].add(tid)
                for m in mods:
                    if m == "audio-non-speech":
                        n_ans += 1
                        continue
                    w = TAG2CH.get(m)
                    if w and txt not in cell[(ch, w)]:
                        cell[(ch, w)].append(txt)
                        attested[ch][txt].add(w)

    chunks = sorted({ch for ch, _ in cell})
    have = [c for c in chunks if frames_of(c)]
    missing = set(chunks) - set(have)
    print(f"cells: {len(cell)}  chunks: {len(chunks)}  with artifacts: {len(have)} "
          f"(missing {len(missing)})")

    # ---- topic split
    topics = sorted({t for c in have for t in chunk_topics[c]})
    rng.shuffle(topics)
    n_ood = max(1, int(len(topics) * a.ood_frac))
    ood_topics = set(topics[:n_ood])
    split_of = {}
    id_pool = []
    for c in have:
        if chunk_topics[c] & ood_topics:
            split_of[c] = "test_ood"   # any ood-topic contact -> eval, never train
        else:
            id_pool.append(c)
    rng.shuffle(id_pool)
    n_id = int(len(id_pool) * a.id_frac)
    for i, c in enumerate(id_pool):
        split_of[c] = "test_id" if i < n_id else "train"

    # ---- items
    items = {s: [] for s in ("train", "test_id", "test_ood")}
    rot = {ch: 0 for ch in CHS}   # per-channel round-robin over the other two
    absence_ct = collections.Counter()
    for c in have:
        asr_txt, asr_lang = asr_text_of(c)
        ocr_strs = ocr_strings_of(c)
        frames = frames_of(c)
        present = {"ASR": bool(asr_txt), "OCR": bool(ocr_strs), "VIS": True}
        nmod = {ch: len(cell.get((c, ch), [])) for ch in CHS}
        for ch in CHS:
            claims = cell.get((c, ch), [])
            if not claims:
                # absence ONLY when the channel's raw content is empty
                if present[ch] or ch == "VIS":
                    continue
                cond = "absent"
            else:
                cond = "attested"
            others = [o for o in CHS if o != ch]
            neg_ch, neg_claims = None, []
            if cond == "attested":
                order = [others[rot[ch] % 2], others[(rot[ch] + 1) % 2]]
                rot[ch] += 1
                own = {t.casefold() for t in claims}
                for o in order:
                    cand = [t for t in cell.get((c, o), [])
                            if t.casefold() not in own]
                    if cand:
                        neg_ch, neg_claims = o, cand
                        break
            it = {"chunk_id": c, "video_id": c.rsplit("_", 1)[0],
                  "topics": sorted(chunk_topics[c]), "channel": ch, "cond": cond,
                  "claims": claims, "neg_channel": neg_ch, "neg_claims": neg_claims,
                  "asr_text": asr_txt, "asr_lang": asr_lang,
                  "ocr_strings": ocr_strs, "frames": frames,
                  "n_mod_claims": nmod,
                  "attested_channels": sorted({w for t in claims
                                               for w in attested[c][t]}) if claims else [],
                  "perm6": rng.randrange(6)}
            if cond == "absent":
                absence_ct[ch] += 1
            items[split_of[c]].append(it)

    # cap + weight: equal channel mass; absence capped relative to attested
    for s in items:
        att = [it for it in items[s] if it["cond"] == "attested"]
        ab = [it for it in items[s] if it["cond"] == "absent"]
        by = collections.Counter(it["channel"] for it in att)
        rng.shuffle(ab)
        cap = {ch: max(1, int(by[ch] * a.absence_cap_frac)) for ch in CHS}
        kept_ab, used = [], collections.Counter()
        for it in ab:
            if used[it["channel"]] < cap.get(it["channel"], 0):
                kept_ab.append(it)
                used[it["channel"]] += 1
        items[s] = att + kept_ab
        for it in items[s]:
            base = 1.0 / max(by.get(it["channel"], 1), 1)
            it["weight"] = base * (0.3 if it["cond"] == "absent" else 1.0)

    # ---- evidence-swap twin pairs, mined WITHIN each split (train pairs feed
    # lam_ctm; test_id/test_ood pairs feed the flip/stay eval -- pairing across
    # splits would leak eval contexts into training)
    pairs, seen_pair = [], set()
    for s in items:
      split_cells = collections.defaultdict(list)
      for it in items[s]:
        if it["cond"] == "attested":
            split_cells[it["channel"]].append(it)
      for ch, lst in split_cells.items():
        by_topic = collections.defaultdict(list)
        for it in lst:
            for t in it["topics"]:
                by_topic[t].append(it)
        for it in lst:
            own = {t.casefold() for t in it["claims"]}
            mates_h = [m for t in it["topics"] for m in by_topic[t]
                       if m["chunk_id"] != it["chunk_id"]
                       and not (own & {x.casefold() for x in m["claims"]})]
            mates = mates_h[:3]
            hard = [True] * len(mates)
            if not mates:
                pool = [m for m in lst if m["chunk_id"] != it["chunk_id"]
                        and not (own & {x.casefold() for x in m["claims"]})]
                if pool:
                    mates, hard = [rng.choice(pool)], [False]
            for m, h in zip(mates, hard):
                key = (ch, *sorted((it["chunk_id"], m["chunk_id"])))
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                pairs.append({"channel": ch, "chunk_a": it["chunk_id"],
                              "chunk_b": m["chunk_id"], "hard": h, "split": s})

    os.makedirs(OUT, exist_ok=True)
    for s in items:
        with open(f"{OUT}/{s}.jsonl", "w") as f:
            for it in items[s]:
                f.write(json.dumps(it) + "\n")
    with open(f"{OUT}/pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    json.dump({c: {t: sorted(v) for t, v in m.items()} for c, m in attested.items()},
              open(f"{OUT}/attestation.json", "w"))

    man = {"seed": a.seed, "ood_topics": sorted(ood_topics),
           "n_topics": len(topics), "chunks": len(have),
           "chunks_missing_artifacts": sorted(missing),
           "dropped_untagged_claims": n_drop_untagged,
           "audio_non_speech_claims_excluded": n_ans,
           "splits": {s: dict(collections.Counter(
               (it["channel"], it["cond"]) for it in items[s]))
               and {f"{ch}|{cd}": sum(1 for it in items[s]
                                      if it["channel"] == ch and it["cond"] == cd)
                    for ch in CHS for cd in ("attested", "absent")
                    if any(it["channel"] == ch and it["cond"] == cd
                           for it in items[s])} for s in items},
           "neg_channel_rotation": {f"{ch}->{ng}": n
               for (ch, ng), n in collections.Counter(
                   (it["channel"], it["neg_channel"])
                   for it in items["train"] if it["neg_channel"]).items()},
           "pairs": {"total": len(pairs),
                     "hard": sum(p["hard"] for p in pairs),
                     "by_channel": dict(collections.Counter(p["channel"]
                                                            for p in pairs))},
           "asr_lang_train": dict(collections.Counter(
               it["asr_lang"] for it in items["train"]
               if it["channel"] == "ASR").most_common(12))}
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
    print(json.dumps(man["splits"], indent=1))
    print(f"pairs: {man['pairs']}")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
