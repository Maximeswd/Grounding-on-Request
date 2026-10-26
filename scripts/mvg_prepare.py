#!/usr/bin/env python3
"""Materialize the MultiVENT-G channel-grounding dataset (stage 1: DESC/OCR/VIS).

Reads  data/multiventG/claims.jsonl (channel-pure claims, QA'd),
       data/multiventG/frames_manifest.jsonl, ../multivent_g descriptions.
Writes data/mvg/{train,test_id,test_ood}.jsonl + manifest.json + weights.json.

Each ITEM = one (video, role, channel) claim, carrying EVERYTHING the collate
needs, with all anti-shortcut machinery decided HERE (reproducible, auditable):

  - sources: desc_text (full description), ocr_strings (video's OCR
    source_texts), vis_frame (path). All three sources presented per item.
  - cond agree|conflict (50/50 by hash bit). Conflict = the item's OWN channel
    source perturbed with a donor value (source-level, never the claim):
      DESC: answer span substituted inside desc_text (evidence.span)
      OCR : the claim's OCR string replaced by the donor's string
      VIS : vis_frame replaced by the donor claim's frame
    Donor: same role+channel+event_type, casefolded value differs, generic-head
    filter, drawn deterministically (hash-seeded). Gold ALWAYS = faithful
    report of the (possibly perturbed) source.
  - rotation: perm6 in 0..5 -> ordering of the three source blocks (position
    is never informative about channel).
  - independent bits: cond, perm6, donor choice, trust designation all come
    from different bytes of sha1(video_id|role|channel|claim_id).
  - answer casing NORMALIZED (casefold) in gold targets: OCR answers are
    lowercase and DESC answers cased in the raw claims -- a target-side
    channel marker a probe could read. Original casing kept in 'answer_raw'.
  - balancing: per-(event_type, role, channel) inverse-frequency weights
    written to weights.json (loader-side weighted sampling; cap 70/cell);
    structural zeros documented in manifest, never fabricated.
  - splits: by VIDEO. OOD-event = ALL videos of held-out event types
    (protests, launch_discovery). Remaining videos: 85/15 train/test_id.
"""
import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from scsrc.fps import frame_to_seconds

GENERIC_HEADS = {"person", "people", "man", "woman", "men", "women", "crowd",
                 "group", "event", "gathering", "attendee", "attendees"}
HELDOUT_TYPES = {"protests", "launch_discovery"}


def hbits(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).digest()


def head_noun(v):
    toks = re.findall(r"[a-z]+", str(v).lower())
    return toks[-1] if toks else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="data/multiventG/claims.jsonl")
    ap.add_argument("--mvg", default="data/multiventG/multivent_g.json")
    ap.add_argument("--frames", default="data/multiventG/frames_manifest.jsonl")
    ap.add_argument("--out", default="data/mvg")
    ap.add_argument("--cap", type=int, default=70)
    ap.add_argument("--id_frac", type=float, default=0.15)
    a = ap.parse_args()

    mvg = json.load(open(a.mvg))
    # 2026-10-02: the rebuilt annotation file (data/multiventG/multivent_g_rebuilt.json) carries no event_type; take it
    # from the existing materialized items (data/mvg_hard/*.jsonl), which record it per video. Videos without any item
    # (DESC-only claims) are dropped exactly as the original build dropped them.
    et = {v: mvg[v]["event_type"] for v in mvg if isinstance(mvg[v], dict) and "event_type" in mvg[v]}
    if len(et) < len(mvg):
        import glob as _g
        for _f in _g.glob("data/mvg_hard/*.jsonl"):
            for _l in open(_f):
                _r = json.loads(_l)
                if _r.get("event_type") and _r["video_id"] in mvg:
                    et.setdefault(_r["video_id"], _r["event_type"])
        print(f"[mvg_prepare] event_type from the annotation file for {sum(1 for v in mvg if 'event_type' in (mvg[v] if isinstance(mvg[v], dict) else {}))} videos, "
              f"from data/mvg_hard items for the rest -> {len(et)} of {len(mvg)} videos typed (untyped videos are dropped)")
        mvg = {v: mvg[v] for v in mvg if v in et}
    # 2026-10-02: the rebuilt json has description=None for some videos; drop them (no DESC block can be rendered)
    _ok = {v for v in mvg if isinstance(mvg[v], dict) and mvg[v].get("description")}
    if len(_ok) < len(mvg):
        print(f"[mvg_prepare] videos without a description dropped: {len(mvg) - len(_ok)}")
        mvg = {v: mvg[v] for v in _ok}; et = {v: et[v] for v in _ok if v in et}
    templates = {v: mvg[v]["template"] for v in mvg}
    desc = {v: mvg[v]["description"] for v in mvg}

    frames = defaultdict(list)          # video -> [(t, path)]
    for l in open(a.frames):
        r = json.loads(l)
        frames[r["video_id"]].append((r["t"], r["path"]))

    claims = [json.loads(l) for l in open(a.claims)]
    _n0 = len(claims); claims = [c for c in claims if c["video_id"] in et]     # 2026-10-02: untyped videos carry no items
    if _n0 != len(claims):
        print(f"[mvg_prepare] claims of untyped videos dropped: {_n0 - len(claims)} of {_n0}")
    by_vid = defaultdict(list)
    for c in claims:
        by_vid[c["video_id"]].append(c)

    # attestation map: (video, role, value) -> channel set (see
    # scripts/_checks/build_attestation.py). Items record attested_channels so
    # (a) the loader can down-weight cross-attested AGREE items (26% of them are
    # non-discriminative for routing: every token would answer the same gold);
    # (b) evals can restrict mismatched-token tests to uniquely-grounded facts.
    _att = {}
    try:
        for _v, _fl in json.load(open("data/multiventG/attestation.json")).items():
            for _f in _fl:
                for _val in _f["values"]:
                    _att[(_v, _f["role"], _val)] = _f["channels"]
    except Exception:
        pass

    # SPATIAL INDEX for the bbox join. bboxes live only in the raw annotations
    # (14,297 entries, percent coords 0-100 as [x0,y0,x1,y1]); the claims file
    # carries certainty but not bbox. Both were being dropped at prep, which is
    # why MVG could not use AAT (gaze supervision) or certainty weighting.
    _spx = defaultdict(list)
    for _v, _m in mvg.items():
        for _e in (_m.get("spatial") or []):
            if _e.get("bbox") and _e.get("frame") is not None:
                _spx[(_v, str(_e.get("role", "")))].append(_e)

    def bbox_for(c):
        """Nearest-in-time spatial box for this claim's (video, role), preferring
        an entity-string match with the claim's answer. Returns percent coords."""
        v, role = c["video_id"], str(c.get("role", ""))
        cands = _spx.get((v, role)) or []
        if not cands:
            return None, None
        ans = str(c.get("answer", "")).strip().lower()
        ev = c.get("evidence", {}) or {}
        t = None
        if ev.get("time"):
            t0, t1 = ev["time"]; t = (t0 + t1) / 2
        elif ev.get("frame") is not None:
            t = frame_to_seconds(v, ev["frame"])
        exact = [e for e in cands
                 if str(e.get("entity", "")).strip().lower() == ans] or cands
        if t is None:
            e = exact[0]
        else:
            e = min(exact, key=lambda x: abs(frame_to_seconds(v, x["frame"]) - t))
        return e.get("bbox"), e.get("certainty")

    def frame_for(c):
        """Nearest extracted frame to the claim's evidence time/frame."""
        v = c["video_id"]
        if not frames[v]:
            return None
        ev = c.get("evidence", {})
        t = None
        if ev.get("time"):
            t0, t1 = ev["time"]; t = (t0 + t1) / 2
        elif ev.get("frame") is not None:
            t = frame_to_seconds(v, ev["frame"])
        if t is None:
            return frames[v][0][1]
        return min(frames[v], key=lambda x: abs(x[0] - t))[1]

    # donor pools: (event_type, role, channel) -> [claims]
    pools = defaultdict(list)
    for c in claims:
        pools[(et[c["video_id"]], c["role"], c["channel"])].append(c)

    def pick_donor(c, h):
        pool = pools[(et[c["video_id"]], c["role"], c["channel"])]
        cands = [d for d in pool
                 if d["video_id"] != c["video_id"]
                 and str(d["answer"]).casefold() != str(c["answer"]).casefold()
                 and head_noun(d["answer"]) not in GENERIC_HEADS
                 and (c["channel"] != "VIS" or frame_for(d))]
        if not cands:
            return None
        return cands[h % len(cands)]

    items, skipped = [], Counter()
    for c in claims:
        v, role, ch = c["video_id"], c["role"], c["channel"]
        bits = hbits(v, role, ch, c["claim_id"])
        cond = "conflict" if bits[0] % 2 else "agree"
        perm6 = bits[1] % 6
        vis_frame = frame_for(c) if ch == "VIS" else None
        # every item presents ALL sources; pick a representative frame for
        # non-VIS items too (first frame) so the VIS block is never empty
        item_frame = vis_frame or (frames[v][0][1] if frames[v] else None)
        attested = _att.get((v, c["role"], str(c["answer"]).lower().strip()),
                            [c["channel"]])
        if item_frame is None:
            skipped["no_frames"] += 1
            continue
        ocr_strings = sorted({x["source_text"] for x in by_vid[v]
                              if x["channel"] == "OCR"})[:8]
        d_text = desc[v][:1200]
        donor = None
        if cond == "conflict":
            donor = pick_donor(c, bits[2] + 256 * bits[3])
            if donor is None:
                cond = "agree"          # no valid donor: fall back, record it
                skipped["no_donor"] += 1
        gold = str(c["answer"]).casefold()
        pert = {}
        if cond == "conflict":
            dval_raw = str(donor["answer"])
            dval = dval_raw.casefold()
            if ch == "DESC":
                sp = c.get("evidence", {}).get("span")
                src = c["source_text"]
                if sp and desc[v][sp[0]:sp[1]] == src:
                    d_text = (desc[v][:sp[0]] + dval_raw
                              + desc[v][sp[1]:])[:1200]
                else:                    # span drifted: substring fallback
                    if src in d_text:
                        d_text = d_text.replace(src, dval_raw, 1)
                    else:
                        skipped["desc_span_miss"] += 1
                        continue
                pert = {"donor_id": donor["claim_id"], "donor_value": dval}
                gold = dval
            elif ch == "OCR":
                ocr_strings = [dval_raw if s == c["source_text"] else s
                               for s in ocr_strings]
                if c["source_text"] not in set(json.loads(json.dumps(ocr_strings))) \
                        and dval_raw not in ocr_strings:
                    ocr_strings = [dval_raw] + ocr_strings[:7]
                pert = {"donor_id": donor["claim_id"], "donor_value": dval}
                gold = dval
            else:                        # VIS: swap the frame
                item_frame = frame_for(donor)
                pert = {"donor_id": donor["claim_id"], "donor_value": dval,
                        "donor_frame": item_frame}
                gold = dval
        items.append({
            "item_id": c["claim_id"], "video_id": v, "role": role,
            "channel": ch, "attested_channels": attested, "event_type": et[v], "cond": cond,
            "role_question": templates[v].get(role, ""),
            "desc_text": d_text, "ocr_strings": ocr_strings,
            "vis_frame": item_frame, "perm6": perm6,
            "gold": gold, "answer_raw": str(c["answer"]),
            "claim": c["claim"], "perturbation": pert,
            "trust_bit": bits[4] % 3,
            # NEW: carried through for gaze supervision (AAT) and certainty
            # weighting. bbox is percent coords [x0,y0,x1,y1]; VIS items only.
            "bbox": (bbox_for(c)[0] if ch == "VIS" else None),
            "certainty": (c.get("certainty")
                          if c.get("certainty") is not None
                          else (bbox_for(c)[1] if ch == "VIS" else None)),
        })

    # splits by video
    vids = sorted({i["video_id"] for i in items})
    ood_vids = {v for v in vids if et[v] in HELDOUT_TYPES}
    rest = [v for v in vids if v not in ood_vids]
    def vh(v): return hbits("split", v)[0] / 255.0
    test_id_vids = {v for v in rest if vh(v) < a.id_frac}
    split_of = {}
    for v in vids:
        split_of[v] = ("test_ood" if v in ood_vids
                       else "test_id" if v in test_id_vids else "train")

    os.makedirs(a.out, exist_ok=True)
    out = {s: open(os.path.join(a.out, f"{s}.jsonl"), "w")
           for s in ("train", "test_id", "test_ood")}
    counts = Counter()
    cell = Counter()
    for i in items:
        s = split_of[i["video_id"]]
        k = (i["event_type"], i["role"], i["channel"])
        if s == "train" and cell[k] >= a.cap:
            counts["capped"] += 1
            continue
        if s == "train":
            cell[k] += 1
        out[s].write(json.dumps(i, ensure_ascii=False) + "\n")
        counts[s] += 1
    for f in out.values():
        f.close()

    # loader-side balancing weights (train only): inverse cell frequency
    weights = {f"{e}|{r}|{c}": round(1.0 / n, 6) for (e, r, c), n in cell.items()}
    json.dump(weights, open(os.path.join(a.out, "weights.json"), "w"), indent=1)
    man = {"counts": dict(counts), "skipped": dict(skipped),
           "train_cells": len(cell), "cap": a.cap,
           "heldout_types": sorted(HELDOUT_TYPES),
           "cond_balance": dict(Counter(i["cond"] for i in items)),
           "deviant_by_channel": dict(Counter(
               i["channel"] for i in items if i["cond"] == "conflict")),
           "structural_zero_cells": sorted(
               f"{e}|{r}|{c}" for e in set(et.values())
               for r in {i["role"] for i in items}
               for c in ("DESC", "OCR", "VIS")
               if (e, r, c) not in pools or not pools[(e, r, c)])[:100]}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("counts", "skipped", "cond_balance",
                                          "deviant_by_channel")}, indent=1))


if __name__ == "__main__":
    main()
