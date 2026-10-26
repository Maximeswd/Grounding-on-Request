#!/usr/bin/env python3
"""MVG counterfactual-twin pairs mined from MultiVENT-G's own annotations.

The insight (user, 2026-07-23): the spatial annotations (entity + frame + bbox)
and temporal role spans give WITHIN-VIDEO counterfactuals -- things true in one
frame and absent/different in another -- which is exactly the twin structure
CTM/flip-stay need, per (event_type, role, channel):

  vis_value_swap : same (video, role), two spatial entries at distant frames
                   with DIFFERENT entities -> frame A vs frame B, golds
                   entity_A vs entity_B (+ bboxes for the causal-crop metric).
                   The strongest pair: everything about the video constant,
                   only the evidence frame changes.
  vis_presence   : a frame inside the role's temporal span vs a frame >=5s
                   outside ALL spans of that role -> evidence present/absent;
                   gold flips value -> 'unavailable' (honesty target).
  ocr_ablate     : the OCR claim's source string present vs removed from the
                   on-screen-text block (text channels live in the PROMPT, so
                   their twins are exact string edits).
  desc_swap      : the DESC claim's answer replaced in the description by a
                   donor value from another video of the same (event_type,
                   role) -> exact text twin with a real alternative value.

Writes data/mvg_ctm/pairs.jsonl + a per-(event_type, role, channel, type)
count manifest. Usable for BOTH CTM-style training margins and flip/stay eval.
"""
import argparse, json, os, re
from collections import defaultdict, Counter
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from scsrc.fps import frame_to_seconds, fps_of

FPS = 30.0   # fallback only; real per-video fps comes from scsrc.fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvg", default="data/multiventG/multivent_g.json")
    ap.add_argument("--claims", default="data/multiventG/claims.jsonl")
    ap.add_argument("--frames", default="data/multiventG/frames_manifest.jsonl")
    ap.add_argument("--out", default="data/mvg_ctm")
    ap.add_argument("--min_gap_s", type=float, default=3.0)
    ap.add_argument("--max_frame_err_s", type=float, default=1.0,
                    help="reject a pair if the nearest EXTRACTED frame is more "
                         "than this far from the ANNOTATED frame. nearest() "
                         "previously had no bound, so a 'positive frame' could "
                         "be seconds away from where the entity was annotated.")
    ap.add_argument("--min_certainty", type=int, default=4,
                    help="annotator certainty floor (1-5). Was captured and "
                         "never used.")
    ap.add_argument("--max_box_iou", type=float, default=0.5,
                    help="reject pairs whose two boxes cover nearly the same "
                         "region -- there is no localised difference to see.")
    ap.add_argument("--exclusive_window_s", type=float, default=2.0,
                    help="THE BIG ONE: reject a pair if the OTHER side's entity "
                         "is annotated within this window of THIS side's time. "
                         "If both values are visible at both times no frame swap "
                         "can flip the answer -- the twin is degenerate. 72% of "
                         "the original vis_value_swap pairs failed this test, "
                         "which is why vis-flip sat at chance.")
    ap.add_argument("--out_suffix", default="",
                    help="write pairs<suffix>.jsonl (keep the old set for A/B)")
    ap.add_argument("--presence_margin_s", type=float, default=5.0)
    a = ap.parse_args()

    mvg = json.load(open(a.mvg))
    frames = defaultdict(list)
    for l in open(a.frames):
        r = json.loads(l); frames[r["video_id"]].append((r["t"], r["path"]))
    for v in frames:
        frames[v].sort()

    claims = [json.loads(l) for l in open(a.claims)]
    claim_by = defaultdict(dict)      # (video, role) -> {channel: claim}
    ocr_by_vid = defaultdict(set)
    for c in claims:
        if c["channel"] == "OCR":
            ocr_by_vid[c["video_id"]].add(c.get("source_text", ""))
        claim_by[(c["video_id"], c["role"])].setdefault(c["channel"], c)

    def nearest(v, t):
        fs = frames.get(v)
        if not fs:
            return None
        return min(fs, key=lambda x: abs(x[0] - t))

    def _iou(b1, b2):
        if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
            return 0.0
        c = lambda v: max(0.0, min(100.0, float(v)))
        ax0, ay0, ax1, ay1 = [c(v) for v in b1[:4]]
        bx0, by0, bx1, by1 = [c(v) for v in b2[:4]]
        ix0, iy0 = max(min(ax0, ax1), min(bx0, bx1)), max(min(ay0, ay1), min(by0, by1))
        ix1, iy1 = min(max(ax0, ax1), max(bx0, bx1)), min(max(ay0, ay1), max(by0, by1))
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        a1 = abs(ax1 - ax0) * abs(ay1 - ay0); a2 = abs(bx1 - bx0) * abs(by1 - by0)
        u = a1 + a2 - inter
        return inter / u if u > 0 else 0.0

    # entity -> sorted times it is annotated as VISIBLE, per video (exclusivity)
    seen_t = defaultdict(lambda: defaultdict(list))
    for _v, _meta in mvg.items():
        for _e in (_meta.get("spatial") or []):
            if _e.get("frame") is not None and _e.get("entity"):
                seen_t[_v][str(_e["entity"]).strip().lower()].append(
                    _e["frame"] / fps_of(_v, FPS))
    for _v in seen_t:
        for _e in seen_t[_v]:
            seen_t[_v][_e].sort()

    def _exclusive(v, ent, t, win):
        """True if `ent` is NOT annotated within `win` seconds of time t."""
        for tt in seen_t.get(v, {}).get(ent, ()):
            if abs(tt - t) <= win:
                return False
        return True

    REJ = Counter()
    pairs = []

    def add(pt, v, role, ch, ctx_a, ctx_b, gold_a, gold_b, extra=None):
        pairs.append({"pair_type": pt, "video_id": v, "role": role,
                      "event_type": mvg[v]["event_type"], "channel": ch,
                      "ctx_a": ctx_a, "ctx_b": ctx_b,
                      "gold_a": gold_a, "gold_b": gold_b, **(extra or {})})

    # donor pool for desc_swap: (event_type, role) -> distinct DESC answers
    donor = defaultdict(set)
    for (v, role), ch in claim_by.items():
        if "DESC" in ch:
            donor[(mvg[v]["event_type"], role)].add(
                str(ch["DESC"]["answer"]).strip().lower())

    for v, meta in mvg.items():
        sp = meta.get("spatial") or []
        tp = meta.get("temporal") or []
        if not frames.get(v):
            continue
        # ---- vis_value_swap: role-grouped spatial entries, distinct entities
        by_role = defaultdict(list)
        for e in sp:
            if e.get("bbox") and e.get("frame") is not None and not e.get("ocr_flag"):
                by_role[e["role"]].append(e)
        for role, es in by_role.items():
            es = sorted(es, key=lambda e: e["frame"])
            for i in range(len(es)):
                for j in range(i + 1, len(es)):
                    ea, eb = es[i], es[j]
                    ta, tb = ea["frame"] / fps_of(v, FPS), eb["frame"] / fps_of(v, FPS)
                    va = ea["entity"].strip().lower(); vb = eb["entity"].strip().lower()
                    if abs(tb - ta) < a.min_gap_s or va == vb:
                        continue
                    fa, fb = nearest(v, ta), nearest(v, tb)
                    if fa is None or fb is None or fa[1] == fb[1]:
                        REJ["no_frame"] += 1; continue
                    if (abs(fa[0] - ta) > a.max_frame_err_s
                            or abs(fb[0] - tb) > a.max_frame_err_s):
                        REJ["frame_too_far"] += 1; continue
                    if min(ea.get("certainty") or 3,
                           eb.get("certainty") or 3) < a.min_certainty:
                        REJ["low_certainty"] += 1; continue
                    if _iou(ea.get("bbox"), eb.get("bbox")) > a.max_box_iou:
                        REJ["box_overlap"] += 1; continue
                    # EXCLUSIVITY: each value must be visible at ITS OWN time and
                    # NOT at the other side's time, or the swap cannot flip.
                    if not (_exclusive(v, vb, fa[0], a.exclusive_window_s)
                            and _exclusive(v, va, fb[0], a.exclusive_window_s)):
                        REJ["not_exclusive"] += 1; continue
                    add("vis_value_swap", v, role, "VIS",
                        {"frame": fa[1], "t": fa[0]}, {"frame": fb[1], "t": fb[0]},
                        va, vb, {"bbox_a": ea["bbox"], "bbox_b": eb["bbox"],
                                 "certainty": min(ea.get("certainty") or 3,
                                                  eb.get("certainty") or 3)})
                    break   # one pair per anchor entry keeps the set balanced
        # ---- vis_presence: in-span vs far-outside-span frame per role
        spans = defaultdict(list)
        for e in tp:
            if e.get("time"):
                spans[e["role"]].append(tuple(e["time"]))
        for role, ss in spans.items():
            cl = claim_by.get((v, role), {}).get("VIS")
            if cl is None:
                continue
            t_in = sum(ss[0]) / 2
            fin = nearest(v, t_in)
            fout = None
            for (t, p) in frames[v]:
                if all(t < s0 - a.presence_margin_s or t > s1 + a.presence_margin_s
                       for (s0, s1) in ss):
                    fout = (t, p); break
            if fin is None or fout is None or fin[1] == fout[1]:
                continue
            add("vis_presence", v, role, "VIS",
                {"frame": fin[1], "t": fin[0]}, {"frame": fout[1], "t": fout[0]},
                str(cl["answer"]).strip().lower(), "unavailable")
        # ---- ocr_ablate: remove the answer-bearing on-screen string
        for role, ch in ((r, c) for (vv, r), c in claim_by.items() if vv == v):
            oc = ch.get("OCR")
            if oc and oc.get("source_text") and oc["source_text"] in ocr_by_vid[v]:
                strings = sorted(ocr_by_vid[v])[:8]
                without = [s for s in strings if s != oc["source_text"]]
                add("ocr_ablate", v, role, "OCR",
                    {"ocr_strings": strings}, {"ocr_strings": without},
                    str(oc["answer"]).strip().lower(), "unavailable")
            # ---- desc_swap: replace the answer span with a donor value
            dc = ch.get("DESC")
            if dc:
                ans = str(dc["answer"]).strip()
                desc = meta["description"][:1200]
                m = re.search(re.escape(ans), desc, re.IGNORECASE)
                cands = [d for d in donor[(meta["event_type"], role)]
                         if d != ans.lower()]
                if m and cands:
                    dv = sorted(cands)[hash(v + role) % len(cands)]
                    add("desc_swap", v, role, "DESC",
                        {"desc": desc}, {"desc": desc[:m.start()] + dv + desc[m.end():]},
                        ans.lower(), dv)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"pairs{a.out_suffix}.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    cnt = Counter((p["pair_type"]) for p in pairs)
    by_cell = Counter((p["event_type"], p["role"], p["channel"], p["pair_type"])
                      for p in pairs)
    man = {"n_pairs": len(pairs), "by_type": dict(cnt),
           "n_cells": len(by_cell),
           "filters": {"max_frame_err_s": a.max_frame_err_s,
                       "min_certainty": a.min_certainty,
                       "max_box_iou": a.max_box_iou,
                       "exclusive_window_s": a.exclusive_window_s,
                       "min_gap_s": a.min_gap_s},
           "rejected": dict(REJ),
           "note": "within-video counterfactual twins; vis pairs carry bboxes "
                   "for the causal-crop grounding metric"}
    json.dump({**man, "by_cell": {"|".join(k): c for k, c in by_cell.items()}},
              open(os.path.join(a.out, f"manifest{a.out_suffix}.json"), "w"), indent=1)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
