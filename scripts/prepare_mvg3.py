#!/usr/bin/env python3
"""MVG 3-way (aligned-triple) materialization — fixes the null-selectivity.

The per-channel eval's distractor (answer_raw) is absent from all sources, so
any token beats it → selectivity ~0. Here the distractor is ANOTHER CHANNEL's
PRESENT value: group claims by (video, role); where >=2 channels have a value
for the same (video, role), that group is a selectivity item {DESC:v, OCR:v,
VIS:v}. <desc> should pick v_DESC over v_OCR/v_VIS, etc. — a real 3-way (or
2-way) routing test. Optional per-channel perturbation (donor) gives the 3x3
flip/stay; the natural cross-channel disagreement already gives selectivity.

Writes data/mvg3/{test_id,test_ood}.jsonl. Eval-only (uses stage-1 tokens).
"""
import argparse, hashlib, json, os
from collections import defaultdict

HELDOUT = {"protests", "launch_discovery"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="data/multiventG/claims.jsonl")
    ap.add_argument("--mvg", default="data/multiventG/multivent_g.json")
    ap.add_argument("--frames", default="data/multiventG/frames_manifest.jsonl")
    ap.add_argument("--out", default="data/mvg3")
    a = ap.parse_args()

    mvg = json.load(open(a.mvg))
    et = {v: mvg[v]["event_type"] for v in mvg}
    desc = {v: mvg[v]["description"][:1200] for v in mvg}
    templates = {v: mvg[v]["template"] for v in mvg}
    frames = defaultdict(list)
    for l in open(a.frames):
        r = json.loads(l); frames[r["video_id"]].append((r["t"], r["path"]))

    claims = [json.loads(l) for l in open(a.claims)]
    by = defaultdict(dict)      # (video,role) -> {channel: (value, claim)}
    ocr_by_vid = defaultdict(set)
    for c in claims:
        if c["channel"] == "OCR":
            ocr_by_vid[c["video_id"]].add(c.get("source_text", ""))
        by[(c["video_id"], c["role"])].setdefault(c["channel"], c)

    def frame_for(c, v):
        if not frames[v]:
            return None
        ev = c.get("evidence", {}); t = None
        if ev.get("time"):
            t = sum(ev["time"]) / 2
        elif ev.get("frame") is not None:
            t = ev["frame"] / 30.0
        return (min(frames[v], key=lambda x: abs(x[0] - t))[1] if t is not None
                else frames[v][0][1])

    items = []
    for (v, role), ch in by.items():
        present = [x for x in ("DESC", "OCR", "VIS") if x in ch]
        vals = {x: str(ch[x]["answer"]).strip().lower() for x in present}
        if len(present) < 2 or len(set(vals.values())) < 2:
            continue                                    # need >=2 channels, disagreeing
        vis_claim = ch.get("VIS") or next(iter(ch.values()))
        vf = frame_for(vis_claim, v)
        if vf is None:
            continue
        items.append({
            # perm6 = deterministic source-block rotation (eval_mvg.user_content)
            "perm6": int(hashlib.sha1(f"{v}|{role}".encode()).hexdigest()[:6], 16) % 6,
            "video_id": v, "role": role, "event_type": et.get(v, "?"),
            "role_question": templates.get(v, {}).get(role, ""),
            "desc_text": desc.get(v, ""),
            "ocr_strings": sorted(ocr_by_vid[v])[:8],
            "vis_frame": vf, "channel_values": vals, "present": present,
            "n_distinct": len(set(vals.values()))})

    os.makedirs(a.out, exist_ok=True)
    counts = {}
    for split in ("test_id", "test_ood"):
        rows = [it for it in items
                if (it["event_type"] in HELDOUT) == (split == "test_ood")]
        counts[split] = len(rows)
        with open(os.path.join(a.out, f"{split}.jsonl"), "w") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    man = {"task": "mvg3", "n_items": len(items), "counts": counts,
           "n_triples": sum(1 for it in items if len(it["present"]) == 3),
           "note": "aligned (video,role) groups w/ >=2 disagreeing channels; "
                   "distractor = other-channel PRESENT value (fixes null selectivity)"}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
