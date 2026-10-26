#!/usr/bin/env python
"""Stage 1 of the MultiVENT-G pipeline: turn multivent_g.json into per-video
claim-generation prompts (for a strong LLM) + a manifest of gold annotations.

Usage:
  python scripts/prepare_multiventg.py --json data/multiventG/multivent_g.json \
      [--out data/multiventG] [--languages english] [--min_certainty 4]

Writes:
  data/multiventG/prompts/<video_id>.txt   one self-contained prompt per video
  data/multiventG/gold_manifest.jsonl      the filtered annotations we prompted on
  data/multiventG/claims/                  (empty) paste LLM outputs here, one
                                           <video_id>.jsonl each; cat into claims.jsonl
"""
import argparse, json, os

PROMPT_HEADER = """You convert GOLD annotations of news videos into atomic, verifiable claims.
You may reason first. Then write the line CLAIMS: and after it output ONLY JSON Lines
(one object per annotation, all videos in this batch), no prose, no markdown fences.
Use ONLY the information in each annotation (plus its role question). NEVER add
facts, names, numbers, or context from elsewhere in the record.

For EACH annotation below, output ONE claim:
- a single declarative sentence, present tense, at most 15 words;
- keep the annotated wording of the entity/span (resolve pronouns only if the
  role question forces it);
- if an annotation cannot yield a self-contained factual sentence, emit a skip
  object instead. Do not merge annotations. Do not invent claims.

OUTPUT: JSON Lines, exactly one object per annotation, no prose, no markdown:
{"video_id": "...", "claim_id": "<video_id>#<channel>#<n>", "role": "...",
 "channel": "DESC|OCR|VIS", "claim": "...", "source_text": "...",
 "evidence": {"span": [s,e]} | {"frame": N} | {"time": [t0,t1]},
 "certainty": <int or null>}
or {"video_id": "...", "skipped": "<role>", "reason": "..."}

RULES for quality: claims state facts about the EVENT, not about the annotation
("Floods are occurring", never "The description mentions floods"); no world
knowledge beyond the annotation (do not expand names/abbreviations); strip
hashtag marks (#Australia -> Australia); OCR claims paraphrase the transcribed
string only.

EXAMPLE -- annotation:
{"n": 1, "channel": "DESC", "role": "where", "source_text": "the coastal city of Townsville", "evidence": {"span": [254, 284]}, "role_question": "Where is the emergency occurring?"}
output line:
{"video_id": "qDUDMd_yfgk", "claim_id": "qDUDMd_yfgk#DESC#1", "role": "where", "channel": "DESC", "claim": "The emergency is occurring in the coastal city of Townsville.", "source_text": "the coastal city of Townsville", "evidence": {"span": [254, 284]}, "certainty": null}

Quality rules:
- begin the JSON Lines IMMEDIATELY after the line CLAIMS: (nothing between);
- number claim_id per channel in annotation order (#DESC#1, #DESC#2, ...);
- hashtags: drop the '#' in the claim text, keep source_text verbatim;
- ALL-CAPS or lowercase spans: write the claim in normal sentence case;
- near-duplicate annotations still get one claim EACH (no merging);
- the claim must read as an answer to the role_question;
- evidence/certainty may be omitted (they are re-joined from gold).

Example annotation -> claim:
{"n": 3, "channel": "OCR", "role": "effect", "source_text": "u.s. downed $32 million drone in black sea", ...}
CLAIMS:
{"video_id": "X", "claim_id": "X#OCR#1", "role": "effect", "channel": "OCR", "claim": "The U.S. downed a $32 million drone in the Black Sea.", "source_text": "u.s. downed $32 million drone in black sea"}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="data/multiventG")
    ap.add_argument("--languages", default="english")
    ap.add_argument("--min_certainty", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8, help="videos per batch prompt file")
    a = ap.parse_args()
    langs = {x.strip().lower() for x in a.languages.split(",")}
    data = json.load(open(a.json))
    pdir = os.path.join(a.out, "prompts"); os.makedirs(pdir, exist_ok=True)
    os.makedirs(os.path.join(a.out, "claims"), exist_ok=True)
    man = open(os.path.join(a.out, "gold_manifest.jsonl"), "w")
    kept = skipped_vids = 0
    batch_blocks = []
    for vid, rec in data.items():
        if (rec.get("metadata", {}).get("language") or "").lower() not in langs:
            skipped_vids += 1
            continue
        template = rec.get("template", {})
        items = []
        for t in rec.get("text", []):        # DESC channel
            items.append({"channel": "DESC", "role": t["role"], "source_text": t["text"],
                          "evidence": {"span": t.get("index")}, "certainty": None})
        # temporal spans indexed by role for VIS evidence lookup
        tspans = {}
        for t in rec.get("temporal", []):
            tspans.setdefault(t["role"], []).append(t.get("time"))
        for sp in rec.get("spatial", []):
            cert = sp.get("certainty")
            if cert is not None and cert < a.min_certainty:
                continue
            ch = "OCR" if sp.get("ocr_flag") else "VIS"
            ev = {"frame": sp.get("frame")}
            if not sp.get("ocr_flag") and tspans.get(sp["role"]):
                ev["time"] = tspans[sp["role"]][0]
            items.append({"channel": ch, "role": sp["role"], "source_text": sp["entity"],
                          "evidence": ev, "certainty": cert})
        if not items:
            continue
        kept += 1
        lines = [PROMPT_HEADER, f'VIDEO_ID: {vid}', f'EVENT_TYPE: {rec.get("event_type")}',
                 "ROLE QUESTIONS: " + json.dumps(template), "", "ANNOTATIONS:"]
        for i, it in enumerate(items, 1):
            q = template.get(it["role"], "")
            lines.append(json.dumps({"n": i, **it, "role_question": q}))
            man.write(json.dumps({"video_id": vid, **it}) + "\n")
        open(os.path.join(pdir, f"{vid}.txt"), "w").write("\n".join(lines) + "\n")
        batch_blocks.append("\n".join(lines[1:]))   # header goes once per batch
    man.close()
    bdir = os.path.join(a.out, "batches"); os.makedirs(bdir, exist_ok=True)
    for i in range(0, len(batch_blocks), a.batch):
        blk = batch_blocks[i:i + a.batch]
        open(os.path.join(bdir, f"batch_{i // a.batch + 1:04d}.txt"), "w").write(
            PROMPT_HEADER + "\n\n" + "\n\n=== NEXT VIDEO ===\n\n".join(blk) + "\n")
    print(f"[multiventG] {(len(batch_blocks) + a.batch - 1) // a.batch} batch prompts "
          f"({a.batch} videos each) -> {bdir}")
    print(f"[multiventG] prompts written for {kept} videos "
          f"({skipped_vids} skipped by language) -> {pdir}")
    print(f"[multiventG] paste LLM outputs into {a.out}/claims/<video_id>.jsonl, then:")
    print(f"             cat {a.out}/claims/*.jsonl > {a.out}/claims.jsonl")

if __name__ == "__main__":
    main()
