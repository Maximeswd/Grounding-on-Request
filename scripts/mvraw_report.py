#!/usr/bin/env python
"""multivent-raw ORACLE report generation with channel words -> leaderboard
submission format.

For each query: oracle chunks = query.qrels positives. Every chunk is run
through the three trained channel words (vis/ocr/asr), each asked to list its
own source's grounded claims. The report is the deduplicated union of claim
lines, each carrying its chunk citation -- extraction-IS-the-report mode (the
model shares all its observations per modality, per chunk; no rewriting stage,
so every sentence is traceable by construction).

Output layout (matches amartin__dummy-test-multivent-raw exactly):
    predictions/<name>/run.json    {query_id: [{"text", "citations",
                                    "claims": [{"claim", "decontextualized_claim"}]}]}
    predictions/<name>/meta.json   {run_name, submitter, submitted_at,
                                    eval_type, metrics, mode, notes}

Submission itself (copy into <cluster>/evals/submissions/generation/oracle/
${USER}__<name>/ + chmod -R g+w) is deliberately MANUAL: check the run first.
"""
import argparse, collections, datetime, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from scsrc.mvraw import mvraw_user_content, ABSENT
from scsrc.tokens import trigger
from evaluate import apply_checkpoint
from scripts.eval_mvraw import gen_blocks, _norm

R = os.environ.get("MVR_ROOT", "data/multivent_raw_src")
CHS = ("vis", "ocr", "asr")


def load_chunk_item(chunk, art="data/multivent_raw/artifacts"):
    """Minimal item dict for mvraw_user_content from the extracted artifacts."""
    from scripts.mvraw_prepare import asr_text_of, ocr_strings_of, frames_of
    frames = frames_of(chunk)
    if not frames:
        return None
    asr_txt, asr_lang = asr_text_of(chunk)
    return {"chunk_id": chunk, "asr_text": asr_txt, "asr_lang": asr_lang,
            "ocr_strings": ocr_strings_of(chunk), "frames": frames, "perm6": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--name", default=None, help="submission name (default: run basename)")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--n_tasks", type=int, default=1)
    ap.add_argument("--max_queries", type=int, default=0, help="0 = all 222")
    ap.add_argument("--style", default="neo", choices=["neo", "instr", "noop"],
                    help="neo = the channel words (the submission we care about); "
                         "instr = English source-naming instruction; noop = no "
                         "source named (floor). instr/noop need --frozen_model_id.")
    ap.add_argument("--frozen_model_id", default=None,
                    help="run the base model with no learned rows (requires "
                         "--style instr|noop)")
    a = ap.parse_args()
    dev = "cuda"
    # BASELINE MODE. Without this the leaderboard has nothing to compare the
    # words against: every claim would be word-generated, so a CiteF1 number
    # would be unattributable. instr = the same per-channel extraction driven by
    # an English source-naming instruction; noop = one generic extraction with no
    # source named at all (the floor).
    if a.frozen_model_id is not None:
        assert a.style != "neo", "--frozen_model_id requires --style instr|noop"
        from scsrc import model as M
        model, proc = M.load(model_id=(a.frozen_model_id or None)
                             if a.frozen_model_id != "base" else None)
        model.eval(); n = 0
    else:
        model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, dev)
        n = meta["n_tokens"]
    name = a.name or (os.path.basename(a.run_dir.rstrip("/"))
                      + ("" if a.style == "neo" else f"_{a.style}"))

    per_q = collections.defaultdict(set)
    for ln in open(f"{R}/annotations/query.qrels"):
        q, _, doc, rel = ln.split()
        if int(rel) > 0:
            per_q[q].add(doc)
    qids = sorted(per_q)
    if a.max_queries:
        qids = qids[:a.max_queries]
    qids = qids[a.task_id::a.n_tasks]

    outd = f"predictions/{name}"
    os.makedirs(outd, exist_ok=True)
    run = {}
    part = f"{outd}/run_part{a.task_id}.json"
    for q in tqdm(qids, desc="queries"):
        sents, seen = [], set()
        for chunk in sorted(per_q[q]):
            it = load_chunk_item(chunk)
            if it is None:
                continue
            chans = CHS if a.style != "noop" else CHS[:1]   # noop: one pass
            for ch in chans:
                if a.style == "neo":
                    trig = trigger(f"{ch}_ctrl", n)
                    ct = mvraw_user_content(it, trig, None, a.n_frames)
                else:
                    from scsrc.mvraw import INSTR, NOOP
                    ct = mvraw_user_content(it, "", None, a.n_frames,
                                            a.style, ch)
                txt = gen_blocks(model, proc, ct, dev, max_new=220)
                for line in txt.strip().split("\n"):
                    line = line.strip("-* \t").strip()
                    if (not line or _norm(line) == _norm(ABSENT)
                            or len(line.split()) < 3):
                        continue
                    key = _norm(line)
                    if key in seen:
                        continue
                    seen.add(key)
                    sents.append({"text": line, "citations": [chunk],
                                  "claims": [{"claim": line,
                                              "decontextualized_claim": line}]})
        run[q] = sents
    json.dump(run, open(part, "w"), indent=1)
    print(f"[saved] {part} ({len(run)} queries)")

    # merge when all parts are present
    parts = [f for f in os.listdir(outd) if re.match(r"run_part\d+\.json$", f)]
    if len(parts) == a.n_tasks:
        merged = {}
        for f in sorted(parts):
            merged.update(json.load(open(f"{outd}/{f}")))
        json.dump(merged, open(f"{outd}/run.json", "w"), indent=1)
        json.dump({"run_name": name, "submitter": os.environ.get("USER", "user"),
                   "submitted_at": datetime.datetime.now().isoformat(),
                   "eval_type": "reference", "metrics": "info cite",
                   "mode": "oracle",
                   "notes": ("neologism channel words (vis/ocr/asr) extraction-"
                             "as-report; one claim line per sentence, chunk-level "
                             f"citations; ckpt {a.run_dir}@{a.step}")},
                  open(f"{outd}/meta.json", "w"), indent=1)
        print(f"[merged] {outd}/run.json ({len(merged)} queries) + meta.json")
        print(f"[submit] cp -r {outd} <cluster>/evals/submissions/generation/"
              f"oracle/$USER__{name}/ && chmod -R g+w "
              f"<cluster>/evals/submissions/generation/oracle/$USER__{name}")


if __name__ == "__main__":
    main()
