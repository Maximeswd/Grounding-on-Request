#!/usr/bin/env python
"""MVG 3-way selectivity eval — the null-selectivity fix.

For each aligned item, query with each PRESENT channel token and force-choose
over the PRESENT channel values (the distractor is another channel's real
value). A selective token picks its OWN channel's value. Builds the routing
matrix M[query c][true-source c'] = P(<c> picks v_{c'}); the diagonal should
dominate. selectivity = mean(diagonal) - mean(off-diagonal).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from collections import defaultdict, Counter
from scsrc.tokens import trigger
from scsrc.mvg import mvg_word, assert_words_registered
from evaluate import apply_checkpoint
# reuse the multi-block MVG prompt + scorer
from scripts.eval_mvg import user_content, loglik_mvg, cand, blocks, _PERMS

def _words(s):
    """Word set, punctuation-stripped -- the unit containment is tested on."""
    import re
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


CH = {"DESC": "desc_ctrl", "OCR": "ocr_ctrl", "VIS": "vis_ctrl"}

# The baseline carriers. `mvg_word()` only makes sense for a run that OWNS those
# rows: on a frozen/instruction run n_tokens=0, so a neologism trigger would
# tokenize into subwords and the eval would return plausible-looking garbage
# (the bug that silently invalidated the gen-arm evals). Each style therefore
# builds its own user turn. Wording mirrors scsrc/mvraw.py INSTR/NOOP so the
# MVG and MVR baselines are the same baseline.
INSTR = {
    "VIS":  ("Answer using ONLY the video imagery. Ignore the description and "
             "the on-screen text."),
    "OCR":  ("Answer using ONLY the on-screen text. Ignore the video imagery "
             "and the description."),
    "DESC": ("Answer using ONLY the description. Ignore the video imagery and "
             "the on-screen text."),
}
NOOP = "Answer the question."


def user_content_style(item, qc, n, style):
    """neo -> the learned word; instr -> English source-naming; noop -> neither.

    noop is the measurement that matters for the modality-default question: the
    request carries NO source at all, so whatever the model answers from IS its
    fixed default. If the three query channels then return identical rows,
    selectivity is 0 by construction and the default is unaddressable.
    """
    if style == "neo":
        return user_content(item, mvg_word(CH[qc], n))
    from PIL import Image
    b = blocks(item); order = _PERMS[item["perm6"] % 6]
    content, buf = [], []
    for x in order:
        if x == "vis":
            if buf:
                content.append({"type": "text", "text": "\n".join(buf)}); buf = []
            content.append({"type": "image",
                            "image": Image.open(item["vis_frame"]).convert("RGB")})
        else:
            buf.append(b[x])
    tail = INSTR[qc] if style == "instr" else NOOP
    buf.append(f'{item["role_question"]} {tail}' if item.get("role_question") else tail)
    content.append({"type": "text", "text": "\n".join(buf)})
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--data_dir", default="data/mvg3")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=200)
    ap.add_argument("--single_attested", type=int, default=1,
                    help="restrict to facts attested by exactly ONE channel; a\n                         multi-attested fact cannot show which source was read")
    ap.add_argument("--require_distinct", type=int, default=1,
                    help="skip (item,query) cells whose value is shared by "
                         "another present channel -- those are decided by dict "
                         "order, not by the model")
    ap.add_argument("--style", default="neo", choices=["neo", "instr", "noop"],
                    help="neo = the learned word (default); instr = English "
                         "source-naming instruction; noop = no source named at "
                         "all, i.e. the model's fixed modality default")
    ap.add_argument("--tag", default=None,
                    help="output filename stem (default: sel, or sel_<style>)")
    ap.add_argument("--frozen_model_id", default=None,
                    help="skip checkpoints: FROZEN base model; --run_dir only "
                         "names the output location. Requires --style instr|noop "
                         "(a frozen model has no word rows to trigger).")
    a = ap.parse_args()

    if a.frozen_model_id is not None:
        assert a.style != "neo", "--frozen_model_id requires --style instr|noop"
        from scsrc import model as M
        model, proc = M.load(model_id=(a.frozen_model_id or None)
                             if a.frozen_model_id != "base" else None)
        model.eval()
        n = 0
    else:
        model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
        n = meta["n_tokens"]
    items = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))][:a.max_n]

    # M[query][picked] counts; and hit = query picks its own value
    M = defaultdict(lambda: defaultdict(int)); tot = defaultdict(int)
    diag, offd = defaultdict(list), defaultdict(list)
    skipped = defaultdict(int); skipped_multi = defaultdict(int)
    arity = defaultdict(int)          # how many channels were on offer
    # (video, role, value) -> attesting channels. Same map mvg_prepare.py built.
    ATT = {}
    try:
        for _v, _fl in json.load(open("data/multiventG/attestation.json")).items():
            for _f in _fl:
                for _val in _f["values"]:
                    ATT[(_v, _f["role"], str(_val))] = _f["channels"]
    except Exception as _e:
        print(f"[warn] no attestation map ({_e}); --single_attested disabled")
    for it in tqdm(items, desc="mvg3"):
        present = it["present"]; vals = it["channel_values"]
        # THE COMMENT HERE USED TO CLAIM A DEDUP THAT WAS NEVER IMPLEMENTED.
        # If two present channels hold the SAME value string they receive an
        # identical log-likelihood, so `max()` resolves the tie by dict insertion
        # order: one channel wins for free and the other cannot be picked at all,
        # regardless of what the model did. That is 6.6% of test_id items and
        # 10.6% of test_ood, and it is an artifact of iteration order, not a
        # measurement. Such (item, query) cells are now excluded and counted.
        # Exact equality catches only a third of it. "earthquake" vs "a 7.0
        # magnitude earthquake" is not a forced choice between two SOURCES, it is
        # a choice between two spellings of one answer -- and word-set
        # containment finds those: 97 further cells in test_id on top of the 46
        # exact ones (29.7% together), and 38.9% in test_ood.
        norm = {c: _words(vals[c]) for c in present}
        for qc in present:                                  # query channel
            if a.require_distinct and any(
                    c != qc and (norm[qc] <= norm[c] or norm[c] <= norm[qc])
                    for c in present):
                skipped[qc] += 1
                continue
            # SINGLE-ATTESTED ONLY. mvg_prepare.py:84 records attested_channels
            # so "evals can restrict mismatched-token tests to uniquely-grounded
            # facts" -- and nothing ever did. If the queried value is attested by
            # more than one channel, answering it correctly says nothing about
            # WHICH source was read, so the cell cannot measure routing. 446 of
            # 3,253 facts are multi-attested.
            if a.single_attested and len(ATT.get((it["video_id"], it.get("role"),
                                                  str(vals[qc])), ["?"])) > 1:
                skipped_multi[qc] += 1
                continue
            arity[len(present)] += 1
            um = user_content_style(it, qc, n, a.style)
            scores = {c: loglik_mvg(model, proc, um, cand(vals[c]), "cuda")
                      for c in present}
            picked = max(scores, key=scores.get)            # which channel's value won
            M[qc][picked] += 1; tot[qc] += 1
            for c in present:
                (diag if c == qc else offd)[qc].append(int(picked == c))
    if sum(skipped.values()):
        print(f"   [distinct] excluded {sum(skipped.values())} (item,query) cells whose "
              f"value is shared by another channel: {dict(skipped)}")

    print(f"\n== MVG 3-way selectivity  {os.path.basename(a.run_dir)}  {a.split} step {a.step}")
    print("   routing matrix  P(query picks source's value):")
    print("           picks:  DESC   OCR    VIS")
    for qc in ("DESC", "OCR", "VIS"):
        row = "  ".join(f"{(M[qc][pc]/tot[qc]):.2f}" if tot[qc] else " -- "
                        for pc in ("DESC", "OCR", "VIS"))
        print(f"   <{qc:4s}> ({tot[qc]:3d}):  {row}")
    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "n": len(items),
           "matrix": {qc: {pc: round(M[qc][pc] / tot[qc], 3) for pc in ("DESC", "OCR", "VIS") if tot[qc]}
                      for qc in ("DESC", "OCR", "VIS") if tot[qc]}}
    d = [x for qc in diag for x in diag[qc]]; o = [x for qc in offd for x in offd[qc]]
    out["diag_acc"] = m(d); out["offdiag_acc"] = m(o)
    out["require_distinct"] = bool(a.require_distinct)
    out["skipped_value_collision"] = dict(skipped)
    # CHANCE IS NOT 1/3. 78% of items offer only two channels, so the blended
    # floor is ~.46, and a diagonal of .509 is +.05 over chance rather than the
    # +.18 it appears to be against a 3-way prior. Reported explicitly so the
    # number can never be read against the wrong floor again.
    n_cells = sum(arity.values())
    chance = (sum(v_ / k_ for k_, v_ in arity.items()) / n_cells
              if n_cells else None)
    out["arity"] = dict(arity)
    out["chance"] = round(chance, 4) if chance is not None else None
    out["above_chance"] = (round(out["diag_acc"] - chance, 4)
                           if chance is not None else None)
    out["skipped_multi_attested"] = dict(skipped_multi)
    out["single_attested"] = bool(a.single_attested)
    out["selectivity"] = round(m(d) - m(o), 3)
    print(f"   diag(own) {out['diag_acc']}  off-diag {out['offdiag_acc']}  "
          f"SELECTIVITY {out['selectivity']}")
    print(f"   cells by #channels offered {dict(arity)}  ->  CHANCE {out['chance']}  "
          f"ABOVE CHANCE {out['above_chance']}")
    if sum(skipped_multi.values()):
        print(f"   [single-attested] excluded {sum(skipped_multi.values())} cells whose "
              f"value is attested by >1 channel: {dict(skipped_multi)}")
    outd = os.path.join(a.run_dir, "eval", f"mvg3_{a.split}"); os.makedirs(outd, exist_ok=True)
    stem = a.tag or ("sel" if a.style == "neo" else f"sel_{a.style}")
    out["style"] = a.style
    json.dump(out, open(os.path.join(outd, f"{stem}_step_{a.step}.json"), "w"), indent=1)
    print(f"[saved] {outd}/{stem}_step_{a.step}.json")


if __name__ == "__main__":
    main()
