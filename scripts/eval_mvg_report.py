#!/usr/bin/env python
"""Stage-2 <report> eval: one-pass provenance-tagged claim generation on MVG.

Metrics mirror the MultiVENT-G paper (Sanders et al., EMNLP-Findings 2024) so
our numbers sit next to theirs:
  P0/R0/F0      span-retrieval style, macro-averaged over role fillers F:
                P0 = 1/|F| sum_f |Rf n Sf|/|Sf| (relaxed intersection, below)
  PC/RC/FC      CEAF-RME: Kuhn-Munkres bipartite matching of predicted to gold
                role fillers WITHIN each role, phi = token-Dice of the values,
                a pair counts when phi >= --phi (their 'partially overlapping
                entities' relaxation); micro P/R/F over the matching
  provenance    OUR addition, the grounding metric: among CEAF-matched claims,
                P(predicted channel tag == gold channel). The paper's nearest
                analogue is the per-entity OCR flag; no baseline reports it.
  queries/video latency column: report token = 1; assembler = 3 x |roles|.

Modes:
  report     (default) stage-2 run_dir with <report_ctrl>; one decode per video
  assembler  --assembler_ckpt = stage-1 run (desc/ocr/vis tokens); one decode
             per (channel, role) from the event-type role inventory (built on
             TRAIN videos only), 'unavailable'-style answers dropped
  frozen     --frozen_model_id: base model, English instruction + explicit
             format spec (tier-2 row, same parser and metrics)
Eval videos = report items whose video_id is NOT in data/mvg/train.jsonl:
unseen by the report token AND by the stage-1 tokens the assembler uses
(the mvg3 test files are item-level, not video-level, so they cannot serve
as the stage-2 holdout).
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from scsrc.mvg import _video_items, REPORT_INSTR
from scsrc.tokens import trigger
from scsrc.mvg import mvg_word, assert_words_registered
from evaluate import apply_checkpoint
from scripts.eval_mvg import user_content

CH = ("DESC", "OCR", "VIS")
CH_TOK = {"DESC": "desc_ctrl", "OCR": "ocr_ctrl", "VIS": "vis_ctrl"}
UNAVAIL = {"unavailable", "unknown", "none", "", "n/a", "not available",
           "unknown operator"}
LINE_RE = re.compile(r"^\s*\[?\s*(DESC|OCR|VIS)\s*\]?\s*[:\-]?\s*([^:]+?)\s*:\s*(.+?)\s*$",
                     re.IGNORECASE)


def parse_report(text):
    """'[CH] role: value' lines -> deduped (channel, role, value) triples."""
    seen, out = set(), []
    for line in text.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        t = (m.group(1).upper(), m.group(2).strip().lower(),
             m.group(3).strip().lower().rstrip("."))
        if t[2] in UNAVAIL or t in seen:
            continue
        seen.add(t); out.append(t)
    return out


def dice(a, b):
    ta, tb = set(a.split()), set(b.split())
    return 2 * len(ta & tb) / max(len(ta) + len(tb), 1)


def hungarian(cost):
    try:
        from scipy.optimize import linear_sum_assignment
        return list(zip(*linear_sum_assignment(cost)))
    except Exception:                       # greedy fallback
        import numpy as np
        c = np.array(cost, dtype=float, copy=True); pairs = []
        while c.size and (c.min() < 1e8):
            i, j = divmod(int(c.argmin()), c.shape[1])
            pairs.append((i, j)); c[i, :] = 1e9; c[:, j] = 1e9
        return pairs


def match_video(pred, gold, phi_thr):
    """CEAF-RME within-role matching. Returns (matched pairs, prov_hits)."""
    matched, prov = [], 0
    roles = {r for _, r, _ in gold} | {r for _, r, _ in pred}
    for r in roles:
        P = [p for p in pred if p[1] == r]
        G = [g for g in gold if g[1] == r]
        if not P or not G:
            continue
        cost = [[1.0 - dice(p[2], g[2]) for g in G] for p in P]
        for i, j in hungarian(cost):
            if dice(P[i][2], G[j][2]) >= phi_thr:
                matched.append((P[i], G[j]))
                prov += int(P[i][0] == G[j][0])
    return matched, prov


@torch.no_grad()
def gen(model, proc, umsg, device, max_new=300):
    # enable_thinking=False: Qwen3.5 templates default to a reasoning block in
    # free generation, which eats the budget before any claim line (LL evals
    # are unaffected). Gemma templates ignore the kwarg.
    try:
        enc = proc.apply_chat_template(umsg, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt",
                                       enable_thinking=False).to(device)
    except (TypeError, ValueError):
        enc = proc.apply_chat_template(umsg, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt").to(device)
    kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    out = model.generate(**kw, max_new_tokens=max_new, do_sample=False)
    return proc.decode(out[0, enc["input_ids"].shape[1]:],
                       skip_special_tokens=True)


def report_msg(item, trig):
    umsg = user_content(item, trig)
    txt = umsg[0]["content"][-1]["text"].rsplit("\n", 1)[0]
    umsg[0]["content"][-1]["text"] = txt + \
        f"\n{REPORT_INSTR} Give me a {trig} answer."
    return umsg


def frozen_msg(item):
    umsg = user_content(item, "PLACEHOLDER")
    txt = umsg[0]["content"][-1]["text"].rsplit("\n", 1)[0]
    umsg[0]["content"][-1]["text"] = txt + (
        f"\n{REPORT_INSTR} Format each line exactly as "
        "'[DESC] role: value', '[OCR] role: value' or '[VIS] role: value' "
        "according to which source grounds the claim.")
    return umsg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--composed_ckpt", default=None,
                    help="run trained with --report_channel_words: decode one "
                         "[CH]-tagged claim list PER CHANNEL WORD and concatenate "
                         "(3 queries/video). Matched-supervision composition test: "
                         "same target format as the report word, split by channel.")
    ap.add_argument("--assembler_ckpt", default=None,
                    help="stage-1 run_dir: multi-pass baseline instead of <report>")
    ap.add_argument("--frozen_model_id", default=None,
                    help="'base' or HF id: instructed frozen baseline")
    ap.add_argument("--claims", default="data/multiventG/claims.jsonl")
    ap.add_argument("--mvg", default="data/multiventG/multivent_g.json")
    ap.add_argument("--frames", default="data/multiventG/frames_manifest.jsonl")
    ap.add_argument("--train_split", default="data/mvg/train.jsonl")
    ap.add_argument("--max_n", type=int, default=60)
    ap.add_argument("--phi", type=float, default=0.5)
    ap.add_argument("--attestation", default="data/multiventG/attestation.json",
                    help="fact -> channel-set map. 14% of facts are attested by "
                         ">1 channel; gold is deduplicated to FACT level and "
                         "provenance scores tag-IN-SET, so a model that states a "
                         "shared fact once with either valid tag is not punished.")
    ap.add_argument("--ablate", default="none", choices=["none", "vis", "ocr", "desc"],
                    help="GROUNDING TEST: remove one channel from the prompt. If the "
                         "claims tagged with that channel do NOT collapse, the report "
                         "token is generating from a prior, not grounding.")
    ap.add_argument("--swap", default="none", choices=["none", "vis", "ocr", "desc"],
                    help="CAUSAL TEST: replace one channel's content with another "
                         "video's. Claims tagged with that channel should follow the "
                         "swapped evidence (flip); the other channels should stay.")
    a = ap.parse_args()

    # FACT -> ATTESTING-CHANNEL-SET map. Defined as an arg since the start but
    # never loaded -> `att` was undefined at the fact-collapse step and every
    # report eval crashed with NameError (killed 6/9 tasks of wave 426195).
    att = {}
    try:
        _raw = json.load(open(a.attestation))
        for _vid, _facts in _raw.items():
            for _f in _facts:
                _chs = {str(c).upper() for c in (_f.get("channels") or [])}
                for _v in (_f.get("values") or [_f.get("value")]):
                    if _v is None:
                        continue
                    _k = (str(_vid), str(_f.get("role", "")).strip().lower(),
                          str(_v).strip().lower())
                    att.setdefault(_k, set()).update(_chs)
        print(f"[attest] {len(att)} (video,role,value) -> channel-set entries")
    except FileNotFoundError:
        print(f"[attest] {a.attestation} missing -- falling back to "
              f"single-channel gold (provenance is then stricter, not wrong)")

    held = {json.loads(l)["video_id"] for l in open(a.train_split)}
    items = [it for it in _video_items(a.claims, a.mvg, a.frames)
             if it["video_id"] not in held][:a.max_n]
    print(f"[report-eval] {len(items)} held-out videos")
    if a.ablate != "none" or a.swap != "none":
        import random as _r
        donors = list(items)
        _r.Random(11).shuffle(donors)
        for k, it in enumerate(items):
            don = donors[(k + 1) % len(donors)]
            if a.ablate == "desc":
                it["desc_text"] = ""
            elif a.ablate == "ocr":
                it["ocr_strings"] = []
            elif a.ablate == "vis":
                it["vis_frame"] = None          # handled below
            if a.swap == "desc":
                it["desc_text"] = don["desc_text"]
            elif a.swap == "ocr":
                it["ocr_strings"] = don["ocr_strings"]
            elif a.swap == "vis":
                it["vis_frame"] = don["vis_frame"]
        if a.ablate == "vis":
            # a blank frame keeps the prompt structure identical while removing
            # all visual evidence (structure-matched ablation)
            from PIL import Image
            blank = os.path.join(os.path.dirname(items[0]["vis_frame"] or "."), "_blank.jpg") \
                if items[0]["vis_frame"] else "/tmp/_blank.jpg"
            blank = "data/multiventG/_blank_ablate.jpg"
            if not os.path.exists(blank):
                Image.new("RGB", (448, 448), (127, 127, 127)).save(blank)
            for it in items:
                it["vis_frame"] = blank
        print(f"[report-eval] ablate={a.ablate} swap={a.swap}")

    if a.frozen_model_id:
        from scsrc import model as M
        model, proc = M.load(model_id=None if a.frozen_model_id == "base"
                             else a.frozen_model_id)
        model.eval(); mode, n = "frozen", 0
    elif a.composed_ckpt:
        model, proc, meta, _ = apply_checkpoint(a.composed_ckpt, a.step, "cuda")
        mode, n = "composed", meta["n_tokens"]
        model, proc, meta, _ = apply_checkpoint(a.assembler_ckpt, a.step, "cuda")
        mode, n = "assembler", meta["n_tokens"]
        templates = {v: (m.get("template") or {})
                     for v, m in json.load(open(a.mvg)).items()}
        # role inventory per event type from TRAIN videos only (no leakage)
        from collections import defaultdict
        ev = {v: m["event_type"] for v, m in json.load(open(a.mvg)).items()}
        roles_by_ev = defaultdict(set)
        for l in open(a.claims):
            c = json.loads(l)
            if c["video_id"] in held and c["video_id"] in ev:
                roles_by_ev[ev[c["video_id"]]].add(c["role"])
        # OOD event types (protests/launch_discovery) have EMPTY train
        # inventories -- the assembler cannot query them at all, while the
        # report token generalizes role names. Fall back to the train-wide
        # role UNION (still leakage-safe) so the assembler gets a fair shot.
        all_roles = sorted(set().union(*roles_by_ev.values())) if roles_by_ev else []
    else:
        model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
        mode, n = "report", meta["n_tokens"]

    S = {"tp_exact": 0, "n_pred": 0, "n_gold": 0, "ceaf_m": 0, "prov": 0,
         "queries": 0}
    p0s, r0s = [], []
    pervid = []
    per_ch = {c: [0, 0] for c in CH}          # [matched, gold] recall by channel
    dumps = []
    for it in tqdm(items, desc=f"mvg-report({mode})"):
        gold = [(c, r, v) for c, r, v in it["gold_claims"]]
        if mode == "composed":
            pred, q = [], 0
            for ch in CH:
                um = report_msg(it, mvg_word(CH_TOK[ch], n))
                raw = gen(model, proc, um, "cuda")
                pred += parse_report(raw); q += 1
            if len(dumps) < 5:
                dumps.append({"video": it["video_id"], "raw": raw[:400]})
        elif mode == "assembler":
            pred, q = [], 0
            tpl = templates.get(it["video_id"], {})
            for ch in CH:
                for role, q in sorted(tpl.items()):
                    # the TEMPLATE question = the phrasing the stage-1 words
                    # were trained with (mvg_prepare uses templates[v][role])
                    it2 = {**it, "role_question": q}
                    txt = gen(model, proc,
                              user_content(it2, mvg_word(CH_TOK[ch], n)),
                              "cuda", max_new=40)
                    q += 1
                    m = re.search(r"final answer:\s*(.+)", txt, re.I)
                    v = (m.group(1) if m else txt).strip().lower().rstrip(".")
                    if v and v not in UNAVAIL:
                        pred.append((ch, role.lower(), v))
        else:
            umsg = (frozen_msg(it) if mode == "frozen"
                    else report_msg(it, mvg_word("report_ctrl", n)))
            raw = gen(model, proc, umsg, "cuda")
            pred, q = parse_report(raw), 1
            if len(dumps) < 5:
                dumps.append({"video": it["video_id"], "raw": raw[:600]})
        pred = list(dict.fromkeys(pred))
        # snap free-text roles onto the video's closed template-role set
        troles = sorted({r for _, r, _ in gold})
        def snap(role):
            best, bs = role, 0.0
            for tr in troles:
                s = dice(role.replace("-", " "), tr.replace("-", " "))
                if s > bs:
                    best, bs = tr, s
            return best if bs >= 0.5 else role
        pred = [(c, snap(r), v) for c, r, v in pred]
        # FACT-LEVEL gold: collapse channel-duplicates of the same (role, value)
        # into one line carrying the attested channel SET
        seenf, gold_f = set(), []
        for c, r, v in gold:
            chs = att.get((str(it["video_id"]), str(r).strip().lower(),
                           str(v).strip().lower()), {c})
            key = (r, v)
            if key in seenf:
                continue
            seenf.add(key)
            gold_f.append((tuple(sorted(chs)), r, v))
        gold = [(chs[0], r, v) for chs, r, v in gold_f]      # matching uses role+value
        chset = {(r, v): set(chs) for chs, r, v in gold_f}
        matched, prov = match_video(pred, gold, a.phi)
        # RESCORE provenance set-valued: predicted tag correct if IN the set
        prov = sum(1 for p, g in matched
                   if p[0] in chset.get((g[1], g[2]), {g[0]}))
        m_strict, _ = match_video(pred, gold, 0.7)
        S["ceaf_strict"] = S.get("ceaf_strict", 0) + len(m_strict)
        pervid.append({"video": it["video_id"], "etype": it["event_type"],
                       "n_pred": len(pred), "n_gold": len(gold),
                       "n_matched": len(matched), "prov": prov})
        S["tp_exact"] += len({(r, v) for _, r, v in pred} &
                             {(r, v) for _, r, v in gold})
        S["n_pred"] += len(pred); S["n_gold"] += len(gold)
        S["ceaf_m"] += len(matched); S["prov"] += prov; S["queries"] += q
        mg = {g for _, g in matched}
        for c, r, v in gold:
            per_ch[c][1] += 1; per_ch[c][0] += int((c, r, v) in mg)
        # paper-style macro over role fillers: one filler = (video, role)
        for role in {r for _, r, _ in gold} | {r for _, r, _ in pred}:
            mr = sum(1 for p, g in matched if g[1] == role)
            sp_ = sum(1 for p in pred if p[1] == role)
            rf = sum(1 for g in gold if g[1] == role)
            if sp_: p0s.append(mr / sp_)
            if rf: r0s.append(mr / rf)

    f = lambda p, r: round(2 * p * r / max(p + r, 1e-9), 3)
    P0 = sum(p0s) / max(len(p0s), 1); R0 = sum(r0s) / max(len(r0s), 1)
    PC = S["ceaf_m"] / max(S["n_pred"], 1); RC = S["ceaf_m"] / max(S["n_gold"], 1)
    pe = S["tp_exact"] / max(S["n_pred"], 1); re_ = S["tp_exact"] / max(S["n_gold"], 1)
    out = {"run": os.path.basename(a.assembler_ckpt or a.run_dir), "mode": mode,
           "ablate": a.ablate, "swap": a.swap,
           "step": a.step, "n_videos": len(items), "phi": a.phi,
           "P0": round(P0, 3), "R0": round(R0, 3), "F0": f(P0, R0),
           "PC": round(PC, 3), "RC": round(RC, 3), "FC": f(PC, RC),
           "P_exact": round(pe, 3), "R_exact": round(re_, 3), "F_exact": f(pe, re_),
           "FC_phi07": f(S.get("ceaf_strict", 0) / max(S["n_pred"], 1),
                         S.get("ceaf_strict", 0) / max(S["n_gold"], 1)),
           "provenance_acc": round(S["prov"] / max(S["ceaf_m"], 1), 3),
           "recall_by_channel": {c: round(m / max(g, 1), 3)
                                 for c, (m, g) in per_ch.items()},
           "claims_pred_per_video": round(S["n_pred"] / max(len(items), 1), 1),
           "claims_gold_per_video": round(S["n_gold"] / max(len(items), 1), 1),
           "queries_per_video": round(S["queries"] / max(len(items), 1), 1)}
    print(json.dumps(out, indent=1))
    outd = os.path.join(a.run_dir, "eval", "mvg_report")
    os.makedirs(outd, exist_ok=True)
    tag = mode + ("" if a.ablate == "none" else f"_ablate-{a.ablate}") \
               + ("" if a.swap == "none" else f"_swap-{a.swap}")
    json.dump({**out, "per_video": pervid, "samples": dumps},
              open(os.path.join(outd, f"step_{a.step}_{tag}.json"), "w"), indent=1)
    print(f"[saved] {outd}/step_{a.step}_{tag}.json")


if __name__ == "__main__":
    main()
