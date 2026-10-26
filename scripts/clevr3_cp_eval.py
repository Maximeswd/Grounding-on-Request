#!/usr/bin/env python
"""Ground-CF conflict-preserving protocol with ONE candidate set per item (STATUS v1.94, 2026-09-22).

For every conflict item (image value A != caption value B) and its four-skill image twin (value A', where one exists):
  caption-edit value  w  = a third value no shown source states (deterministic, skill vocabulary)
  colliding twins (A' == B) move the caption to a fresh value in BOTH contexts at the item level, so both requests
  share the same item and the same candidate set
  Y = {A, B, w, A'}   -- scored in the original, the image-edited and the caption-edited context, for both requests

  <look>: Accuracy = pick(orig) == A;  Flip = Accuracy and pick(image twin) == A';  Stay = pick(caption edit) == pick(orig)
  <read>: Accuracy = pick(orig) == B;  Flip = Accuracy and pick(caption edit) == w; Stay = pick(image twin) == pick(orig)
  conjunctive Stay (*_c) additionally requires the original pick to be correct; leak = a text value picked on <look>.
Chance is 1/|Y|. Denominators: Accuracy over all items; Flip/Stay of <look> and Stay of <read> over items with a twin.
The presence diagnostic (gray image) is kept when a callable is supplied. Detector margins are the log-likelihood
of the gold minus the alternative (w for the caption, B for the image) in the original and the edited context.
"""
import collections
from scsrc.clevr3 import _edited_item, cp_text_edit_value, cp_fix_look_collision, set_c3_vocab, eval_design
from scsrc.tokens import c3_sources


def cp_protocol(rows, pairs, pick, R, SK, PK, max_n=10**9, presence=None):
    # v1.161 (author: old-vs-new on the same checkpoints and items). SC_GCF_AGREE_TWIN=1 adds, for the image request, an
    # AGREEMENT version of the image twin: the edited image (value tv) with the caption ALSO set to tv, as the old
    # agreement-resolving twins had it. look_flip_agree = correct on the original AND tv on that version (old-style
    # Flip), next to look_flip (conflict-preserving: caption still says B). SC_GCF_LOOK_ONLY=1 skips the caption request
    # (the diagnostic needs only the image request). Tag these files _agree: master_grid.py never harvests them.
    import os as _os
    AGREE = _os.environ.get("SC_GCF_AGREE_TWIN", "0") == "1"
    LOOK_ONLY = _os.environ.get("SC_GCF_LOOK_ONLY", "0") == "1"
    """Fills R (dict of lists), SK (skill -> dict of lists) and PK (picks) in place; returns the item count."""
    set_c3_vocab(rows)
    act = c3_sources()
    twin_by_id = {p["id"]: p for p in pairs}
    seen = 0
    for i, it in enumerate(rows):
        if seen >= max_n:
            break
        ch = {k: str(v) for k, v in it["channels"].items()}
        if ch["look"] == ch["read"]:
            continue
        pr = twin_by_id.get(it["id"])
        tv = str(pr["gold_b"]) if pr else None
        if tv is not None and tv in {ch[k] for k in ch if k in act and k != "look"}:
            fixed, _moved = cp_fix_look_collision(it, tv)
            if fixed is None:
                R.setdefault("look_cp_unfixable", []).append(1); tv, pr = None, None
            else:
                R.setdefault("look_cp_moved", []).append(1); it = fixed
                ch = {k: str(v) for k, v in it["channels"].items()}
        w, fb = cp_text_edit_value(it, "read")
        if w is None:
            R.setdefault("cp_skipped", []).append(1); continue
        R.setdefault("cp_fallback_read", []).append(int(fb))
        seen += 1
        A, B = ch["look"], ch["read"]
        Y = tuple(dict.fromkeys([A, B, w] + ([tv] if tv else [])))
        if __import__("os").environ.get("SC_CP_CANDS", "twin") == "full":
            # 2026-10-02 (protocol note item 6): the FULL skill inventory as the candidate set -- every permitted colour /
            # shape / count / relation -- with the item's own values first so the twin candidates keep their order.
            # chance = 1/|V| (cand_n records it). Opt-in: eval_clevr3_flip.py --cands full (needs --tag).
            from scsrc.clevr3 import _C3_VOCAB as _V
            Y = tuple(dict.fromkeys(list(Y) + [str(x) for x in _V.get(it.get("skill"), [])]))
        R.setdefault("cand_n", []).append(len(Y))
        sk = it.get("skill", "?")
        base = it
        cap_edit = _edited_item(it, "read", w)
        img_twin = None
        if pr:
            img_twin = dict(it); img_twin["image"] = pr["ctx_b"]["image"]
            chx = dict(img_twin["channels"]); chx["look"] = tv; img_twin["channels"] = chx
        # ---- image request
        pick.ctx = "base"   # v1.143: context label for the item log
        p0, s0 = pick(base, "look", Y, i)
        pick.ctx = "capedit"   # v1.143: context label for the item log
        pc, s_c = pick(cap_edit, "look", Y, i)
        acc = int(p0 == A)
        R["look_acc"].append(acc); SK[sk]["look_acc"].append(acc)
        R["look_stay"].append(int(pc == p0)); SK[sk]["look_stay"].append(int(pc == p0))
        R["look_stay_c"].append(int(acc and pc == A)); SK[sk]["look_stay_c"].append(int(acc and pc == A))
        R["look_leak_text"].append(int(p0 in (B, w)))
        R.setdefault("look_margin", []).append(float(s0[A] - s0[B]))
        _r4 = lambda d: {str(k): round(float(v), 4) for k, v in d.items()}   # v1.130: keep every candidate score
        rec_look = {"id": it["id"], "skill": sk, "p0": p0, "p_capedit": pc, "A": A, "B": B, "w": w, "Y": list(Y),
                    "scores": {"base": _r4(s0), "capedit": _r4(s_c)}}
        if img_twin is not None:
            pick.ctx = "twin"   # v1.143: context label for the item log
            pt_, s_t = pick(img_twin, "look", Y, i)
            R["look_twin_acc"].append(acc); SK[sk]["look_twin_acc"].append(acc)
            R["look_flip"].append(int(acc and pt_ == tv)); SK[sk]["look_flip"].append(int(acc and pt_ == tv))
            rec_look.update({"p_twin": pt_, "twin": tv}); rec_look["scores"]["twin"] = _r4(s_t)
            if AGREE:
                agree = _edited_item(img_twin, "read", tv)          # image AND caption now say tv
                pick.ctx = "twin_agree"
                pa_, s_a = pick(agree, "look", Y, i)
                R.setdefault("look_flip_agree", []).append(int(acc and pa_ == tv))
                SK[sk].setdefault("look_flip_agree", []).append(int(acc and pa_ == tv))
                rec_look.update({"p_twin_agree": pa_}); rec_look["scores"]["twin_agree"] = _r4(s_a)
        # v1.131 (author): each context's source values, gold and the gold's score, explicit per record
        rec_look["values"] = {"base": {"image": A, "caption": B}, "capedit": {"image": A, "caption": w}}
        rec_look["gold"] = {"base": A, "capedit": A}
        if img_twin is not None:
            rec_look["values"]["twin"] = {"image": tv, "caption": B}; rec_look["gold"]["twin"] = tv
            if AGREE:
                rec_look["values"]["twin_agree"] = {"image": tv, "caption": tv}; rec_look["gold"]["twin_agree"] = tv
        rec_look["lp_gold"] = {c: rec_look["scores"][c].get(g) for c, g in rec_look["gold"].items()}
        rec_look["cap_first"] = eval_design(it, i)[1]                  # v1.133: block order of this item's prompts
        PK["look"].append(rec_look)
        if LOOK_ONLY:
            continue
        # ---- caption request
        pick.ctx = "base"   # v1.143: context label for the item log
        r0, t0 = pick(base, "read", Y, i)
        pick.ctx = "capedit"   # v1.143: context label for the item log
        rc, tc = pick(cap_edit, "read", Y, i)
        racc = int(r0 == B)
        R["read_acc"].append(racc); SK[sk]["read_acc"].append(racc)
        R["read_flip"].append(int(racc and rc == w)); SK[sk]["read_flip"].append(int(racc and rc == w))
        R.setdefault("read_margin", []).append(float(t0[B] - t0[w]))
        R.setdefault("read_margin_edit", []).append(float(tc[B] - tc[w]))
        rec_read = {"id": it["id"], "skill": sk, "p0": r0, "p1": rc, "old": B, "new": w, "Y": list(Y),
                    "scores": {"base": _r4(t0), "capedit": _r4(tc)}}
        if img_twin is not None:
            pick.ctx = "twin"   # v1.143: context label for the item log
            rt, t_t = pick(img_twin, "read", Y, i)
            R["read_stay"].append(int(rt == r0)); SK[sk]["read_stay"].append(int(rt == r0))
            R["read_stay_c"].append(int(racc and rt == B)); SK[sk]["read_stay_c"].append(int(racc and rt == B))
            R.setdefault("read_stay_img", []).append(int(rt == r0))
            rec_read.update({"p_twin": rt}); rec_read["scores"]["twin"] = _r4(t_t)
        rec_read["values"] = {"base": {"image": A, "caption": B}, "capedit": {"image": A, "caption": w}}
        rec_read["gold"] = {"base": B, "capedit": w}
        if img_twin is not None:
            rec_read["values"]["twin"] = {"image": tv, "caption": B}; rec_read["gold"]["twin"] = B
        rec_read["lp_gold"] = {c: rec_read["scores"][c].get(g) for c, g in rec_read["gold"].items()}
        rec_read["cap_first"] = rec_look["cap_first"]
        if len(PK["read"]) < 50 and getattr(pick, "last_prompts", None):
            rec_look["prompt_base"] = pick.last_prompts.get(("look", it["id"]))
            rec_read["prompt_base"] = pick.last_prompts.get(("read", it["id"]))
        PK["read"].append(rec_read)
        if presence is not None:
            R["look_presence_drop"].append(float(presence(base)))
    return seen


def macro_over_skills(SK, keys=("look_acc", "look_flip", "look_stay_c", "read_acc", "read_flip", "read_stay_c")):
    """Unweighted mean over skills of each metric (skills with no rows are left out), so unequal twin availability
    does not let one attribute dominate the summary."""
    out = {}
    for k in keys:
        vals = [sum(v[k]) / len(v[k]) for v in SK.values() if v.get(k)]
        out[k] = round(sum(vals) / len(vals), 3) if vals else None
    return out
