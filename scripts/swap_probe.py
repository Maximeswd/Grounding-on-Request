#!/usr/bin/env python
"""UNIFIED causal swap battery for neo-ctrl, PT, and LoRA runs -- the same 2x2
(img_swap_look_flip / img_swap_read_stay / cap_swap_read_flip / cap_swap_look_stay),
the same flip-to-correct scoring, the same items, regardless of method.
This is the differentiation gate: the paper's claim needs neo >> pt on the look
pair, and documents where LoRA tracks vs collapses.

Usage: srun ... python scripts/swap_probe.py --run_dir runs/color_pt_na_1tok_sft_seed0 \
             --task color --data_dir data/color --split test_id --step 400 [--max_n 80]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from scsrc.data import load_task
from scsrc.tokens import trigger
from scsrc import prompts as P, metrics as Me
from scsrc.infer import generate
from evaluate import apply_checkpoint, _pt_user

def norm_key(k):
    """Normalize a gold/pred key dict: subject-only objects, lowercase string values."""
    out = {}
    for o, v in (k or {}).items():
        out[str(o).split("|")[0].strip().lower().rstrip("s")] = str(v).strip().lower().rstrip(".")
    return out


def make_answerers(method, arm, model, processor, n, device, forbid, task):
    TASK = task
    """Return g(image, op, caption) -> parsed claim, per method, matching each
    method's OWN trained interface exactly (same as its battery uses)."""
    if method == "neo" and arm == "ctrl":
        def g(im, op, cap):
            tok = trigger(f"{op}_ctrl", n)
            o = generate(model, processor, im, P.render_ctrl(P.CTRL_CANONICAL, tok, cap),
                         device, 160, forbid_ids=forbid, stop_anchor="final answer:")
            return Me.parse_final_answer(o, TASK), o
        return g
    if method == "pt":
        def g(im, op, cap):
            o = generate(model, processor, im, _pt_user(op, n, P.pt_template(0), cap),
                         device, 160, forbid_ids=forbid, stop_anchor="final answer:")
            return Me.parse_final_answer(o, TASK), o
        return g
    if method == "lora":
        ATTR = {"color": "color", "counting": "count", "spatial": "spatial relation"}[task]
        def g(im, op, cap):
            u = ("Report each object's %s according to the image and the text, and whether they "
                 "agree or conflict. Caption: \"%s\". Format: IMAGE: obj: %s, ...; TEXT: obj: %s, "
                 "...; RELATION: agree/conflict." % (ATTR, cap, ATTR, ATTR))
            o = generate(model, processor, im, u, device, 192)
            if op == "look":
                seg = o.split("IMAGE:", 1)[1].split("TEXT:", 1)[0] if ("IMAGE:" in o and "TEXT:" in o) else ""
            else:
                seg = o.split("TEXT:", 1)[1].split("RELATION:", 1)[0] if "TEXT:" in o else ""
            return Me.parse_final_answer(seg, TASK), o[:200]
        return g
    if method == "instructed":
        # frozen model + natural-language source instructions; MUST stay
        # byte-identical to scripts/baselines.py INSTR/Q (the banked
        # instructed baseline) so this is its causal counterpart.
        INSTR = {
            "look": 'Answer using ONLY what you can see in the image; ignore the caption. ',
            "read": 'Answer using ONLY what the caption states; ignore the image. ',
            "compare": 'State whether the image and the caption agree or conflict about the object. ',
        }
        Q = {
            "color": 'Caption: "{cap}". What color is each object? Answer as object: color. final answer: ',
            "counting": 'Caption: "{cap}". How many are there? Answer as object: number. final answer: ',
            "spatial": 'Caption: "{cap}". Where is the subject relative to the other object? '
                       'Answer as object: relation. final answer: ',
        }
        def g(im, op, cap):
            u = INSTR[op] + Q[task].format(cap=cap)
            o = generate(model, processor, im, u, device, 160,
                         stop_anchor="final answer:")
            return Me.parse_final_answer(o, TASK), o
        return g
    raise SystemExit(f"unsupported method/arm: {method}/{arm}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=400)
    ap.add_argument("--max_n", type=int, default=80)
    ap.add_argument("--debug", type=int, default=0, help="print first K raw generations per condition")
    ap.add_argument("--instructed", action="store_true",
                    help="frozen model + baselines.py instructions; run_dir is "
                         "only used as the output location (e.g. the "
                         "base_frozen run dir); no checkpoint is loaded")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--with_resolve", action="store_true",
                    help="checkpoint has a <resolve_ctrl> row: add its flip/stay")
    ap.add_argument("--with_settle", action="store_true",
                    help="checkpoint has a <settle_ctrl> row: trust-flip 2x2 "
                         "(designation decides the answer on conflicts, "
                         "is inert on agreements)")
    ap.add_argument("--with_arbiter", action="store_true",
                    help="checkpoint has an <arbiter_ctrl> row: test the "
                         "argument-flip 2x2 (sensitivity on conflict, "
                         "invariance on agree)")
    ap.add_argument("--with_fillers", action="store_true",
                    help="THE CRUX TABLE: run clean fillers (please/trombone/"
                         "blicket -- verified NOT dataset values) through the "
                         "causal battery. Does a junk word in the slot produce "
                         "the token's [tracks-image] signature, or default to "
                         "one modality? Uses the base-conflict vs both-swap "
                         "discriminator (both cells stay conflicts, so a source "
                         "follower is separable from a modality-default).")
    a = ap.parse_args()
    FILLERS_CLEAN = ["please", "trombone", "blicket"]

    if a.instructed:
        from scsrc import model as M
        model, processor = M.load(model_id=a.model_id)
        method, arm, n, forbid = "instructed", "na", 0, None
    else:
        model, processor, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
        method, arm, n = meta["method"], meta["arm"], meta["n_tokens"]
    ds = load_task(a.data_dir, a.split)
    g = make_answerers(method, arm, model, processor, n, "cuda", forbid, a.task)

    # SPEC-DRIVEN pairing (works for every task, same machinery as training/eval):
    # shown image = image_{img_idx}; counterfactual twin = image_{3-img_idx};
    # conflict caption = fed_caption; matching caption = the row caption for the shown image.
    from scsrc.data import build_specs
    specs = [sp for sp in build_specs(ds, a.split, conditions=("agree", "conflict"))
             if sp.cond == "conflict"][:a.max_n]
    skipped = 0

    iflip, istay, cflip, cstay = [], [], [], []
    v_base, v_iflip, v_cflip, v_bstay = [], [], [], []
    r_base, r_iflip, r_cstay = [], [], []
    ab_look, ab_read, ab_flip, ab_inv = [], [], [], []
    st_img, st_cap, st_flip, st_inv = [], [], [], []
    gs = None
    if a.with_settle:
        def gs(im, cap, trust_img):
            clause = ("Trusted source: the image. " if trust_img
                      else "Trusted source: the caption. ")
            u = clause + P.render_ctrl(P.CTRL_CANONICAL,
                                       trigger("settle_ctrl", n), cap)
            o = generate(model, processor, im, u, "cuda", 160,
                         forbid_ids=forbid, stop_anchor="final answer:")
            return Me.parse_final_answer(o, a.task), o
    ga = None
    if a.with_arbiter:
        from scsrc.prompts import render_arbiter
        def ga(im, cap, arg):
            u = render_arbiter(trigger("arbiter_ctrl", n),
                               trigger(f"{arg}_ctrl", n), cap)
            o = generate(model, processor, im, u, "cuda", 160,
                         forbid_ids=forbid, stop_anchor="final answer:")
            return Me.parse_final_answer(o, a.task), o
    rec = []
    for sp in tqdm(specs, desc=f"swaps({method})"):
        row = ds[sp.ex_index]
        i1 = row.get(f"image_{sp.img_idx}")
        i2 = row.get(f"image_{3 - sp.img_idx}")
        if i1 is None or i2 is None:
            skipped += 1
            continue
        c2 = sp.fed_caption                                  # conflicts with i1
        c1 = row.get(f"caption_{sp.img_idx}", c2)            # matches i1
        k1, k2 = norm_key(sp.image_key), norm_key(sp.text_key)
        idx = sp.ex_index
        (look0, r0) = g(i1, "look", c2)
        (lookI, rI), (readI, _rRI) = g(i2, "look", c2), g(i2, "read", c2)
        ok = lambda pred, gold: Me.object_accuracy(norm_key(pred), gold) == 1.0
        iflip.append(int(ok(lookI, k2))); istay.append(int(ok(readI, k2)))
        (read0, rr0) = g(i1, "read", c2)
        (lookC, _rLC), (readC, rRC) = g(i1, "look", c1), g(i1, "read", c1)
        cflip.append(int(ok(readC, k1))); cstay.append(int(ok(lookC, k1)))
        # --- compare-verdict 2x2: the verdict must be causally bound to the
        # agreement STATE, not a class prior. On a conflict item:
        #   base (i1,c2)      -> "conflict"
        #   image swap (i2,c2)-> sources now agree -> verdict FLIPS to "agree"
        #   cap swap (i1,c1)  -> sources now agree -> FLIPS to "agree"
        #   both  (i2,c1)     -> i2 says k2, c1 says k1 -> still "conflict" (STAY)
        vb = Me.parse_relation(g(i1, "compare", c2)[1])
        vi = Me.parse_relation(g(i2, "compare", c2)[1])
        vc = Me.parse_relation(g(i1, "compare", c1)[1])
        vs = Me.parse_relation(g(i2, "compare", c1)[1])
        v_base.append(int(vb == "conflict"))
        v_iflip.append(int(vi == "agree"))
        v_cflip.append(int(vc == "agree"))
        v_bstay.append(int(vs == "conflict"))
        if a.with_settle:
            # trust-flip 2x2: designation decides on conflict, inert on agree
            (sI, _), (sC, _) = gs(i1, c2, True), gs(i1, c2, False)
            okI, okC = ok(sI, k1), ok(sC, k2)
            st_img.append(int(okI)); st_cap.append(int(okC))
            st_flip.append(int(okI and okC))
            (gI, _), (gC, _) = gs(i1, c1, True), gs(i1, c1, False)
            st_inv.append(int(ok(gI, k1) and ok(gC, k1)))
        if a.with_arbiter:
            # argument 2x2: on the conflict input the argument decides the
            # answer; on the agree input (i1 + its own caption) it must not
            (aL, _), (aR, _) = ga(i1, c2, "look"), ga(i1, c2, "read")
            okL, okR = ok(aL, k1), ok(aR, k2)
            ab_look.append(int(okL)); ab_read.append(int(okR))
            ab_flip.append(int(okL and okR))
            (gL, _), (gR, _) = ga(i1, c1, "look"), ga(i1, c1, "read")
            ab_inv.append(int(ok(gL, k1) and ok(gR, k1)))
        if a.with_resolve:
            # image-wins arbitration word: must TRACK the image (flip under
            # image swap) and IGNORE the caption (stay under caption swap)
            (res0, rr0_), (resI, rrI_), (resC, rrC_) = (g(i1, "resolve", c2),
                                                        g(i2, "resolve", c2),
                                                        g(i1, "resolve", c1))
            r_base.append(int(ok(res0, k1)))
            r_iflip.append(int(ok(resI, k2)))
            r_cstay.append(int(ok(resC, k1)))
            rec.append({"idx": idx, "test": "resolve",
                        "raw_res0": rr0_[:200], "raw_resI": rrI_[:200],
                        "raw_resC": rrC_[:200]})
        rec.append({"idx": idx, "k1": {str(a_): b_ for a_, b_ in (k1 or {}).items()},
                    "k2": {str(a_): b_ for a_, b_ in (k2 or {}).items()},
                    "look0": look0, "lookI": lookI, "readC": readC,
                    "raw_look0": r0[:180], "raw_lookI": rI[:180], "raw_readC": rRC[:180]})
        if a.debug and len(rec) <= a.debug:
            print(f"\n[debug idx={idx}] gold_shown={k1} gold_twin={k2}")
            print(f"  look(orig) raw: {r0[:150]!r} -> parsed {look0}")
            print(f"  look(swap) raw: {rI[:150]!r} -> parsed {lookI}")
            print(f"  read(capsw) raw: {rRC[:150]!r} -> parsed {readC}")

    filler_table = {}
    if a.with_fillers:
        okk = lambda pred, gold: Me.object_accuracy(norm_key(pred), gold) == 1.0

        def slot_answer(im, slot, cap):
            o = generate(model, processor, im,
                         P.render_ctrl(P.CTRL_CANONICAL, slot, cap), "cuda", 160,
                         forbid_ids=forbid, stop_anchor="final answer:")
            return Me.parse_final_answer(o, a.task)

        # token look/read = the reference signature; fillers wrapped in ()
        slots = {"TOKEN_look": trigger("look_ctrl", n),
                 "TOKEN_read": trigger("read_ctrl", n)}
        for f in FILLERS_CLEAN:
            slots[f"filler:{f}"] = f"({f})"
        for label, slot in slots.items():
            ti = tt = base_img = nseen = 0
            for sp in specs[:min(a.max_n, 40)]:
                row = ds[sp.ex_index]
                i1 = row.get(f"image_{sp.img_idx}")
                i2 = row.get(f"image_{3 - sp.img_idx}")
                if i1 is None or i2 is None:
                    continue
                c2 = sp.fed_caption
                c1 = row.get(f"caption_{sp.img_idx}", c2)
                k1, k2 = norm_key(sp.image_key), norm_key(sp.text_key)
                base = slot_answer(i1, slot, c2)   # image=k1, caption=k2
                both = slot_answer(i2, slot, c1)   # image=k2, caption=k1 (still conflict)
                nseen += 1
                base_img += int(okk(base, k1))
                ti += int(okk(base, k1) and okk(both, k2))   # flips WITH the image
                tt += int(okk(base, k2) and okk(both, k1))   # flips WITH the caption
            filler_table[label] = {"base_follows_image": round(base_img / max(nseen, 1), 3),
                                   "tracks_image": round(ti / max(nseen, 1), 3),
                                   "tracks_text": round(tt / max(nseen, 1), 3), "n": nseen}
        print("\n== filler-through-swap-battery (the crux table) ==")
        print(f"  {'slot':20s} {'base→img':>9s} {'tracks_img':>11s} {'tracks_txt':>11s}")
        for label, v in filler_table.items():
            print(f"  {label:20s} {v['base_follows_image']:>9.3f} "
                  f"{v['tracks_image']:>11.3f} {v['tracks_text']:>11.3f}")

    for name, vals in (("iflip", iflip), ("istay", istay), ("cflip", cflip), ("cstay", cstay)):
        if vals and (all(v == 0 for v in vals) or all(v == 1 for v in vals)):
            print(f"[tripwire] {name} is degenerate ({vals[0]}) -- sample records:")
            for x in rec[:3]:
                print("   ", {kk: x[kk] for kk in ("k1", "k2", "look0", "lookI", "raw_look0") if kk in x})
    if skipped:
        print(f"[warn] {skipped} specs lacked a twin image and were skipped")
    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    table = {"method": method, "img_swap_look_flip": m(iflip), "img_swap_read_stay": m(istay),
             "cap_swap_read_flip": m(cflip), "cap_swap_look_stay": m(cstay),
             "compare_base_conflict": m(v_base), "img_swap_compare_flip": m(v_iflip),
             "cap_swap_compare_flip": m(v_cflip), "both_swap_compare_stay": m(v_bstay),
             "n": len(iflip)}
    if a.with_resolve:
        table.update({"resolve_base_img_wins": m(r_base),
                      "img_swap_resolve_flip": m(r_iflip),
                      "cap_swap_resolve_stay": m(r_cstay)})
    if a.with_arbiter:
        table.update({"arbiter_favor_look_acc": m(ab_look),
                      "arbiter_favor_read_acc": m(ab_read),
                      "arbiter_argflip_effect": m(ab_flip),
                      "arbiter_agree_invariance": m(ab_inv)})
    if a.with_fillers:
        table["fillers"] = filler_table
    if a.with_settle:
        table.update({"settle_trust_img_acc": m(st_img),
                      "settle_trust_cap_acc": m(st_cap),
                      "settle_trustflip_effect": m(st_flip),
                      "settle_agree_invariance": m(st_inv)})
    print(f"\n== unified swap battery  {a.run_dir}  step {a.step}")
    for k, v in table.items():
        print(f"  {k:22s} {v}")
    outd = os.path.join(a.run_dir, "eval", f"{a.task}_{a.split}")
    os.makedirs(outd, exist_ok=True)
    with open(os.path.join(outd, f"swap_probe_step_{a.step}.json"), "w") as f:
        json.dump(table, f, indent=2)
    with open(os.path.join(outd, f"swap_probe_records_step_{a.step}.jsonl"), "w") as f:
        for x in rec:
            f.write(json.dumps(x, default=str) + "\n")
    with open(os.path.join(outd, f"swap_probe_records_step_{a.step}.jsonl"), "w") as f:
        for x in rec:
            f.write(json.dumps(x, default=str) + "\n")
    print(f"[saved] {outd}/swap_probe_step_{a.step}.json")

if __name__ == "__main__":
    main()
