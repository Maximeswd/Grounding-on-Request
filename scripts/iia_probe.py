#!/usr/bin/env python
"""IIA probe: is a trained word's contextual state a causal VARIABLE for its
source, or does the word steer without denoting?

Twin conflict pairs (same pairing as swap_probe): base = image i1 + conflict
caption c2; the look donor shows the exact twin i2 (same object, same caption,
only the value differs), the read donor shows the matching caption c1 (same
image). We cache the donor's <op_ctrl> hidden state at layer L and transplant
it into the base prefill at the word's position. If the state CARRIES the
source value, the patched answer must equal the donor's source value --
although the pixels/caption the model actually saw are unchanged.

Per layer L and op in {look, read}:
  T (target transfer)      iia_*        = P[patched answer == donor value]
  N (non-target retention) npres_*      = P[answer objects == base objects]
  joint (strict IIA)       iia_*_joint  = P[T and N]
  *_cond                   = T restricted to pairs answered correctly
                             UNPATCHED on both base and donor (competence)
  a_input_*                = P[unpatched donor answer == donor value]
                             (the direct-input counterfactual oracle)
  riia_*                   = iia_* / a_input_*   (normalized IIA)
  change_*                 = P[patched != unpatched base]  (site liveness)

Controls (look op, first --control_n pairs, every sweep layer):
  samival_stay   same-value donor (different scene/object, same color values):
                 answer must STAY base-correct; drops = nuisance leakage
  wrongtok_follow  donor <read> state patched into the <look> position:
                 value transfer across roles = state carries value sans role
  filler_iia     state swap at a matched filler word occupying the slot:
                 ~chance = causal control is localized to the learned word

Anchors (first --anchor_n pairs): self-swap must reproduce the base string at
~1.00 AND next-token logits to <1e-2; norm-matched random vectors must change
the output on most items. anchors_ok=false poisons the file -- do not read
IIA cells then.

Pre-registered readout (cet-pivot ruling): SFT/v3 rows with high behavioral
flips but LOW joint IIA at every live layer = "steers without denoting".
Usage:
  python scripts/iia_probe.py --run_dir runs/color_neo_ctrl_1tok_sft_seed0 \
    --task color --data_dir data/color --split test_id --step 400 \
    --layers 4,8,12,16,20,24 --max_n 40
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

from scsrc.data import load_task, build_specs
from scsrc.tokens import trigger, token_strings
from scsrc import prompts as P, metrics as Me, iit
from scsrc.infer import generate
from scsrc.collate import NEG_SLOT_WORDS
from evaluate import apply_checkpoint
from scripts.swap_probe import norm_key


def enc_for(processor, image, user_text, device):
    """EXACTLY the encoding infer.generate builds -- same msgs, same chat
    template -- so positions found here are valid inside generate()."""
    from scsrc.collate import _user_content
    msgs = [{"role": "user", "content": _user_content(image, user_text)}]
    return processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt").to(device)


def find_subseq(input_ids, sub_ids):
    seq = input_ids[0].tolist()
    for i in range(len(seq) - len(sub_ids) + 1):
        if seq[i:i + len(sub_ids)] == sub_ids:
            return list(range(i, i + len(sub_ids)))
    return []


def pick_filler(template_probe):
    """First slot-negative word that does not occur in the rendered prompt
    (e.g. 'give' collides with the template's own 'Give me a ... answer')."""
    for w in NEG_SLOT_WORDS:
        if w.lower() not in template_probe.lower():
            return w
    return NEG_SLOT_WORDS[-1]


def filler_positions(enc, tok, filler):
    """Positions of the filler's token block; require a UNIQUE match (leading
    -space variant first -- that is how it tokenizes after 'a')."""
    seq = enc["input_ids"][0].tolist()
    for pre in (" ", ""):
        sub = tok(pre + filler, add_special_tokens=False)["input_ids"]
        hits = [i for i in range(len(seq) - len(sub) + 1)
                if seq[i:i + len(sub)] == sub]
        if len(hits) == 1:
            return list(range(hits[0], hits[0] + len(sub)))
    return []


def next_logits(model, enc, patch_ctx=None):
    import torch
    kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    with torch.no_grad():
        if patch_ctx is None:
            return model(**kw, use_cache=False).logits[0, -1].float()
        with patch_ctx:
            return model(**kw, use_cache=False).logits[0, -1].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=400)
    ap.add_argument("--layers", default="4,8,12,16,20,24")
    ap.add_argument("--max_n", type=int, default=40)
    ap.add_argument("--anchor_n", type=int, default=8)
    ap.add_argument("--control_n", type=int, default=12)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    model, processor, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    n_layers = len(iit.lm_layers(model))
    layers = [L for L in layers if L < n_layers]
    tok = processor.tokenizer
    ids = {op: tok.convert_tokens_to_ids(token_strings(f"{op}_ctrl", n))
           for op in ("look", "read")}
    filler = pick_filler(P.render_ctrl(P.CTRL_CANONICAL, "XSLOTX", "probe"))

    def ask(im, user_text, patch_ctx=None):
        if patch_ctx is None:
            o = generate(model, processor, im, user_text, "cuda", 160,
                         forbid_ids=forbid, stop_anchor="final answer:")
        else:
            with patch_ctx:
                o = generate(model, processor, im, user_text, "cuda", 160,
                             forbid_ids=forbid, stop_anchor="final answer:")
        return norm_key(Me.parse_final_answer(o, a.task)), o

    def u_of(op, cap):
        return P.render_ctrl(P.CTRL_CANONICAL, trigger(f"{op}_ctrl", n), cap)

    ds = load_task(a.data_dir, a.split)
    all_specs = [sp for sp in build_specs(ds, a.split,
                                          conditions=("agree", "conflict"))
                 if sp.cond == "conflict"]
    specs = all_specs[:a.max_n]
    # same-value donor pool: color-value signature -> spec list
    sig = lambda sp: tuple(sorted(norm_key(sp.image_key).values()))
    pool = {}
    for sp in all_specs:
        pool.setdefault(sig(sp), []).append(sp)

    ok = lambda pred, gold: Me.object_accuracy(pred, gold) == 1.0
    z = lambda: {L: [] for L in layers}
    hit = {op: z() for op in ("look", "read")}
    hitc = {op: z() for op in ("look", "read")}
    joint = {op: z() for op in ("look", "read")}
    npres = {op: z() for op in ("look", "read")}
    chg = {op: z() for op in ("look", "read")}
    a_input = {"look": [], "read": []}
    sv_stay, wt_follow, fl_iia = z(), z(), z()
    fl_base, bl_donor = [], []      # unpatched control baselines
    anc_self, anc_rand, anc_logit = z(), z(), z()
    rec, skipped = [], 0

    for pi, sp in enumerate(tqdm(specs, desc="iia")):
        row = ds[sp.ex_index]
        i1, i2 = row.get(f"image_{sp.img_idx}"), row.get(f"image_{3 - sp.img_idx}")
        if i1 is None or i2 is None:
            skipped += 1
            continue
        c2 = sp.fed_caption
        c1 = row.get(f"caption_{sp.img_idx}", c2)
        k1, k2 = norm_key(sp.image_key), norm_key(sp.text_key)

        # per op: (donor image, donor caption, base gold, cf gold)
        setup = {"look": (i2, c2, k1, k2), "read": (i1, c1, k2, k1)}
        enc_b_look = None
        for op, (di, dc, base_gold, cf_gold) in setup.items():
            u_base, u_don = u_of(op, c2), u_of(op, dc)
            enc_b = enc_for(processor, i1, u_base, "cuda")
            enc_d = enc_for(processor, di, u_don, "cuda")
            pos_b = iit.token_positions(enc_b["input_ids"], ids[op])
            pos_d = iit.token_positions(enc_d["input_ids"], ids[op])
            if not pos_b or not pos_d:
                skipped += 1
                continue
            if op == "look":
                enc_b_look = (enc_b, pos_b, u_base)
            don_states = iit.cache_states(model, enc_d, layers, pos_d)
            base_ans, base_raw = ask(i1, u_base)
            don_ans, _ = ask(di, u_don)
            a_input[op].append(int(ok(don_ans, cf_gold)))
            if op == "look":
                bl_donor.append(int(ok(base_ans, cf_gold)))
            competent = ok(base_ans, base_gold) and ok(don_ans, cf_gold)
            for L in layers:
                pans, praw = ask(i1, u_base,
                                 iit.patch(model, L, pos_b, don_states[L]))
                h = int(ok(pans, cf_gold))
                npr = int(set(pans.keys()) == set(base_ans.keys())
                          and len(pans) > 0)
                hit[op][L].append(h)
                joint[op][L].append(int(h and npr))
                npres[op][L].append(npr)
                if competent:
                    hitc[op][L].append(h)
                chg[op][L].append(int(pans != base_ans))
                rec.append({"idx": sp.ex_index, "op": op, "layer": L,
                            "cell": "iia", "cf_gold": cf_gold,
                            "base": base_ans, "donor": don_ans,
                            "patched": pans, "competent": competent,
                            "raw": praw[:160]})

        if enc_b_look is None:
            continue
        enc_b, pos_b, u_base = enc_b_look
        base_ans, base_raw = ask(i1, u_base)

        # ---- controls (look op only, first control_n pairs) ----
        if pi < a.control_n:
            # same-value donor: different scene, same color values
            cands = [d for d in pool.get(sig(sp), [])
                     if d.ex_index != sp.ex_index]
            if cands:
                dsp = cands[pi % len(cands)]
                drow = ds[dsp.ex_index]
                dim = drow.get(f"image_{dsp.img_idx}")
                if dim is not None:
                    u_d = u_of("look", dsp.fed_caption)
                    e_d = enc_for(processor, dim, u_d, "cuda")
                    p_d = iit.token_positions(e_d["input_ids"], ids["look"])
                    if p_d:
                        st = iit.cache_states(model, e_d, layers, p_d)
                        for L in layers:
                            pans, praw = ask(i1, u_base,
                                             iit.patch(model, L, pos_b, st[L]))
                            sv_stay[L].append(int(ok(pans, k1)))
                            rec.append({"idx": sp.ex_index, "layer": L,
                                        "cell": "samival", "patched": pans,
                                        "base_gold": k1, "raw": praw[:120]})
            # wrong-token swap: donor <read> state into the <look> position
            u_dr = u_of("read", c2)
            e_dr = enc_for(processor, i2, u_dr, "cuda")
            p_dr = iit.token_positions(e_dr["input_ids"], ids["read"])
            if p_dr:
                st = iit.cache_states(model, e_dr, layers, p_dr)
                for L in layers:
                    pans, praw = ask(i1, u_base,
                                     iit.patch(model, L, pos_b, st[L]))
                    wt_follow[L].append(int(ok(pans, k2)))
                    rec.append({"idx": sp.ex_index, "layer": L,
                                "cell": "wrongtok", "patched": pans,
                                "cf_gold": k2, "raw": praw[:120]})
            # filler-state swap: same slot, filler word both sides
            u_bf = P.render_ctrl(P.CTRL_CANONICAL, filler, c2)
            e_bf = enc_for(processor, i1, u_bf, "cuda")
            e_df = enc_for(processor, i2, u_bf, "cuda")
            p_bf = filler_positions(e_bf, tok, filler)
            p_df = filler_positions(e_df, tok, filler)
            if p_bf and p_df:
                st = iit.cache_states(model, e_df, layers, p_df)
                fb_ans, _ = ask(i1, u_bf)
                fl_base.append(int(ok(fb_ans, k2)))
                for L in layers:
                    pans, praw = ask(i1, u_bf,
                                     iit.patch(model, L, p_bf, st[L]))
                    fl_iia[L].append(int(ok(pans, k2)))
                    rec.append({"idx": sp.ex_index, "layer": L,
                                "cell": "filler", "patched": pans,
                                "cf_gold": k2, "fbase": fb_ans,
                                "raw": praw[:120]})

        # ---- anchors (look op, first anchor_n pairs) ----
        if pi < a.anchor_n:
            base_states = iit.cache_states(model, enc_b, layers, pos_b)
            lg0 = next_logits(model, enc_b)
            for L in layers:
                _, sraw = ask(i1, u_base,
                              iit.patch(model, L, pos_b, base_states[L]))
                anc_self[L].append(int(sraw == base_raw))
                lg1 = next_logits(model, enc_b,
                                  iit.patch(model, L, pos_b, base_states[L]))
                anc_logit[L].append(float((lg1 - lg0).abs().max()))
                rans, _ = ask(i1, u_base,
                              iit.patch(model, L, pos_b,
                                        iit.norm_matched_random(
                                            base_states[L], seed=pi)))
                anc_rand[L].append(int(rans != base_ans))

    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    self_ok = all(m(anc_self[L]) >= 0.9 for L in layers if anc_self[L])
    logit_ok = all(max(anc_logit[L], default=0.0) < 1e-2
                   for L in layers if anc_logit[L])
    live_layers = [L for L in layers if anc_rand[L] and m(anc_rand[L]) >= 0.5]
    table = {"run": a.run_dir, "step": a.step, "n_pairs": len(specs) - skipped,
             "anchors_ok": bool(self_ok and logit_ok),
             "selfswap_by_layer": {L: m(anc_self[L]) for L in layers},
             "selfswap_logit_maxdiff": {L: round(max(anc_logit[L], default=0.0), 5)
                                        for L in layers},
             "randchange_by_layer": {L: m(anc_rand[L]) for L in layers},
             "live_layers": live_layers,
             "samival_stay": {L: m(sv_stay[L]) for L in layers},
             "wrongtok_follow": {L: m(wt_follow[L]) for L in layers},
             "filler_iia": {L: m(fl_iia[L]) for L in layers},
             "filler_base_follow": m(fl_base),
             "base_look_donor_rate": m(bl_donor)}
    for op in ("look", "read"):
        ai = m(a_input[op])
        table[f"a_input_{op}"] = ai
        table[f"iia_{op}"] = {L: m(hit[op][L]) for L in layers}
        table[f"iia_{op}_joint"] = {L: m(joint[op][L]) for L in layers}
        table[f"npres_{op}"] = {L: m(npres[op][L]) for L in layers}
        table[f"riia_{op}"] = {L: round(m(hit[op][L]) / max(ai, 1e-6), 3)
                               for L in layers}
        table[f"iia_{op}_cond"] = {L: m(hitc[op][L]) for L in layers}
        table[f"n_competent_{op}"] = {L: len(hitc[op][L]) for L in layers}
        table[f"change_{op}"] = {L: m(chg[op][L]) for L in layers}
    if not (self_ok and logit_ok):
        print("[ANCHOR FAILED] self-swap not identity (string or logits) -- "
              "position/layer indexing is wrong; IIA cells are NOT readable.")
    if all(v == 0 for L in layers for v in hit["look"][L]) and \
            any(chg["look"][L] for L in layers):
        print("[tripwire] iia_look all-zero across layers; sample records:")
        for x in [r for r in rec if r["op"] == "look" and r["cell"] == "iia"][:3]:
            print("   ", {k: x[k] for k in ("cf_gold", "base", "patched", "raw")})

    print(f"\n== IIA probe  {a.run_dir}  step {a.step}")
    for k in ("anchors_ok", "live_layers", "a_input_look", "a_input_read"):
        print(f"  {k:18s} {table[k]}")
    for op in ("look", "read"):
        print(f"  iia_{op:5s}       {table[f'iia_{op}']}")
        print(f"  riia_{op:5s}      {table[f'riia_{op}']}")
        print(f"  iia_{op:5s}_joint {table[f'iia_{op}_joint']}")
    for k in ("samival_stay", "wrongtok_follow", "filler_iia",
              "filler_base_follow", "base_look_donor_rate"):
        print(f"  {k:20s} {table[k]}")
    outd = os.path.join(a.run_dir, "eval", f"{a.task}_{a.split}")
    os.makedirs(outd, exist_ok=True)
    json.dump(table, open(os.path.join(outd, f"iia_probe_step_{a.step}.json"),
                          "w"), indent=2, default=str)
    with open(os.path.join(outd, f"iia_probe_records_step_{a.step}.jsonl"), "w") as f:
        for x in rec:
            f.write(json.dumps(x, default=str) + "\n")
    print(f"[saved] {outd}/iia_probe_step_{a.step}.json")


if __name__ == "__main__":
    main()
