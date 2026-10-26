#!/usr/bin/env python
"""PATTERN TRANSPLANT: give the INSTRUCTION condition the WORD's image-reading
attention pattern, and see whose behaviour results.

The open mechanism question: on `look`, the word wins with LESS image mass and
EQUAL precision, so the attention PATTERN does not separate the methods. The
decisive intervention is to hold the pattern fixed and vary the carrier:

  pass 1 (record): token-condition forward; at every mid-band layer, store the
         head-averaged attention DISTRIBUTION over the 256 image-key positions,
         averaged over continuation queries.
  pass 2 (patch):  instruction-condition forward on the SAME item; at each
         mid-band layer, replace every continuation query's attention over the
         image keys with the recorded distribution, rescaled to preserve that
         query's own total image mass (shape transplanted, budget kept), then
         renormalise off-image mass so rows still sum to 1.

  If instruction behaviour under the word's pattern -> word-level flips, the
  pattern was the carrier after all (and our mass/precision metrics missed the
  relevant feature). If behaviour does NOT follow the pattern, the difference
  is in what the model DOES with attention (value/OV pathway) -- the digest
  reading. Either outcome resolves the open question.

Image keys are locatable in both conditions (gemma: 256 fixed image tokens),
so the transplant is well-defined across prompts of different lengths.
Gemma-only (the monkeypatch targets gemma3 eager attention, like sap_flip).

Output: runs/<run>/eval/<ds>_<split>/transplant_step_<STEP>.json
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger
from scsrc import prompts as P, score as SC
from evaluate import apply_checkpoint

INSTR = {"look": "Answer using ONLY what you can see in the image; ignore the caption.",
         "read": "Answer using ONLY what the caption states; ignore the image."}
T = {"mode": None, "layers": set(), "img": None, "plen": 0, "rec": {}, "hits": 0}


def install():
    import transformers.models.gemma3.modeling_gemma3 as G3
    try:
        repeat_kv = G3.repeat_kv
    except AttributeError:
        from transformers.models.llama.modeling_llama import repeat_kv
    orig = G3.eager_attention_forward

    def hook(module, query, key, value, attention_mask, *args, **kwargs):
        out, w = orig(module, query, key, value, attention_mask, *args, **kwargs)
        li = getattr(module, "layer_idx", None)
        if li not in T["layers"] or T["img"] is None or w.shape[2] <= T["plen"]:
            return out, w
        i0, i1 = T["img"]
        if T["mode"] == "record":
            # head-avg distribution over image keys, averaged over continuation queries
            d = w[:, :, T["plen"]:, i0:i1].mean(dim=(1, 2))            # (B, n_img)
            s = d.sum(-1, keepdim=True).clamp_min(1e-9)
            T["rec"][li] = (d / s).detach()
            return out, w
        if T["mode"] == "ablate":
            T["hits"] += 1
            vv = repeat_kv(value, module.num_key_value_groups)
            nw = w.clone()
            # ROWS: which QUERIES lose access. None = continuation queries
            # (the original ablation); a list = specific prompt positions
            # (e.g. the ctrl word's own queries, for the dir2img relay hop).
            rows = T.get("abl_rows")
            if rows is None:
                q = nw[:, :, T["plen"]:, :]
            else:
                q = nw[:, :, rows, :]
            # COLS: which KEYS become invisible to those queries.
            if T.get("abl_idx") is not None:            # scattered key indices
                q[:, :, :, T["abl_idx"]] = 0.0
            else:
                q[:, :, :, i0:i1] = 0.0
            q = q / q.sum(-1, keepdim=True).clamp_min(1e-9)
            if rows is None:
                nw[:, :, T["plen"]:, :] = q
            else:
                nw[:, :, rows, :] = q
            out = torch.matmul(nw, vv).transpose(1, 2).contiguous()
            return out, nw
        if T["mode"] == "patch" and li in T["rec"]:
            T["hits"] += 1
            vv = repeat_kv(value, module.num_key_value_groups)
            nw = w.clone()
            q = nw[:, :, T["plen"]:, :]
            img = q[:, :, :, i0:i1]
            budget = img.sum(-1, keepdim=True)                          # keep own mass
            shape = T["rec"][li][:, None, None, :]                      # (B,1,1,n_img)
            q[:, :, :, i0:i1] = budget * shape
            nw[:, :, T["plen"]:, :] = q
            out = torch.matmul(nw, vv).transpose(1, 2).contiguous()
            return out, nw
        return out, w

    G3.eager_attention_forward = hook


def img_span(proc, ids):
    tok = proc.tokenizer
    cands = [v for k, v in tok.get_vocab().items() if "image_soft_token" in k or k == "<image_soft_token>"]
    if not cands:
        enc = tok("<image_soft_token>", add_special_tokens=False)["input_ids"]
        cands = enc
    iid = cands[0]
    pos = (ids == iid).nonzero(as_tuple=True)[-1].tolist()
    return (pos[0], pos[-1] + 1) if pos else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=60)
    ap.add_argument("--mode", default="transplant", choices=["transplant", "ablate"],
                    help="ablate = zero the attention to --span keys at all "
                         "continuation queries (renormalised), for BOTH the word "
                         "and the instruction condition. The mechanism ablation: "
                         "if a token's advantage runs through that span, removing "
                         "it should erase the advantage.")
    ap.add_argument("--span", default="evid", choices=["evid", "capval", "img", "ans2dir", "dir2img"],
                    help="evid = twin-difference image patches (the <look> "
                         "mechanism site); capval = the caption's asserted value "
                         "tokens (the <read> precision site); img = whole image")
    a = ap.parse_args()
    dev = "cuda"
    install()
    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, dev)
    n = meta["n_tokens"]
    L = model.config.text_config.num_hidden_layers if hasattr(model.config, "text_config") \
        else model.config.num_hidden_layers
    T["layers"] = set(range(L // 3, 2 * L // 3))
    for m_ in model.modules():
        if hasattr(m_, "config") and hasattr(m_.config, "_attn_implementation"):
            m_.config._attn_implementation = "eager"

    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",))
             if s.cond == "conflict"]
    R = collections.defaultdict(list)
    seen = 0
    for sp in tqdm(specs, desc="transplant"):
        if seen >= a.max_n:
            break
        ik = {str(k).split("|")[0].strip().lower(): str(v).strip().lower()
              for k, v in (sp.image_key or {}).items()}
        tk = {str(k).split("|")[0].strip().lower(): str(v).strip().lower()
              for k, v in (sp.text_key or {}).items()}
        conf = [o for o in ik if o in tk and ik[o] != tk[o]]
        if not conf:
            continue
        seen += 1
        obj = conf[seen % len(conf)]
        ci = P.fmt_key_task({obj: ik[obj]}, a.task)
        ct = P.fmt_key_task({obj: tk[obj]}, a.task)
        img = get_image(ds, sp)
        u_tok = P.render_ctrl(P.CTRL_CANONICAL, trigger("look_ctrl", n), sp.fed_caption)
        u_ins = f'Caption: "{sp.fed_caption}". {INSTR["look"]}'

        def pick(user, mode):
            T["mode"] = mode
            la, meta1, _ = SC.loglik(model, proc, img, user, ci, dev)
            lb, _, _ = SC.loglik(model, proc, img, user, ct, dev)
            return "i" if la > lb else "t", la - lb

        # locate spans per condition via a dry encode
        def spans(user, cont):
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": user}]}]
            enc = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=True, return_dict=True,
                                           return_tensors="pt")
            ids = enc["input_ids"][0]
            return img_span(proc, ids), ids.shape[0], ids

        s_tok, p_tok, ids_tok = spans(u_tok, ci)
        s_ins, p_ins, ids_ins = spans(u_ins, ci)

        # DIRECTIVE positions, per condition. tok: the ctrl word's own token
        # positions (added-vocab ids, exact match). ins: the English
        # instruction sentence's positions (subsequence match). These are the
        # two ends of the relay hypothesis: dir2img severs directive->image
        # (hop 1), ans2dir severs answer->directive (hop 2). The instruction
        # condition gets the SYMMETRIC cut -- lose access to ITS directive --
        # so differential_drop reads as 'whose directive is load-bearing'.
        def word_positions(ids):
            wid = set(proc.tokenizer(trigger("look_ctrl", n),
                                     add_special_tokens=False)["input_ids"])
            return [i for i, t in enumerate(ids.tolist()) if t in wid]

        def instr_positions(ids):
            sub = proc.tokenizer(INSTR["look"], add_special_tokens=False)["input_ids"]
            hay, k = ids.tolist(), min(len(sub), 6)
            for i in range(len(hay) - k + 1):
                if hay[i:i + k] == sub[:k]:
                    return list(range(i, min(i + len(sub), len(hay))))
            return []

        dir_tok = word_positions(ids_tok)
        dir_ins = instr_positions(ids_ins)
        if s_tok is None or s_ins is None:
            continue
        if a.mode == "ablate":
            # span key indices per condition
            def abl(span_img, user, plen):
                if a.span == "img":
                    return None, span_img            # contiguous
                if a.span == "evid":
                    from scripts.attn_flip_probe import evidence_patches
                    imgp = ds[sp.ex_index][f"image_{3 - sp.img_idx}"]
                    eidx, _ = evidence_patches(img, imgp)
                    return [span_img[0] + i for i in eidx], span_img
                # capval: value tokens inside the prompt text -- locate by re-encode
                tok = proc.tokenizer
                ids = tok(user, add_special_tokens=False)["input_ids"]
                vt = tok(" " + ct, add_special_tokens=False)["input_ids"]
                # fall back to whole-caption ablation if not found
                return None, span_img
            for cond, (user, spn, pl, dpos) in {
                    "tok": (u_tok, s_tok, p_tok, dir_tok),
                    "ins": (u_ins, s_ins, p_ins, dir_ins)}.items():
                if a.span in ("ans2dir", "dir2img") and not dpos:
                    continue                      # directive not locatable
                base, _ = pick(user, None)
                if a.span == "ans2dir":
                    # answer tokens lose the DIRECTIVE keys (relay hop 2)
                    T.update(img=spn, plen=pl, mode="ablate", hits=0,
                             abl_rows=None, abl_idx=dpos)
                elif a.span == "dir2img":
                    # the directive's own queries lose the IMAGE keys (hop 1)
                    T.update(img=spn, plen=pl, mode="ablate", hits=0,
                             abl_rows=dpos, abl_idx=None)
                else:
                    idx, span_img = abl(spn, user, pl)
                    T.update(img=span_img, plen=pl, mode="ablate", hits=0,
                             abl_rows=None, abl_idx=idx)
                after, m_af = pick(user, "ablate")
                T["mode"] = None
                R[f"{cond}_base_img"].append(int(base == "i"))
                R[f"{cond}_ablate_img"].append(int(after == "i"))
                R[f"{cond}_drop"].append(int(base == "i") - int(after == "i"))
            continue
        # pass 1: record the word's pattern
        T.update(img=s_tok, plen=p_tok, rec={}, mode="record")
        pick(u_tok, "record")
        # baselines
        T["mode"] = None
        b_tok, m_tok = pick(u_tok, None)
        b_ins, m_ins = pick(u_ins, None)
        # pass 2: instruction with the word's pattern
        T.update(img=s_ins, plen=p_ins, mode="patch", hits=0)
        b_tx, m_tx = pick(u_ins, "patch")
        if T["hits"] == 0:
            R["patch_miss"].append(1)
            continue
        R["patch_miss"].append(0)
        R["tok_img"].append(int(b_tok == "i"))
        R["ins_img"].append(int(b_ins == "i"))
        R["transplant_img"].append(int(b_tx == "i"))
        R["delta_margin"].append(m_tx - m_ins)

    m = lambda v: round(sum(v) / len(v), 3) if v else None
    out = {"run": os.path.basename(a.run_dir), "n": seen, "mode": a.mode,
           "span": a.span,
           "tok_img": m(R["tok_img"]), "ins_img": m(R["ins_img"]),
           "transplant_img": m(R["transplant_img"]),
           "mean_margin_shift": (round(sum(R["delta_margin"]) / len(R["delta_margin"]), 3)
                                 if R["delta_margin"] else None),
           "patch_miss_rate": m(R["patch_miss"]),
           "note": ("transplant_img >> ins_img -> the PATTERN carries the look "
                    "advantage; transplant_img ~= ins_img -> the carrier is what "
                    "the model DOES with attention (value/OV pathway)")}
    # ABLATE ARM WAS SILENTLY DISCARDED. The ablate branch fills
    # R["{tok,ins}_{base,ablate}_img"] and R["{tok,ins}_drop"], none of which
    # the dict above reads -- so both banked runs (color/apo, vcf/ctm) wrote
    # all-null files and the mechanism question was never actually answered.
    # n_ablate is reported separately because the ablate branch keeps BOTH
    # conditions per item, so its denominator is not `seen`.
    if a.mode == "ablate":
        out["ablate"] = {
            c: {"base_img": m(R[f"{c}_base_img"]),
                "ablate_img": m(R[f"{c}_ablate_img"]),
                "drop": m(R[f"{c}_drop"]),
                "n": len(R[f"{c}_base_img"])}
            for c in ("tok", "ins")}
        t, i = out["ablate"]["tok"], out["ablate"]["ins"]
        if t["drop"] is not None and i["drop"] is not None:
            # THE MECHANISM TEST: does killing this span cost the WORD more than
            # it costs the instruction? >0 = the word's advantage runs through
            # the span; ~0 = it does not, and the carrier is elsewhere.
            out["ablate"]["differential_drop"] = round(t["drop"] - i["drop"], 3)
        out["note"] = ("differential_drop >> 0 -> the word's look advantage is "
                       "carried by attention to this span; ~0 -> removing the "
                       "span costs both conditions equally, so the span is not "
                       "what separates them (value/OV pathway reading)")
    print(json.dumps(out, indent=1))
    od = os.path.join(a.run_dir, "eval", f"{os.path.basename(a.data_dir)}_{a.split}")
    os.makedirs(od, exist_ok=True)
    # mode+span in the filename: the three (mode, span) variants of one run
    # previously all wrote transplant_step_800.json and OVERWROTE each other --
    # the first F_mech wave lost its transplant + ablate-evid results this way.
    tag = a.mode if a.mode == "transplant" else f"ablate_{a.span}"
    json.dump(out, open(os.path.join(od, f"transplant_{tag}_step_{a.step}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
