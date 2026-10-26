#!/usr/bin/env python
"""SAP-noise x flip: attention-score randomization meets the causal battery.

'Lost in Attention' (ICML'26) showed benchmark ACCURACY survives replacing
decoder attention scores with Gaussian noise (their SAP intervention) -- but
they never ran a causal eval under it. We do: the flip_ll protocol under
  none : intact attention (sanity anchor -- must reproduce flip_ll)
  mid  : noise scores in the middle third of layers (the token's action band)
  all  : noise scores in every layer
Noise mode per LiA: scores ~ N(0,1), causal/sliding mask still applied, then
softmax -- visual content still flows in (uniform-ish mixing), only SELECTION
is destroyed. Pre-registered: control accuracy largely survives (replicating
LiA on our tasks) while flips/margins collapse -> selective attention is
necessary for evidence-TRACKING but not for accuracy; flip/stay detects the
mechanism accuracy metrics are blind to. If flips also survive, attention is
demoted to correlate and the readout pathway carries grounding.

Implementation: monkeypatch gemma3 eager_attention_forward; recompute the
noised layers' outputs; dispatch forced to eager on every submodule config.
A hit-counter asserts the patch actually fires (catches sdpa dispatch).
Output: runs/<run>/eval/<ds>_<split>/sap_flip_step_<STEP>.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from scsrc.data import load_task, build_specs, get_image
from scsrc.tokens import trigger
from scsrc import prompts as P, score as SC
from evaluate import apply_checkpoint

COLORS = {"red","orange","yellow","green","blue","purple","pink","brown","black",
          "white","gray","grey","gold","silver","tan","beige","violet","cyan",
          "maroon","navy"}
SAP = {"layers": set(), "hits": 0}


def install_sap():
    import torch.nn.functional as F
    import transformers.models.gemma3.modeling_gemma3 as G3
    try:
        repeat_kv = G3.repeat_kv
    except AttributeError:
        from transformers.models.llama.modeling_llama import repeat_kv
    orig = G3.eager_attention_forward

    def sap_eager(module, query, key, value, attention_mask, *args, **kwargs):
        out, w = orig(module, query, key, value, attention_mask, *args, **kwargs)
        li = getattr(module, "layer_idx", None)
        if li in SAP["layers"]:
            SAP["hits"] += 1
            vv = repeat_kv(value, module.num_key_value_groups)
            scores = torch.randn(w.shape, device=w.device, dtype=torch.float32)
            if attention_mask is not None:
                m = attention_mask[:, :, :, : vv.shape[-2]]
                scores = scores + m.to(scores.dtype)
            nw = F.softmax(scores, dim=-1).to(vv.dtype)
            out = torch.matmul(nw, vv).transpose(1, 2).contiguous()
            w = nw
        return out, w

    G3.eager_attention_forward = sap_eager


def norm(k):
    return {str(o).split("|")[0].strip().lower(): str(v).strip().lower()
            for o, v in (k or {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="color")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800)
    ap.add_argument("--max_n", type=int, default=60)
    a = ap.parse_args()
    torch.manual_seed(0)

    model, proc, meta, _ = apply_checkpoint(a.run_dir, a.step, "cuda")
    n = meta["n_tokens"]
    # force eager dispatch everywhere so the patch is on the code path
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    install_sap()
    tcfg = getattr(model.config, "text_config", model.config)
    L = tcfg.num_hidden_layers
    conds = {"none": set(), "mid": set(range(L // 3, 2 * L // 3)),
             "all": set(range(L))}

    ds = load_task(a.data_dir, a.split)
    specs = [s for s in build_specs(ds, a.split, conditions=("conflict",))
             if s.cond == "conflict"]
    is_vcf = "vcf" in a.data_dir

    out = {"run": os.path.basename(a.run_dir), "split": a.split, "step": a.step,
           "n_layers": L, "sap_mode": "gaussian-scores+mask (LiA Noise)"}
    for cname, layers in conds.items():
        SAP["layers"] = layers
        SAP["hits"] = 0
        res = {k: [] for k in ("look_acc", "read_acc", "look_flip", "read_flip",
                               "look_margin", "read_margin")}
        n_seen = 0
        for sp in tqdm(specs, desc=f"sap[{cname}]"):
            ik, tk = norm(sp.image_key), norm(sp.text_key)
            if not ik or set(ik) != set(tk):
                continue
            conf = [o for o in ik if ik[o] != tk[o]
                    and not (is_vcf and any(c in o.split() for c in COLORS))]
            if not conf:
                continue
            if n_seen >= a.max_n:
                break
            n_seen += 1
            obj = conf[n_seen % len(conf)]
            v_i, v_t = ik[obj], tk[obj]
            img = get_image(ds, sp)
            imgp = ds[sp.ex_index][f"image_{3 - sp.img_idx}"]
            capp = ds[sp.ex_index][f"caption_{sp.img_idx}"]
            ci = P.fmt_key_task({obj: v_i}, a.task)
            ct = P.fmt_key_task({obj: v_t}, a.task)

            def pick(op, image, cap):
                u = P.render_ctrl(P.CTRL_CANONICAL, trigger(f"{op}_ctrl", n), cap)
                la, _, _ = SC.loglik(model, proc, image, u, ci, "cuda")
                lb, _, _ = SC.loglik(model, proc, image, u, ct, "cuda")
                return ("i" if la > lb else "t"), la - lb

            li_, m_li = pick("look", img, sp.fed_caption)
            lp_, _ = pick("look", imgp, sp.fed_caption)
            ri_, m_ri = pick("read", img, sp.fed_caption)
            rc_, _ = pick("read", img, capp)
            res["look_acc"].append(int(li_ == "i"))
            res["read_acc"].append(int(ri_ == "t"))
            res["look_flip"].append(int(li_ == "i" and lp_ == "t"))
            res["read_flip"].append(int(ri_ == "t" and rc_ == "i"))
            res["look_margin"].append(m_li)
            res["read_margin"].append(-m_ri)
            if n_seen == 1 and layers and SAP["hits"] == 0:
                raise RuntimeError("SAP patch not firing -- dispatch is not eager")
        m = lambda v: round(sum(v) / max(len(v), 1), 3)
        out[cname] = {"n": n_seen, "sap_layers": len(layers),
                      "patch_hits": SAP["hits"],
                      **{k: m(v) for k, v in res.items()}}
        print(f"[{cname}] {json.dumps(out[cname])}")

    outd = os.path.join(a.run_dir, "eval", f"{os.path.basename(a.data_dir)}_{a.split}")
    os.makedirs(outd, exist_ok=True)
    json.dump(out, open(os.path.join(outd, f"sap_flip_step_{a.step}.json"), "w"), indent=1)
    print(f"[saved] {outd}/sap_flip_step_{a.step}.json")


if __name__ == "__main__":
    main()
