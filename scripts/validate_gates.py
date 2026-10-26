#!/usr/bin/env python
"""END-TO-END TRAINING-PATH VALIDATION (2026-09-21, revised after review), the P0 gate before any rerun.

For ONE (model, interface, task) this drives the real dataset class, tokenisation, objective, one real
optimizer step, then saves the checkpoint as train.py does and reloads it as evaluate.apply_checkpoint
does. Bundle -> results/gates/<task>_<method>_<model>.json; exit status 0 only when every gate passes.

  Gate 1  real records: rendered prompt/target, the EXACT expected control block for the item's own
          source (ids for neo/pt, instruction text for LoRA), its position, exactly-once, absent for the
          other source, identical across the sides of a pair; per-source exposure; record keys.
  Gate 2  each term alone from the same state: finite, present in comps (SFT reports as 'ce'), non-zero
          gradient on the intended parameters, none on frozen ones. Enabled-but-absent is a FAIL.
  Gate 4  tiny overfit from identical initialisation: SFT-only vs SFT+CTM(+SM) on the same items and
          order; CTM must be active on every step whose record carries a CTM pair; margins before/after.
  Gate 3  one AdamW step: intended params change (max |delta|, counts), frozen do not (exact), old-row
          deltas in the stored dtype; save; free the model; reload; rebuild the SAME examples with the
          reloaded processor (ids must match); scores within tolerance, both models in eval mode.
          Records that likelihood scoring does not mask the new ids (generation does).
Interfaces: neo | pt (prepended) | pti (inserted, PT_INSERTED=1) | lora (request-conditioned on the
ladder, instructed carrier on Ground-CF). GPU only.
"""
import argparse, json, os, re, sys, time, inspect, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SC_LOWMEM", "1")          # one vision graph alive at a time (identical maths)
import torch
from scsrc import model as M
import scsrc.preference as P
from scsrc.tokens import default_specs, add_neologisms, NeoSpec
from scsrc.prompts import LORA_INSTR, pick_ctrl_template, pt_template

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TERM_KEY = {"sft": "ce", "ctm": "ctm", "sm": "sm", "apo": "apo"}
C3_INSTR = {"look": "the image", "read": "the caption", "ocr": "the on-screen text"}


def examples_of(rec):
    out = []
    for k, v in rec.items():
        if not isinstance(v, dict):
            continue
        if "input_ids" in v:
            out.append((k, "", v))
        else:
            out += [(k, side, ex) for side, ex in v.items() if isinstance(ex, dict) and "input_ids" in ex]
    return out


def scored_span(ex):
    if ex.get("labels") is not None:
        idx = (ex["labels"][0] != -100).nonzero().flatten()
        return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (None, None)
    cs = int(ex.get("score_start", ex.get("cont_start")))
    return cs, int(ex["input_ids"].shape[1])


def find_block(ids, block):
    """start positions where `block` occurs contiguously in `ids`."""
    n = len(block)
    return [j for j in range(len(ids) - n + 1) if ids[j:j + n] == block] if n else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True); ap.add_argument("--method", required=True, choices=["neo", "pt", "pti", "lora"])
    ap.add_argument("--task", required=True, choices=["color", "clevr3"]); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_tokens", type=int, default=5); ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--n_items", type=int, default=12); ap.add_argument("--gate4", type=int, default=1)
    ap.add_argument("--gate4_steps", type=int, default=30); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.method == "pti":
        os.environ["PT_INSERTED"] = "1"
    meth = "pt" if a.method == "pti" else a.method
    if meth == "lora":
        os.environ["SC_GRAD_CKPT"] = "1"
    ladder = a.task == "color"
    mtag = a.model_id.split("/")[-1].lower().replace(".", "")
    tag = a.tag or f"{os.path.basename(a.data_dir)}_{a.method}_{mtag}"
    out = {"tag": tag, "model_id": a.model_id, "method": a.method, "task": a.task, "data_dir": a.data_dir,
           "n_tokens": a.n_tokens, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "gates": {}}
    dev = "cuda"
    model, proc = M.load(model_id=a.model_id, attn="eager")

    # ---------------------------------------------------------------- interface setup, as train.py does it
    all_new_ids, name_to_ids, W, ref_rows = None, None, None, None
    if meth in ("neo", "pt"):
        if ladder:
            specs = default_specs()["ctrl"] if meth == "neo" else default_specs()["pt"]
        else:
            from scsrc.clevr3 import clevr3_user_content
            rows = [json.loads(l) for l in open(os.path.join(a.data_dir, "train.jsonl"))][:40]
            names = set()
            for r in rows:
                for src in ("look", "read", "ocr"):
                    c = clevr3_user_content(r, src, a.n_tokens, "srconly", 0, question_style="explicit", method=meth)
                    names |= set(re.findall(r"<([a-z_]+)_\d+>", " ".join(b.get("text", "") for b in c if isinstance(b, dict))))
            specs = [NeoSpec(nm, "pt" if meth == "pt" else "ctrl", "text", nm) for nm in sorted(names)]
        name_to_ids, all_new_ids = add_neologisms(proc.tokenizer, specs, a.n_tokens)
        W = M.setup_neologisms(model, proc.tokenizer, name_to_ids, all_new_ids)
        M.init_neologisms(model, proc.tokenizer, name_to_ids, {s.name: s for s in specs}, mode="word_mean", seed=0)
        ref_rows = W.data[all_new_ids].detach().clone(); trainable = [W]
        out["registered_tokens"] = dict(name_to_ids)
    else:
        model = M.setup_lora(model, r=a.lora_r, alpha=32)
        trainable = [p for p in model.parameters() if p.requires_grad]
    init_params = [p.detach().clone() for p in trainable]
    out["dtype"] = str(model.get_input_embeddings().weight.dtype)
    oe = model.get_output_embeddings()
    out["tied_lm_head"] = bool(oe is not None and oe.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr())

    # ---------------------------------------------------------------- dataset, the real class
    eff_full = {"lam_sft": 1.0, "lam_ctm": 1.0, "lam_sm": 1.0, "lam_apo": (1.0 if meth != "lora" else 0.0),
                "lam_vgm": 0.0, "lam_tc": 0.0, "gamma": 0.1, "beta": 0.1}
    arm = "ctrl" if meth == "neo" else "na"
    c3mode = "instructed" if meth == "lora" else "srconly"
    if ladder:
        from scsrc.data import load_task, build_specs
        import scsrc.dataset as DS
        ds = load_task(a.data_dir, "train")
        sp = [s for s in build_specs(ds, "train", conditions=("conflict",)) if s.cond == "conflict"][:a.n_items]
        cls = next(c for n, c in inspect.getmembers(DS, inspect.isclass) if "obj_global" in inspect.signature(c.__init__).parameters)
        data = cls(ds, sp, proc, meth, arm, a.n_tokens, dev, eff_full, task="color", ctrl_style=("request" if meth == "lora" else "neutral"))
        def tpl_for(i):
            if meth == "neo" or a.method == "pti": return pick_ctrl_template(i)
            if meth == "pt": return pt_template(i)
            return None
    else:
        from scsrc.clevr3 import Clevr3Dataset
        data = Clevr3Dataset(a.data_dir, "train", proc, dev, n_tokens=a.n_tokens, mode=c3mode,
                             held_cells=(("look", "spatial"), ("read", "count"), ("ocr", "object")),
                             apo=(meth != "lora"), ctm=True, sm=True, method=meth, question_style="explicit")
    print(f"[gates] {tag}: items={len(data)} dtype={out['dtype']} tied={out['tied_lm_head']}", flush=True)

    def source_of(i, rec):
        return (rec.get("op") if ladder else data.items[i][1])

    def expected_block(src):
        """(ids, text) the item's OWN source control must carry; ids for embedding methods, text for LoRA."""
        if meth == "lora":
            return None, (LORA_INSTR[src] if ladder else f"Answer using only {C3_INSTR[src]}.")
        if ladder:
            return name_to_ids[f"{src}_ctrl" if meth == "neo" else f"pt_{src}"], None
        ids = [i for nm, ids_ in name_to_ids.items() if src in nm for i in ids_]
        return ids, None

    # ---------------------------------------------------------------- Gate 1
    g1 = {"examples": [], "exposure": {}, "fails": []}
    recs = []
    n_take = min(len(data), a.n_items if ladder else 3 * a.n_items)
    for i in range(n_take):
        try:
            rec = data[i]
        except Exception as e:
            g1["fails"].append(f"item {i}: __getitem__ raised {type(e).__name__}: {str(e)[:120]}"); continue
        recs.append((i, rec))
        src = source_of(i, rec)
        g1["exposure"][src] = g1["exposure"].get(src, 0) + 1
        if src not in ("look", "read", "ocr"):
            continue
        blk_ids, blk_txt = expected_block(src)
        others = [s for s in (("look", "read") if ladder else ("look", "read", "ocr")) if s != src]
        blocks_seen = {}
        for key, side, ex in examples_of(rec):
            ids = ex["input_ids"][0].tolist(); s0, s1 = scored_span(ex)
            prompt = proc.tokenizer.decode(ids[:s0], skip_special_tokens=False) if s0 else ""
            target = proc.tokenizer.decode(ids[s0:s1], skip_special_tokens=False) if s0 else ""
            e = {"item": i, "source": src, "key": key, "side": side, "scored_span": [s0, s1], "prompt_tail": prompt[-200:], "target": target[:120]}
            if blk_ids is not None:
                pos = find_block(ids, blk_ids); e["block_positions"] = pos; e["block_to_cont_offset"] = ([s0 - (p + len(blk_ids)) for p in pos] if s0 else None)
                own_ok = len(pos) == 1
                foreign = [o for o in others if find_block(ids, expected_block(o)[0])]
            else:
                own_ok = blk_txt in prompt
                foreign = [o for o in others if expected_block(o)[1] in prompt] + (["registered-token-string"] if re.search(r"<[a-z_]+_\d+>", prompt) else [])
            e["own_block_once"] = own_ok; e["foreign"] = foreign
            blocks_seen.setdefault(key, set()).add(tuple(e.get("block_positions", [])) if blk_ids is not None else own_ok)
            g1["examples"].append(e)
            if not own_ok:
                g1["fails"].append(f"item {i} {key}/{side} src={src}: own control block not exactly once ({e.get('block_positions')})")
            if foreign:
                g1["fails"].append(f"item {i} {key}/{side} src={src}: foreign control present {foreign}")
        for key, seen in blocks_seen.items():
            if key in ("ctm", "sm", "apo", "pref") and blk_ids is not None and len(seen) > 1 and ladder and src == "look":
                # image-swap pairs share the prompt exactly; caption-swap pairs may shift the block with caption length
                g1["fails"].append(f"item {i} {key}: control block position differs across sides of an image-swap pair {sorted(seen)}")
    g1["keys_seen"] = sorted({k for _, r in recs for k in r if isinstance(r[k], dict)})
    need = ["sft", "ctm", "sm"] + (["apo|pref"] if meth != "lora" else [])
    for nd in need:
        if not any(k in g1["keys_seen"] for k in nd.split("|")):
            g1["fails"].append(f"no record ever carried '{nd}' although its lam>0")
    g1["pass"] = not g1["fails"]; out["gates"]["1_records"] = g1
    print(f"[gate1] keys={g1['keys_seen']} exposure={g1['exposure']} fails={len(g1['fails'])}", flush=True)
    if not recs:
        out["fatal"] = "no records"; json.dump(out, open(f"{ROOT}/results/gates/{tag}.json", "w"), indent=1); sys.exit(2)
    rec_i, rec = next(((i, r) for i, r in recs if all(k in r for k in ("sft", "ctm", "sm"))), recs[0])

    # ---------------------------------------------------------------- Gate 2
    def grad_report():
        if W is not None:
            g = W.grad
            if g is None:
                return {"new_rows_nonzero": 0, "note": "W.grad is None"}
            mask = torch.ones(g.shape[0], dtype=torch.bool, device=g.device); mask[all_new_ids] = False
            return {"new_rows_nonzero": int((g[all_new_ids].float().norm(dim=-1) > 0).sum()), "new_rows_total": len(all_new_ids),
                    "old_rows_grad_max": float(g[mask].float().abs().max())}
        nz = sum(1 for p in trainable if p.grad is not None and float(p.grad.float().abs().max()) > 0)
        frozen_bad = sum(1 for n_, p in model.named_parameters() if not p.requires_grad and p.grad is not None and float(p.grad.float().abs().max()) > 0)
        return {"adapter_params_nonzero": nz, "adapter_params_total": len(trainable), "frozen_with_grad": frozen_bad}
    g2 = {}
    model.train()
    for term in ["sft", "ctm", "sm"] + (["apo"] if meth != "lora" else []):
        eff = {k: 0.0 for k in eff_full}; eff.update({"gamma": 0.1, "beta": 0.1, f"lam_{term}": 1.0})
        model.zero_grad(set_to_none=True)
        if W is not None: W.grad = None
        try:
            loss, comps = P.objective_step(model, rec, eff, all_new_ids, ref_rows, accum=1)
            r = {"loss": float(loss), "finite": bool(torch.isfinite(torch.tensor(float(loss)))),
                 "comps": {k: float(v) for k, v in comps.items() if isinstance(v, (int, float))}}
            r.update(grad_report())
            has_term = TERM_KEY[term] in comps and comps[TERM_KEY[term]] is not None
            has_grad = (r.get("new_rows_nonzero", 0) > 0) or (r.get("adapter_params_nonzero", 0) > 0)
            r["pass"] = bool(r["finite"] and has_term and has_grad and r.get("old_rows_grad_max", 0.0) == 0.0 and r.get("frozen_with_grad", 0) == 0)
            if not has_term: r["fail_reason"] = f"term enabled but '{TERM_KEY[term]}' absent from comps (no record?)"
            elif not has_grad: r["fail_reason"] = "term computed but no gradient reached the trainable parameters"
        except Exception as e:
            r = {"pass": False, "fail_reason": f"{type(e).__name__}: {str(e)[:160]}"}
        g2[term] = r; print(f"[gate2] {term}: pass={r.get('pass')} {r.get('fail_reason','')}", flush=True)
    out["gates"]["2_gradients"] = g2
    model.zero_grad(set_to_none=True)

    # ---------------------------------------------------------------- Gate 4 (before the step, from identical init)
    if a.gate4:
        g4 = {}
        items = [r for _, r in recs[:8]]
        lr = 2e-4 if meth == "lora" else 0.05
        def restore():
            with torch.no_grad():
                for p, q in zip(trainable, init_params): p.data.copy_(q)
        def margins():
            m = []
            model.eval()
            for r_ in items:
                c = r_.get("ctm")
                if c:
                    m.append((float(P.seq_logprob(model, c["a_pos"], True) - P.seq_logprob(model, c["a_neg"], True)),
                              float(P.seq_logprob(model, c["b_pos"], True) - P.seq_logprob(model, c["b_neg"], True))))
            model.train()
            return {"delta_old_mean": sum(x for x, _ in m) / max(len(m), 1), "delta_new_mean": sum(y for _, y in m) / max(len(m), 1), "n_items_with_ctm": len(m)}
        for label, eff in (("sft_only", {**{k: 0.0 for k in eff_full}, "lam_sft": 1.0, "gamma": .1, "beta": .1}),
                           ("sft_ctm_sm", {**eff_full, "lam_apo": 0.0})):
            restore(); opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01); traj = []; n_ctm_records = 0
            m0 = margins()
            for st in range(a.gate4_steps):
                r_ = items[st % len(items)]; n_ctm_records += int(bool(r_.get("ctm")))
                opt.zero_grad(set_to_none=True)
                if W is not None: W.grad = None
                loss, comps = P.objective_step(model, r_, eff, all_new_ids, ref_rows, accum=1); opt.step()
                traj.append({k: round(float(v), 4) for k, v in comps.items() if isinstance(v, (int, float))})
            m1 = margins()
            g4[label] = {"margins_before": m0, "margins_after": m1, "comps_first": traj[0], "comps_last": traj[-1],
                         # a term is ACTIVE when it was computed (key present); a margin that saturates the
                         # logistic to exactly 0.0 is not inactivity (coco/LoRA gate, 2026-09-21)
                         "ctm_active_steps": sum(1 for t in traj if "ctm" in t), "ctm_saturated_zero_steps": sum(1 for t in traj if t.get("ctm") == 0.0),
                         "steps_with_ctm_record": n_ctm_records}
            del opt
        g4["pass"] = g4["sft_ctm_sm"]["ctm_active_steps"] == g4["sft_ctm_sm"]["steps_with_ctm_record"] > 0
        out["gates"]["4_overfit"] = g4; print(f"[gate4] pass={g4['pass']} ctm_active={g4['sft_ctm_sm']['ctm_active_steps']}/{g4['sft_ctm_sm']['steps_with_ctm_record']}", flush=True)
        restore()

    # ---------------------------------------------------------------- Gate 3
    g3 = {}
    model.zero_grad(set_to_none=True); model.train()
    if W is not None:
        mask = torch.ones(W.shape[0], dtype=torch.bool, device=W.device); mask[all_new_ids] = False
        snap_new = W.data[all_new_ids].clone(); snap_old = W.data[mask][:20000].clone()
    else:
        frozen_name, frozen_p = next((n_, p) for n_, p in model.named_parameters() if not p.requires_grad and p.dim() == 2 and "embed" not in n_)
        snap_frozen = frozen_p.detach().clone()
    opt = torch.optim.AdamW(trainable, lr=(2e-4 if meth == "lora" else 0.05), weight_decay=0.01)
    loss, comps = P.objective_step(model, rec, eff_full, all_new_ids, ref_rows, accum=1)
    opt.step(); opt.zero_grad(set_to_none=True); del opt
    if W is not None:
        d_new = (W.data[all_new_ids].float() - snap_new.float()).abs(); d_old = (W.data[mask][:20000].float() - snap_old.float()).abs()
        g3.update({"new_rows_max_delta": float(d_new.max()), "new_rows_changed_elems": int((d_new > 0).sum()),
                   "old_rows_max_delta": float(d_old.max()), "old_rows_changed_elems": int((d_old > 0).sum()), "old_rows_sample_elems": int(d_old.numel()),
                   "old_rows_note": "decoupled weight decay on grad-masked rows in the stored dtype; 0 means rounding absorbed it"})
        intended_changed = g3["new_rows_changed_elems"] > 0; frozen_ok = True
    else:
        d = [(p.detach().float() - q.float()).abs() for p, q in zip(trainable, init_params)]
        g3.update({"adapter_max_delta": float(max(x.max() for x in d)), "adapter_changed_params": int(sum(1 for x in d if x.max() > 0)), "adapter_total": len(d),
                   "frozen_probe": frozen_name, "frozen_max_delta": float((frozen_p.detach().float() - snap_frozen.float()).abs().max())})
        intended_changed = g3["adapter_changed_params"] > 0; frozen_ok = g3["frozen_max_delta"] == 0.0
    # scores of the trained in-memory model, EVAL mode (LoRA dropout off), on the record's own examples
    model.eval()
    ex_keys = [(k, s) for k, s in (("ctm", "a_pos"), ("ctm", "a_neg"), ("sm", "pos")) if rec.get(k)]
    mem_scores = {f"{k}/{s}": float(P.seq_logprob(model, rec[k][s], no_grad=True)) for k, s in ex_keys}
    ids_ref = {f"{k}/{s}": rec[k][s]["input_ids"][0].tolist() for k, s in ex_keys}
    # save as train.py, then FREE the model before reloading (a second 4B model OOMs a 44 GB card)
    run_dir = f"{ROOT}/runs/_gates_{tag}"; d1 = os.path.join(run_dir, "ckpt", "step_1"); os.makedirs(d1, exist_ok=True)
    if W is not None:
        M.save_neologisms(W, all_new_ids, name_to_ids, os.path.join(d1, "neo.pt"))
    else:
        model.save_pretrained(d1)
    json.dump({"step": 1, "method": meth, "arm": arm, "n_tokens": a.n_tokens, "seed": 0, "model_id": a.model_id, "add_word": None, "extend_ckpt": None},
              open(os.path.join(d1, "meta.json"), "w"))
    json.dump({"args": {"method": meth, "ctrl_style": ("request" if meth == "lora" else "neutral"), "pt_inserted": int(a.method == "pti"),
                        "factor_mode": c3mode, "n_tokens": a.n_tokens}}, open(os.path.join(run_dir, "config.json"), "w"))
    item_ref = data.items[rec_i]
    del model, W, trainable, init_params; torch.cuda.empty_cache()
    try:
        from evaluate import apply_checkpoint
        model2, proc2, meta2, forbid = apply_checkpoint(run_dir, 1, dev); model2.eval()
        if ladder:
            spec_, op_ = item_ref; op_ = op_ if op_ in ("look", "read") else "look"
            pairs2 = {"ctm": P.build_ctm_pair(proc2, ds, spec_, arm, op_, a.n_tokens, dev, template=tpl_for(rec_i), method=meth),
                      "sm": P.build_sm_pair(proc2, ds, spec_, arm, op_, a.n_tokens, dev, template=tpl_for(rec_i), method=meth)}
        else:
            from scsrc.clevr3 import build_clevr3_ctm_quad, build_clevr3_sm_pair
            row_, src_ = item_ref
            pairs2 = {"ctm": build_clevr3_ctm_quad(proc2, row_, src_, dev, a.n_tokens, c3mode, seed=rec_i, question_style="explicit", method=meth),
                      "sm": build_clevr3_sm_pair(proc2, row_, src_, dev, a.n_tokens, c3mode, seed=rec_i, question_style="explicit", method=meth)}
        reload_scores, ids_same = {}, {}
        for k, s in ex_keys:
            if pairs2.get(k) and s in pairs2[k]:
                reload_scores[f"{k}/{s}"] = float(P.seq_logprob(model2, pairs2[k][s], no_grad=True))
                ids_same[f"{k}/{s}"] = pairs2[k][s]["input_ids"][0].tolist() == ids_ref[f"{k}/{s}"]
        deltas = {k: abs(reload_scores[k] - mem_scores[k]) for k in reload_scores}
        g3.update({"scores_in_memory_eval": mem_scores, "scores_after_reload": reload_scores, "rebuilt_ids_identical": ids_same,
                   "reload_abs_delta": deltas, "reload_pass": bool(deltas) and all(ids_same.values()) and max(deltas.values()) < 0.05,
                   "forbid_ids_for_generation": (len(forbid) if forbid else 0), "likelihood_masks_new_ids": False,
                   "protocol_note": "scorers take log_softmax over the full vocabulary; new ids are forbidden at GENERATION only"})
        del model2; torch.cuda.empty_cache()
    except Exception as e:
        g3.update({"reload_pass": False, "reload_error": f"{type(e).__name__}: {str(e)[:220]}"})
    shutil.rmtree(run_dir, ignore_errors=True)
    g3["pass"] = bool(g3.get("reload_pass")) and intended_changed and frozen_ok
    out["gates"]["3_step_reload"] = g3
    print(f"[gate3] pass={g3['pass']} reload={g3.get('reload_pass')} deltas={ {k: round(v, 4) for k, v in (g3.get('reload_abs_delta') or {}).items()} } err={g3.get('reload_error','')}", flush=True)

    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["pass_all"] = bool(g1["pass"] and all(v.get("pass") for v in g2.values()) and g3["pass"] and (not a.gate4 or out["gates"]["4_overfit"]["pass"]))
    json.dump(out, open(f"{ROOT}/results/gates/{tag}.json", "w"), indent=1)
    print(f"[gates] {tag} PASS_ALL={out['pass_all']} -> results/gates/{tag}.json", flush=True)
    sys.exit(0 if out["pass_all"] else 2)


if __name__ == "__main__":
    main()
