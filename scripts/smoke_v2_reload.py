#!/usr/bin/env python
"""Protocol-v2 training-step-and-reload check (2026-09-29, audit).

train.py --selfcheck_items K writes ckpt/step_<n>/selfcheck.json: for K training rows and both requests, the rendered
prompt, the target continuation, the candidate continuations and their in-process log-likelihoods. This script reloads
the checkpoint through evaluate.apply_checkpoint (the path every evaluator uses), re-renders the same prompts and
re-scores the same candidates, and reports:
  * prompt text identical after reload (the v2 switches were restored from config)      -> must be 100%
  * target format identical (SC_GCF_CONT restored)                                       -> must match
  * max |score_in_process - score_reloaded| and pick agreement                            -> ~1e-3 / 100%
  * the active loss terms from config (obj_global) and the last logged ce / ce_twin / ctm from the .out file (if given)
Usage: python scripts/smoke_v2_reload.py --run_dir runs/<run> --step <n> [--log logs/<file>.out]
"""
import argparse, json, os, re, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import apply_checkpoint
from scsrc.clevr3 import clevr3_user_content, c3_cont
from scsrc.collate import _encode, _prompt_len


@torch.no_grad()
def score(model, proc, um, cont):
    fm = um + [{"role": "assistant", "content": [{"type": "text", "text": cont}]}]
    enc = _encode(proc, fm, "cuda"); plen = _prompt_len(proc, um, "cuda")
    ids = enc["input_ids"]; kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    lg = model(**kw, use_cache=False).logits.float()
    lp = torch.log_softmax(lg[0, plen - 1:ids.shape[1] - 1, :], -1)
    return float(lp.gather(-1, ids[0, plen:].unsqueeze(-1)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--step", type=int, required=True); ap.add_argument("--log", default="")
    a = ap.parse_args()
    sc_path = os.path.join(a.run_dir, "ckpt", f"step_{a.step}", "selfcheck.json")
    S = json.load(open(sc_path)); cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    args = cfg.get("args", {}); g = cfg.get("obj_global", {})
    model, proc, meta, forbid = apply_checkpoint(a.run_dir, a.step, "cuda"); model.eval()
    print(f"[reload] {a.run_dir} step {a.step}: method={args.get('method')} format={os.environ.get('SC_GCF_FORMAT')} "
          f"cont={os.environ.get('SC_GCF_CONT')} neo_template={os.environ.get('SC_NEO_TEMPLATE')} pt_carrier={os.environ.get('SC_GCF_PT_CARRIER')}")
    print("[reload] active loss terms:", {k: v for k, v in g.items() if isinstance(v, (int, float)) and v})
    rows = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(args.get("data_dir") or "data/clevr3", "train.jsonl")))}
    meth = args.get("method") if args.get("method") in ("neo", "pt") else "lora"
    same_prompt = same_target = agree = 0; maxd = 0.0; maxd_same = 0.0; dumped = []
    for rec in S["records"]:
        r = rows[rec["id"]]
        um = [{"role": "user", "content": clevr3_user_content(r, rec["source"], args.get("n_tokens", 1), args.get("factor_mode"), 0, method=meth, question_style="explicit")}]
        txt = " ".join("[IMAGE]" if c.get("type") == "image" else c.get("text", "") for c in um[0]["content"])
        same_prompt += int(txt == rec["prompt"]); same_target += int(c3_cont(str(r["channels"][rec["source"]])) == rec["target"])
        sc = {v: score(model, proc, um, c3_cont(v)) for v in rec["cands"]}
        d = max(abs(sc[v] - rec["scores"][v]) for v in rec["cands"]); maxd = max(maxd, d)
        offs = [sc[v] - rec["scores"][v] for v in rec["cands"]]
        dumped.append({"id": rec["id"], "source": rec["source"], "inproc": rec["scores"], "reload": sc,
                       "offset_mean": sum(offs) / len(offs), "offset_spread": max(offs) - min(offs)})
        if txt == rec["prompt"]: maxd_same = max(maxd_same, d)
        else: print(f"[reload] PROMPT DIFF {rec['source']} {rec['id']}:\n   in-process: {rec['prompt'].replace(chr(10),' | ')[:300]}\n   reloaded  : {txt.replace(chr(10),' | ')[:300]}")
        agree += int(max(sc, key=sc.get) == rec["pick"])
    n = len(S["records"])
    print(f"[reload] prompts identical {same_prompt}/{n}; targets identical {same_target}/{n}; picks agree {agree}/{n}; "
          f"max |dlogp| {maxd:.4f} (over identical prompts {maxd_same:.4f})")
    import statistics as _st
    print(f"[reload] offset analysis: mean per-record offset {_st.mean(d['offset_mean'] for d in dumped):+.4f}, "
          f"max within-record spread {max(d['offset_spread'] for d in dumped):.4f} "
          f"(a constant offset = prompt-side encoding difference; spread = candidate-side difference)")
    json.dump(dumped, open(os.path.join(a.run_dir, "ckpt", f"step_{a.step}", "selfcheck_reload.json"), "w"), indent=1)
    print(f"[reload] first prompt:\n{S['records'][0]['prompt'][:500]}\n[reload] target: {S['records'][0]['target']!r}")
    if not a.log:   # find the training log by run name (train_array prints the manifest line)
        import glob as _g
        for f in sorted(_g.glob("logs/sc-train-*.out"), key=os.path.getmtime, reverse=True)[:400]:
            try:
                if f"--run_name {os.path.basename(a.run_dir)} " in open(f, errors="ignore").read(4000) or f"--run_name {os.path.basename(a.run_dir)}\n" in open(f, errors="ignore").read(4000):
                    a.log = f; break
            except Exception:
                pass
    if a.log and os.path.exists(a.log):
        lines = [l for l in open(a.log, errors="ignore") if "ce_twin=" in l or "ce=" in l]
        print("[reload] last progress line:", (lines[-1].strip()[-220:] if lines else "none found"))
        print("[reload] [pt]/[v2]/[optim] lines:", *[l.strip()[:200] for l in open(a.log, errors="ignore") if l.startswith(("[pt]", "[v2]", "[optim]", "[selfcheck]"))][:6], sep="\n  ")
    ok = same_prompt == n and same_target == n and agree == n and maxd_same < 5e-2
    print("[reload] RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
