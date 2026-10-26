#!/usr/bin/env python
"""CPU DRY RUN of an evaluation script (2026-09-23, STATUS v1.130): runs the script's real main() end to end with the
model replaced by a deterministic stub, so every Python-level path (prompt building, pairing, candidate sets, protocol,
record writing, output fields) is exercised without a GPU. Scores are a hash of (prompt text, candidate) -- the
NUMBERS are meaningless by design; only "it runs and writes the fields we expect" is being checked. Outputs are written
under the tag given in the arguments (use a `_dryrun` tag: the harvester skips `_smoke`/`_dryrun`-free? -> see below).

Usage: python scripts/dryrun_eval.py scripts/eval_clevr3_switch.py --run_dir runs/<run> ... --tag _dryrun
The run's config.json is read for the environment exactly as evaluate.apply_checkpoint does.
"""
import importlib.util, json, os, sys, zlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _env_from_config(run_dir):
    try:   # v1.153: frozen-baseline run dirs only name the output location and carry no config
        a = (json.load(open(os.path.join(run_dir, "config.json"))).get("args") or {})
    except FileNotFoundError:
        a = {}
    if a.get("c3_sources"):
        os.environ["SC_C3_SOURCES"] = a["c3_sources"]
    os.environ["SC_NEO_TEMPLATE"] = a.get("neo_template") or "bare"
    os.environ["SC_CAPTION_STYLE"] = os.environ.get("SC_CAPTION_STYLE_OVERRIDE") or a.get("caption_style") or "template"
    os.environ["SC_LORA_ADAPTER"] = "1" if (a.get("method") == "lora" and a.get("ctrl_style") == "adapter") else "0"
    for k in ("SC_INSTR_ROTATE", "SC_Q_ROTATE", "SC_TRAIN_SOURCES"):
        os.environ.pop(k, None)
    return a


def _text(umsg):
    out = []
    for m in umsg:
        for c in m.get("content", []):
            out.append("[IMG]" if c.get("type") == "image" else c.get("text", ""))
    return " ".join(out)


def fake_score(model, proc, umsg, candidate, device="cpu", *a, **k):
    # v1.153: SC_DRY_CONST=1 -> an INPUT-INSENSITIVE scorer (the candidate text alone sets its score). Under a
    # fixed-candidate protocol it must get zero Flip; under per-version candidate sets it can score Flip.
    key = str(candidate) if os.environ.get("SC_DRY_CONST") == "1" else (_text(umsg) + "||" + str(candidate))
    return -float(zlib.crc32(key.encode()) % 1000) / 100.0


def fake_tokens(umsg, candidate):
    """shared lead-in (5 ids) + candidate-specific answer ids + shared end-of-turn (2 ids); lps sum to fake_score"""
    ans = [1000 + (zlib.crc32(w.encode()) % 5000) for w in str(candidate).split()[-2:]]
    ids = [1, 2, 3, 4, 5] + ans + [99, 100]
    tot = fake_score(None, None, umsg, candidate)
    lps = [-0.01] * 5 + [round((tot + 0.07) / len(ans), 4)] * len(ans) + [-0.01, -0.01]
    return ids, lps


def main():
    script, argv = sys.argv[1], sys.argv[2:]
    run_dir = argv[argv.index("--run_dir") + 1]
    cfg = _env_from_config(run_dir)
    from transformers import AutoProcessor
    from scsrc import model as M, paths
    mid = (argv[argv.index("--frozen_model_id") + 1] if "--frozen_model_id" in argv else None) or cfg.get("model_id") or paths.MODEL_ID
    proc = AutoProcessor.from_pretrained(M._resolve_cached(mid))
    _mp = os.path.join(run_dir, "ckpt", "step_800", "meta.json")
    meta = json.load(open(_mp)) if os.path.exists(_mp) else {"n_tokens": 0, "method": "frozen", "model_id": mid}
    import torch

    class _Emb:
        def __init__(self): self.weight = torch.zeros(len(proc.tokenizer) + 4096, 8)
    class _FakeModel:                               # only what evaluators query outside scoring (dtype, eval())
        _e = None
        def parameters(self):
            yield torch.zeros(1, dtype=torch.bfloat16)
        def eval(self):
            return self
        def get_input_embeddings(self):             # v1.141: the metalinguistic controls add rows
            if self._e is None: self._e = _Emb()
            return self._e
    M.resize_for_new_tokens = lambda model, tok, seed=0: None
    M.load = lambda *a_, **k_: (_FakeModel(), proc)   # v1.153: frozen baselines (--frozen_model_id) never load weights
    fake_ckpt = lambda rd, step, dev: (_FakeModel(), proc, meta, None)
    spec = importlib.util.spec_from_file_location("target", os.path.join(ROOT, script))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for name in ("apply_checkpoint",):
        if hasattr(mod, name):
            setattr(mod, name, fake_ckpt)
    for name in ("loglik", "loglik_norm"):
        if hasattr(mod, name):
            setattr(mod, name, fake_score)
    def _lb0(model, proc, um, c, dev="cpu", tokens=False):
        v = fake_score(model, proc, um, c)
        if tokens:
            ids, lps = fake_tokens(um, c); return v, v / len(ids), len(ids), ids, lps
        return v, v, 1
    if hasattr(mod, "loglik_both"):
        setattr(mod, "loglik_both", _lb0)
    if hasattr(mod, "generate_free"):
        # a fixed composed-style reply: exercises every parser branch without a model
        setattr(mod, "generate_free", lambda model, proc, content, dev, max_new, forbid, grace:
                ("From the first requested source, the answer is red. From the second requested source, the answer is 4. "
                 "The requested sources conflict. final answer: red; 4", 12))
    if hasattr(mod, "loglik_batch"):
        setattr(mod, "loglik_batch", lambda model, proc, um, cands, dev: {c: (fake_score(model, proc, um, c), 0) for c in cands})
    if hasattr(mod, "SC"):                         # v1.141: ladder evaluators score through scsrc.score
        def _sl(model, proc, image, u, c, dev="cpu", tokens=False):
            um = [{"content": [{"type": "text", "text": u}]}]; v = fake_score(model, proc, um, c)
            if tokens:
                ids, lps = fake_tokens(um, c); return v, v / len(ids), len(ids), ids, lps
            return v, v, 1
        mod.SC.loglik = _sl
    if hasattr(mod, "generate"):
        mod.generate = lambda *x, **k: "The caption and the image disagree. final answer: red"
    import scripts.eval_clevr3_generate as _G      # v1.141: evaluators that import loglik_both inside main()
    def _lb(model, proc, um, c, dev="cpu", tokens=False):
        v = fake_score(model, proc, um, c)
        if tokens:
            ids, lps = fake_tokens(um, c); return v, v / len(ids), len(ids), ids, lps
        return v, v, 1
    _G.loglik_both = _lb
    sys.argv = [script] + argv
    mod.main()


if __name__ == "__main__":
    main()
