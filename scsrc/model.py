"""Model loading + parameter-isolation for the three training methods.

neo : add tokens, FREEZE everything, train ONLY the new embedding rows (a
      gradient hook zeros grads on all pre-existing rows). Gemma ties input and
      output embeddings, so one matrix carries both the read- and write-vector;
      that is intentional within an arm. We keep ctrl and gen on SEPARATE tokens
      across arms so behavioral equivalence can be tested, not assumed.
lora: standard PEFT adapter on the language tower (the "just fine-tune" baseline
      Hewitt insisted on). Vision tower frozen.
pt  : soft-prompt prefix (EXPERIMENTAL). Structurally comparable to ctrl only:
      a prefix cannot be emitted, so it cannot express the generation arm.
"""
from . import paths  # sets env first
import os
import torch

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
except Exception:  # older transformers
    from transformers import AutoProcessor
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText  # noqa


def load(model_id=None, dtype=torch.bfloat16, attn="eager", device="cuda"):
    model_id = model_id or paths.MODEL_ID
    # SC_DTYPE=float32|float16|bfloat16 overrides the weight dtype (2026-09-21): the V100 pool has no bf16,
    # so evaluations there score in float32. Gemma-4-E4B (per-layer table ~2.8B params) does not fit a
    # 32 GB card in float32 and stays on A100/L40S. Training never sets this.
    import os as _os0
    _dt = _os0.environ.get("SC_DTYPE", "").strip().lower()
    if _dt:
        dtype = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16,
                 "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[_dt]
        print(f"[model] SC_DTYPE={_dt}: loading weights as {dtype}")
    kw = {}
    if "qwen" in model_id.lower():
        # Qwen's dynamic-resolution ViT emits image tokens proportional to
        # pixels (Gemma is fixed at 256); native-res ColorSwap images blow the
        # sequence length and OOM a 44G L40S at step 1. ~1024 visual tokens:
        import os as _os
        # Qwen uses DYNAMIC visual tokenization: max_pixels/28^2 = tokens/image
        # (gemma is FIXED at 256). Default 1024 tok/frame OOM'd MVG multi-frame
        # runs (4 frames x CTM's 4 sequences on 80GB). QWEN_MAX_TOK caps it;
        # 256 = gemma parity, which is also the FAIR comparison setting.
        _mt = int(_os.environ.get("QWEN_MAX_TOK", "1024"))
        kw = {"min_pixels": 256 * 28 * 28, "max_pixels": _mt * 28 * 28}
    model_id = _resolve_cached(model_id)
    processor = AutoProcessor.from_pretrained(model_id, **kw)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation=attn)
    model.to(device)
    # GRADIENT CHECKPOINTING, opt-in via SC_GRAD_CKPT=1.
    # It was never enabled anywhere, and MultiVENT-Raw cannot train without it: a
    # PLAIN SFT smoke at --n_frames 4 died with CUDA OOM at 42.71 of 44.39 GiB
    # (job 431359, 83 s in). Not the 8-frame CTM quad from the old note -- four
    # frames and a single CE term were already enough, because the MVR prompt
    # carries all three channels plus N images while the text is tiny (ASR median
    # 27 chars). So the cost is entirely visual tokens, and recomputing
    # activations is the right trade: ~40-60% less VRAM for ~20-30% slower steps.
    # use_cache must go off or the cache silently defeats it.
    if os.environ.get("SC_GRAD_CKPT", "") == "1":
        try:
            model.config.use_cache = False
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            print("[model] gradient checkpointing ON (use_cache=False)")
        except Exception as e:                     # never fail the run over this
            print(f"[model] gradient checkpointing unavailable: {str(e)[:90]}")
    return model, processor


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


# ---- neologism (embedding-only) --------------------------------------------
def _resolve_cached(model_id):
    """Repo id -> a complete local snapshot when the cache's default revision is incomplete (2026-09-22, v1.96).

    The Gemma-4-E4B cache holds two snapshots and `refs/main` points at the one WITHOUT processor files, so on a
    compute node (no internet) `AutoProcessor.from_pretrained("google/gemma-4-E4B-it")` raises "Can't load feature
    extractor" and every frozen-baseline evaluation of that model died. Trained runs happened to resolve differently.
    If the id loads, it is returned unchanged; otherwise the newest snapshot directory that has a processor config
    and weights is used. Local paths are returned as they are.
    """
    import os as _o, glob as _g
    if _o.path.isdir(model_id):
        return model_id
    try:
        from transformers import AutoProcessor as _AP
        _AP.from_pretrained(model_id)
        return model_id
    except Exception:
        pass
    home = _o.environ.get("HF_HOME") or _o.path.expanduser("~/.cache/huggingface")
    slug = "models--" + model_id.replace("/", "--")
    cands = []
    for d in _g.glob(_o.path.join(home, "hub", slug, "snapshots", "*")):
        has_proc = any(_o.path.exists(_o.path.join(d, f)) for f in ("processor_config.json", "preprocessor_config.json"))
        has_w = bool(_g.glob(_o.path.join(d, "*.safetensors")) or _g.glob(_o.path.join(d, "*.bin")))
        if has_proc and has_w:
            cands.append((_o.path.getmtime(d), d))
    if not cands:
        return model_id
    best = sorted(cands)[-1][1]
    print(f"[model] {model_id}: cache default revision is incomplete -> {best}")
    return best


def per_layer_table(model):
    """Gemma-4 (E4B) carries a SECOND embedding table, `embed_tokens_per_layer` (hidden_size_per_layer_input
    x n_layers per token), that resize_token_embeddings extends with RANDOM rows for the new ids. Gemma-3 and
    Qwen-3.5 have none. Returns that table's weight or None (2026-09-21, validate_gates Gate 3 on E4B)."""
    for getter in ("get_per_layer_input_embeddings",):
        for obj in (model, getattr(model, "model", None), getattr(getattr(model, "model", None), "language_model", None)):
            fn = getattr(obj, getter, None) if obj is not None else None
            if callable(fn):
                try:
                    t = fn()
                    if t is not None and hasattr(t, "weight"):
                        return t.weight
                except Exception:
                    pass
    return None


def per_layer_module(model):
    """The embed_tokens_per_layer MODULE (E4B) or None; per_layer_table() returns its weight."""
    for obj in (model, getattr(model, "model", None), getattr(getattr(model, "model", None), "language_model", None)):
        fn = getattr(obj, "get_per_layer_input_embeddings", None) if obj is not None else None
        if callable(fn):
            try:
                t = fn()
                if t is not None and hasattr(t, "weight"):
                    return t
            except Exception:
                pass
    return None


def train_per_layer_rows(model, tokenizer, name_to_ids, trainable_ids, specs_by_name, mode="word_mean", seed=0):
    """v1.157 (2026-09-24, author): make the NEW tokens' per-layer embedding rows (Gemma-4 E4B) trainable.

    Before this, resize_token_embeddings gave the new ids RANDOM per-layer rows (mean_resizing=False) that were never
    trained, so on E4B every word / prefix token fed noise into its per-layer input while its main row trained
    (identical rows at step 100 and 800 in every E4B word and prefix run). The table is ~262k x 10752, far too large to
    hand to AdamW, so only the new rows become a small Parameter P that the patched embed_tokens_per_layer lookup uses
    at new-token positions (same embed_scale); every other row is the frozen table. Init mirrors init_neologisms: the
    mean of the init word's per-layer rows + 0.01 N(0,1) (word_mean), else 0.02 N(0,1). Returns (P, sync); sync()
    writes P into the table so save_neologisms stores the trained rows and a reload (load_neologisms_by_name) restores
    them; (None, None) when the model has no per-layer table."""
    mod = per_layer_module(model)
    if mod is None:
        return None, None
    pl = mod.weight
    ids = [int(i) for i in trainable_ids]
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)
    rows = pl.data[ids].detach().clone().float()
    pos = {i: k for k, i in enumerate(ids)}
    with torch.no_grad():
        for name, nids in name_to_ids.items():
            for i in nids:
                if i not in pos:
                    continue
                if mode == "word_mean" and name in specs_by_name:
                    sub = tokenizer(specs_by_name[name].init_word, add_special_tokens=False)["input_ids"]
                    base = pl.data[sub].float().mean(0).cpu()
                    rows[pos[i]] = (base + 0.01 * torch.randn(pl.shape[1], generator=g)).to(rows.device)
                else:
                    rows[pos[i]] = (0.02 * torch.randn(pl.shape[1], generator=g)).to(rows.device)
    P = torch.nn.Parameter(rows.to(pl.device, pl.dtype), requires_grad=True)
    lut = torch.full((pl.shape[0],), -1, dtype=torch.long, device=pl.device)
    lut[torch.tensor(ids, device=pl.device)] = torch.arange(len(ids), device=pl.device)
    orig = mod.forward

    def forward(input_ids):
        out = orig(input_ids)
        k = lut[input_ids]
        m = k >= 0
        if not bool(m.any()):
            return out
        new = P[k.clamp(min=0)] * mod.embed_scale.to(P.dtype)
        return torch.where(m.unsqueeze(-1), new.to(out.dtype), out)
    mod.forward = forward

    def sync():
        with torch.no_grad():
            pl.data[ids] = P.data.to(pl.dtype)
    sync()
    print(f"[ple] training the per-layer rows of {len(ids)} new ids: P {tuple(P.shape)} {P.dtype}, "
          f"init={mode}, norm {P.data.float().norm(dim=-1).mean().item():.3f}")
    return P, sync


def resize_for_new_tokens(model, tokenizer, seed=0):
    """resize_token_embeddings with the torch RNG re-seeded immediately before it. The main-table rows are
    overwritten by init_neologisms / the checkpoint anyway; the per-layer rows (E4B) are NOT trained and were
    a fresh random draw at every load, so a reloaded E4B word was never exactly the trained one (Gate 3 delta
    up to .9 nats; Gemma-3/Qwen unaffected). Re-seeding makes the draw a function of the seed alone, in
    training and at evaluation. New checkpoints additionally store the per-layer rows (save_neologisms)."""
    torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    pl = per_layer_table(model)
    if pl is not None:
        print(f"[model] per-layer embedding table present {tuple(pl.shape)}: new rows drawn under seed {seed}")


def setup_neologisms(model, tokenizer, name_to_ids, all_new_ids, seed=0):
    """Resize, freeze, train only new rows. Returns the embedding weight."""
    resize_for_new_tokens(model, tokenizer, seed=seed)
    freeze_all(model)
    emb = model.get_input_embeddings()
    W = emb.weight
    W.requires_grad_(True)
    mask = torch.zeros(W.shape[0], 1, device=W.device, dtype=W.dtype)
    mask[all_new_ids] = 1.0

    def hook(grad):
        return grad * mask
    W.register_hook(hook)
    return W


def init_neologisms(model, tokenizer, name_to_ids, specs_by_name,
                    mode="word_mean", seed=0):
    """Init new rows from the mean embedding of an existing word's subtokens
    (skill-neologism style), or small random."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for name, ids in name_to_ids.items():
            if mode == "word_mean":
                word = specs_by_name[name].init_word
                sub = tokenizer(word, add_special_tokens=False)["input_ids"]
                base = emb[sub].mean(0)
                for i in ids:
                    emb[i] = base + 0.01 * torch.randn(emb.shape[1], generator=g).to(emb.device, emb.dtype)
            else:
                for i in ids:
                    emb[i] = 0.02 * torch.randn(emb.shape[1], generator=g).to(emb.device, emb.dtype)


def save_neologisms(W, all_new_ids, name_to_ids, path, per_layer=None):
    """per_layer: the model's per-layer embedding table (E4B) or None; its rows for the new ids are stored
    too, so a reload restores exactly what training used (2026-09-21)."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = {"rows": W.data[all_new_ids].detach().cpu(), "ids": list(all_new_ids), "name_to_ids": name_to_ids}
    if per_layer is not None:
        d["per_layer_rows"] = per_layer.data[all_new_ids].detach().cpu()
    torch.save(d, path)


def _resolve_neo_path(path):
    """Accept either the neo.pt file or its checkpoint DIRECTORY (…/ckpt/step_400)."""
    if os.path.isdir(path):
        path = os.path.join(path, "neo.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no neologism checkpoint at {path!r} -- pass either the step directory "
            f"or the neo.pt file inside it; did the gen run finish?")
    return path


def load_neologisms_by_name(model, path, name_to_ids, alias=None):
    """Copy saved rows into THIS model's ids for matching names (raw ids differ
    when a run adds a different token set, e.g. link runs add ctrl+gen).

    alias: {saved_name -> name_to_use}. The rows are the learned object; the name is
    only a dictionary key, so renaming lets a word trained under one task's vocabulary
    (ladder: look_ctrl/read_ctrl) be tested under another's (Ground-CF: src_look/src_read)
    WITHOUT retraining. That is the cross-dataset transfer test (2026-09-16)."""
    ckpt = torch.load(_resolve_neo_path(path), map_location="cpu")
    alias = alias or {}
    src = {}
    for name, ids in ckpt["name_to_ids"].items():
        id_to_row = {i: r for i, r in zip(ckpt["ids"], ckpt["rows"])}
        src[alias.get(name, name)] = [id_to_row[i] for i in ids]
    emb = model.get_input_embeddings().weight
    if src:
        _w = next(iter(src.values()))[0].shape[-1]
        if _w != emb.shape[-1]:
            raise SystemExit(f"[STOP] row width {_w} != embedding width {emb.shape[-1]}: "
                             f"these checkpoints are not transplantable between these models")
    loaded = []
    pl = per_layer_table(model)
    plsrc = {}
    if pl is not None and "per_layer_rows" in ckpt:
        id_to_pl = {i: r for i, r in zip(ckpt["ids"], ckpt["per_layer_rows"])}
        for name, ids in ckpt["name_to_ids"].items():
            plsrc[alias.get(name, name)] = [id_to_pl[i] for i in ids]
    with torch.no_grad():
        for name, rows in src.items():
            if name in name_to_ids and len(name_to_ids[name]) == len(rows):
                for i, r in zip(name_to_ids[name], rows):
                    emb[i] = r.to(emb.device, emb.dtype)
                if name in plsrc:
                    for i, r in zip(name_to_ids[name], plsrc[name]):
                        pl[i] = r.to(pl.device, pl.dtype)
                loaded.append(name)
    print(f"[neo] loaded by name: {loaded}" + ("" if pl is None else
          (" (+ stored per-layer rows)" if plsrc else " (per-layer rows re-drawn under the run seed; checkpoint predates storing them)")))
    return loaded


def load_neologisms(model, path):
    ckpt = torch.load(path, map_location="cpu")
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for row, i in zip(ckpt["rows"], ckpt["ids"]):
            emb[i] = row.to(emb.device, emb.dtype)
    return ckpt


# ---- LoRA -------------------------------------------------------------------
def setup_lora(model, r=16, alpha=32, dropout=0.05,
               targets=("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj")):
    """LoRA on the LANGUAGE MODEL ONLY. Suffix targets like "q_proj" match
    model-wide -- including SigLIP's attention, whose 4096-token maps then get
    stored for backward (~1GB/layer) and OOM a 48GB card at the first image.
    Resolving to exact language_model module names keeps the vision tower and
    projector frozen AND grad-free, which is also the fair baseline (same
    parameter locus as the neologism embeddings)."""
    import torch.nn as _nn
    from peft import LoraConfig, get_peft_model
    freeze_all(model)
    suff = tuple("." + t for t in targets)
    names = [n for n, m in model.named_modules()
             if isinstance(m, _nn.Linear) and "language_model" in n and n.endswith(suff)]
    assert names, "no language_model linear layers matched -- check model naming"
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=names, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    # RE-APPLY AFTER WRAPPING. M.load() sets use_cache=False and turns on
    # gradient checkpointing, but get_peft_model() returns a NEW outer module:
    # neither setting survives the rewrap, so the LoRA path silently trained with
    # checkpointing OFF and the KV cache ON. Measured consequence on MVR: memory
    # grew every step and hit 136.8 GiB of a 139.8 GiB H200 by step 114 -- an
    # accumulation, not a capacity limit. enable_input_require_grads() is
    # required for peft + checkpointing: with every base weight frozen the
    # checkpointed segments have no grad-requiring input and produce no graph.
    if os.environ.get("SC_GRAD_CKPT", "") == "1":
        try:
            model.config.use_cache = False
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            print("[lora] gradient checkpointing RE-ENABLED after peft wrap "
                  "(use_cache=False)")
        except Exception as e:
            print(f"[lora] WARNING: could not re-enable checkpointing: {e}")
    model.print_trainable_parameters()
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and ("vision" in n.lower() or "multi_modal_projector" in n.lower())]
    assert not bad, f"vision/projector params trainable: {bad[:3]}"
    print(f"[lora] adapters on {len(names)} language_model linears; vision tower grad-free")
    return model


# ---- prompt tuning (experimental, ctrl-comparable only) --------------------
class SoftPrompt(torch.nn.Module):
    def __init__(self, n_virtual, hidden, init=None):
        super().__init__()
        w = torch.randn(n_virtual, hidden) * 0.02 if init is None else init
        self.prompt = torch.nn.Parameter(w)

    def forward(self, inputs_embeds):
        b = inputs_embeds.shape[0]
        pre = self.prompt.unsqueeze(0).expand(b, -1, -1).to(inputs_embeds.dtype)
        return torch.cat([pre, inputs_embeds], dim=1)


# ---- generation helper: forbid ctrl tokens from being emitted --------------
def forbid_ids_processor(forbidden_ids):
    from transformers import LogitsProcessor

    class _Forbid(LogitsProcessor):
        def __init__(self, ids):
            self.ids = torch.tensor(ids, dtype=torch.long)

        def __call__(self, input_ids, scores):
            scores[:, self.ids] = float("-inf")
            return scores
    return _Forbid(forbidden_ids)
