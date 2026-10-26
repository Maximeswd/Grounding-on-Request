"""Attention-head output scaling: the inference-time control baseline of Hua et al. (2025),
"How Do Vision-Language Models Process Conflicting Information Across Modalities?" (arXiv 2507.01790,
code MIT-licensed at github.com/ethahtz/vlm_conflicting_info_processing).

THE INTERVENTION. Each attention head's contribution to a layer's output is the slice of the input to
o_proj that belongs to that head. Multiplying that slice by a scalar alpha amplifies (alpha > 1) or
suppresses (alpha < 1) the head while leaving every other component untouched -- their `alpha`,
`layer_idx`, `head_idx` arguments. They report two useful head types:
  router head      modality-AGNOSTIC: amplifying it promotes whatever modality the instruction names
  promotion head   modality-SPECIFIC: amplifying it promotes one modality whatever the instruction says

POSITION. Their evaluation predicts a single next token, so they scale at the last prompt position only.
Our batteries score a multi-token continuation by likelihood, where "the last position" would touch only
the final continuation token. We therefore scale at EVERY position by default, which is the faithful
generalisation for a deployed intervention and keeps the head search and the evaluation matched;
last_token_only=True reproduces their exact placement for a single-token probe.

GQA. o_proj consumes num_attention_heads * head_dim values (query heads), so a query-head index selects
the right slice even when key/value heads are shared.
"""
import contextlib

import torch


# 2026-09-16: the original helper hard-coded self_attn.o_proj, which is Gemma/Llama naming. Both Qwen
# models crashed here with "no decoder layers with self_attn.o_proj found" (job 652381 tasks 2 and 3,
# dead in 15 s), so the Hua et al. baseline silently covered only two of the four models. These are the
# attention-block and output-projection names actually in use across the families we run.
_ATTN_NAMES = ("self_attn", "attn", "self_attention", "attention")
_PROJ_NAMES = ("o_proj", "out_proj", "proj", "c_proj", "dense", "wo")


def attn_out_proj(layer):
    """(attention module, output projection) for one decoder layer, or (None, None)."""
    for an in _ATTN_NAMES:
        attn = getattr(layer, an, None)
        if attn is None:
            continue
        for pn in _PROJ_NAMES:
            proj = getattr(attn, pn, None)
            if isinstance(proj, torch.nn.Linear):
                return attn, proj
    return None, None


def decoder_layers(model):
    """The language tower's decoder layers: the longest ModuleList whose entries carry an attention
    block with a Linear output projection, under any of the names the families we run actually use."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            if attn_out_proj(mod[0])[1] is not None:
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
    if best is None:
        have = sorted({n.split(".")[-1] for n, _ in model.named_modules()})[:40]
        raise RuntimeError(
            "no decoder layers with a recognised attention output projection. "
            f"tried {_ATTN_NAMES} x {_PROJ_NAMES}; leaf module names present include {have}")
    return best[1]


def head_dim_of(layer):
    attn, proj = attn_out_proj(layer)
    if attn is None:
        attn = layer.self_attn
    hd = getattr(attn, "head_dim", None)
    if hd:
        return int(hd)
    nh = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None)
    if not nh:
        raise RuntimeError("cannot determine head_dim")
    return int(proj.in_features // nh)


def geometry(model):
    """(n_layers, n_query_heads, head_dim) of the language tower."""
    layers = decoder_layers(model)
    hd = head_dim_of(layers[0])
    return len(layers), int(attn_out_proj(layers[0])[1].in_features // hd), hd


@contextlib.contextmanager
def scaled_heads(model, spec, last_token_only=False):
    """spec: iterable of (layer_idx, head_idx, alpha). No-op for an empty spec."""
    spec = [tuple(x) for x in (spec or [])]
    if not spec:
        yield
        return
    layers = decoder_layers(model)
    by_layer = {}
    for l, h, a in spec:
        by_layer.setdefault(int(l), []).append((int(h), float(a)))
    handles = []

    def make(heads, hd):
        def pre(module, args):
            if not args or not torch.is_tensor(args[0]) or args[0].dim() != 3:
                return None
            x = args[0].clone()
            for h, a in heads:
                sl = slice(h * hd, (h + 1) * hd)
                if last_token_only:
                    x[:, -1, sl] = x[:, -1, sl] * a
                else:
                    x[..., sl] = x[..., sl] * a
            return (x,) + tuple(args[1:])
        return pre

    try:
        for l, heads in by_layer.items():
            layer = layers[l]
            handles.append(attn_out_proj(layer)[1].register_forward_pre_hook(
                make(heads, head_dim_of(layer))))
        yield
    finally:
        for h in handles:
            h.remove()


def parse_spec(s):
    """'11:14:10' or '11:14:10,20:16:10' -> [(11, 14, 10.0), ...]; '' -> []."""
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        l, h, a = part.split(":")
        out.append((int(l), int(h), float(a)))
    return out
