"""Interchange-intervention instrument (Track B, rows-only IIT).

The unit of analysis: the CONTEXTUAL hidden state of a trained word at a
decoder layer L -- not the static row. cache_states runs one plain forward
with capture hooks (all requested layers at once); patch() is a context
manager that replaces the state(s) at the word's position(s) during the
PREFILL forward only. Decode steps have seq_len 1 and are skipped, so the
patched activation persists through generation via the KV cache of layers
above L (standard activation-patching semantics: one site, one layer).

House sanity anchors (a new instrument class pays its debugging tax before
any number is believed): self-swap must reproduce the base output at ~1.00;
a norm-matched random vector must CHANGE the output on most items (else the
position is dead and downstream never reads it).
"""
import torch


def lm_layers(model):
    """Locate the decoder layer ModuleList across HF layout variants."""
    for path in ("model.language_model.layers", "language_model.model.layers",
                 "language_model.layers", "model.layers"):
        obj, ok = model, True
        for p in path.split("."):
            if not hasattr(obj, p):
                ok = False
                break
            obj = getattr(obj, p)
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj
    raise AttributeError("cannot locate decoder layer list on this model")


def token_positions(input_ids, token_ids):
    """All positions of the word's id(s) in a [1, T] input_ids tensor."""
    want = set(int(t) for t in token_ids)
    return [i for i, t in enumerate(input_ids[0].tolist()) if int(t) in want]


@torch.no_grad()
def cache_states(model, enc, layer_idxs, positions):
    """One forward; capture layer-output states at `positions` for every
    layer in layer_idxs. Returns {layer_idx: tensor [k, d]} (detached)."""
    layers = lm_layers(model)
    out, handles = {}, []

    def mk(li):
        def fn(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            out[li] = hs[0, positions, :].detach().clone()
        return fn

    for li in layer_idxs:
        handles.append(layers[li].register_forward_hook(mk(li)))
    try:
        kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
        model(**kw, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return out


class patch:
    """with patch(model, L, positions, states): ... generate/forward ...

    Replaces layer-L output rows at `positions` with `states` on every
    forward whose sequence length exceeds max(positions) -- i.e. the
    prefill; single-token decode steps pass through untouched."""

    def __init__(self, model, layer_idx, positions, states):
        self.layers = lm_layers(model)
        self.li, self.pos, self.st = layer_idx, positions, states
        self._h = None

    def __enter__(self):
        pos, st = self.pos, self.st

        def fn(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] > max(pos):
                hs[0, pos, :] = st.to(device=hs.device, dtype=hs.dtype)
            return output

        self._h = self.layers[self.li].register_forward_hook(fn)
        return self

    def __exit__(self, *exc):
        self._h.remove()
        return False


class patch_train:
    """Grad-preserving patch for IIT TRAINING (the eval `patch` mutates in
    place under no_grad; training must NOT). Returns a NEW hidden tensor with
    the rows replaced -- `states` stays in the autograd graph (NO detach; a
    detached donor makes the IIT loss inert w.r.t. the row: the base state is
    overwritten and everything downstream is frozen). Also captures the
    PRE-replacement rows in .captured (in-graph), so one patched forward
    yields both the counterfactual loss and this side's donor state for the
    symmetric swap. states=None -> capture-only."""

    def __init__(self, model, layer_idx, positions, states=None):
        self.layers = lm_layers(model)
        self.li, self.pos, self.st = layer_idx, positions, states
        self.captured = None
        self._h = None

    def __enter__(self):
        def fn(module, args, output):
            tup = isinstance(output, tuple)
            hs = output[0] if tup else output
            if hs.shape[1] <= max(self.pos):
                return output
            self.captured = hs[0, self.pos, :]
            if self.st is None:
                return output
            new = hs.clone()
            new[0, self.pos, :] = self.st.to(dtype=new.dtype)
            return (new,) + tuple(output[1:]) if tup else new

        self._h = self.layers[self.li].register_forward_hook(fn)
        return self

    def __exit__(self, *exc):
        self._h.remove()
        return False


def norm_matched_random(states, seed=0):
    """Random direction, per-row norm matched to `states` -- the dead-position
    control: if THIS doesn't change the output, nothing read the site."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(states.shape, generator=g).to(states.device, states.dtype)
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return r * states.norm(dim=-1, keepdim=True)
