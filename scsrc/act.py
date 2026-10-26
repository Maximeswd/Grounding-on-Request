"""Activation-level control (2026-10-19, author go): request-conditioned vectors added to the residual stream at ONE
decoder layer, on the span of the requested evidence -- the image soft tokens for <look>, the caption block for <read>.
A hybrid with the neologism: the word rows still carry the request in the prompt (they tell the hook which vector to
add and are trained as usual); delta acts where the evidence is encoded, i.e. it can re-encode the evidence instead of
only steering the readout. Inspired by the marker/content interventions of Hua, Yun & Pavlick (COLM 2026), who found
the strongest causal handle on modality attribution at the image-span positions in early/middle layers.

Install: a forward pre-hook on the input embedding captures the input ids of the current forward; a forward hook on
decoder layer `layer` adds delta[request] on the span. The request is read from the ids themselves (which word rows are
present), so the hook is batch- and order-safe for every builder and evaluator. Parameters: delta (2 x d), zero-init."""
import torch
import torch.nn as nn

SOURCES = ("look", "read")
STATE = {"ids": None, "found": 0, "missed": 0, "look_hits": 0, "read_hits": 0, "handles": None, "layer_path": None}


class ActControl(nn.Module):
    def __init__(self, d, n_src=2):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(n_src, d))


def decoder_layers(model):
    """(name, ModuleList) of the language model's decoder layers: the longest `*.layers` list outside the vision tower."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith("layers") and "visual" not in name and "vision" not in name:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError("[act] no decoder layer list found")
    return best


def image_token_id(model, tokenizer):
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int):
            return v
    for s in ("<image_soft_token>", "<|image_pad|>"):
        i = tokenizer.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id:
            return i
    raise RuntimeError("[act] image token id not found")


def _variants(tokenizer, s):
    out = []
    out.append(tokenizer(s, add_special_tokens=False)["input_ids"])
    nl = tokenizer("\n", add_special_tokens=False)["input_ids"]
    b = tokenizer("\n" + s, add_special_tokens=False)["input_ids"]
    if nl and b[:len(nl)] == nl:
        b = b[len(nl):]
    out.append(b)
    out.append(tokenizer(" " + s, add_special_tokens=False)["input_ids"])
    uniq = []
    for v in out:
        if v and v not in uniq:
            uniq.append(v)
    return uniq


def _find(seq, pat, start=0):
    n, m = len(seq), len(pat)
    for i in range(start, n - m + 1):
        if seq[i:i + m] == pat:
            return i
    return -1


def span_mask(row_ids, req):
    """Boolean mask over one sequence: image soft tokens for look; the caption block (after 'Caption:' up to the next
    block label or the image) for read. row_ids: 1-D LongTensor."""
    row = row_ids.tolist()
    if req == "look":
        return row_ids == STATE["img_id"]
    mask = torch.zeros_like(row_ids, dtype=torch.bool)
    s = -1
    for v in STATE["label_variants"]["Caption:"]:
        j = _find(row, v)
        if j >= 0:
            s = j + len(v)
            break
    if s < 0:
        return mask
    e = len(row)
    for lab in ("Image:", "Question:"):
        for v in STATE["label_variants"][lab]:
            j = _find(row, v, s)
            if j >= 0:
                e = min(e, j)
    for k in range(s, e):
        if row[k] == STATE["img_id"]:
            e = k
            break
    if e > s:
        mask[s:e] = True
    return mask


def install(model, ctrl, layer, name_to_ids, tokenizer, img_id, labels=("Caption:", "Image:", "Question:")):
    req_of_id = {}
    for nm, ids in name_to_ids.items():
        r = "look" if "look" in nm else ("read" if "read" in nm else None)
        if r:
            for i in ids:
                req_of_id[int(i)] = r
    STATE.update({"req_of_id": req_of_id, "img_id": int(img_id), "layer": int(layer),
                  "label_variants": {l: _variants(tokenizer, l) for l in labels}})
    if STATE.get("handles"):
        for h in STATE["handles"]:
            h.remove()
    emb = model.get_input_embeddings()

    def _capture(module, inputs):
        x = inputs[0] if inputs else None
        STATE["ids"] = x.detach() if torch.is_tensor(x) and x.dim() == 2 else None

    h1 = emb.register_forward_pre_hook(_capture)
    name, layers = decoder_layers(model)
    if not (0 <= layer < len(layers)):
        raise RuntimeError(f"[act] layer {layer} out of range (0..{len(layers) - 1})")

    def hook(module, inputs, output):
        ids = STATE.get("ids")
        hs = output[0] if isinstance(output, tuple) else output
        if ids is None or hs.shape[:2] != ids.shape:
            return None
        out, changed = hs, False
        for b in range(ids.shape[0]):
            req = next((STATE["req_of_id"][t] for t in ids[b].tolist() if t in STATE["req_of_id"]), None)
            if req is None:
                if ids.shape[1] > 8:
                    STATE["missed"] += 1
                continue
            mask = span_mask(ids[b], req)
            if not bool(mask.any()):
                STATE["missed"] += 1
                continue
            STATE["found"] += 1
            STATE[f"{req}_hits"] += 1
            if not changed:
                out = hs.clone()
                changed = True
            out[b] = out[b] + mask.unsqueeze(-1).to(hs.dtype) * ctrl.delta[SOURCES.index(req)].to(hs.dtype)
        if not changed:
            return None
        return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out

    h2 = layers[layer].register_forward_hook(hook)
    STATE["handles"] = (h1, h2)
    STATE["layer_path"] = f"{name}[{layer}] of {len(layers)}"
    return STATE["handles"]


def save(ctrl, path, layer, span="content"):
    torch.save({"delta": ctrl.delta.detach().cpu(), "layer": int(layer), "span": span}, path)


def load(path, device):
    d = torch.load(path, map_location="cpu")
    ctrl = ActControl(d["delta"].shape[1], d["delta"].shape[0])
    ctrl.delta.data.copy_(d["delta"])
    return ctrl.to(device), d


def diag():
    return {"span_found": STATE["found"], "span_missed": STATE["missed"], "look_hits": STATE["look_hits"],
            "read_hits": STATE["read_hits"], "layer_path": STATE.get("layer_path")}
