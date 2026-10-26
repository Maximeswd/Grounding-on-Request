"""IIT training: symmetric graph-preserving interchange on exact twins.

Pairs come from conflict specs, whose twin structure makes the counterfactual
targets exactly the spec's OTHER key -- no textual surgery:

  look  base=(i1, c2) target image_key | donor=(i2, c2) target text_key
        (the twin image i2 carries the caption's value by construction)
  read  base=(i1, c2) target text_key  | donor=(i1, c1) target image_key
        (the matching caption c1 states the image's value)

Each pair contributes four CEs (all rendered by build_train_example, so
targets match the SFT arm byte-for-byte):

  L = (obs_b + obs_d)/2 + lam_iit * (cf_bd + cf_db)/2

cf_bd = base forward with the donor's layer-L word state transplanted
(grad-preserving, iit.patch_train), target = donor's value; cf_db symmetric.
Backward is staged (obs_b alone, then the entangled three) so peak memory
holds three graphs -- the same envelope as the VGM arm on this hardware.
Per-token normalization is HF's own mean-over-supervised-tokens loss.
"""
from types import SimpleNamespace

from . import iit
from .collate import build_train_example, collate  # noqa: F401 (collate re-export)
from .tokens import trigger


def _clone(sp, **over):
    d = dict(fed_caption=sp.fed_caption, image_key=sp.image_key,
             text_key=sp.text_key, relation=sp.relation,
             ex_index=sp.ex_index, img_idx=sp.img_idx, cond=sp.cond,
             task=getattr(sp, "task", None))
    d.update(over)
    return SimpleNamespace(**d)


class IITPairDataset:
    """Yields {"op", "iit": {base, donor, base_cf, donor_cf, pos_b, pos_d}}.
    Ops alternate look/read so both directions train. bs=1 + accumulation,
    same as every other arm."""

    def __init__(self, ds, specs, processor, n_tokens, device, task,
                 ctrl_style="neutral"):
        self.ds = ds
        self.specs = []
        for sp in specs:
            if sp.cond != "conflict":
                continue
            row = ds[sp.ex_index]
            if row.get(f"image_{sp.img_idx}") is None or \
                    row.get(f"image_{3 - sp.img_idx}") is None:
                continue
            self.specs.append(sp)
        if not self.specs:
            raise ValueError("IITPairDataset: no twin-complete conflict specs")
        self.processor, self.n, self.device = processor, n_tokens, device
        self.task, self.style = task, ctrl_style
        self.ids = {op: processor.tokenizer.convert_tokens_to_ids(
            [t for t in _tok_strings(op, n_tokens)]) for op in ("look", "read")}

    def __len__(self):
        return 2 * len(self.specs)

    def _build(self, sp, image):
        return build_train_example(self.processor, sp, image, "neo", "ctrl",
                                   self.n, self.device, task=self.task,
                                   ctrl_style=self.style)

    def __getitem__(self, i):
        sp = self.specs[i % len(self.specs)]
        op = ("look", "read")[i % 2]
        row = self.ds[sp.ex_index]
        i1 = row.get(f"image_{sp.img_idx}")
        i2 = row.get(f"image_{3 - sp.img_idx}")
        c1 = row.get(f"caption_{sp.img_idx}", sp.fed_caption)
        if op == "look":
            base = self._build(_clone(sp, op=op), i1)
            donor = self._build(_clone(sp, op=op, image_key=sp.text_key), i2)
            base_cf = self._build(_clone(sp, op=op, image_key=sp.text_key), i1)
            donor_cf = self._build(_clone(sp, op=op), i2)
        else:
            base = self._build(_clone(sp, op=op), i1)
            donor = self._build(
                _clone(sp, op=op, fed_caption=c1, text_key=sp.image_key), i1)
            base_cf = self._build(_clone(sp, op=op, text_key=sp.image_key), i1)
            donor_cf = self._build(_clone(sp, op=op, fed_caption=c1), i1)
        pos_b = iit.token_positions(base["input_ids"], self.ids[op])
        pos_d = iit.token_positions(donor["input_ids"], self.ids[op])
        return {"op": op, "iit": {"base": base, "donor": donor,
                                  "base_cf": base_cf, "donor_cf": donor_cf,
                                  "pos_b": pos_b, "pos_d": pos_d}}


def _tok_strings(op, n):
    from .tokens import token_strings
    return token_strings(f"{op}_ctrl", n)


def _feed(enc):
    kw = {k: v for k, v in enc.items()
          if k not in ("token_type_ids", "labels")}
    return kw, enc["labels"]


def iit_step(model, rec, lam_iit, layer, accum):
    """Four forwards, staged backward. Returns (None, comps) -- backward
    already done (train.py convention: skip its own backward on non-tensor)."""
    d = rec["iit"]
    pos_b, pos_d = d["pos_b"], d["pos_d"]
    if not pos_b or not pos_d:
        return None, {}
    # 1) base observational: independent graph, free it immediately
    kw, lab = _feed(d["base"])
    ce_b = model(**kw, labels=lab).loss
    ((0.5 * ce_b) / accum).backward()
    ce_b_f = float(ce_b.detach())
    # 2) donor observational, capturing the donor word state IN-GRAPH
    cap_d = iit.patch_train(model, layer, pos_d, states=None)
    kw, lab = _feed(d["donor"])
    with cap_d:
        ce_d = model(**kw, labels=lab).loss
    h_d = cap_d.captured
    # 3) base counterfactual: donor state in, base word state captured out
    swap_b = iit.patch_train(model, layer, pos_b, states=h_d)
    kw, lab = _feed(d["base_cf"])
    with swap_b:
        ce_bd = model(**kw, labels=lab).loss
    h_b = swap_b.captured
    # 4) donor counterfactual: base state in
    kw, lab = _feed(d["donor_cf"])
    with iit.patch_train(model, layer, pos_d, states=h_b):
        ce_db = model(**kw, labels=lab).loss
    total = 0.5 * ce_d + lam_iit * 0.5 * (ce_bd + ce_db)
    (total / accum).backward()
    comps = {"ce": ce_b_f + float(ce_d.detach()),
             "iit": float(ce_bd.detach()) + float(ce_db.detach())}
    return None, comps
