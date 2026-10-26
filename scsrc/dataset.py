"""Torch Dataset producing ready-to-train examples for a given method/arm."""
import copy, os
from torch.utils.data import Dataset
from .data import get_image
from .collate import build_train_example
from .prompts import gen_template, pt_template


class TrainDataset(Dataset):
    def __init__(self, hf_ds, specs, processor, method, arm, n_tokens, device, task="color", absence_frac=0.0, cap_first_prob=1.0, slot_negative_frac=0.0, mask_ctrl_lead=False):
        self.ds = hf_ds
        self.processor = processor
        self.method, self.arm = method, arm
        self.n = n_tokens
        self.device = device
        self.task = task
        self.absence_frac = float(absence_frac)
        self.cap_first_prob = float(cap_first_prob)
        self.slot_negative_frac = float(slot_negative_frac)
        import random as _random
        self._absrng = _random.Random(1234)
        # ctrl (and the pt baseline) expand each spec into 3 operator examples
        if (method == "neo" and arm in ("ctrl", "link")) or method == "pt":
            self.items = []
            for s in specs:
                for op in ("look", "read", "compare"):
                    s2 = copy.copy(s)
                    s2.op = op
                    self.items.append(s2)
        else:
            self.items = list(specs)
        self.mask_ctrl_lead = mask_ctrl_lead

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.items[i]
        img = get_image(self.ds, s)
        _cf = (self._absrng.random() < self.cap_first_prob) if self.cap_first_prob < 1.0 else True
        _neg = (self._absrng.random() < self.slot_negative_frac) if self.slot_negative_frac > 0 else False
        return build_train_example(self.processor, s, img, self.method,
                                   self.arm, self.n, self.device, task=self.task,
                                   cap_first=_cf, slot_negative=_neg,
                                   mask_ctrl_lead=self.mask_ctrl_lead)


class ObjectiveDataset(Dataset):
    """Yields {'op':..., 'sft':..., 'pref':... , 'vgm':...}. Builds pref/vgm for
    an op only if that op's resolved coefficient is > 0 (so nothing is wasted)."""
    def __init__(self, hf_ds, specs, processor, method, arm, n_tokens, device,
                 obj_global, per_op=None, task="color", ctrl_style="neutral",
                 absence_frac=0.0, cap_first_prob=1.0, slot_negative_frac=0.0,
                 ops=None, ctrl_site="user", lookread_conflict_only=False,
                 disentangle_frac=0.0, mask_ctrl_lead=False):
        from .preference import (build_pref_pair, build_vgm_pair, build_tc_pair,
                                 build_ctm_pair, build_sm_pair, resolve_obj)
        from .prompts import pick_ctrl_template
        self.mask_ctrl_lead = mask_ctrl_lead
        self._pref_fn, self._vgm_fn, self._resolve = build_pref_pair, build_vgm_pair, resolve_obj
        self._tc_fn = build_tc_pair
        self._ctm_fn = build_ctm_pair
        self._sm_fn = build_sm_pair
        self.ctrl_style = ctrl_style
        self.ctrl_site = ctrl_site
        self._pick_ctrl = pick_ctrl_template
        self.ds, self.processor = hf_ds, processor
        self.method, self.arm, self.n, self.device = method, arm, n_tokens, device
        self.obj_global, self.per_op = obj_global, per_op
        self.task = task
        self.absence_frac = float(absence_frac)
        self.cap_first_prob = float(cap_first_prob)
        self.slot_negative_frac = float(slot_negative_frac)
        import random as _random
        self._absrng = _random.Random(1234)
        # VCF prior-disentangling: with prob f rewrite a conflict caption's
        # colour to a THIRD colour (!= shown, != asserted) so <read> must READ
        # (prior-echo fails) and <look> cannot be anti-prior. Training-side
        # augmentation because the row schema binds caption_i to image_i.
        self.disentangle_frac = float(disentangle_frac)
        self._dis_palette = ["red", "orange", "yellow", "green", "blue",
                             "purple", "pink", "brown", "black", "white", "gray"]
        if (method == "neo" and arm in ("ctrl", "link")) or method == "pt" \
                or (method == "lora" and ctrl_style in ("compose", "request", "adapter")):
            self.items = []
            _both = os.environ.get("SC_COMPOSE_LEGACY_ORDER", "0") != "1"   # default: both orders
            if method == "lora" and ctrl_style == "adapter":
                # v1.125: one adapter per request -- only this adapter's request is trained
                _ts = [x for x in os.environ.get("SC_TRAIN_SOURCES", "").split(",") if x.strip()]
                assert _ts and all(x in ("look", "read") for x in _ts), "ctrl_style=adapter needs --train_sources look|read"
                ops = tuple(_ts)
            elif method == "lora" and ctrl_style == "request":
                # request-conditioned LoRA (2026-09-21): one item per selected source, like pt
                ops = ("look", "read")
            elif method == "lora":
                # LoRA keeps its own recipe (op None -> the fixed separated format)
                # and ADDS the composed items, so every interface sees the same
                # composed supervision on top of its own.
                self.items += [(s, None) for s in specs]
                ops = ("compose", "compose_r") if _both else ("compose",)
            # ops override: --extend_ckpt runs train ONLY the added word's op
            # (e.g. ("resolve",)) -- the frozen rows' items would waste compute
            for s in specs:
                # 2026-09-16: ctrl_style="compose" gave the COMPOSED one-prompt format to LoRA ONLY,
                # because `ops = ("compose",)` sits inside the `if method == "lora"` block above. The
                # neologisms and the learned prefixes were therefore SCORED by the composed conflict
                # detector in a format they had never been TRAINED in, while LoRA had been trained in
                # exactly it -- so the one-prompt comparison measured the recipe, not the interface.
                # Every interface now gets its own ops PLUS the composed item, which is what the
                # comment above has always claimed. Only ctrl_style="compose" runs are affected.
                for op in (ops or (("look", "read", "compare")
                                   + (("compose",) if ctrl_style == "compose" else ())
                                   + (("compose_r",) if (ctrl_style == "compose" and _both) else ()))):
                    # agree items give look and read IDENTICAL golds -> zero
                    # discriminating gradient that actively pulls the two
                    # tokens together; optionally restrict them to compare
                    if lookread_conflict_only and op in ("look", "read")                             and s.cond != "conflict":
                        continue
                    s2 = copy.copy(s); s2.op = op
                    self.items.append((s2, op))
        else:
            self.items = [(s, None) for s in specs]
        self._pair_coverage_check(specs)

    def _pair_coverage_check(self, specs):
        """HARD-FAIL when a requested pair objective cannot be built for this
        (arm, style) -- the builders return None for unsupported combinations,
        which otherwise trains a run that is silently plain SFT while its name
        claims an objective. This exact class of bug cost the APO axis once
        (lam_apo read rec['pref'], datasets wrote rec['apo']) and would recur
        for e.g. arm=gen + lam_ctm (build_ctm_pair has no gen branch).
        Probe up to 16 conflict specs x the trained ops; legitimate per-item
        Nones (agree items, wrong op) pass as long as ANY pair builds."""
        if self.method not in ("neo", "pt", "lora"):   # lora added 2026-09-14: LoRA + CTM
            return
        eff = self._resolve(self.obj_global, self.per_op, "look")
        want = {"apo": (eff.get("lam_apo", 0) or 0) > 0,
                "ctm": (eff.get("lam_ctm", 0) or 0) > 0,
                "sm": (eff.get("lam_sm", 0) or 0) > 0,
                "tc": (eff.get("lam_tc", 0) or 0) > 0,
                "vgm": (eff.get("lam_vgm", 0) or 0) > 0}
        if not any(want.values()):
            return
        fns = {"apo": self._pref_fn, "ctm": self._ctm_fn, "sm": self._sm_fn,
               "tc": self._tc_fn, "vgm": self._vgm_fn}
        probe = [s for s in specs if getattr(s, "cond", None) == "conflict"][:16] or list(specs)[:16]
        for name, on in want.items():
            if not on:
                continue
            built = 0
            for s in probe:
                for op in ("look", "read"):
                    try:
                        kw = dict(ctrl_style=self.ctrl_style, task=self.task)
                        if name in ("apo", "ctm", "sm"):
                            kw["method"] = self.method
                        if fns[name](self.processor, self.ds, s, self.arm, op,
                                     self.n, "cpu", **kw) is not None:
                            built += 1
                    except Exception:
                        pass
                if built:
                    break
            if not built:
                raise SystemExit(
                    f"[dataset] lam_{name} > 0 but the {name} builder returned None "
                    f"for ALL of {len(probe)} probe specs (arm={self.arm}, "
                    f"ctrl_style={self.ctrl_style}, task={self.task}). This run "
                    f"would train as plain SFT while claiming +{name}. Either the "
                    f"combination is unsupported (e.g. arm=gen has no ctm branch) "
                    f"or the data has no conflict items. Refusing to start.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, op = self.items[i]
        eff = self._resolve(self.obj_global, self.per_op, op)
        if self.method == "neo" and self.arm in ("ctrl", "link"):
            tpl = self._pick_ctrl(i)
        elif self.method == "neo" and self.arm == "gen":
            tpl = gen_template(self.task, i)
        elif self.method == "pt":
            # inserted-PT control: the WORD's rotating template at the SAME index as a neologism item,
            # so SFT and pair prompts differ from the neologism's in nothing but the token ids (2026-09-21)
            tpl = self._pick_ctrl(i) if os.environ.get("PT_INSERTED", "0") == "1" else pt_template(i)
        elif self.method == "lora" and self.ctrl_style == "adapter":
            tpl = pt_template(i)          # v1.125: the prefix's rotating neutral templates, prefix removed
        else:
            tpl = None
        absence = None
        if (self.absence_frac > 0 and op in ("look", "read")
                and self.method in ("neo", "pt") and self.arm in ("ctrl", "na")
                and self._absrng.random() < self.absence_frac):
            absence = "image" if op == "look" else "caption"
        if (self.disentangle_frac > 0 and op in ("look", "read")
                and s.cond == "conflict" and absence is None
                and self._absrng.random() < self.disentangle_frac):
            old = next(iter(s.text_key.values()), None)
            shown = next(iter(s.image_key.values()), None)
            third = [c for c in self._dis_palette
                     if c not in (str(old).lower(), str(shown).lower())]
            if old is not None and str(old).lower() in s.fed_caption.lower() and third:
                nv = self._absrng.choice(third)
                s = copy.copy(s)
                import re as _re
                s.fed_caption = _re.sub(_re.escape(str(old)), nv, s.fed_caption,
                                        count=1, flags=_re.IGNORECASE)
                s.text_key = {k: nv for k in s.text_key}
        if absence == "caption":
            s = copy.copy(s); s.fed_caption = "(no caption provided)"
        img = None if absence == "image" else get_image(self.ds, s)
        cap_first = (i % 2 == 0)   # rotate caption position: channel != position
        if self.cap_first_prob < 1.0:
            cap_first = self._absrng.random() < self.cap_first_prob
        _neg = (self.slot_negative_frac > 0 and absence is None
                and self._absrng.random() < self.slot_negative_frac)
        rec = {"op": op,
               "sft": build_train_example(self.processor, s, img, self.method,
                                          self.arm if self.method == "neo" else "na",
                                          self.n, self.device, template=tpl, task=self.task,
                                          cap_first=cap_first, ctrl_style=self.ctrl_style,
                                          mask_ctrl_lead=self.mask_ctrl_lead,
                                          absence=absence, slot_negative=_neg,
                                          ctrl_site=self.ctrl_site)}
        if op in ("compose", "compose_r"):
            # 2026-09-15: composed items WERE CE-only, which meant the counterfactual objective never
            # touched the format the one-prompt conflict detector is scored in -- so CTM could not help
            # there by construction. With lam_ctm on, the composed prompt now gets its own twin.
            if self.method in ("neo", "pt", "lora") and eff.get("lam_ctm", 0) > 0:
                from .preference import build_compose_ctm_pair
                _both = os.environ.get("SC_COMPOSE_LEGACY_ORDER", "0") != "1"
                _order = (("read", "look") if op == "compose_r" else (("look", "read") if _both else None))
                # WITHIN every spec the two orders get COMPLEMENTARY twins: one order edits its first
                # slot (composed Flip), the other its second (composed Stay); which is which alternates
                # across specs by ex_index, so over the data each (order, slot) is exactly a quarter.
                _slot = ((int(getattr(s, "ex_index", i)) + (op == "compose_r")) % 2) if _both else None
                _c = build_compose_ctm_pair(self.processor, self.ds, s, self.n, self.device,
                                            task=self.task, cap_first=cap_first, method=self.method,
                                            srcs=_order, edit_slot=_slot)
                if _c is not None:
                    rec["ctm"] = _c
            return rec
        if _neg:
            return rec   # slot-negative items are SFT-only (no pair terms for a non-operator)
        if absence is not None:
            return rec   # absence items are SFT-only: no pair terms without both sources
        # LoRA items carry op=None (their SFT prompt is op-less by design, and stays so); the pair
        # terms need an operator, so alternate look/read on the item index. neo/pt: pop == op.
        pop = op if op is not None else ("look", "read")[i % 2]
        if self.method in ("neo", "pt", "lora") and eff.get("lam_apo", 0) > 0:
            rec["pref"] = self._pref_fn(self.processor, self.ds, s, self.arm, pop, self.n,
                                        self.device, template=tpl, ctrl_style=self.ctrl_style,
                                        task=self.task, method=self.method)
        if self.method in ("neo", "pt") and eff.get("lam_vgm", 0) > 0:
            rec["vgm"] = self._vgm_fn(self.processor, self.ds, s, self.arm, pop, self.n,
                                      self.device, template=tpl, ctrl_style=self.ctrl_style, task=self.task)
        if self.method in ("neo", "pt") and eff.get("lam_tc", 0) > 0:
            rec["tc"] = self._tc_fn(self.processor, self.ds, s, self.arm, pop, self.n,
                                    self.device, template=tpl, ctrl_style=self.ctrl_style, task=self.task)
        if self.method in ("neo", "pt", "lora") and eff.get("lam_ctm", 0) > 0:
            rec["ctm"] = self._ctm_fn(self.processor, self.ds, s, self.arm, pop, self.n,
                                      self.device, template=tpl, ctrl_style=self.ctrl_style,
                                      task=self.task, method=self.method)
        if self.method in ("neo", "pt", "lora") and eff.get("lam_sm", 0) > 0:
            rec["sm"] = self._sm_fn(self.processor, self.ds, s, self.arm, pop, self.n,
                                    self.device, template=tpl, ctrl_style=self.ctrl_style,
                                    task=self.task, method=self.method)
        return rec
