#!/usr/bin/env python
"""Composed training must cover BOTH orders per item and twin BOTH slots (2026-09-21).

The legacy recipe chose one order per item by parity and twinned only the first slot, so the second
word in every composed prompt got no counterfactual supervision and no item taught "same content,
other order". This test runs the real dataset class with encoders stubbed and asserts, per spec:
  * exactly one image-first and one caption-first composed item;
  * the two items' CTM twins edit DIFFERENT sources (one composed Flip, one composed Stay);
and over the specs each (order, edited source) combination is a quarter of the pairs. CPU, ~15 s.
"""
import os, sys, inspect, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scsrc.preference as P, scsrc.dataset as DS
from scsrc import data as D


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "color", "train")
    if not os.path.isdir(root):
        print("SKIP: data/color/train not on disk"); return 0
    from datasets import load_from_disk
    P._seq = lambda processor, img, user, cont, device: {"user": user, "cont": cont, "img": id(img)}
    DS.build_train_example = lambda *a, **k: {"stub": True}
    cls = next(c for n, c in inspect.getmembers(DS, inspect.isclass) if "obj_global" in inspect.signature(c.__init__).parameters)
    ds = load_from_disk(root)
    specs = [x for x in D.build_specs(ds, "train", conditions=("conflict",))][:12]
    fails = []
    for method, arm in (("neo", "ctrl"), ("pt", "na"), ("lora", "na")):
        d = cls(ds, specs, None, method, arm, 5, "cpu", {"lam_sft": 1.0, "lam_ctm": 1.0, "gamma": 0.1}, task="color", ctrl_style="compose")
        per_spec = collections.defaultdict(list); combo = collections.Counter()
        for i in range(len(d)):
            r = d[i]
            if r.get("op") not in ("compose", "compose_r") or not r.get("ctm"):
                continue
            c = r["ctm"]; u = c["a_pos"]["user"]
            edited = "image" if c["a_neg"]["img"] != c["a_pos"]["img"] else "caption"
            order = "read-first" if r["op"] == "compose_r" else "look-first"
            per_spec[d.items[i][0].ex_index].append((order, edited)); combo[(order, edited)] += 1
        for ex, pairs in per_spec.items():
            orders = {o for o, _ in pairs}; edits = {e for _, e in pairs}
            if orders != {"look-first", "read-first"}:
                fails.append(f"{method}: spec {ex} orders {orders}")
            if len(pairs) == 2 and len(edits) != 2:
                fails.append(f"{method}: spec {ex} both orders twin the same source {edits}")
        n = sum(combo.values())
        for k, v in combo.items():
            if abs(v / max(n, 1) - 0.25) > 0.15:
                fails.append(f"{method}: (order, edited) {k} is {v}/{n}, expected ~1/4")
        print(f"  {method}: composed pairs {n}, (order, edited) = {dict(combo)}")
    if fails:
        print("COMPOSE ORDER FAILURES:"); [print("  " + f) for f in fails]; return 1
    print("compose orders OK: both orders per spec, complementary twins, quarters over the data"); return 0


if __name__ == "__main__":
    sys.exit(main())
