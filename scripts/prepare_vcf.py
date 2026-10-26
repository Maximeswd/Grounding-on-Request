#!/usr/bin/env python3
"""Visual-CounterFact as the ATOMIC rung of the modality-conflict ladder.

VCF gives the SAME object in two colours (original_image = prior colour,
counterfact_image = edited colour). That is a colour minimal pair, so we
attach synthetic captions and build the identical look-vs-READ conflict used
by ColorSwap -- the axis is modality everywhere, kept constant on purpose.

The prior stays in the data as a COVARIATE, not a trained <know> token:
image_1 always shows the prior-congruent colour and image_2 the
counterfactual, so a spec's img_idx recovers prior-congruence for free
(baseline model performance quantifies the prior pull). klass = prior colour.

Emits the ColorSwap/UnifiedExample schema via to_hf + save_to_disk, so the
existing load_task / build_specs / collate / eval pipeline runs unchanged.
Split by held-out OBJECT (test_ood = unseen entities).
Writes data/vcf/{train,test_id,test_ood} + manifest.json.
"""
import argparse, ast, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scsrc.tasks import UnifiedExample

# to_hf lives in prepare_task; import without triggering its argparse main
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "prepare_task", os.path.join(os.path.dirname(__file__), "prepare_task.py"))
_pt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_pt)
to_hf = _pt.to_hf


def h(s):
    return int(hashlib.sha1(str(s).encode()).hexdigest()[:8], 16)


def first_val(x):
    """VCF's correct_answer arrives as a STRING literal ("['white']"), not a
    list; incorrect_answer is a plain string. Normalize both to one word."""
    if isinstance(x, list):
        return str(x[0]) if x else ""
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            return str(v[0]) if isinstance(v, (list, tuple)) and v else s
        except (ValueError, SyntaxError):
            return s.strip("[]'\" ")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vcf")
    ap.add_argument("--subset", default="color", choices=["color", "size"])
    ap.add_argument("--ood_frac", type=float, default=0.2, help="fraction of OBJECTS held out")
    ap.add_argument("--id_frac", type=float, default=0.15)
    a = ap.parse_args()

    os.environ.setdefault("HF_DATASETS_CACHE",
                          "~/.cache/huggingface/datasets")
    from datasets import load_dataset
    ds = load_dataset("mgolov/Visual-Counterfact", split=a.subset)

    objs = sorted({r["object"] for r in ds})
    ood_obj = {o for o in objs if (h("obj|" + o) % 100) < int(a.ood_frac * 100)}

    per_split = {"train": [], "test_id": [], "test_ood": []}
    for i, r in enumerate(ds):
        obj = str(r["object"]).strip()
        prior = first_val(r["correct_answer"]).strip().lower()
        cf = first_val(r["incorrect_answer"]).strip().lower()
        if not prior or not cf or prior == cf:
            continue
        ex = UnifiedExample(
            image_1=r["original_image"].convert("RGB"),      # shows prior colour
            image_2=r["counterfact_image"].convert("RGB"),   # shows cf colour
            caption_1=f"a {prior} {obj}", caption_2=f"a {cf} {obj}",
            key_1={obj: prior}, key_2={obj: cf},
            task="color", klass=prior, id=i, has_swap=True,
            caption_source="synthetic")
        if obj in ood_obj:
            s = "test_ood"
        else:
            s = "test_id" if (h(f"row|{i}") % 100) < int(a.id_frac * 100) else "train"
        per_split[s].append(ex)

    os.makedirs(a.out, exist_ok=True)
    counts = {}
    for s, exs in per_split.items():
        if not exs:
            counts[s] = 0; continue
        hf, n = to_hf(exs)
        hf.save_to_disk(os.path.join(a.out, s))
        counts[s] = n
    man = {"task": "vcf", "subset": a.subset, "source": "mgolov/Visual-Counterfact",
           "axis": "look_vs_read (modality); prior=covariate via img_idx",
           "n_objects": len(objs), "n_ood_objects": len(ood_obj),
           "counts": counts, "prompt_task": "color"}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
