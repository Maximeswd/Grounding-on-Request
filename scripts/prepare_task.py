#!/usr/bin/env python
"""Materialize a processed task dataset (LOGIN NODE; needs internet + the raw
data). Produces ColorSwap-shaped splits the compute nodes read locally via
data.load_task -> train.py --data_dir.

Pipeline (see scsrc/tasks.py):
  raw rows --adapter--> SlotItem stream
  --split_by_class--> {train, test_id, test_ood}  (held-out CLASSES -> test_ood)
  --build_examples (per split, swap-matched)--> ColorSwap-shaped rows
  --save_to_disk--> <out>/{train,test_id,test_ood} + manifest.json

Each row: image_1, image_2 (Image), caption_1, caption_2 (natural text stating
the slot value), key_1, key_2 (JSON {obj: value}), task, klass, id,
caption_source="human", has_swap. data.keys_from_row reads key_1/key_2 directly,
so values may be ints / relations / attributes -- no value-specific parser.

Examples
  # counting (the gate task); hold out counts 4 and 7 for compositional OOD
  python scripts/prepare_task.py --task counting --source vikhyatk/tallyqa-test \
      --split_name test --out $SC_ROOT/data/counting --heldout 4,7 --simple_only

  # counting OOD eval set (separate dataset; eval only)
  python scripts/prepare_task.py --task counting_ood --source vikhyatk/CountBenchQA \
      --split_name train --out $SC_ROOT/data/counting_countbench

  # spatial from a VSR jsonl + a local COCO mirror (or --download)
  python scripts/prepare_task.py --task spatial --vsr_jsonl vsr_random_train.jsonl \
      --image_dir /path/to/coco --out $SC_ROOT/data/spatial --heldout "above,below"

  # attributes (color) from GQA balanced (heaviest; preliminary)
  python scripts/prepare_task.py --task attr --source lmms-lab/GQA \
      --gqa_instr train_balanced_instructions --gqa_images train_balanced_images \
      --by_attr color --out $SC_ROOT/data/attr_color --heldout "color:purple,color:pink"
"""
import argparse, io, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scsrc.tasks import (build_examples, split_by_class, values_of, SlotItem,  # noqa: E402
                        UnifiedExample)
from scsrc import task_adapters as A  # noqa: E402


# ---- image helpers ---------------------------------------------------------
def to_pil(x):
    from PIL import Image
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, str) and os.path.exists(x):
        return Image.open(x).convert("RGB")
    return None


def coco_loader(image_dir, cache_dir, download):
    """Return loader(row)->PIL using a local COCO mirror first, then download."""
    from PIL import Image
    seen = {}

    def load(row):
        name = row.get("image")
        if name in seen:
            return seen[name]
        img = None
        if image_dir:
            for sub in ("", "train2017", "val2017", "images"):
                p = os.path.join(image_dir, sub, name) if sub else os.path.join(image_dir, name)
                if os.path.exists(p):
                    img = Image.open(p).convert("RGB"); break
        if img is None and download and row.get("image_link"):
            try:
                import urllib.request
                os.makedirs(cache_dir, exist_ok=True)
                cp = os.path.join(cache_dir, name)
                if not os.path.exists(cp):
                    urllib.request.urlretrieve(row["image_link"], cp)
                img = Image.open(cp).convert("RGB")
            except Exception as e:
                print(f"[vsr] image fetch failed {name}: {e}")
        seen[name] = img
        return img
    return load


# ---- raw -> SlotItems per task ---------------------------------------------
def _load_split(source, split_name, config=None):
    """Load a split. If the requested split name is absent (e.g. CountBenchQA /
    tallyqa-test do not expose 'train'), fall back to the first available split.
    The offline-miss hint fires ONLY when the whole dataset can't be loaded
    (genuinely not cached) -- a wrong split name must not trigger it."""
    from datasets import load_dataset
    offline = (os.environ.get("HF_DATASETS_OFFLINE") == "1"
               or os.environ.get("HF_HUB_OFFLINE") == "1")
    # 1) try the requested split directly; ANY failure just falls through
    try:
        return (load_dataset(source, config, split=split_name) if config
                else load_dataset(source, split=split_name))
    except Exception:
        pass
    # 2) load all splits and pick one. This is the only place a not-cached
    #    dataset actually fails, so the offline hint belongs here.
    try:
        dd = load_dataset(source, config) if config else load_dataset(source)
    except Exception as e:
        if offline:
            raise SystemExit(
                f"[prepare] '{source}' is not in the offline cache. Download it on "
                f"the LOGIN node first, e.g.\n"
                f"    python scripts/fetch_all.py --out $SC/data --hf_cache $HF_HOME --datasets <name>\n"
                f"then re-run with --offline. (original: {type(e).__name__})") from e
        raise
    if hasattr(dd, "keys"):
        key = split_name if split_name in dd else list(dd.keys())[0]
        print(f"[prepare] split '{split_name}' not found (have {list(dd.keys())}); using '{key}'")
        return dd[key]
    return dd

def load_items(args):
    if args.task == "counting":
        ds = _load_split(args.source, args.split_name)
        items = list(A.iter_tallyqa(ds, simple_only=args.simple_only,
                                    max_count=args.max_count))
    elif args.task == "counting_ood":
        ds = _load_split(args.source, args.split_name)
        items = list(A.iter_countbench(ds, max_count=args.max_count))
    elif args.task == "spatial":
        rows = [json.loads(l) for l in open(args.vsr_jsonl) if l.strip()]
        loader = coco_loader(args.image_dir, os.path.join(args.out, "_coco_cache"),
                             args.download)
        items = list(A.iter_vsr(rows, loader))
    elif args.task == "attr":
        instr = (_load_split(args.source, "train", config=args.gqa_instr)
                 if args.gqa_instr else _load_split(args.source, args.split_name))
        imgs = {}
        if args.gqa_images:
            ids = _load_split(args.source, "train", config=args.gqa_images)
            for r in ids:
                imgs[r["id"]] = r.get("image")
            instr = [{"question": r["question"], "answer": r.get("answer", ""),
                      "image": imgs.get(r.get("imageId"))} for r in instr]
        items = list(A.iter_gqa(instr, by_attr=args.by_attr))
    else:
        raise SystemExit(f"unknown task {args.task}")
    # task label on disk: counting_ood is just counting
    task = "counting" if args.task in ("counting", "counting_ood") else args.task
    for it in items:
        it.task = task
    return items, task


# ---- HF dataset writer -----------------------------------------------------
def to_hf(examples):
    from datasets import Dataset, Features, Image, Value
    data = {k: [] for k in ("image_1", "image_2", "caption_1", "caption_2",
                            "key_1", "key_2", "task", "klass", "id",
                            "caption_source", "has_swap")}
    for e in examples:
        i1, i2 = to_pil(e.image_1), to_pil(e.image_2)
        if i1 is None or i2 is None:
            continue
        data["image_1"].append(i1); data["image_2"].append(i2)
        data["caption_1"].append(e.caption_1); data["caption_2"].append(e.caption_2)
        data["key_1"].append(json.dumps(e.key_1)); data["key_2"].append(json.dumps(e.key_2))
        data["task"].append(e.task); data["klass"].append(e.klass)
        data["id"].append(e.id); data["caption_source"].append(e.caption_source)
        data["has_swap"].append(bool(e.has_swap))
    feats = Features({
        "image_1": Image(), "image_2": Image(),
        "caption_1": Value("string"), "caption_2": Value("string"),
        "key_1": Value("string"), "key_2": Value("string"),
        "task": Value("string"), "klass": Value("string"),
        "id": Value("int64"), "caption_source": Value("string"),
        "has_swap": Value("bool")})
    return Dataset.from_dict(data, features=feats), len(data["id"])


def prepare_colorswap(args):
    """ColorSwap has NATIVE minimal pairs (image_1/image_2 + caption_1/caption_2),
    so we preserve rows as-is (cleanest VGM swap) and only add a held-out-COLOR
    split: any row containing a held-out color -> test_ood, rest -> train/test_id.
    Stored key_1/key_2 come from the color parser."""
    import json, os, random
    from scsrc.colors import valid_swap
    rng = random.Random(args.seed)
    ds = _load_split(args.source or "stanfordnlp/colorswap", args.split_name)
    heldout = {x.strip().lower() for x in args.heldout.split(",") if x.strip()}
    rows = []
    for r in ds:
        ok, k1, k2 = valid_swap(r["caption_1"], r["caption_2"])
        if ok:
            rows.append((r, k1, k2))

    def is_ood(k1, k2):
        return bool((set(k1.values()) | set(k2.values())) & heldout)

    seen = [t for t in rows if not is_ood(t[1], t[2])]
    ood = [t for t in rows if is_ood(t[1], t[2])]
    rng.shuffle(seen)
    if args.max:
        seen = seen[:args.max]
    n_test = max(1, int(round(len(seen) * args.id_frac))) if seen else 0
    parts = {"train": seen[n_test:], "test_id": seen[:n_test], "test_ood": ood}

    os.makedirs(args.out, exist_ok=True)
    manifest = {"task": "color", "heldout": sorted(heldout), "id_frac": args.id_frac,
                "source": args.source or "stanfordnlp/colorswap", "splits": {}}
    print(f"[prepare] color: {len(rows)} valid rows, heldout={sorted(heldout)}")
    for split, items in parts.items():
        if not items:
            print(f"[prepare] {split}: 0 rows, skipped"); continue
        exs = [UnifiedExample(
            image_1=r["image_1"], image_2=r["image_2"],
            caption_1=r["caption_1"], caption_2=r["caption_2"],
            key_1=k1, key_2=k2, task="color",
            klass=",".join(sorted(set(k1.values()) | set(k2.values()))),
            id=i, has_swap=True) for i, (r, k1, k2) in enumerate(items)]
        hf, n = to_hf(exs)
        hf.save_to_disk(os.path.join(args.out, split))
        manifest["splits"][split] = {"rows": len(items), "examples": n}
        print(f"[prepare] {split}: {len(items)} rows -> {n} examples")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prepare] wrote {args.out}/manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["color", "counting", "counting_ood", "spatial", "attr"])
    ap.add_argument("--source", default=None, help="HF dataset id (counting/attr)")
    ap.add_argument("--split_name", default="train", help="HF split to read")
    ap.add_argument("--vsr_jsonl", default=None, help="VSR jsonl path (spatial)")
    ap.add_argument("--image_dir", default=None, help="local COCO mirror (spatial)")
    ap.add_argument("--download", action="store_true", help="fetch COCO via image_link")
    ap.add_argument("--gqa_instr", default=None, help="GQA instructions config (attr)")
    ap.add_argument("--gqa_images", default=None, help="GQA images config (attr)")
    ap.add_argument("--by_attr", default="color", help="attribute type to keep (attr)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default="", help="comma list of held-out CLASSES")
    ap.add_argument("--id_frac", type=float, default=0.15)
    ap.add_argument("--max", type=int, default=0, help="cap #items (0=all)")
    ap.add_argument("--max_count", type=int, default=12)
    ap.add_argument("--simple_only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline", action="store_true",
                    help="read the HF cache only (set this on compute nodes with no internet)")
    args = ap.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    if args.task == "color":
        return prepare_colorswap(args)
    rng = random.Random(args.seed)

    items, task = load_items(args)
    rng.shuffle(items)
    if args.max:
        items = items[:args.max]
    heldout = {x.strip() for x in args.heldout.split(",") if x.strip()}
    all_vals = values_of(items)
    print(f"[prepare] task={task} items={len(items)} values={all_vals} heldout={sorted(heldout)}")

    splits = split_by_class(items, heldout, args.id_frac, rng)
    os.makedirs(args.out, exist_ok=True)
    manifest = {"task": task, "heldout": sorted(heldout), "id_frac": args.id_frac,
                "all_values": all_vals, "source": args.source or args.vsr_jsonl,
                "splits": {}}
    for split, sitems in splits.items():
        if not sitems:
            print(f"[prepare] {split}: 0 items, skipped"); continue
        # train/test_id draw conflict values from their own (no-heldout) pool;
        # test_ood may use any value (held-out classes appear as v_img there).
        vocab = all_vals if split == "test_ood" else values_of(sitems)
        exs = build_examples(sitems, task, vocab, rng, require_swap=True)
        ds, n = to_hf(exs)
        ds.save_to_disk(os.path.join(args.out, split))
        manifest["splits"][split] = {"items": len(sitems), "examples": n,
                                     "values": values_of(sitems)}
        print(f"[prepare] {split}: {len(sitems)} items -> {n} examples")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prepare] wrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
