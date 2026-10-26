#!/usr/bin/env python
"""Download ColorSwap into the HF cache. Run on the LOGIN node (has internet);
compute nodes are offline. Forces online mode regardless of paths.py defaults."""
import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
USER = os.environ.get("USER", "user")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

from datasets import load_dataset

for split in ("train", "test"):
    ds = load_dataset("stanfordnlp/colorswap", split=split)
    print(f"[{split}] n={len(ds)} columns={ds.column_names}")
    r = ds[0]
    print("  sample keys:", list(r.keys()))
    print("  caption_1:", r.get("caption_1"))
    print("  caption_2:", r.get("caption_2"))
    print("  caption_source:", r.get("caption_source"))
print("done. cached under", os.environ["HUGGINGFACE_HUB_CACHE"])
