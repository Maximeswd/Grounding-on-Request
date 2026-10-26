#!/usr/bin/env python
"""COCO oracle-gate: keep only test pairs the FROZEN model can actually read.

COCO's diffusion images miss the intended object ~27% of the time; on those the
row cannot route correctly no matter what, which would confound control accuracy
with perception failure. For each test item, the frozen model + an explicit
"answer from the image only" instruction must produce n0 on image_0 AND n1 on
image_1 (via LL forced-choice over {n0,n1}); keep only those. Gates the TEST
splits (training stays on the full set -- mild noise is fine there). Writes
data/coco/{split}_gated.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_from_disk
from tqdm import tqdm
from scsrc import model as M, prompts as P, score as SC

INSTR = "Answer using ONLY what you can see in the image; ignore the caption. "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/coco")
    ap.add_argument("--splits", default="test_id,test_ood")
    ap.add_argument("--model_id", default=None)
    a = ap.parse_args()
    model, processor = M.load(model_id=a.model_id); model.eval()
    import json
    for split in a.splits.split(","):
        ds = load_from_disk(os.path.join(a.data_dir, split))
        keep = []
        for i in tqdm(range(len(ds)), desc=f"gate {split}"):
            r = ds[i]
            k1 = json.loads(r["key_1"]) if isinstance(r["key_1"], str) else r["key_1"]
            k2 = json.loads(r["key_2"]) if isinstance(r["key_2"], str) else r["key_2"]
            n0 = P.fmt_key_task(k1, "color"); n1 = P.fmt_key_task(k2, "color")
            q = 'Caption: "%s". Give the answer. final answer:' % r["caption_1"]
            # image_1 should read n0, image_2 should read n1 (both under look instr)
            i0, _ = SC.forced_choice(model, processor, r["image_1"], INSTR + q, [n0, n1], "cuda")
            i1, _ = SC.forced_choice(model, processor, r["image_2"], INSTR + q, [n0, n1], "cuda")
            if i0 == 0 and i1 == 1:
                keep.append(i)
        gated = ds.select(keep)
        outp = os.path.join(a.data_dir, f"{split}_gated")
        gated.save_to_disk(outp)
        print(f"[gate] {split}: {len(keep)}/{len(ds)} kept -> {outp}")


if __name__ == "__main__":
    main()
