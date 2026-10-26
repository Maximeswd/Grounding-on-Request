#!/usr/bin/env python
"""Small multimodal retrieval component over the MultiVENT-G pool.

Queries: each video's description FIRST SENTENCE (the MultiVENT event-query
style). Index: OCR strings (e5-base-v2) + up to 3 frames per video (SigLIP) --
the DESCRIPTION IS EXCLUDED from the index, otherwise retrieval is circular.
Fusion: z-scored score sum. Outputs:
  results/mvg_retrieval.json          R@1/5/10 + MRR, per channel and fused
  data/multiventG/retrieval_map.jsonl per video: top-5 retrieved + 1 distractor
The map feeds the RAG-stress ablation: routing/report evals conditioned on
{oracle | top1-retrieved | distractor} context.
"""
import argparse, json, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import torch
import torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvg", default="data/multiventG/multivent_g.json")
    ap.add_argument("--claims", default="data/multiventG/claims.jsonl")
    ap.add_argument("--frames", default="data/multiventG/frames_manifest.jsonl")
    ap.add_argument("--max_frames", type=int, default=3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    mvg = json.load(open(a.mvg))
    ocr_by_vid = defaultdict(set)
    for l in open(a.claims):
        c = json.loads(l)
        if c["channel"] == "OCR" and c.get("source_text"):
            ocr_by_vid[c["video_id"]].add(c["source_text"])
    frames = defaultdict(list)
    for l in open(a.frames):
        r = json.loads(l); frames[r["video_id"]].append((r["t"], r["path"]))
    vids = [v for v in mvg if frames.get(v)]
    print(f"pool: {len(vids)} videos with frames")

    # ---- queries: description first sentence (event-query style) ----
    def first_sent(t):
        s = re.split(r"(?<=[.!?])\s", (t or "").strip())[0]
        return s[:300]
    queries = {v: first_sent(mvg[v]["description"]) for v in vids}

    # ---- text retriever: e5-base-v2 over OCR strings ----
    from transformers import AutoTokenizer, AutoModel
    et = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
    em = AutoModel.from_pretrained("intfloat/e5-base-v2").to(dev).eval()

    @torch.no_grad()
    def e5(texts, pre):
        out = []
        for i in range(0, len(texts), 64):
            b = [f"{pre}: {t}" for t in texts[i:i+64]]
            enc = et(b, padding=True, truncation=True, max_length=256,
                     return_tensors="pt").to(dev)
            h = em(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            v = (h * mask).sum(1) / mask.sum(1)
            out.append(F.normalize(v, dim=-1).cpu())
        return torch.cat(out)

    ocr_docs = [" | ".join(sorted(ocr_by_vid[v]))[:800] or "(none)" for v in vids]
    D_ocr = e5(ocr_docs, "passage")
    Q_txt = e5([queries[v] for v in vids], "query")
    S_ocr = Q_txt @ D_ocr.T                     # [n, n]

    # ---- visual retriever: SigLIP text->frames ----
    from transformers import AutoProcessor, AutoModel as AM
    sp = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    sm = AM.from_pretrained("google/siglip-base-patch16-224").to(dev).eval()
    from PIL import Image

    def _pool(o):
        # transformers >=4.5x returns BaseModelOutputWithPooling here
        return o if torch.is_tensor(o) else o.pooler_output

    @torch.no_grad()
    def sig_img(paths):
        ims = [Image.open(p).convert("RGB") for p in paths]
        enc = sp(images=ims, return_tensors="pt").to(dev)
        return F.normalize(_pool(sm.get_image_features(**enc)), dim=-1).cpu()

    @torch.no_grad()
    def sig_txt(texts):
        out = []
        for i in range(0, len(texts), 64):
            enc = sp(text=texts[i:i+64], padding="max_length",
                     truncation=True, return_tensors="pt").to(dev)
            out.append(F.normalize(_pool(sm.get_text_features(**enc)), dim=-1).cpu())
        return torch.cat(out)

    vid_emb = []
    for v in vids:
        fs = [p for _, p in sorted(frames[v])[:a.max_frames]]
        vid_emb.append(sig_img(fs).mean(0))
    D_vis = F.normalize(torch.stack(vid_emb), dim=-1)
    Q_vis = sig_txt([queries[v][:64] for v in vids])
    S_vis = Q_vis @ D_vis.T

    # ---- fuse (z-sum) + metrics ----
    def z(S): return (S - S.mean(1, keepdim=True)) / S.std(1, keepdim=True).clamp_min(1e-6)
    S_fuse = z(S_ocr) + z(S_vis)

    def metrics(S):
        n = S.shape[0]
        ranks = []
        for i in range(n):
            order = S[i].argsort(descending=True)
            ranks.append(int((order == i).nonzero()[0]) + 1)
        r = torch.tensor(ranks, dtype=torch.float)
        return {"R@1": float((r <= 1).float().mean()), "R@5": float((r <= 5).float().mean()),
                "R@10": float((r <= 10).float().mean()), "MRR": float((1 / r).mean()),
                "n": n}

    res = {"index": "OCR(e5) + frames(SigLIP); description EXCLUDED (anti-circularity)",
           "query": "description first sentence (MultiVENT event-query style)",
           "ocr_only": metrics(S_ocr), "vis_only": metrics(S_vis), "fused": metrics(S_fuse)}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/mvg_retrieval.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

    # ---- retrieval map for the RAG-stress ablation ----
    import random
    rng = random.Random(0)
    with open("data/multiventG/retrieval_map.jsonl", "w") as f:
        for i, v in enumerate(vids):
            order = S_fuse[i].argsort(descending=True).tolist()
            top5 = [vids[j] for j in order[:5]]
            # distractor: same event_type, not self (hard); else random
            same = [u for u in vids if u != v and mvg[u]["event_type"] == mvg[v]["event_type"]]
            distractor = rng.choice(same) if same else rng.choice([u for u in vids if u != v])
            f.write(json.dumps({"video_id": v, "query": queries[v], "top5": top5,
                                "rank_of_gold": order.index(i) + 1,
                                "distractor": distractor}) + "\n")
    print("[saved] results/mvg_retrieval.json + data/multiventG/retrieval_map.jsonl")


if __name__ == "__main__":
    main()
