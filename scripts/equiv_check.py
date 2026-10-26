#!/usr/bin/env python
"""EQUIVALENCE CONTROL (co-author step 1, 2026-09-16).

Claim under test: "a learned vocabulary row" and "an injected activation" are two
NAMES for the same thing once both are input embeddings at the same positions.

Route A: the word's token ids go in as input_ids; the model looks the rows up.
Route B: the SAME rows are gathered and handed back as inputs_embeds at the same
positions, with the same attention mask and position ids.

If the downstream model receives identical tensors and metadata the two routes must
agree to numerical tolerance. Any gap is an implementation/input-processing
difference, NOT evidence that vocabulary membership matters. Separately TRAINED
words and prefixes are not this control -- that comparison confounds route with
training.
"""
import os, sys, json, glob, torch
sys.path.insert(0, ".")
from evaluate import apply_checkpoint
from scsrc.tokens import trigger

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/vcf_neo_ctrl_5tok_ctm_seed0_qwen2b"
DEV = os.environ.get("EQ_DEV", "cpu")
torch.set_grad_enabled(False)

model, proc, meta, _ = apply_checkpoint(RUN, 800, DEV)
model.eval()
print(f"[run] {RUN}  model={meta.get('model_id')}  method={meta['method']} L={meta['n_tokens']}")

tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
word = trigger("look_ctrl", meta["n_tokens"])
text = f"{word} The caption says the ball is green. final answer:"
ids = tok(text, return_tensors="pt").input_ids.to(DEV)
attn = torch.ones_like(ids)

emb_layer = model.get_input_embeddings()
lm = model.language_model if hasattr(model, "language_model") else model

# Route A: ids -> model
outA = lm(input_ids=ids, attention_mask=attn)
# Route B: the SAME rows, handed in as activations
embs = emb_layer(ids)
outB = lm(inputs_embeds=embs, attention_mask=attn)

la, lb = outA.logits.float(), outB.logits.float()
d = (la - lb).abs()
print(f"[tensors] ids={tuple(ids.shape)} embs={tuple(embs.shape)} dtype={embs.dtype}")
print(f"[logits ] max|A-B| = {d.max().item():.3e}   mean|A-B| = {d.mean().item():.3e}")
print(f"[logits ] argmax agrees on {(la.argmax(-1)==lb.argmax(-1)).float().mean().item()*100:.2f}% of positions")
top = 1e-3
print(f"[VERDICT] {'IDENTICAL within tolerance -> route is NOT the explanation' if d.max().item()<top else 'DIFFERENT -> an input-processing distinction exists, investigate'}")
