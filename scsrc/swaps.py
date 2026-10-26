"""Swap-based faithfulness tests (used by evaluate).

image-swap  : hold caption, swap the image  -> <look> claim must FLIP, <read> stay
caption-swap: hold image,   swap the caption -> <read> claim must FLIP, <look> stay
token-swap  : replace operator token with another / a random untrained token
              -> certifies the *token* carries the operation, not the format.
"""
from .colors import valid_swap
from . import metrics as Me


def conflict_examples(ds, legset, limit):
    """Swap-eval example iterator. legset is an OPTIONAL id filter -- empty/None
    means no filtering (the task-generic path has no legibility list; the old
    behavior silently yielded ZERO items there). Prefers the keys stored by
    prepare_task (task-generic) over color-caption parsing (color-only)."""
    from .data import keys_from_row
    out = []
    for idx in range(min(limit * 3, len(ds))):
        if len(out) >= limit:
            break
        row = ds[idx]
        if legset and int(row.get("id", idx)) not in legset:
            continue
        k1 = k2 = None
        try:
            k1, k2 = keys_from_row(row)          # prepared datasets store keys
        except Exception:
            pass
        if not k1 or not k2:
            ok, k1, k2 = valid_swap(row["caption_1"], row["caption_2"])
            if not ok:
                continue
        if row.get("image_2") is None:
            continue
        out.append((idx, row, k1, k2))
    return out


def flip_score(claim_orig, claim_swapped, key_before, key_after):
    """Did the claim move from key_before toward key_after? Returns (flipped, stayed)."""
    flipped = Me.key_match(claim_swapped, key_after)
    stayed = Me.key_match(claim_orig, key_before)
    return flipped, stayed
