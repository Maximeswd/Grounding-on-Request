"""MultiVENT-G stage-1 dataset: channel-grounding words on real event video.

Items come pre-materialized (scripts/mvg_prepare.py): every anti-shortcut
decision (perm6 rotation, conflict/donor, casefolded golds, independent hash
bits) lives in the data, so this class only renders and tokenizes.

Interface-compatible with ObjectiveDataset: __getitem__ -> {"op", "sft"}.
Ops are the item's own channel: desc | ocr | vis. SFT-only (the banked
objective verdict); v3 slot-negatives supported.
"""
import itertools
import json
import os
import random

from PIL import Image

from .collate import _encode, _prompt_len, NEG_SLOT_WORDS
from .tokens import trigger

# ---- request-word rendering (2026-09-17) ---------------------------------------------------------
# MVG prompt-tuning runs are trained with --pt_inserted 1: the learned rows sit in the SAME slot as
# the neologism, but the checkpoint registers them as pt_<channel>, not <channel>_ctrl. Until
# 2026-09-17 every MVG site rendered <channel>_ctrl regardless of method, so on a PT run the
# tokenizer split "<vis_ctrl_0>" into literal text and the PT rows never appeared in any prompt
# (train or eval); they moved only through the tied softmax. All MVG prompt-tuning numbers banked
# before this date are void (STATUS 2026-09-17). The ladder (collate.py) and Ground-CF (clevr3.py)
# were already method-aware.
_WORD_METHOD = {"method": "neo"}
_PT_ROW = {"vis": "pt_vis", "desc": "pt_desc", "ocr": "pt_ocr"}


def set_word_method(method):
    """'neo' (default) renders <channel>_ctrl; 'pt' renders the pt_<channel> rows instead."""
    _WORD_METHOD["method"] = method


def mvg_word(name, n_tokens):
    """The request string for channel word `name` under the active method."""
    if _WORD_METHOD["method"] == "pt" and name.endswith("_ctrl"):
        ch = name[:-5]
        if ch not in _PT_ROW:
            raise SystemExit(f"[mvg] a prompt-tuning run has no row for '{name}'")
        name = _PT_ROW[ch]
    return trigger(name, n_tokens)


def assert_words_registered(tokenizer, n_tokens, chans=("vis", "desc", "ocr")):
    """Refuse to train or evaluate with an unregistered request string (the 2026-09-17 bug class).
    LoRA is exempt BY DESIGN: it adds no embedding rows, so its request is the literal string
    "<vis_ctrl_0>..." as ordinary text and the adapted weights learn to respond to it."""
    if _WORD_METHOD["method"] == "lora":
        return
    for ch in chans:
        w = mvg_word(f"{ch}_ctrl", n_tokens)
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) != n_tokens:
            raise SystemExit(f"[mvg] request word {w[:22]}... is NOT a registered row: {len(ids)} ids, "
                             f"expected {n_tokens} (method={_WORD_METHOD['method']}). Refusing to run.")

_PERMS = list(itertools.permutations(("desc", "ocr", "vis")))


MVG_CH = {"DESC": "desc", "OCR": "ocr", "VIS": "vis"}


def mvg_sources():
    """Active MultiVENT-G evidence blocks (2026-10-02): SC_MVG_SOURCES, default "desc,ocr,vis" = the banked three-block
    prompt; the author's two-modality setting is "desc,vis" (no on-screen-text block, no OCR items). train.py sets it from
    --mvg_sources and records it in config.json; evaluate.apply_checkpoint restores it."""
    import os as _os
    want = [x for x in _os.environ.get("SC_MVG_SOURCES", "desc,ocr,vis").split(",") if x]
    return tuple(s for s in ("desc", "ocr", "vis") if s in want)


def _blocks(item):
    b = {
        "desc": f'Description: "{item["desc_text"]}"',
        "ocr": ("On-screen text: " +
                " | ".join(f'"{s}"' for s in item["ocr_strings"])
                if item["ocr_strings"] else "On-screen text: (none)"),
        "vis": None,   # the image itself
    }
    act = mvg_sources()
    return {k: v for k, v in b.items() if k in act}


def mvg_user_content(item, op_trig, override=None):
    """Multi-block user content (desc/ocr/vis in perm6 order + question + token).
    `override` swaps channel CONTENT without touching structure -- the hook the
    counterfactual-twin pairs use: {'desc_text':..., 'ocr_strings':...,
    'vis_frame':...} replaces just that channel's evidence."""
    it = {**item, **(override or {})}
    blocks = _blocks(it)
    order = _PERMS[int(it["perm6"]) % 6]
    content, buf = [], []
    def _load(x):
        return x.convert("RGB") if hasattr(x, "convert") else Image.open(x).convert("RGB")
    _vf = it["vis_frame"]
    # MULTI-FRAME: vis_frame may be a LIST -> the visual channel is a short clip
    # (k frames in temporal order) instead of a single snapshot. Both gemma-3 and
    # qwen3.5 accept several images in one turn, so the same word is now grounded
    # in video rather than in one sampled frame.
    imgs = [_load(x) for x in _vf] if isinstance(_vf, (list, tuple)) else [_load(_vf)]
    for b in order:
        if b not in blocks:          # inactive block (two-modality setting): the perm6 order of the rest is kept
            continue
        if b == "vis":
            if buf:
                content.append({"type": "text", "text": "\n".join(buf)})
                buf = []
            if len(imgs) > 1:
                content.append({"type": "text",
                                "text": f"Video frames ({len(imgs)}, in order):"})
            for _im in imgs:
                content.append({"type": "image", "image": _im})
        else:
            buf.append(blocks[b])
    import os as _os
    _prefix = (_WORD_METHOD["method"] == "pt" and _os.environ.get("SC_MVG_PT_PREFIX", "0") == "1")
    _adapter = (_WORD_METHOD["method"] == "lora" and _os.environ.get("SC_LORA_ADAPTER", "0") == "1")
    if _prefix or _adapter:
        # 2026-10-24: standard prompt tuning (soft rows prepended to the user turn, --mvg_pt_prefix 1) and one LoRA
        # adapter per request (--ctrl_style adapter): the question carries no request sentence
        q = it["role_question"] if it.get("role_question") else "Answer the question."
    else:
        q = (f'{it["role_question"]} Give me a {op_trig} answer.'
             if it.get("role_question") else f"Give me a {op_trig} answer.")
    buf.append(q)
    content.append({"type": "text", "text": "\n".join(buf)})
    if _prefix:
        content = [{"type": "text", "text": op_trig}] + content
    return content


MVG_CONT = "I used the requested source to answer the question. final answer: {v}"


def _seq_mvg(processor, content, cont, device):
    """seq_logprob-compatible (input_ids/attention_mask/cont_start) for a
    multi-block MVG user turn + continuation."""
    umsg = [{"role": "user", "content": content}]
    fmsg = umsg + [{"role": "assistant",
                    "content": [{"type": "text", "text": cont}]}]
    enc = _encode(processor, fmsg, device)
    out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
           "cont_start": _prompt_len(processor, umsg, device)}
    for k in enc:
        if k not in ("input_ids", "attention_mask", "token_type_ids") and k not in out:
            out[k] = enc[k]
    return out


def _sft_mvg(processor, content, target, device):
    """SFT record: needs LABELS with the prompt masked to -100.

    Distinct from _seq_mvg above, which produces the (input_ids, attention_mask,
    cont_start) form that seq_logprob consumes for PREFERENCE pairs. Using the
    scoring form as an "sft" record raises KeyError: 'labels' inside
    objective_step -- caught by the temporal smoke gate after it had been
    replicated into mvraw, corroborate and compare-v2."""
    umsg = [{"role": "user", "content": content}]
    fmsg = umsg + [{"role": "assistant",
                    "content": [{"type": "text", "text": target}]}]
    enc = _encode(processor, fmsg, device)
    plen = _prompt_len(processor, umsg, device)
    labels = enc["input_ids"].clone()
    labels[:, :plen] = -100
    out = {"input_ids": enc["input_ids"],
           "attention_mask": enc["attention_mask"], "labels": labels}
    for k in enc:
        if k not in ("input_ids", "attention_mask", "token_type_ids") and k not in out:
            out[k] = enc[k]
    return out


def build_mvg_ctm_pair(processor, item, pair, device, n_tokens=1):
    """Counterfactual-twin margin on real video: symmetric two-sided margins
    over the mined within-video pairs (prepare_mvg_ctm.py).
      vis_value_swap: frame A vs frame B, values entity_A vs entity_B.
      desc_swap     : description vs donor-substituted description.
      ocr_ablate / vis_presence: evidence present vs removed; the 'absent'
      value is the honesty gold ('unavailable').
    Returns {a_pos,a_neg,b_pos,b_neg} for the generic lam_ctm objective term."""
    op = item["channel"].lower()
    op_trig = mvg_word(f"{op}_ctrl", n_tokens)
    pt = pair["pair_type"]
    if pt in ("vis_value_swap", "vis_presence"):
        ov_a = {"vis_frame": pair["ctx_a"]["frame"]}
        ov_b = {"vis_frame": pair["ctx_b"]["frame"]}
    elif pt == "desc_swap":
        ov_a = {"desc_text": pair["ctx_a"]["desc"]}
        ov_b = {"desc_text": pair["ctx_b"]["desc"]}
    else:  # ocr_ablate
        ov_a = {"ocr_strings": pair["ctx_a"]["ocr_strings"]}
        ov_b = {"ocr_strings": pair["ctx_b"]["ocr_strings"]}
    ca = MVG_CONT.format(v=pair["gold_a"])
    cb = MVG_CONT.format(v=pair["gold_b"])
    ct_a = mvg_user_content(item, op_trig, ov_a)
    ct_b = mvg_user_content(item, op_trig, ov_b)
    return {"a_pos": _seq_mvg(processor, ct_a, ca, device),
            "a_neg": _seq_mvg(processor, ct_b, ca, device),
            "b_pos": _seq_mvg(processor, ct_b, cb, device),
            "b_neg": _seq_mvg(processor, ct_a, cb, device)}


def build_mvg_sm_pair(processor, item, pair, device, n_tokens=1):
    """STAY MARGIN on video (2026-09-22, STATUS v1.88): the competing channel is edited with a mined twin
    OF THAT CHANNEL and the requested channel's value must still beat the value the competitor now asserts.
      VIS item  (look): competitor = description -> a desc_swap pair; desc := donor description (gold_b)
                        logP(gold_vis | frame, desc') > logP(gold_b | frame, desc')
      DESC item (read): competitor = frame -> a vis_value_swap pair; frame := the frame whose value differs
                        logP(gold_desc | frame', desc) > logP(v_frame' | frame', desc)
    None when no usable twin (the competitor's new value equals the requested gold, or other pair types)."""
    op = item["channel"].lower()
    gold = str(item["gold"])
    pt = pair["pair_type"]
    if op == "vis" and pt == "desc_swap":
        w = str(pair["gold_b"])
        if w.strip().lower() == gold.strip().lower():
            return None
        ov = {"desc_text": pair["ctx_b"]["desc"]}
    elif op == "desc" and pt == "vis_value_swap":
        cand = [(pair["ctx_b"]["frame"], str(pair["gold_b"])), (pair["ctx_a"]["frame"], str(pair["gold_a"]))]
        cand = [(f, v) for f, v in cand if v.strip().lower() != gold.strip().lower()]
        if not cand:
            return None
        (frame, w) = cand[0]
        ov = {"vis_frame": frame}
    else:
        return None
    ct = mvg_user_content(item, mvg_word(f"{op}_ctrl", n_tokens), ov)
    return {"pos": _seq_mvg(processor, ct, MVG_CONT.format(v=gold), device),
            "neg": _seq_mvg(processor, ct, MVG_CONT.format(v=w), device)}


def _crop_to_bbox(path, bbox, pad=0.25):
    """bbox [x0,y0,x1,y1] in PERCENT -> cropped PIL image with fractional padding."""
    im = path.convert("RGB") if hasattr(path, "convert") else Image.open(path).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = [max(0.0, min(100.0, float(c))) for c in bbox]
    x0, x1 = sorted((x0 * W / 100.0, x1 * W / 100.0))
    y0, y1 = sorted((y0 * H / 100.0, y1 * H / 100.0))
    dw, dh = (x1 - x0) * pad, (y1 - y0) * pad
    box = (max(0, int(x0 - dw)), max(0, int(y0 - dh)),
           min(W, int(x1 + dw)), min(H, int(y1 + dh)))
    return im if (box[2] - box[0] < 8 or box[3] - box[1] < 8) else im.crop(box)


def build_mvg_ecc_pair(processor, item, pair, device, n_tokens=1, pad=0.25):
    """EVIDENCE-CROP CONSISTENCY (ECC).

    The mined visual evidence covers a median 6% of the frame, so a full-frame
    forced choice conflates CONTROL with LOCALISATION. ECC makes the crop a
    TEACHER: the discrimination the model achieves when shown only the evidence
    region is the target for the discrimination it must achieve on the whole
    frame. Returned in the a_pos/a_neg/b_pos/b_neg layout, where side A is the
    FULL frame and side B is the CROP of the same context, both with the same
    two candidate values -- so the generic two-sided margin term pushes the
    full-frame margin toward the crop's margin instead of rewarding a shortcut.
    Only vis_value_swap pairs carry bboxes."""
    if pair.get("pair_type") != "vis_value_swap" or not pair.get("bbox_a"):
        return None
    op_trig = mvg_word("vis_ctrl", n_tokens)
    full = {"vis_frame": pair["ctx_a"]["frame"]}
    crop = {"vis_frame": _crop_to_bbox(pair["ctx_a"]["frame"], pair["bbox_a"], pad)}
    ca, cb = MVG_CONT.format(v=pair["gold_a"]), MVG_CONT.format(v=pair["gold_b"])
    ct_full = mvg_user_content(item, op_trig, full)
    ct_crop = mvg_user_content(item, op_trig, crop)
    return {"a_pos": _seq_mvg(processor, ct_full, ca, device),
            "a_neg": _seq_mvg(processor, ct_full, cb, device),
            "b_pos": _seq_mvg(processor, ct_crop, ca, device),
            "b_neg": _seq_mvg(processor, ct_crop, cb, device)}


MVG_CMP_Q = "Do the provided sources agree on the answer to this question?"


def build_mvg_compare_example(processor, item, device, n_tokens=1, n_frames=1):
    """3-WAY COMPARE on real video -- the MVG analogue of <compare>.

    Why this construction and not the natural cross-channel differences: those
    are mostly PARAPHRASE, not contradiction (measured: of 724 multi-channel
    (video,role) cells only 103 agree while 621 "disagree", but the
    disagreements are things like selector DESC='americans' vs OCR='people' and
    when DESC='2020' vs OCR='election day' -- synonyms and different
    granularities). Training a verdict on those would teach the word to call
    paraphrases conflicts.

    Instead we reuse the item conditions we already built, which ARE genuine:
      cond=agree    -> every channel carries its annotated content  -> the
                       sources corroborate, answer the question.
      cond=conflict -> exactly ONE channel was donor-substituted (desc span
                       swap / within-video twin frame / OCR donor), so the
                       prompt really does contain contradictory sources ->
                       decline, because a contradicted question has no
                       warranted answer.
    Target format matches the ladder's compare v2 (neutral receipts +
    abstention), so the two datasets are directly comparable."""
    from .prompts import CMP_ABSTAIN
    trig = mvg_word("compare_ctrl", n_tokens)
    it = dict(item)
    it["role_question"] = f'{item.get("role_question", "")} {MVG_CMP_Q}'.strip()
    content = mvg_user_content(it, trig)
    gold = str(item.get("gold", "")).strip().lower()
    if item.get("cond") == "conflict":
        cont = ("Based on the requested comparison, the sources conflict and the "
                f"question cannot be answered from this evidence. final answer: {CMP_ABSTAIN}")
    else:
        cont = ("Based on the requested comparison, the sources agree. "
                f"final answer: {gold}")
    return {"op": "compare", "sft": _sft_mvg(processor, content, cont, device)}


def build_mvg_apo_pair(processor, item, other_vals, device, n_tokens=1):
    """APO on channels (the MVG analogue of look-rejects-text-value).

    chosen   = faithful report of THIS channel's value
    rejected = the SAME prompt, but the value ANOTHER channel attests
    The anti-prior is therefore cross-channel: the word must prefer its own
    source's value over a value that is present in the context but belongs to a
    different source. Skipped when no other channel attests a different value
    (nothing to reject) -- which is exactly the cross-attested case."""
    ch = item["channel"]
    gold = str(item.get("gold", "")).strip().lower()
    alts = [v for c, v in other_vals.items() if c != ch and v and v != gold]
    if not alts:
        return None
    op_trig = mvg_word(f"{ch.lower()}_ctrl", n_tokens)
    content = mvg_user_content(item, op_trig)
    pos = _seq_mvg(processor, content, MVG_CONT.format(v=gold), device)
    neg = _seq_mvg(processor, content, MVG_CONT.format(v=alts[0]), device)
    return {"chosen": pos, "rejected": neg}


def build_mvg_tc_pair(processor, item, device, n_tokens=1):
    """TOKEN CONTRAST on channels: identical evidence, identical gold, scored
    under THIS channel's word vs ANOTHER channel's word. Separates what the
    words DO without pushing any output string down globally (APO's failure
    mode). The cleanest channel-discrimination signal available on MVG."""
    ch = item["channel"].lower()
    other = {"desc": "vis", "vis": "desc", "ocr": "desc"}[ch]
    gold = str(item.get("gold", "")).strip().lower()
    cont = MVG_CONT.format(v=gold)
    c_own = mvg_user_content(item, mvg_word(f"{ch}_ctrl", n_tokens))
    c_oth = mvg_user_content(item, mvg_word(f"{other}_ctrl", n_tokens))
    return {"pos": _seq_mvg(processor, c_own, cont, device),
            "neg": _seq_mvg(processor, c_oth, cont, device)}


def build_mvg_vgm_pair(processor, item, pair, device, n_tokens=1):
    """VGM on channels: one-sided margin, continuation FIXED to this channel's
    value, the channel's own EVIDENCE swapped to its twin. Exactly the a_pos /
    a_neg half of the CTM quad, so the twins we already mined supply it."""
    q = build_mvg_ctm_pair(processor, item, pair, device, n_tokens)
    if q is None:
        return None
    return {"pos": q["a_pos"], "neg": q["a_neg"]}


def build_mvg_example(processor, item, device, n_tokens=1,
                      slot_negative=False):
    op = item["channel"].lower()
    op_trig = mvg_word(f"{op}_ctrl", n_tokens)
    if slot_negative:
        rng = random.Random(item["item_id"])
        op_trig = rng.choice(NEG_SLOT_WORDS)
    content = mvg_user_content(item, op_trig)
    if slot_negative:
        target = ("I do not recognize the requested operator, so I cannot "
                  "select a source. final answer: unknown operator")
    else:
        target = (f'I used the requested source to answer the question. '
                  f'final answer: {item["gold"]}')
    umsg = [{"role": "user", "content": content}]
    fmsg = umsg + [{"role": "assistant",
                    "content": [{"type": "text", "text": target}]}]
    enc = _encode(processor, fmsg, device)
    plen = _prompt_len(processor, umsg, device)
    labels = enc["input_ids"].clone()
    labels[:, :plen] = -100
    out = {"input_ids": enc["input_ids"],
           "attention_mask": enc["attention_mask"], "labels": labels}
    for k in enc:
        if k not in ("input_ids", "attention_mask", "token_type_ids") \
                and k not in out:
            out[k] = enc[k]
    return out


def clip_frames(video_id, centre_t, frames_by_vid, k):
    """k frames nearest centre_t, returned in temporal order (a short clip)."""
    fs = frames_by_vid.get(video_id) or []
    if not fs or k <= 1:
        return None
    near = sorted(fs, key=lambda x: abs(x[0] - centre_t))[:k]
    return [p for _, p in sorted(near)]


class MvgDataset:
    def __init__(self, data_dir, split, processor, device, n_tokens=1,
                 slot_negative_frac=0.0, ctm_pairs=None, n_frames=1, ecc=False,
                 apo=False, vgm=False, tc=False, compare_frac=0.0,
                 aat=False, aat_max_area=0.5, cert_weight=0.0,
                 frames_manifest="data/multiventG/frames_manifest.jsonl",
                 method="neo", sm=False):
        self.items = [json.loads(l)
                      for l in open(os.path.join(data_dir, f"{split}.jsonl"))]
        _act = mvg_sources()
        _ts = os.environ.get("SC_TRAIN_SOURCES", "")
        if _ts:                      # 2026-10-24: one LoRA adapter per request -> train only that source's items
            _keep = {{"look": "vis", "read": "desc"}.get(x.strip(), x.strip()) for x in _ts.split(",") if x.strip()}
            _n1 = len(self.items)
            self.items = [it for it in self.items if MVG_CH.get(it["channel"]) in _keep]
            print(f"[mvg] train_sources={sorted(_keep)}: {_n1} items -> {len(self.items)}")
        if len(_act) < 3:            # 2026-10-02: train only the active channels' items (two-modality MVG)
            _n0 = len(self.items)
            self.items = [it for it in self.items if MVG_CH.get(it["channel"]) in _act]
            print(f"[mvg] sources={','.join(_act)}: {_n0} items -> {len(self.items)} (inactive channels' items dropped)")
        self.processor, self.device, self.n = processor, device, n_tokens
        set_word_method(method)
        assert_words_registered(processor.tokenizer, n_tokens)
        print(f"[mvg] request words rendered for method={method}: "
              f"{mvg_word('vis_ctrl', n_tokens)[:40]}...")
        self.snf = float(slot_negative_frac)
        self.n_frames = int(n_frames)
        self.ecc = bool(ecc)
        # (video, role) -> {channel: value}, for the cross-channel APO negative
        self.apo, self.vgm, self.tc = bool(apo), bool(vgm), bool(tc)
        self.aat, self.aat_max_area = bool(aat), float(aat_max_area)
        self.cert_weight = float(cert_weight)
        self._img_tok_id = None
        if self.aat:
            try:                       # gemma image placeholder id (AAT is gemma-only:
                                       # qwen's visual token count varies per image)
                _v = processor.tokenizer.get_vocab()
                self._img_tok_id = next(i for k, i in _v.items()
                                        if "image_soft_token" in k)
            except Exception:
                self._img_tok_id = None
        # fraction of items rendered as 3-way COMPARE items instead of
        # single-channel items (0 = channel words only, the banked setting)
        self.compare_frac = float(compare_frac)
        self.other_vals = {}
        try:
            from collections import defaultdict as _dd
            ov = _dd(dict)
            for l in open("data/multiventG/claims.jsonl"):
                c = json.loads(l)
                ov[(c["video_id"], c["role"])][c["channel"]] = \
                    str(c["answer"]).strip().lower()
            self.other_vals = dict(ov)
        except Exception:
            pass
        self._rng = random.Random(4242)
        self.frames_by_vid = {}
        if self.n_frames > 1 and os.path.exists(frames_manifest):
            from collections import defaultdict as _dd
            fb = _dd(list)
            for l in open(frames_manifest):
                r = json.loads(l); fb[r["video_id"]].append((r["t"], r["path"]))
            self.frames_by_vid = {k: sorted(v) for k, v in fb.items()}
            print(f"[mvg] multi-frame ON: {self.n_frames} frames/item "
                  f"over {len(self.frames_by_vid)} videos")
        w = {}
        wp = os.path.join(data_dir, "weights.json")
        if os.path.exists(wp):
            w = json.load(open(wp))
        self.weights = [
            w.get(f'{i["event_type"]}|{i["role"]}|{i["channel"]}', 1.0)
            for i in self.items]
        # counterfactual-twin pairs indexed by (video, role, channel); one
        # rotating pair per key so repeated visits see different twins
        self.ctm = {}
        self._ctm_seen = {}
        self.sm = bool(sm); self._sm_seen = {}
        if ctm_pairs and os.path.exists(ctm_pairs):
            from collections import defaultdict
            idx = defaultdict(list)
            for l in open(ctm_pairs):
                p = json.loads(l)
                idx[(p["video_id"], p["role"], p["channel"])].append(p)
            self.ctm = dict(idx)
            print(f"[mvg-ctm] {sum(len(v) for v in self.ctm.values())} twin "
                  f"pairs over {len(self.ctm)} (video,role,channel) keys")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = dict(self.items[i])
        if self.n_frames > 1 and self.frames_by_vid:
            # centre the clip on the item's own frame timestamp when known
            t = item.get("vis_t")
            if t is None:
                p = str(item.get("vis_frame") or "")
                try:
                    t = float(p.rsplit("_", 1)[1].rsplit(".", 1)[0])
                except Exception:
                    t = 0.0
            cl = clip_frames(item["video_id"], float(t), self.frames_by_vid,
                             self.n_frames)
            if cl:
                item["vis_frame"] = cl
        neg = self.snf > 0 and self._rng.random() < self.snf
        rec = {"op": item["channel"].lower(),
               "sft": build_mvg_example(self.processor, item, self.device,
                                        self.n, slot_negative=neg)}
        key = (item["video_id"], item["role"], item["channel"])
        if not neg and key in self.ctm:
            ps = self.ctm[key]
            k = self._ctm_seen.get(key, 0)
            self._ctm_seen[key] = k + 1
            _p = ps[k % len(ps)]
            rec["ctm"] = build_mvg_ctm_pair(self.processor, item, _p, self.device, self.n)
            if self.ecc:
                _e = build_mvg_ecc_pair(self.processor, item, _p, self.device, self.n)
                if _e is not None:
                    rec["ecc"] = _e
            if self.vgm:
                _v = build_mvg_vgm_pair(self.processor, item, _p, self.device, self.n)
                if _v is not None:
                    rec["vgm"] = _v
        if self.sm and not neg:
            # Stay margin (v1.88): twins of the OTHER channel for the same (video, role), rotating
            _other = {"VIS": "DESC", "DESC": "VIS"}.get(str(item["channel"]).upper())
            _ps = self.ctm.get((item["video_id"], item["role"], _other)) if _other else None
            if _ps:
                _k = self._sm_seen.get(key, 0); self._sm_seen[key] = _k + 1
                for _t in range(len(_ps)):
                    _s = build_mvg_sm_pair(self.processor, item, _ps[(_k + _t) % len(_ps)], self.device, self.n)
                    if _s is not None:
                        rec["sm"] = _s; break
        # AAT: gaze supervision toward the annotated evidence box. TRAIN-TIME
        # ONLY -- the bbox never appears at eval, so this is auxiliary
        # supervision, not oracle test information. Skipped for whole-scene
        # boxes (area >= aat_max_area), which would teach "attend everywhere".
        if (self.aat and not neg and self._img_tok_id is not None
                and str(item.get("channel", "")).upper() == "VIS"
                and item.get("bbox")):
            _bb = [max(0.0, min(100.0, float(x))) for x in item["bbox"][:4]]
            _area = abs(_bb[2] - _bb[0]) * abs(_bb[3] - _bb[1]) / 1e4
            if 0.0 < _area < self.aat_max_area:
                rec["aat"] = {"ex": rec["sft"], "bbox": _bb,
                              "img_size": 100,          # bboxes are PERCENT coords
                              "image_token_id": self._img_tok_id}
        # certainty weighting: annotator confidence 1-5 -> per-record loss scale.
        # Low-certainty items are noisy targets; down-weighting them sharpens the
        # gradient on evidence the annotators could actually see.
        if self.cert_weight > 0:
            _c = item.get("certainty")
            if _c is not None:
                rec["w"] = float((int(_c) / 5.0) ** self.cert_weight)
        if not neg and self.apo:
            _a = build_mvg_apo_pair(self.processor, item,
                                    self.other_vals.get((item["video_id"], item["role"]), {}),
                                    self.device, self.n)
            if _a is not None:
                rec["apo"] = _a
        if not neg and self.tc:
            rec["tc"] = build_mvg_tc_pair(self.processor, item, self.device, self.n)
        return rec


# ---------------- stage-2: <report> (one-pass, provenance-tagged) ----------------
REPORT_INSTR = "List every grounded claim from all three sources, one per line."


def _video_items(claims_path, mvg_path, frames_path, min_claims=2):
    """Group claims by video -> one report item per video."""
    import itertools as _it
    from collections import defaultdict as _dd
    mvg = json.load(open(mvg_path))
    frames = _dd(list)
    for l in open(frames_path):
        r = json.loads(l); frames[r["video_id"]].append((r["t"], r["path"]))
    ocr_by_vid, by_vid = _dd(set), _dd(list)
    for l in open(claims_path):
        c = json.loads(l)
        if c["channel"] == "OCR":
            ocr_by_vid[c["video_id"]].add(c.get("source_text", ""))
        by_vid[c["video_id"]].append(c)
    items = []
    for v, cl in by_vid.items():
        if len(cl) < min_claims or not frames.get(v) or v not in mvg:
            continue
        gold = sorted({(c["channel"], c["role"], str(c["answer"]).strip().lower())
                       for c in cl})
        items.append({"video_id": v, "perm6": hash(v) % 6,
                      "desc_text": mvg[v]["description"][:1200],
                      "ocr_strings": sorted(ocr_by_vid[v])[:8],
                      "vis_frame": sorted(frames[v])[0][1],
                      "role_question": "", "event_type": mvg[v]["event_type"],
                      "gold_claims": gold})
    return items


def report_gold(item):
    return "\n".join(f"[{ch}] {role}: {val}" for ch, role, val in item["gold_claims"])


def build_mvg_report_example(processor, item, device, n_tokens=5):
    trig = mvg_word("report_ctrl", n_tokens)
    content = mvg_user_content(item, trig)   # question slot renders the trigger
    # override the trailing question line with the report instruction
    content[-1]["text"] = content[-1]["text"].rsplit("\n", 1)[0] + \
        f"\n{REPORT_INSTR} Give me a {trig} answer."
    target = report_gold(item)
    umsg = [{"role": "user", "content": content}]
    fmsg = umsg + [{"role": "assistant",
                    "content": [{"type": "text", "text": target}]}]
    enc = _encode(processor, fmsg, device)
    plen = _prompt_len(processor, umsg, device)
    labels = enc["input_ids"].clone(); labels[:, :plen] = -100
    out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
           "labels": labels}
    for k in enc:
        if k not in ("input_ids", "attention_mask", "token_type_ids") and k not in out:
            out[k] = enc[k]
    return out


CH_WORD = {"DESC": "desc_ctrl", "OCR": "ocr_ctrl", "VIS": "vis_ctrl"}


def build_mvg_channel_report_example(processor, item, channel, device, n_tokens=5):
    """MATCHED-SUPERVISION composition: the CHANNEL word is prompted with the
    same report instruction (channel-neutral co-text) and teacher-forced on the
    [CH]-tagged claim lines OF ITS OWN CHANNEL ONLY. Concatenating the three
    words' outputs reconstructs exactly the report word's target format, so
    report-vs-composition is compared at IDENTICAL supervision, format and
    metric -- the only difference left is one word vs three."""
    trig = mvg_word(CH_WORD[channel], n_tokens)
    content = mvg_user_content(item, trig)
    content[-1]["text"] = content[-1]["text"].rsplit("\n", 1)[0] + \
        f"\nList every grounded claim from the requested source, one per line. " \
        f"Give me a {trig} answer."
    lines = [f"[{ch}] {role}: {val}" for ch, role, val in item["gold_claims"]
             if ch == channel]
    target = "\n".join(lines) if lines else "no grounded claims from this source"
    umsg = [{"role": "user", "content": content}]
    fmsg = umsg + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]
    enc = _encode(processor, fmsg, device)
    plen = _prompt_len(processor, umsg, device)
    labels = enc["input_ids"].clone(); labels[:, :plen] = -100
    out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
           "labels": labels}
    for k in enc:
        if k not in ("input_ids", "attention_mask", "token_type_ids") and k not in out:
            out[k] = enc[k]
    return out


class MvgReportDataset:
    """One item per video: 3 channels in, provenance-tagged claim lines out.
    VIDEO-LEVEL split inherited from stage-1: train on the data/mvg train
    videos (the mvg3 test files are item-level and reuse most of these), eval
    on the videos NO stage ever trained on -- the only split under which
    report-token vs stage-1-assembler is a fair comparison."""

    def __init__(self, processor, device, n_tokens=5, mode="report",
                 claims="data/multiventG/claims.jsonl",
                 mvg="data/multiventG/multivent_g.json",
                 frames="data/multiventG/frames_manifest.jsonl",
                 train_split="data/mvg/train.jsonl"):
        tv = {json.loads(l)["video_id"] for l in open(train_split)}
        allitems = _video_items(claims, mvg, frames)
        base = [it for it in allitems if it["video_id"] in tv]
        self.mode = mode
        if mode == "channel":
            # one training item per (video, channel) so each word sees every video
            self.items = [{**it, "_ch": ch} for it in base
                          for ch in ("DESC", "OCR", "VIS")]
        else:
            self.items = base
        self.processor, self.device, self.n = processor, device, n_tokens
        self.weights = [1.0] * len(self.items)
        print(f"[mvg-report] {len(self.items)} train videos "
              f"({len(allitems) - len(self.items)} held out)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if self.mode == "channel":
            return {"op": it["_ch"].lower(),
                    "sft": build_mvg_channel_report_example(
                        self.processor, it, it["_ch"], self.device, self.n)}
        return {"op": "report",
                "sft": build_mvg_report_example(self.processor, it,
                                                self.device, self.n)}
