"""multivent-raw stage-1: channel words on multilingual event video at CHUNK
granularity.

Structural contrast with MVG (scsrc/mvg.py): MVG attribution is per-FRAME with
bboxes, so items show ONE evidence frame and frame-level labels are load-bearing
(the fps bug was fatal there). multivent-raw attribution is per-CHUNK, so items
show the WHOLE chunk (all uniform-5s keyframes, capped) -- label granularity
equals input granularity and no frame selection exists to get wrong.

Channels and their words:
    VIS <- video-non-text  (keyframe clip)
    OCR <- video-text      (per-frame OCR strings, deduped)
    ASR <- audio-speech    (WhisperX transcript; often NON-English while claims
                            are English -> the word is cross-lingually grounded)
audio-non-speech has no input rendering shipped and is EXCLUDED from words.

Targets are the chunk's OWN-channel claim sentences, one per line (multi-claim:
the report behaviour), or the absence string when the channel's raw content is
empty. Cells with content but no claims are NOT absence items -- claim-centric
annotation means missing-claim != absent-evidence.

Interface-compatible with ObjectiveDataset: __getitem__ -> {"op","sft",...} with
optional "apo"/"tc"/"ctm" pair keys consumed by the generic objective terms.
"""
import glob
import itertools
import json
import os
import random

from .mvg import _seq_mvg, _sft_mvg
from .tokens import trigger

_PERMS = list(itertools.permutations(("asr", "ocr", "vis")))
ABSENT = "no grounded claims from this source."
_QUESTION = "List every grounded claim from the requested source, one per line."
# BASELINE carriers. The neologism condition says "the requested source" and lets
# the WORD pick it out; the instruction condition must name the source in English
# (that is the whole point of the comparison), and it also gets the explicit
# ignore-the-others clause, which is the strongest form of the baseline.
INSTR = {
    "vis": ("List every grounded claim you can make from the VIDEO IMAGERY only, "
            "one per line. Ignore the speech transcript and the on-screen text."),
    "ocr": ("List every grounded claim you can make from the ON-SCREEN TEXT only, "
            "one per line. Ignore the video imagery and the speech transcript."),
    "asr": ("List every grounded claim you can make from the SPEECH TRANSCRIPT "
            "only, one per line. Ignore the video imagery and the on-screen text."),
}
NOOP = "List every grounded claim from this video, one per line."


def _trig(ch, n_tokens, style="neo"):
    """The addressable handle for this channel under each method."""
    if style == "neo":
        return trigger(f"{ch}_ctrl", n_tokens)
    if style == "pt":
        return trigger(f"pt_{ch}", n_tokens)
    return ""          # instr / noop carry no learned rows


def _target(claims):
    return "\n".join(claims) if claims else ABSENT


def _blocks(item):
    asr = (item.get("asr_text") or "").strip()
    ocr = [s for s in (item.get("ocr_strings") or []) if s.strip()]
    return {
        "asr": (f'Speech transcript (auto): "{asr}"' if asr
                else "Speech transcript: (none)"),
        "ocr": ("On-screen text: " + " | ".join(f'"{s}"' for s in ocr)
                if ocr else "On-screen text: (none)"),
        "vis": None,
    }


def _cap_frames(paths, n):
    """Evenly spaced sample of n frames, endpoints included.

    n == 1 used to divide by zero here, which meant `--n_frames 1` crashed the
    dataloader 15 s in -- and n_frames 1 is the obvious lever to reach for when
    MVR runs out of GPU memory, so the crash landed exactly when it was least
    welcome. One frame takes the MIDDLE rather than the first: chunks average a
    scene change every few seconds, so frame 0 is often a cut-in or a title card
    and the midpoint is the more representative single sample."""
    if n <= 0:
        return []
    if len(paths) <= n:
        return paths
    if n == 1:
        return [paths[len(paths) // 2]]
    idx = [round(i * (len(paths) - 1) / (n - 1)) for i in range(n)]
    return [paths[i] for i in dict.fromkeys(idx)]


def mvraw_user_content(item, op_trig, override=None, n_frames=8,
                       style="neo", channel=None):
    """style: neo   -> "<..._ctrl> answer" (the word does the selecting)
              pt    -> soft-prefix rows PREPENDED to the user turn, then a
                       neutral question (prompt-tuning baseline placement)
              instr -> English source-naming instruction, no new tokens
                       (the LoRA / instruction / frozen baseline carrier)
              noop  -> no source specified at all (frozen no-op floor)

    Three-block user content (asr/ocr/vis in perm6 order + instruction).
    `override` swaps one channel's CONTENT without touching structure --
    {'asr_text':...} / {'ocr_strings':...} / {'frames':[...]} -- the hook the
    evidence-swap twin pairs use."""
    from PIL import Image
    it = {**item, **(override or {})}
    blocks = _blocks(it)
    order = _PERMS[int(it["perm6"]) % 6]
    frames = _cap_frames(it["frames"], n_frames)
    content, buf = [], []
    for b in order:
        if b == "vis":
            if buf:
                content.append({"type": "text", "text": "\n".join(buf)})
                buf = []
            if len(frames) > 1:
                content.append({"type": "text",
                                "text": f"Video frames ({len(frames)}, in order):"})
            for p in frames:
                content.append({"type": "image", "image": Image.open(p).convert("RGB")})
        else:
            buf.append(blocks[b])
    ch = (channel or it.get("channel", "")).lower()
    if style == "neo":
        buf.append(f"{_QUESTION} Give me a {op_trig} answer.")
    elif style == "pt":
        buf.append(_QUESTION)
    elif style == "instr":
        buf.append(INSTR[ch])
    else:
        buf.append(NOOP)
    content.append({"type": "text", "text": "\n".join(buf)})
    if style == "pt":
        # prompt tuning is a PREFIX method: rows lead the turn
        content = [{"type": "text", "text": op_trig}] + content
    return content


def _override_for(channel, src_item):
    ch = channel.lower()
    if ch == "vis":
        return {"frames": src_item["frames"]}
    if ch == "asr":
        return {"asr_text": src_item.get("asr_text", "")}
    return {"ocr_strings": src_item.get("ocr_strings", [])}


def build_mvraw_ctm_pair(processor, item, mate, device, n_tokens=1, n_frames=8,
                         style="neo"):
    """Evidence-swap twin at chunk granularity: ctx_a = the item's own context;
    ctx_b = the SAME context with only this channel's content replaced by the
    mate chunk's. Golds are each chunk's own-channel claim lists. Symmetric
    two-sided margins (a_pos/a_neg/b_pos/b_neg) for the generic lam_ctm term."""
    ch = item["channel"]
    trig = _trig(ch.lower(), n_tokens, style)
    ga, gb = _target(item["claims"]), _target(mate["claims"])
    ct_a = mvraw_user_content(item, trig, None, n_frames, style, ch)
    ct_b = mvraw_user_content(item, trig, _override_for(ch, mate), n_frames, style, ch)
    return {"a_pos": _seq_mvg(processor, ct_a, ga, device),
            "a_neg": _seq_mvg(processor, ct_b, ga, device),
            "b_pos": _seq_mvg(processor, ct_b, gb, device),
            "b_neg": _seq_mvg(processor, ct_a, gb, device)}


def build_mvraw_sm_pair(processor, item, mate, device, n_tokens=1, n_frames=8, style="neo", other=None):
    """STAY MARGIN on noisy real video (2026-09-22, STATUS v1.102): the COMPETING channel's content is replaced by
    the mate chunk's while the requested channel is untouched, and the requested channel's own claims must still beat
    the claims the competitor now carries. Mirrors build_clevr3_sm_pair and build_mvg_sm_pair. `other` names the
    competing channel (the paper's two sources are VIS and ASR); None -> the first channel that is not this one."""
    ch = item["channel"]
    oc = (other or ("ASR" if ch.upper() == "VIS" else "VIS")).upper()
    if oc == ch.upper():
        return None
    gold, neg = _target(item["claims"]), _target(mate["claims"])
    if not gold or not neg or gold.strip() == neg.strip():
        return None
    trig = _trig(ch.lower(), n_tokens, style)
    ct = mvraw_user_content(item, trig, _override_for(oc, mate), n_frames, style, ch)
    return {"pos": _seq_mvg(processor, ct, gold, device),
            "neg": _seq_mvg(processor, ct, neg, device)}


def build_mvraw_apo_pair(processor, item, device, n_tokens=1, n_frames=8,
                         style="neo"):
    """chosen = own channel's claim lines; rejected = the rotation-assigned
    OTHER channel's claims for the SAME chunk (cross-attested lines already
    removed at prepare time). The negative modality was balanced round-robin
    in mvraw_prepare so no channel is over-used as the foil."""
    if not item.get("neg_claims"):
        return None
    ch = item["channel"].lower()
    trig = _trig(ch, n_tokens, style)
    ct = mvraw_user_content(item, trig, None, n_frames, style, ch)
    return {"chosen": _seq_mvg(processor, ct, _target(item["claims"]), device),
            "rejected": _seq_mvg(processor, ct, _target(item["neg_claims"]), device)}


def build_mvraw_tc_pair(processor, item, device, n_tokens=1, n_frames=8,
                        style="neo"):
    """Same context, same gold, own word vs the rotation-assigned other
    channel's word. Pushes no output string down globally."""
    co, cn = item["channel"].lower(), item["neg_channel"].lower()
    gold = _target(item["claims"])
    return {"pos": _seq_mvg(processor, mvraw_user_content(
                item, _trig(co, n_tokens, style), None, n_frames, style, co), gold, device),
            "neg": _seq_mvg(processor, mvraw_user_content(
                item, _trig(cn, n_tokens, style), None, n_frames, style, cn), gold, device)}


CORR_Q = ("Which of the provided sources support this claim? "
          "Claim: \"{claim}\"")
CORR_NONE = "none"


def _corr_target(sources):
    """Canonical, parseable, and scored as a SET. Sources must be named -- for
    corroboration the source identity IS the answer, unlike look/read where a
    named source is a co-text shortcut to the value."""
    if not sources:
        return ("Based on the requested corroboration, no provided source "
                f"supports this claim. sources: {CORR_NONE}")
    return ("Based on the requested corroboration, this claim is supported by "
            f"the listed sources. sources: {', '.join(sources)}")


def _corr_seq(processor, item, device, n_tokens, n_frames, override, srcs):
    """One (CONTEXT, GOLD) sequence. Context and gold are independent arguments
    on purpose: the twin margins below must vary exactly one of them at a time,
    otherwise the margin conflates the intervention with the label change."""
    trig = trigger("corroborate_ctrl", n_tokens)
    it = dict(item)
    it["role_question"] = None
    content = mvraw_user_content(it, trig, override, n_frames, "neo", "vis")
    content[-1] = {"type": "text",
                   "text": CORR_Q.format(claim=item["claim"]) +
                           f" Give me a {trig} answer."}
    return _seq_mvg(processor, content, _corr_target(srcs), device)


def build_mvraw_corr_example(processor, item, device, n_tokens=1, n_frames=8):
    """SFT record (labels form). The CTM quad below uses _corr_seq, the scoring
    form, because preference terms go through seq_logprob."""
    from .tokens import trigger as _t
    it = dict(item); it["role_question"] = None
    trig = _t("corroborate_ctrl", n_tokens)
    content = mvraw_user_content(it, trig, None, n_frames, "neo", "vis")
    content[-1] = {"type": "text",
                   "text": CORR_Q.format(claim=item["claim"]) + f" Give me a {trig} answer."}
    return {"op": "corroborate",
            "sft": _sft_mvg(processor, content, _corr_target(list(item["sources"])),
                            device)}


def build_mvraw_corr_ctm(processor, item, ch, mate, device, n_tokens=1,
                         n_frames=8):
    """INTERVENTIONAL twin for corroboration, and the reason this operator is
    worth having: swapping ONE attesting channel's content must make the gold
    set LOSE exactly that source, so the counterfactual is exact and GRADED.

      ctx A = as annotated,          gold A = S
      ctx B = channel `ch` swapped,  gold B = S \ {ch}

    The four sequences vary one factor at a time:
      a_pos (A, S)      a_neg (B, S)      <- same gold, context swapped
      b_pos (B, S-ch)   b_neg (A, S-ch)   <- same gold, context swapped
    A model that ignores the evidence and answers from claim wording alone
    scores 0 on both margins."""
    S = list(item["sources"])
    if ch.lower() not in S:
        return None
    S_drop = [s for s in S if s != ch.lower()]
    ovB = _override_for(ch, mate)
    f = lambda ov, srcs: _corr_seq(processor, item, device, n_tokens, n_frames,
                                   ov, srcs)
    return {"a_pos": f(None, S), "a_neg": f(ovB, S),
            "b_pos": f(ovB, S_drop), "b_neg": f(None, S_drop)}


class MvrawCorrDataset:
    """<corroborate> on multivent-raw. Emits SFT items and, when twins are
    available for an attesting channel, an interventional CTM pair whose two
    sides differ ONLY in that channel's content and whose golds differ by
    exactly that one source."""

    def __init__(self, data_dir, split, processor, device, n_tokens=1,
                 n_frames=8, ctm_pairs=None):
        self.processor, self.device = processor, device
        self.n, self.nf = int(n_tokens), int(n_frames)
        self.items = [json.loads(l) for l in open(f"{data_dir}/corr_{split}.jsonl")]
        self.weights = [it.get("weight", 1.0) for it in self.items]
        self.mates = {}
        if ctm_pairs and os.path.exists(ctm_pairs):
            base = {}
            for l in open(f"{data_dir}/{split}.jsonl"):
                x = json.loads(l)
                base[(x["chunk_id"], x["channel"])] = x
            for l in open(ctm_pairs):
                p = json.loads(l)
                for x, y in ((p["chunk_a"], p["chunk_b"]), (p["chunk_b"], p["chunk_a"])):
                    if (y, p["channel"]) in base:
                        self.mates.setdefault((x, p["channel"]), []).append(
                            base[(y, p["channel"])])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        rec = build_mvraw_corr_example(self.processor, it, self.device,
                                       self.n, self.nf)
        for s in it["sources"]:
            mates = self.mates.get((it["chunk_id"], s.upper()))
            if mates:
                q = build_mvraw_corr_ctm(self.processor, it, s.upper(),
                                         mates[i % len(mates)], self.device,
                                         self.n, self.nf)
                if q is not None:
                    rec["ctm"] = q
                break
        return rec


class MvrawDataset:
    def __init__(self, data_dir, split, processor, device, n_tokens=1,
                 n_frames=8, apo=False, tc=False, ctm_pairs=None, ops=None,
                 style="neo", full_coverage_only=False, sm=False):
        self.processor, self.device = processor, device
        self.n, self.nf = int(n_tokens), int(n_frames)
        self.apo, self.tc = bool(apo), bool(tc)
        self.sm = bool(sm)
        self.style = style
        self.items = [json.loads(l) for l in open(f"{data_dir}/{split}.jsonl")]
        if full_coverage_only:
            # A chunk longer than n_frames is SUBSAMPLED, so a visual claim whose
            # moment falls between shown frames becomes unsupported by the input.
            # OCR/ASR are unaffected (their text covers every frame / the whole
            # transcript), so the filter applies to VIS only.
            before = len(self.items)
            self.items = [it for it in self.items
                          if it["channel"] != "VIS" or len(it["frames"]) <= self.nf]
            print(f"[mvraw] full_coverage_only: VIS items subsampled away "
                  f"{before - len(self.items)} of {before}")
        if ops:
            keep = {o.upper() for o in ops}
            self.items = [it for it in self.items if it["channel"] in keep]
        self.weights = [it.get("weight", 1.0) for it in self.items]
        self.by_chunk = {}
        self.pairs = {}
        if ctm_pairs and os.path.exists(ctm_pairs):
            allowed = {(it["chunk_id"], it["channel"]) for it in self.items}
            items_ix = {(it["chunk_id"], it["channel"]): it for it in self.items}
            for l in open(ctm_pairs):
                p = json.loads(l)
                for a, b in ((p["chunk_a"], p["chunk_b"]), (p["chunk_b"], p["chunk_a"])):
                    k = (a, p["channel"])
                    if k in allowed and (b, p["channel"]) in items_ix:
                        self.pairs.setdefault(k, []).append(items_ix[(b, p["channel"])])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        ch = it["channel"].lower()
        trig = _trig(ch, self.n, self.style)
        ct = mvraw_user_content(it, trig, None, self.nf, self.style, ch)
        rec = {"op": it["channel"].lower(),
               "sft": _sft_mvg(self.processor, ct, _target(it["claims"]), self.device)}
        if it.get("cond") == "absent":
            return rec  # honesty items are SFT-only
        if self.apo:
            a = build_mvraw_apo_pair(self.processor, it, self.device, self.n,
                                     self.nf, self.style)
            if a is not None:
                rec["apo"] = a
        if self.tc and it.get("neg_channel"):
            rec["tc"] = build_mvraw_tc_pair(self.processor, it, self.device,
                                            self.n, self.nf, self.style)
        mates = self.pairs.get((it["chunk_id"], it["channel"]))
        if mates:
            rec["ctm"] = build_mvraw_ctm_pair(self.processor, it,
                                              mates[i % len(mates)],
                                              self.device, self.n, self.nf,
                                              self.style)
        if self.sm:
            # v1.102: the competitor's twins live under the OTHER channel's key for the same chunk
            _oc = "ASR" if str(it["channel"]).upper() == "VIS" else "VIS"
            _m = self.pairs.get((it["chunk_id"], _oc))
            if _m:
                _s = build_mvraw_sm_pair(self.processor, it, _m[i % len(_m)], self.device,
                                         self.n, self.nf, self.style, other=_oc)
                if _s is not None:
                    rec["sm"] = _s
        return rec
