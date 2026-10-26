#!/usr/bin/env python
"""Train one configuration. Objective L = lam_sft*CE + lam_apo*APO + lam_vgm*VGM,
with PER-OPERATOR coefficients: --per_op '{"look":{"lam_apo":0,"lam_vgm":1},
"read":{"lam_apo":1},"compare":{"lam_apo":1}}' assigns VGM to <look> and APO to
<read>/<compare> (each on an SFT base). Globals apply where per_op is silent.
Every term is independently toggleable so SFT-only / APO-only / VGM-only / any
combination all run from this file. Launched in parallel via a Slurm array."""
import argparse, os, json, signal, random
import faulthandler
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from scsrc import paths
from scsrc.config import build_config, save_config
from scsrc.data import load_colorswap, build_specs, load_legible_ids, get_image
from scsrc.dataset import ObjectiveDataset
from scsrc.tokens import default_specs, add_neologisms
from scsrc import model as M
from scsrc.collate import collate
from scsrc.preference import objective_step, resolve_obj, uses

_STOP = {"flag": False}


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _obj_tag(g, per_op):
    base = [k for k in ("sft", "apo", "vgm", "tc", "ctm", "sm", "ecc", "aat", "drift") if g.get("lam_" + k, 0) > 0]
    if per_op:
        for v in per_op.values():
            for k in ("apo", "vgm", "tc"):
                if v.get("lam_" + k, 0) > 0 and k not in base:
                    base.append(k)
    return ("perop_" if per_op else "") + ("".join(sorted(base)) if base else "none")


def _to_dev(x, device):
    """Move every tensor in a (nested) training record to the device; records are built on CPU by the workers."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_dev(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_dev(v, device) for v in x)
    return x


def _worker_init(_wid):
    torch.set_num_threads(1)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--method", required=True, choices=["neo", "lora", "pt", "act"],
                    help="act (2026-10-19) = neologism rows + a request-conditioned vector added at one decoder layer on the requested evidence span (scsrc/act.py)")
    ap.add_argument("--act_layer", type=int, default=9, help="act: decoder layer whose output receives delta (0-based)")
    ap.add_argument("--act_span", default="content", choices=["content"], help="act: which positions of the requested source receive delta")
    ap.add_argument("--act_lr", type=float, default=None, help="act: learning rate of delta (default: the embedding lr)")
    ap.add_argument("--gcf_ctrl_pos", default="native", choices=["native", "first", "between", "pre_src", "post_src", "last"],
                    help="Ground-CF: TRAIN the control at this position (2026-10-19): first = before both evidence blocks, pre_src / post_src = right before / after the requested source's block, between = after the first block, last = after the question; recorded in config and restored at evaluation")
    ap.add_argument("--ladder_ctrl_pos", default="native", choices=["native", "prefix", "postq", "postcap"],
                    help="ladder (VCF/ColorSwap/COCO-CF): TRAIN the word at this position (scsrc/prompts.ladder_ctrl_pos); evaluate with scripts/eval_relocation.py")
    ap.add_argument("--arm", default="ctrl", choices=["ctrl", "gen", "fuse", "link", "na"])
    ap.add_argument("--n_tokens", type=int, default=4)
    ap.add_argument("--lam_ecc", type=float, default=0.0,
                    help="evidence-crop consistency (MVG vis): the full-frame margin "
                         "must match the margin achieved on the bbox crop. Trains "
                         "LOCALISATION, which the crop-eval identified as the visual "
                         "channel's ceiling (evidence = median 6% of the frame)")
    ap.add_argument("--compare_frac", type=float, default=0.0,
                    help="task=mvg: fraction of items rendered as 3-WAY COMPARE "
                         "items (all channels shown, verdict + abstention gold) "
                         "instead of single-channel items. Instantiates "
                         "compare_ctrl alongside the channel words.")
    ap.add_argument("--mask_ctrl_lead", type=int, default=0,
                    help="ctrl arms only: drop the neutral lead from the CE so the gradient goes to the VALUE (~87%% of ctrl loss is on the lead, which is identical for look and read by design). The 'final answer:' anchor stays supervised -- evaluate.py uses it as stop_anchor and scores trunc_* as its absence, so masking it would look like a learning failure.")
    ap.add_argument("--full_coverage_only", type=int, default=0,
                    help="task=mvraw: drop VIS items whose chunk has MORE "
                         "keyframes than --n_frames. Those items are subsampled, "
                         "so a visual claim occurring between shown frames is "
                         "unsupported by the input = label noise. OCR/ASR are "
                         "unaffected (their text spans every frame).")
    ap.add_argument("--n_frames", type=int, default=1,
                    help="task=mvg: present k video frames (a clip) as the visual "
                         "channel instead of one snapshot, centred on the item's "
                         "evidence timestamp -- makes the visual word grounded in "
                         "VIDEO rather than a sampled frame")
    ap.add_argument("--report_channel_words", type=int, default=0,
                    help="task=mvgreport: train the three CHANNEL words to emit "
                         "their own channel's [CH]-tagged claim lines (matched-"
                         "supervision composition baseline; eval with --composed_ckpt)")
    ap.add_argument("--report_with_channels", type=int, default=0,
                    help="task=mvgreport: also instantiate desc/ocr/vis_ctrl so the "
                         "report word can compose them (with --extend_ckpt they are "
                         "frozen = the MVG composition test)")
    ap.add_argument("--lora_r", type=int, default=None,
                    help="LoRA rank override (default 16 from config). Used for "
                         "the MATCHED-PARAMETER control: r=1 gives ~181k params, "
                         "comparable to a 30-token word (230k) -- if LoRA still "
                         "cannot flip, the gap is conditionality, not budget.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=None, help="checkpoint interval in steps (default: train.ckpt_every in the YAML)")
    ap.add_argument("--keep_opt_state", type=int, default=0, help="1: keep opt_state.pt after a run finishes (the resume-fidelity smoke resumes from a finished run)")
    ap.add_argument("--resume_state", default="auto", choices=["auto", "off"],
                    help="v1.300: with --resume_ckpt, auto restores the AdamW moments of the trained rows/adapter and the RNG states "
                         "saved in opt_state.pt (full-state resume); off re-initialises them (restarted continuation = pre-v1.300 behaviour)")
    ap.add_argument("--force", action="store_true", help="retrain even if final ckpt exists")
    ap.add_argument("--train_per_layer", type=int, default=0,
                    help="v1.157: also train the NEW tokens' per-layer embedding rows (Gemma-4 E4B; scsrc.model."
                         "train_per_layer_rows). Without it they stay the random rows resize_token_embeddings drew.")
    ap.add_argument("--word_first", type=int, default=0,
                    help="v1.165 control-position ablation (ladder): the word's instruction BEFORE the caption in every "
                         "SFT example (forces --cap_first_prob 0) and every CTM/SM pair (SC_WORD_FIRST=1). Grid key +wf.")
    ap.add_argument("--gcf_word_pos", default="native", choices=["native", "mix", "first"],
                    help="v1.167 Ground-CF mixed placement: 'mix' puts the word's sentence BEFORE the evidence in half the "
                         "rows (per-row coin shared by SFT and all pairs; scsrc.clevr3._POS_BIT). Grid key +pmix.")
    ap.add_argument("--model_id", default=None, help="HF model id; default scsrc.paths.MODEL_ID (Gemma-3-4B)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--pref_warmup", type=int, default=None,
                    help="steps of SFT-ONLY before APO/VGM ramp in (linear ramp over the "
                         "next warmup/2 steps). Fixes the early format-vs-preference fight: "
                         "the token first learns WHAT to say, then WHERE to ground it.")
    ap.add_argument("--disentangle_frac", type=float, default=0.0,
                    help="VCF: prob of rewriting a conflict caption to a THIRD "
                         "colour (breaks the prior<->caption confound)")
    ap.add_argument("--lookread_conflict_only", type=int, default=0,
                    help="1: train look/read ONLY on conflict items (agree items "
                         "give both tokens identical golds = anti-discriminative)")
    ap.add_argument("--cmp_order", default=None,
                    help="comma list overriding prompts.CMP_ORDER for blockscmp "
                         "targets, e.g. 'lrc,clr' for a 1:1 balance. Default None "
                         "keeps the banked 2:1 lrc bias. Recorded in config.json; "
                         "the 0.0 order_exact cells on compare-first prompts trace "
                         "directly to that 2:1 bias (order-following degrades where "
                         "the requested order is rare AND anti-causal).")
    ap.add_argument("--ctrl_site", default="user", choices=["user", "decision"],
                    help="ctrl token placement: user turn (standard) vs prefilled "
                         "at the head of the assistant turn (decision-site "
                         "leverage ablation; SFT-only)")
    ap.add_argument("--compose_frac", type=float, default=0.0,
                    help="Ground-CF only: fraction of training items replaced by a COMPOSED "
                         "request (two sources in one prompt, either order, both answers + "
                         "verdict). Ladder tasks use --ctrl_style compose instead.")
    ap.add_argument("--ctrl_style", default="neutral", choices=["neutral", "sourced", "blocks", "blockscmp", "markup", "compose", "request", "adapter"],
                    help="ctrl SFT gold style: neutral='Based on the requested source...' "
                         "(clean self-verbalization) | sourced='Based on the image/text...' "
                         "(natural self-attribution; verbalization becomes co-text)")
    ap.add_argument("--extend_ckpt", type=str, default=None,
                    help="ctrl arm: load these trained rows FROZEN; train only --add_word rows")
    ap.add_argument("--resume_ckpt", type=str, default=None,
                    help="v1.210 (author: finish a run stopped by the walltime): a run's ckpt/step_<n> directory (neo/pt rows or a LoRA adapter; v1.298); "
                         "loads its rows, advances the LR schedule and the data stream to step n and trains on to "
                         "--steps. The optimizer state is NOT restored (never saved): AdamW restarts at step n. Writes "
                         "config_resume.json; the run's original config.json is kept.")
    ap.add_argument("--train_ops", type=str, default=None,
                    help="comma list overriding the op expansion, e.g. "
                         "'look,read,compare,resolve' for joint 4-word runs")
    ap.add_argument("--add_word", type=str, default=None,
                    help="comma list of new ctrl words to append, e.g. 'resolve'")
    ap.add_argument("--delim_loss_weight", type=float, default=1.0,
                    help="CE weight on the delimiter-token positions in blocks/"
                         "link/blockscmp targets (1.0 = normal; 0.2 = format "
                         "supervision capped; 0 = delimiters as context only)")
    ap.add_argument("--gen_ckpt", default=None,
                    help="arm=link: path to a trained Arm-B neo.pt whose gen rows are loaded FROZEN")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--norm_clamp", type=float, default=None,
                    help="cap on the new rows' norm as a multiple of the mean old-row norm (default: train.norm_clamp in the YAML; 0 = off)")
    ap.add_argument("--lr_schedule", default=None, choices=[None, "constant", "cosine", "linear"],
                    help="peak-lr schedule; default from config (cosine)")
    ap.add_argument("--warmup", type=int, default=None,
                    help="linear warmup steps from 0 to peak lr; default from config")
    ap.add_argument("--min_lr_frac", type=float, default=None,
                    help="cosine/linear floor as a fraction of peak lr; default from config")
    ap.add_argument("--init", default=None, choices=[None, "word_mean", "random"])
    ap.add_argument("--lam_sft", type=float, default=None)
    ap.add_argument("--vds_weight", type=int, default=0, help="1 = VDS-weighted SFT (VGS-style)")
    ap.add_argument("--lam_iit", type=float, default=0.0,
                    help=">0: interleave symmetric interchange-intervention "
                         "pairs (exact twins) at every other micro-step; "
                         "neo/ctrl only")
    ap.add_argument("--iit_layer", type=int, default=12,
                    help="decoder layer whose word state is swapped (probe: "
                         "live band is 4-20 on Gemma-3-4B)")
    ap.add_argument("--lam_apo", type=float, default=None)
    ap.add_argument("--lam_vgm", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--lam_aat", type=float, default=None,
                    help="attention-alignment: CE(answer-position attention over "
                         "image patches || evidence bbox mask); clevr3 look cells")
    ap.add_argument("--aat_layer", type=int, default=12)
    ap.add_argument("--lam_sft_stay", type=float, default=None,
                    help="context-matched SFT control (2026-10-05, advisor): mean-token cross-entropy on the competing-edit "
                         "context (C_u', v) -- the Stay record's positive sequence -- with NO margin; CTM+Stay vs this isolates the margin")
    ap.add_argument("--lam_sft_twin", type=float, default=None,
                    help="paired-SFT control (2026-09-26, reviewers R2-Q1/R3-W1): plain mean-token cross-entropy on the "
                         "EDITED context with its own gold answer, (C', v') = the CTM record's b_pos. Same twins and "
                         "the same warm-up ramp as CTM, no cross-context coupling. Use with --lam_ctm 0 --cp_pairs 1.")
    ap.add_argument("--lam_ctm", type=float, default=None,
                    help="counterfactual-twin margin: symmetric two-sided source-swap "
                         "margin on exact twins (trains what flip/stay measures)")
    ap.add_argument("--cp_pairs", type=int, default=0,
                    help="conflict-preserving CTM/SM pairs on the ladder (2026-09-22): the objective's captions are "
                         "moved to third/fourth values so the two sources still disagree after every edit; sets "
                         "SC_CP_PAIRS=1 for scsrc.preference.cp_captions and is recorded in config.json (grid key +cp)")
    ap.add_argument("--lam_sm", type=float, default=None,
                    help="stay margin: the competing source is edited as the evaluator edits it and "
                         "the requested value must still beat the alternative (trains the Stay half "
                         "of the criterion; CTM trains the Flip half). 2026-09-21.")
    ap.add_argument("--ctrl_template_set", default="legacy", choices=["legacy", "srcsel"],
                    help="ladder control carriers: legacy (banked) or srcsel = source-selection sentences (2026-09-29)")
    ap.add_argument("--gcf_format", default="legacy", choices=["legacy", "v2"],
                    help="Ground-CF prompt: legacy (banked) or v2 = labelled Image/Caption/Question blocks, the control "
                         "on its own line, 'Return only the answer.' (2026-09-29)")
    ap.add_argument("--gcf_cont", default="leadin", choices=["leadin", "final", "bare"],
                    help="Ground-CF target/candidate format: leadin = the banked '... final answer: <v>'; final = 'Final answer: <v>' "
                         "(protocol-v2 primary); bare = '<v>'. One formatter (scsrc.clevr3.c3_cont) serves training and evaluation.")
    ap.add_argument("--selfcheck_items", type=int, default=0,
                    help="clevr3: after the final save, score K training items in-process (both requests, candidate "
                         "continuations) and write ckpt/step_<n>/selfcheck.json; scripts/smoke_v2_reload.py reloads and compares")
    ap.add_argument("--gcf_pt_carrier", type=int, default=0, choices=[0, 1],
                    help="prompt tuning only: 1 = matched-position control, the soft tokens in the word's carrier slot")
    ap.add_argument("--init_words", default="",
                    help="Ground-CF initialization-word override for BOTH embedding interfaces, e.g. look:look,read:read "
                         "(default: image / caption). Sets SC_C3_INIT_WORDS (2026-10-02 initialization ablation).")
    ap.add_argument("--exclude_scenes", default="", help="file of Ground-CF scene ids to drop from training (the dev partition)")
    ap.add_argument("--train_scenes", type=int, default=0, help="keep only the first N distinct training scenes (nested data-size subsets)")
    ap.add_argument("--mvg_sources", default="desc,ocr,vis",
                    help="MultiVENT-G evidence blocks and training channels: desc,ocr,vis (banked) or desc,vis (two-modality, "
                         "2026-10-02). Sets SC_MVG_SOURCES; recorded in config.json; restored at evaluation.")
    ap.add_argument("--save_step0", type=int, default=0, choices=[0, 1],
                    help="also save ckpt/step_0 right after initialization (untrained controls for the step-0 probes)")
    ap.add_argument("--pt_read_init", default="text", choices=["text", "caption"],
                    help="prompt tuning: initialization word of the caption-request prefix ('caption' matches the word)")
    ap.add_argument("--decay_scope", default="all", choices=["all", "new_rows"],
                    help="neo/pt: 'all' = AdamW decays the whole embedding matrix (banked; frozen rows shrink during "
                         "training, evaluation restores them); 'new_rows' = AdamW runs with weight_decay 0 and the "
                         "decoupled decay lr*wd is applied to the new rows only, after each step (2026-09-29)")
    ap.add_argument("--weight_decay", type=float, default=None,
                    help="AdamW weight decay; default keeps the historical 0.01. NOTE it applies to "
                         "the whole embedding matrix (grad-masked frozen rows included), see STATUS "
                         "2026-09-21 v1.45 F2; pass 0 for the control.")
    ap.add_argument("--ctm_pairs", default="data/mvg_ctm/pairs.jsonl",
                    help="twin-pair file for MVG CTM (use pairs_clean.jsonl -- "
                         "72%% of the original vis pairs are degenerate)")
    ap.add_argument("--aat_max_area", type=float, default=0.5,
                    help="skip AAT on boxes covering >= this fraction of the "
                         "frame (whole-scene annotations teach nothing)")
    ap.add_argument("--cert_weight", type=float, default=0.0,
                    help=">0 enables annotator-certainty loss weighting: "
                         "w = (certainty/5)^cert_weight per record")
    ap.add_argument("--mvg_pt_prefix", type=int, default=0,
                    help="MVG: 1 = standard prompt tuning (soft rows prepended to the user turn, no request sentence); "
                         "0 = the rows in the word's slot (the banked MVG prompt-tuning runs). Restored at evaluation.")
    ap.add_argument("--pt_inserted", type=int, default=0,
                    help="1 = prompt-tuning rows INSERTED at the word slot via "
                         "render_ctrl (sets PT_INSERTED=1 for collate). The "
                         "lexicality-vs-placement control: same params, same "
                         "position, no vocabulary identity.")
    ap.add_argument("--tie_rows", type=int, default=0,
                    help="1 = after every optimizer step, set each word's L rows "
                         "to their mean (projected gradient onto the tied "
                         "subspace). Same positions/attention queries as untied "
                         "L, ~one row of independent parameters -- the capacity "
                         "control that separates contextual WIDTH from raw "
                         "parameter count.")
    ap.add_argument("--lam_tc", type=float, default=None,
                    help="token-contrast margin: own-token vs other-token logprob of the SAME gold")
    ap.add_argument("--lam_drift", type=float, default=None,
                    help="hinge on ||e - e_init|| beyond rho_drift*||e_init|| (soft anchor; 0 = off)")
    ap.add_argument("--rho_drift", type=float, default=None)
    ap.add_argument("--cap_first_prob", type=float, default=1.0,
                    help="prob caption precedes image (rotation already default in ObjectiveDataset)")
    ap.add_argument("--slot_negative_frac", type=float, default=0.0,
                    help="fraction of examples with a filler word in the operator slot and unknown-operator gold")
    ap.add_argument("--absence_frac", type=float, default=0.0,
                    help="fraction of look/read SFT examples with the requested source REMOVED and refusal gold (honesty supervision)")
    ap.add_argument("--per_op", type=str, default=None,
                    help='compact JSON, e.g. {"look":{"lam_vgm":1,"lam_apo":0},"read":{"lam_apo":1}}')
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--task", default=None, choices=[None, "color", "counting", "spatial", "attr", "mvg", "mvgreport", "recheck", "clevr3", "mvraw", "mvrawcorr", "mvgtemporal"],
                    help="slot task; overrides cfg.task")
    ap.add_argument("--factor_mode", default="factored",
                    choices=["factored", "bundled", "srconly", "instructed", "noop"],  # instructed: English carrier (2026-09-21); noop: no request text -- LoRA adapter per request (v1.125)
                    help="clevr3: <src><op> shared tokens (factored) vs one fused "
                         "token per cell (bundled, param-matched at the cell) vs "
                         "source tokens only (srconly = Design A: the question "
                         "carries the skill)")
    ap.add_argument("--clevr3_absence", default="",
                    help="Ground-CF only: absence records from scripts/clevr3_absence.py (train split); "
                         "with --clevr3_absence_frac the word is also trained to ABSTAIN when its source "
                         "lacks the answer (2026-09-15)")
    ap.add_argument("--clevr3_absence_frac", type=float, default=0.0)
    ap.add_argument("--clevr3_image_twins", default="",
                    help="Ground-CF only: pairs file with exact image twins (scripts/clevr3_image_twins.py "
                         "--split train); when set, <look> receives the two sided objective on the image "
                         "edit instead of being skipped")
    ap.add_argument("--question_style", default="explicit",
                    choices=["explicit", "generic"],
                    help="clevr3: explicit questions state the skill (Design A); "
                         "generic questions are skill-free so the operator token "
                         "is the only skill carrier (Design B probe)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SC_WORKERS", "6")),   # 6 since 2026-09-22 18:20 (v1.99)
                    help="DataLoader workers that build the (image-processed) training records on CPU while the GPU trains "
                         "(STATUS v1.99: a CTM+Stay record costs ~7 s of CPU; with 0 workers the GPU waited for it every step)")
    ap.add_argument("--batch_terms", default=os.environ.get("SC_BATCH_TERMS", "0"),   # OFF since 2026-09-23 (v1.104): not numerically equivalent
                    help="auto|0|1: score the CTM quad / Stay pair / APO pair in one batched forward (STATUS v1.99); "
                         "auto = on when the card has >= 70 GB and SC_LOWMEM is not set")
    ap.add_argument("--c3_sources", default="look,read",   # v1.132 (author): OCR skipped everywhere by default
                    help="Ground-CF sources in the prompt, rows and items (STATUS v1.94): 'look,read' is the two-modality "
                         "protocol (no on-screen-text block, no src_ocr rows, no ocr items); recorded in config.json (grid +2s)")
    ap.add_argument("--held_cells", default="",
                    help="clevr3: comma list of held-out src:op cells excluded "
                         "from training, e.g. 'ocr:spatial,look:count' -> the "
                         "zero-shot composition test cells (factored only)")
    ap.add_argument("--data_dir", default=None,
                    help="processed task dir from prepare_task.py; if unset, uses ColorSwap")
    ap.add_argument("--neo_template", default="bare", choices=["bare", "hewitt", "srcsel"],
                    help="Ground-CF word carrier (2026-09-23, STATUS v1.123): bare = '<question> <tokens>' (banked runs); "
                         "hewitt = six rotated carrier templates after Hewitt et al., evaluated on two held-out ones. "
                         "Sets SC_NEO_TEMPLATE; recorded in config.json (grid key +tpl).")
    ap.add_argument("--instr_rotate", default="",
                    help="LoRA / English-carrier runs on Ground-CF: rotate the training instruction over these "
                         "INSTR variants per item, e.g. 0,2,4 (held-out evaluation uses 1 and 3). Sets SC_INSTR_ROTATE; "
                         "recorded in config.json (grid key +rot).")
    ap.add_argument("--caption_style", default="template", choices=["template", "fact"],
                    help="Ground-CF text channel (2026-09-23, STATUS v1.124): template = 'The answer to the question is v.' "
                         "(banked runs); fact = the fact itself ('The diamond is red.'). SC_CAPTION_STYLE; grid key +fc.")
    ap.add_argument("--q_rotate", type=int, default=0,
                    help="Ground-CF: rotate the three training question formulations per item (SC_Q_ROTATE=1); "
                         "evaluation uses the original (in-template) or the two held-out ones.")
    ap.add_argument("--train_sources", default="",
                    help="Ground-CF: train only these requests, e.g. look -- one modality-specific adapter per request "
                         "for the LoRA baseline (SC_TRAIN_SOURCES; grid key +ad, halves merged by the harvester).")
    args = ap.parse_args()
    # 2026-09-23 (v1.125): ctrl_style=adapter -- LoRA as ONE adapter per request (--train_sources look|read) with the
    # prefix's neutral task text and no request sentence; adapter identity selects the modality (ladder + Ground-CF).
    os.environ['SC_LORA_ADAPTER'] = '1' if (args.method == 'lora' and args.ctrl_style == 'adapter') else '0'
    if args.ctrl_style == 'adapter':
        assert args.method == 'lora' and args.train_sources, "--ctrl_style adapter needs --method lora and --train_sources look|read"
    os.environ['SC_CAPTION_STYLE'] = args.caption_style
    os.environ['SC_Q_ROTATE'] = str(int(bool(args.q_rotate)))
    if args.train_sources:
        os.environ['SC_TRAIN_SOURCES'] = args.train_sources
        print(f"[data] training requests restricted to {args.train_sources} (modality-specific adapter)")
    if args.caption_style != "template" or args.q_rotate:
        print(f"[prompt] caption_style={args.caption_style} q_rotate={int(bool(args.q_rotate))}")
    os.environ['SC_NEO_TEMPLATE'] = args.neo_template
    # 2026-09-29 protocol-v2 switches (all default to the banked behaviour; recorded in config.json via args and
    # restored by evaluate.apply_checkpoint): ladder carrier set, Ground-CF explicit format, answer format,
    # matched-position prompt tuning, weight-decay scope.
    os.environ['SC_CTRL_TEMPLATE_SET'] = args.ctrl_template_set
    os.environ['SC_GCF_FORMAT'] = args.gcf_format
    os.environ['SC_GCF_CONT'] = args.gcf_cont
    os.environ['SC_GCF_PT_CARRIER'] = str(args.gcf_pt_carrier)
    os.environ['SC_PT_READ_INIT'] = args.pt_read_init
    _is_act = (args.method == "act")          # 2026-10-19: the act method is the neo pipeline plus scsrc.act hooks
    if _is_act:
        args.method = "neo"
        print(f"[act] activation control: word rows + delta at decoder layer {args.act_layer} on the requested evidence span", flush=True)
    if args.gcf_ctrl_pos != "native":
        os.environ["SC_GCF_CTRL_POS"] = args.gcf_ctrl_pos; os.environ["SC_GCF_CTRL_ROTATE"] = "1"
        print(f"[prompt] Ground-CF control trained at position {args.gcf_ctrl_pos} (lead sentences rotate)", flush=True)
    if args.ladder_ctrl_pos != "native":
        os.environ["SC_LADDER_CTRL_POS"] = args.ladder_ctrl_pos
        print(f"[prompt] ladder word trained at position {args.ladder_ctrl_pos}", flush=True)
    os.environ['SC_C3_INIT_WORDS'] = args.init_words
    os.environ['SC_MVG_SOURCES'] = args.mvg_sources
    if args.mvg_sources != "desc,ocr,vis":
        print(f"[mvg] two-modality setting: sources={args.mvg_sources}")
    os.environ['SC_EXCLUDE_SCENES'] = args.exclude_scenes
    os.environ['SC_TRAIN_SCENES'] = str(args.train_scenes or 0)
    if args.init_words or args.exclude_scenes or args.train_scenes:
        print(f"[dev] init_words={args.init_words or '-'} exclude_scenes={args.exclude_scenes or '-'} train_scenes={args.train_scenes or '-'}")
    if (args.ctrl_template_set, args.gcf_format, args.gcf_cont, args.gcf_pt_carrier) != ("legacy", "legacy", "leadin", 0):
        print(f"[v2] ctrl_template_set={args.ctrl_template_set} gcf_format={args.gcf_format} gcf_cont={args.gcf_cont} "
              f"gcf_pt_carrier={args.gcf_pt_carrier} decay_scope={args.decay_scope}")
    if args.gcf_word_pos in ("mix", "first"):
        os.environ['SC_GCF_WORD_POS'] = args.gcf_word_pos
        print(f"[prompt] gcf_word_pos={args.gcf_word_pos}: the word's sentence precedes the evidence in "
              f"{'half the' if args.gcf_word_pos == 'mix' else 'every'} Ground-CF row(s)")
    if args.word_first:
        os.environ['SC_WORD_FIRST'] = '1'; args.cap_first_prob = 0.0
        print("[prompt] word_first: the word's instruction precedes the caption in SFT and in every pair")
    if args.instr_rotate:
        os.environ['SC_INSTR_ROTATE'] = args.instr_rotate
        print(f"[prompt] LoRA/English carrier rotates INSTR variants {args.instr_rotate} in training")
    if args.neo_template != "bare":
        print(f"[prompt] word carrier templates: {args.neo_template} (six rotated; held-out pair at evaluation)")
    if getattr(args, 'cp_pairs', 0):
        os.environ['SC_CP_PAIRS'] = '1'
    if getattr(args, 'c3_sources', None):
        os.environ['SC_C3_SOURCES'] = args.c3_sources
        print(f'[clevr3] source protocol: {args.c3_sources}')
        print('[pairs] conflict-preserving CTM/SM pairs ON (third/fourth-value captions)')
    if args.cmp_order:
        from scsrc import prompts as _Pm
        _Pm.CMP_ORDER = tuple(x.strip() for x in args.cmp_order.split(","))
        print(f"[obj] CMP_ORDER overridden -> {_Pm.CMP_ORDER}")
    if getattr(args, 'pt_inserted', 0):
        os.environ['PT_INSERTED'] = '1'
    os.environ['SC_MVG_PT_PREFIX'] = str(int(args.mvg_pt_prefix))

    cfg = build_config(args.config, {})
    steps = args.steps or cfg["train"]["steps"]
    lr = args.lr or cfg["train"]["lr"]
    _nc = args.norm_clamp if args.norm_clamp is not None else float(cfg["train"].get("norm_clamp", 2.0))
    accum = cfg["train"]["grad_accum"]; ckpt_every = args.ckpt_every or cfg["train"]["ckpt_every"]
    init_mode = args.init or cfg["train"]["init"]; device = "cuda"
    _seed(args.seed)
    # WATCHDOG: if any step hangs, dump all thread stacks every SC_WATCHDOG_SEC
    # seconds (default 300) so the log shows exactly WHERE it is stuck.
    wd = int(os.environ.get("SC_WATCHDOG_SEC", "300"))
    if wd > 0:
        # heartbeat watchdog: armed once here, RE-ARMED at the end of every
        # step (which resets the timer). It therefore fires ONLY if no step
        # completes within wd seconds -- a true stall -- instead of dumping
        # stacks on a schedule while training is healthy.
        faulthandler.dump_traceback_later(wd)

    oc = cfg["objective"]
    task = args.task or cfg.get("task", "color")
    g = {"vds_weight": args.vds_weight,
         "lam_sft": args.lam_sft if args.lam_sft is not None else oc["lam_sft"],
         "lam_apo": args.lam_apo if args.lam_apo is not None else oc["lam_apo"],
         "lam_vgm": args.lam_vgm if args.lam_vgm is not None else oc["lam_vgm"],
         "beta": args.beta if args.beta is not None else oc["beta"],
         "gamma": args.gamma if args.gamma is not None else oc["gamma"],
         "lam_tc": args.lam_tc if args.lam_tc is not None else oc.get("lam_tc", 0),
         "lam_ctm": args.lam_ctm if args.lam_ctm is not None else oc.get("lam_ctm", 0),
         "lam_sft_twin": args.lam_sft_twin if args.lam_sft_twin is not None else oc.get("lam_sft_twin", 0),
         "lam_sft_stay": args.lam_sft_stay if args.lam_sft_stay is not None else oc.get("lam_sft_stay", 0),
         "lam_sm": args.lam_sm if args.lam_sm is not None else oc.get("lam_sm", 0),
         "lam_ecc": args.lam_ecc if args.lam_ecc is not None else oc.get("lam_ecc", 0),  # was missing: every ctmecc run before 2026-10-06 trained WITHOUT the crop term
         "lam_aat": args.lam_aat if args.lam_aat is not None else oc.get("lam_aat", 0),
         "aat_layer": args.aat_layer,
         "lam_drift": args.lam_drift if args.lam_drift is not None else oc.get("lam_drift", 0),
         "rho_drift": args.rho_drift if args.rho_drift is not None else oc.get("rho_drift", 0.5)}
    per_op = json.loads(args.per_op.strip().strip(chr(39)).strip(chr(34))) if args.per_op else None
    # ensure per_op entries carry beta/gamma if they enable apo/vgm
    if per_op:
        for k, v in per_op.items():
            v.setdefault("lam_sft", g["lam_sft"]); v.setdefault("beta", g["beta"]); v.setdefault("gamma", g["gamma"])
    if args.method not in ("neo", "pt", "lora"):
        # 2026-09-14: pt admitted. Its pairs come from the same builders (dataset.py gates on
        # method in neo/pt/lora; preference._ctrl_user has a pt carrier) and its APO reference is
        # the prefix at initialisation, exactly as the neo reference is the word at initialisation
        # (ref_rows set in the pt branch below). Before this every "pt ... apo" run was SFT-only.
        if uses(g, per_op, "lam_apo") or uses(g, per_op, "lam_vgm"):
            print("[obj] APO/VGM only apply to method=neo/pt/lora; forcing off."); g["lam_apo"] = g["lam_vgm"] = 0.0; per_op = None
    elif args.method == "lora":
        # 2026-09-22 (STATUS v1.96): LoRA admitted for APO -- its reference model is the adapter-disabled base
        # model (scsrc.preference.reference_rows). VGM stays off for LoRA: its pair semantics are defined on the
        # learned rows, and it is an ablation the paper does not run for LoRA.
        if uses(g, per_op, "lam_vgm"):
            print("[obj] VGM does not apply to method=lora; forcing off."); g["lam_vgm"] = 0.0
        if uses(g, per_op, "lam_apo"):
            print("[obj] LoRA APO: reference = the adapter-disabled base model")


    _mslug = ("_" + args.model_id.split("/")[-1].lower().replace(".", "").replace("-", "")) if args.model_id else ""
    run_name = args.run_name or f"{task}_{args.method}_{args.arm}_{args.n_tokens}tok_{_obj_tag(g, per_op)}" + ("_src" if args.ctrl_style == "sourced" else "") + f"_seed{args.seed}{_mslug}"

    _final = os.path.join("runs", run_name, "ckpt",
                          f"step_{args.steps or 400}")
    if os.path.isdir(_final) and os.environ.get("SC_OVERWRITE") != "1":
        print(f"[guard] {run_name} already has a final checkpoint at {_final} -- refusing to retrain.")
        print("[guard] set SC_OVERWRITE=1 in the submission environment to retrain deliberately.")
        return
    out = os.path.join(paths.RUNS, run_name)
    os.makedirs(out, exist_ok=True)
    _start_step = 0
    if args.resume_ckpt:
        assert args.method in ("neo", "pt", "lora"), "--resume_ckpt: neo/pt rows (neo.pt) or a LoRA adapter dir"
        _start_step = int(json.load(open(os.path.join(args.resume_ckpt, "meta.json")))["step"])
        assert 0 < _start_step < steps, (_start_step, steps)
    save_config({**cfg, "args": vars(args), "obj_global": g, "per_op": per_op, **({"act": {"layer": args.act_layer, "span": args.act_span}} if _is_act else {})},
                os.path.join(out, "config_resume.json" if args.resume_ckpt else "config.json"))

    def _on_usr1(signum, frame):
        _STOP["flag"] = True; print("[signal] SIGUSR1 -> checkpoint & exit")
    signal.signal(signal.SIGUSR1, _on_usr1)

    task = args.task or cfg.get("task", "color")
    data_dir = args.data_dir or cfg["data"].get("dir")
    if task == "recheck":
        # standalone task: RecheckDataset builds triples from twin pairs itself
        from scsrc.data import load_task
        ds = load_task(data_dir or "data/vcf", "train"); legible = None; specs = None
        print(f"[data] task=recheck from {data_dir or 'data/vcf'} (twin pairs)")
    elif task == "mvg":
        ds = None; legible = None; specs = None
        print(f"[data] task=mvg from {data_dir} (pre-materialized items)")
    elif task == "mvgtemporal":
        ds = None; legible = None; specs = None
        print("[data] task=mvgtemporal (<before>/<after>: position on the time axis)")
    elif task == "mvrawcorr":
        ds = None; legible = None; specs = None
        print("[data] task=mvrawcorr (<corroborate>: claim -> attesting sources)")
    elif task == "mvraw":
        ds = None; legible = None; specs = None
        print(f"[data] task=mvraw from {data_dir or 'data/multivent_raw'} "
              "(pre-materialized chunk items)")
    elif task == "mvgreport":
        ds = None; legible = None; specs = None
        print("[data] task=mvgreport (per-video report items, video-level split "
              "inherited from data/mvg train)")
    elif task == "clevr3":
        ds = None; legible = None; specs = None
        print(f"[data] task=clevr3 from {data_dir or 'data/clevr3'} "
              f"(mode={args.factor_mode}, held={args.held_cells or 'none'})")
    elif data_dir:
        from scsrc.data import load_task
        if not os.path.isdir(os.path.join(data_dir, "train")):
            raise SystemExit(
                f"[train] data_dir '{data_dir}' has no train/ split.\n"
                "  - If the path starts with '/data/...', the manifest was generated with $SC unset.\n"
                "  - Fix: regenerate manifests with RELATIVE paths (no $SC needed):\n"
                '      sed "s#TASK#<task>#g; s#DATADIR#data/<task>#g" configs/manifest_template.txt > configs/manifest_<task>.txt\n'
                "    (train_array.sbatch cd's into the project, so 'data/<task>' always resolves.)")
        ds = load_task(data_dir, "train")
        legible = None   # legibility gate is ColorSwap-specific; processed tasks are pre-filtered
        print(f"[data] task={task} from {data_dir}/train ({len(ds)} examples)")
    else:
        ds = load_colorswap("train", human_only=cfg["data"]["human_only"])
        legible = load_legible_ids("train") if cfg["data"]["use_legibility_filter"] else None
    if task not in ("mvg", "mvgreport", "recheck", "clevr3", "mvraw", "mvrawcorr", "mvgtemporal"):
        specs = build_specs(ds, "train", legible_ids=legible, conditions=("agree", "conflict"))
    held_cells = tuple(tuple(c.split(":")) for c in args.held_cells.split(",") if ":" in c)
    if os.environ.get("SC_PREFLIGHT") == "parse":
        print(f"[preflight-parse] OK run={args.run_name} method={args.method} task={task} steps={args.steps}"); raise SystemExit(0)
    if os.environ.get("SC_PREFLIGHT") == "1" and task != "clevr3" and task in ("color", "counting", "spatial", "attr"):
        # v1.125: ladder preflight -- processor only; the same ObjectiveDataset the run builds (ops as below).
        from transformers import AutoProcessor
        from scsrc import paths as _paths
        # (ObjectiveDataset is imported at module level; a local import here made the name local to main() and
        #  crashed every non-preflight ladder run at the dataset build -- UnboundLocalError, 2026-09-23 16:20-23:17)
        _mid = args.model_id or _paths.MODEL_ID
        _mid = M._resolve_cached(_mid) if hasattr(M, "_resolve_cached") else _mid
        processor = AutoProcessor.from_pretrained(_mid)
        _pf = ObjectiveDataset(ds, specs, processor, args.method, args.arm, args.n_tokens, "cpu", g, per_op, task=task,
                               ctrl_style=args.ctrl_style, absence_frac=args.absence_frac, cap_first_prob=args.cap_first_prob,
                               slot_negative_frac=args.slot_negative_frac, ops=None, ctrl_site=args.ctrl_site,
                               lookread_conflict_only=bool(args.lookread_conflict_only), disentangle_frac=args.disentangle_frac)
        _ops_seen = sorted({str(op) for _, op in _pf.items})
        _want = [k for k, on in (("ctm", g.get("lam_ctm", 0) > 0 or g.get("lam_sft_twin", 0) > 0), ("apo", g.get("lam_apo", 0) > 0), ("sm", g.get("lam_sm", 0) > 0 or g.get("lam_sft_stay", 0) > 0)) if on]
        _got = {k: 0 for k in _want}; _n = 0; _txt = None
        for _i in __import__("random").Random(0).sample(range(len(_pf)), min(60, len(_pf))):   # random: a fixed stride can alias the item blocks
            _r = _pf[_i]; _n += 1
            assert _r.get("sft") is not None, f"no sft record at item {_i}"
            for k in _want:
                _got[k] += int(_r.get(k) is not None)
            if _txt is None:
                _txt = processor.tokenizer.decode(_r["sft"]["input_ids"][0][-70:])
        _miss = [k for k in _want if _got[k] == 0]
        print(f"[preflight] {'OK' if not _miss else 'FAIL'} run={args.run_name} items={len(_pf)} ops={_ops_seen} sampled={_n} pairs={_got}"
              + (f" MISSING={_miss}" if _miss else ""))
        print(f"[preflight] tail: {_txt!r}")
        raise SystemExit(3 if _miss else 0)
    if os.environ.get("SC_PREFLIGHT") == "1" and task == "clevr3":
        # 2026-09-23 (v1.125): CPU preflight -- processor only, no weights. Builds the SAME Clevr3Dataset the run
        # would build (same kwargs as below) and renders/tokenises real SFT and pair records, so a prompt-path error
        # surfaces before a GPU job starts. Learned tokens are not added (they tokenise as plain text here).
        from transformers import AutoProcessor
        from scsrc import paths as _paths
        from scsrc.clevr3 import Clevr3Dataset
        _mid = args.model_id or _paths.MODEL_ID
        _mid = M._resolve_cached(_mid) if hasattr(M, "_resolve_cached") else _mid
        processor = AutoProcessor.from_pretrained(_mid)
        _pf = Clevr3Dataset(data_dir or "data/clevr3", "train", processor, "cpu", n_tokens=args.n_tokens,
                            method=args.method, compose_frac=args.compose_frac, mode=args.factor_mode,
                            held_cells=held_cells, slot_negative_frac=args.slot_negative_frac,
                            question_style=args.question_style, aat=(g.get("lam_aat", 0) > 0),
                            apo=(g.get("lam_apo", 0) > 0), vgm=(g.get("lam_vgm", 0) > 0),
                            ctm=(g.get("lam_ctm", 0) > 0 or g.get("lam_sft_twin", 0) > 0), sm=(g.get("lam_sm", 0) > 0 or g.get("lam_sft_stay", 0) > 0), tc=(g.get("lam_tc", 0) > 0),
                            image_twins=(args.clevr3_image_twins or None), absence=(args.clevr3_absence or None),
                            absence_frac=args.clevr3_absence_frac)
        _srcs = sorted({src for _, src in _pf.items})
        _want = [k for k, on in (("ctm", g.get("lam_ctm", 0) > 0 or g.get("lam_sft_twin", 0) > 0), ("apo", g.get("lam_apo", 0) > 0), ("sm", g.get("lam_sm", 0) > 0 or g.get("lam_sft_stay", 0) > 0)) if on]
        _got = {k: 0 for k in _want}; _n = 0; _txt = None
        for _i in __import__("random").Random(0).sample(range(len(_pf)), min(60, len(_pf))):   # random: a fixed stride can alias the item blocks
            _r = _pf[_i]; _n += 1
            assert _r.get("sft") is not None, f"no sft record at item {_i}"
            for k in _want:
                _got[k] += int(_r.get(k) is not None)
            if _txt is None:
                try:
                    _txt = processor.tokenizer.decode(_r["sft"]["input_ids"][0][-90:])
                except Exception as _e:
                    _txt = f"(decode n/a: {type(_e).__name__})"
        _miss = [k for k in _want if _got[k] == 0]
        print(f"[preflight] {'OK' if not _miss else 'FAIL'} run={args.run_name} items={len(_pf)} requests={_srcs} "
              f"sampled={_n} pairs={_got}" + (f" MISSING={_miss}" if _miss else ""))
        print(f"[preflight] tail: {_txt!r}")
        raise SystemExit(3 if _miss else 0)

    model, processor = M.load(model_id=args.model_id, attn=cfg["model"]["attn"])
    trainable = None; name_to_ids = all_new_ids = None; ref_rows = None
    _act_ctrl = None
    _pleP, _ple_sync, _ple_init = None, None, None   # v1.157: E4B per-layer rows (--train_per_layer)
    if args.method == "neo":
        if task == "mvg":
            from scsrc.tokens import NeoSpec
            _specs = [NeoSpec("desc_ctrl", "ctrl", "text", "description"),
                      NeoSpec("ocr_ctrl", "ctrl", "text", "screen"),
                      NeoSpec("vis_ctrl", "ctrl", "image", "vision")]
            if args.compare_frac > 0:
                _specs.append(NeoSpec("compare_ctrl", "ctrl", "relation", "compare"))
        elif task == "mvgtemporal":
            from scsrc.tokens import NeoSpec
            _specs = [NeoSpec("before_ctrl", "ctrl", "relation", "earlier"),
                      NeoSpec("after_ctrl", "ctrl", "relation", "later")]
            if args.report_with_channels:
                # COMPOSITION: the visual source word alongside the temporal
                # words, so <vis><before> can be probed. Only vis is added --
                # desc/ocr would narrate the order and leak the answer.
                _specs = [NeoSpec("vis_ctrl", "ctrl", "image", "vision")] + _specs
        elif task == "mvrawcorr":
            from scsrc.tokens import NeoSpec
            _specs = [NeoSpec("corroborate_ctrl", "ctrl", "relation", "corroborate")]
        elif task == "mvraw":
            from scsrc.tokens import NeoSpec
            _specs = [NeoSpec("asr_ctrl", "ctrl", "text", "speech"),
                      NeoSpec("ocr_ctrl", "ctrl", "text", "screen"),
                      NeoSpec("vis_ctrl", "ctrl", "image", "vision")]
        elif task == "clevr3":
            from scsrc.tokens import clevr3_specs
            _specs = clevr3_specs(args.factor_mode, held_cells)
        elif task == "mvgreport":
            from scsrc.tokens import NeoSpec
            _specs = [NeoSpec("report_ctrl", "ctrl", "relation", "report")]
            if args.report_channel_words:
                # matched-supervision composition: NO report word; the three
                # channel words each learn to emit their own channel's lines
                _specs = [NeoSpec("desc_ctrl", "ctrl", "text", "description"),
                          NeoSpec("ocr_ctrl", "ctrl", "text", "screen"),
                          NeoSpec("vis_ctrl", "ctrl", "image", "vision")]
            elif args.extend_ckpt or args.report_with_channels:
                # the three stage-1 channel words are present too, so the report
                # word can COMPOSE them rather than relearn grounding alone.
                # --extend_ckpt loads them FROZEN (link-style composition test);
                # without it they co-train (joint stage-1+stage-2).
                _specs = [NeoSpec("desc_ctrl", "ctrl", "text", "description"),
                          NeoSpec("ocr_ctrl", "ctrl", "text", "screen"),
                          NeoSpec("vis_ctrl", "ctrl", "image", "vision")] + _specs
        elif task == "recheck":
            from scsrc.tokens import NeoSpec
            _specs = [NeoSpec("recheck_ctrl", "ctrl", "image", "recheck")]
        else:
            # ctrl_style=blocks needs the GEN words too: they are the citation
            # delimiters emitted in the ANSWER, and they must be learned rows
            # (not literal markup) so the delimiter itself carries provenance.
            if args.arm == "ctrl" and args.ctrl_style in ("blocks", "blockscmp"):
                _specs = default_specs()["ctrl"] + default_specs()["cite"]
            elif args.arm == "link":
                _specs = default_specs()["ctrl"] + default_specs()["gen"]
            else:
                _specs = default_specs()[args.arm]
        if args.add_word:
            from scsrc.tokens import NeoSpec
            for w in args.add_word.split(","):
                w = w.strip()
                _specs = _specs + [NeoSpec(f"{w}_ctrl", "ctrl", "relation", w)]
        specs_by_name = {s.name: s for s in _specs}
        # bundled clevr3 fuses two slots into one token -> 2x rows per token so
        # it is param-matched with the factored <src>(L)+<op>(L)=2L cell footprint
        _nt = (2 * args.n_tokens
               if (task == "clevr3" and args.factor_mode == "bundled")
               else args.n_tokens)
        _n_by_name = ({s.name: 1 for s in _specs if s.role in ("gen", "cite")}
                      if (args.arm == "ctrl" and args.ctrl_style == "blocks") else None)
        name_to_ids, all_new_ids = add_neologisms(processor.tokenizer, _specs, _nt,
                                                  n_by_name=_n_by_name)
        # link: gradients flow ONLY into the ctrl rows; gen rows stay frozen at
        # the pretrained Arm-B values (loaded below), so ctrl learns to INVOKE
        # the existing block symbols rather than co-adapting them.
        trainable_ids = ([i for nm, ids in name_to_ids.items() if nm.endswith("_ctrl") for i in ids]
                         if args.arm == "link" else all_new_ids)
        if args.extend_ckpt:
            new_names = {f"{w.strip()}_ctrl" for w in (args.add_word or "").split(",") if w.strip()}
            assert new_names, "--extend_ckpt requires --add_word"
            trainable_ids = [i for nm, ids in name_to_ids.items() if nm in new_names for i in ids]
            print(f"[extend] training ONLY {sorted(new_names)}; prior rows frozen from {args.extend_ckpt}")
        W = M.setup_neologisms(model, processor.tokenizer, name_to_ids, trainable_ids, seed=args.seed)
        M.init_neologisms(model, processor.tokenizer, name_to_ids, specs_by_name, mode=init_mode, seed=args.seed)
        if args.arm == "link" and args.gen_ckpt:
            M.load_neologisms_by_name(model, args.gen_ckpt, name_to_ids)
        if args.extend_ckpt:
            M.load_neologisms_by_name(model, args.extend_ckpt, name_to_ids)
        elif args.arm == "link":
            print("[link] WARNING: no --gen_ckpt given; gen rows are word-mean init, "
                  "the linked target blocks use UNTRAINED gen symbols")
        trainable = [W]
        if args.train_per_layer:
            _pleP, _ple_sync = M.train_per_layer_rows(model, processor.tokenizer, name_to_ids, trainable_ids,
                                                      specs_by_name, mode=init_mode, seed=args.seed)
            if _pleP is None:
                raise SystemExit("[STOP] --train_per_layer 1 needs a model with a per-layer embedding table (Gemma-4 E4B)")
            trainable.append(_pleP); _ple_init = _pleP.data.float().clone()
        if _is_act:
            import scsrc.act as A
            _act_ctrl = A.ActControl(W.shape[1]).to(device)
            A.install(model, _act_ctrl, args.act_layer, name_to_ids, processor.tokenizer, A.image_token_id(model, processor.tokenizer))
            trainable.append(_act_ctrl.delta)
            print(f"[act] delta (2 x {W.shape[1]}) zero-init at {A.STATE['layer_path']}; image token id {A.STATE['img_id']}", flush=True)
        if uses(g, per_op, "lam_apo") or uses(g, per_op, "lam_drift") or uses(g, per_op, "lam_tc"):
            ref_rows = W.data[all_new_ids].detach().clone()
    elif args.method == "lora":
        model = M.setup_lora(model, r=(args.lora_r or cfg["lora"]["r"]), alpha=cfg["lora"]["alpha"])
        trainable = [p for p in model.parameters() if p.requires_grad]
    elif args.method == "pt":
        # prompt-tuning baseline as a learned soft PREFIX via added-vocab rows:
        # identical training mechanism to neo (add rows, grad-isolate, init, save)
        # so the forward path is the one smoke_test.py already validated -- the
        # ONLY difference from neo is placement (prepended prefix vs addressable
        # inline operator). Plain SFT (the standard prompt-tuning objective).
        _pt = default_specs()["pt"]
        if task == "mvgtemporal":
            from scsrc.tokens import NeoSpec
            _pt = [NeoSpec("pt_before", "pt", "relation", "earlier"),
                   NeoSpec("pt_after", "pt", "relation", "later")]
        elif task == "mvraw":
            from scsrc.tokens import NeoSpec
            _pt = [NeoSpec("pt_asr", "pt", "text", "speech"),
                   NeoSpec("pt_ocr", "pt", "text", "screen"),
                   NeoSpec("pt_vis", "pt", "image", "vision")]
        elif task == "mvg":
            from scsrc.tokens import NeoSpec
            _pt = [NeoSpec("pt_desc", "pt", "text", "description"),
                   NeoSpec("pt_ocr", "pt", "text", "screen"),
                   NeoSpec("pt_vis", "pt", "image", "vision")]
        elif task == "clevr3":
            # Ground-CF has three sources; the default pt specs cover only
            # look/read/compare, so <ocr> had no prefix row (STATUS 2026-09-11).
            from scsrc.tokens import NeoSpec
            if args.gcf_format == "v2" or args.pt_read_init == "caption":
                # 2026-09-29 (audit item 2): prefix rows for the ACTIVE sources only, initialised like the word
                # (look <- "image", read <- --pt_read_init; caption = matched), so both embedding interfaces register
                # the same number of control rows from the same initialization words
                from scsrc.tokens import c3_sources as _c3s, clevr3_specs as _c3specs
                _init = {"look": "image", "read": args.pt_read_init, "ocr": "screen"}
                _init.update(dict(kv.split(":", 1) for kv in args.init_words.split(",") if ":" in kv))   # matched override
                _modal = {"look": "image", "read": "text", "ocr": "text"}
                _pt = [NeoSpec(f"pt_{s}", "pt", _modal[s], _init[s]) for s in _c3s()]
                _word = _c3specs("srconly", held_cells)
                assert [p.init_word for p in _pt] == [w.init_word for w in _word] or args.pt_read_init != "caption", \
                    ("prefix/word initialization words differ", [p.init_word for p in _pt], [w.init_word for w in _word])
                assert len(_pt) == len(_word), ("prefix/word control-row counts differ", len(_pt), len(_word))
                print(f"[pt] Ground-CF prefix rows for the active sources: " + ", ".join(f"{p.name}<-{p.init_word}" for p in _pt)
                      + f" ({len(_pt)} controls x {args.n_tokens} rows; the word registers {len(_word)} x {args.n_tokens})")
            else:
                _pt = [NeoSpec("pt_look", "pt", "image", "image"),
                       NeoSpec("pt_read", "pt", "text", "text"),
                       NeoSpec("pt_ocr", "pt", "text", "screen")]
        specs_by_name = {s.name: s for s in _pt}
        name_to_ids, all_new_ids = add_neologisms(processor.tokenizer, _pt, args.n_tokens)
        W = M.setup_neologisms(model, processor.tokenizer, name_to_ids, all_new_ids, seed=args.seed)
        M.init_neologisms(model, processor.tokenizer, name_to_ids, specs_by_name, mode=init_mode, seed=args.seed)
        trainable = [W]
        if args.train_per_layer:
            _pleP, _ple_sync = M.train_per_layer_rows(model, processor.tokenizer, name_to_ids, all_new_ids,
                                                      specs_by_name, mode=init_mode, seed=args.seed)
            if _pleP is None:
                raise SystemExit("[STOP] --train_per_layer 1 needs a model with a per-layer embedding table (Gemma-4 E4B)")
            trainable.append(_pleP); _ple_init = _pleP.data.float().clone()
        if uses(g, per_op, "lam_apo") or uses(g, per_op, "lam_drift") or uses(g, per_op, "lam_tc"):
            ref_rows = W.data[all_new_ids].detach().clone()   # the prefix at init is APO's reference

    # reference norm for the new-row clamp (None disables clamping, e.g. lora)
    _norm_ref = None
    if all_new_ids:
        with torch.no_grad():
            _W = model.get_input_embeddings().weight.data
            _mask = torch.ones(_W.shape[0], dtype=torch.bool, device=_W.device)
            _mask[all_new_ids] = False
            _norm_ref = _W[_mask][:20000].float().norm(dim=-1).mean().item()
        print(f"[norm] mean old-row norm ~ {_norm_ref:.3f}; clamping new rows to "
              f"{_nc}x after each step")

    _wd = args.weight_decay if args.weight_decay is not None else float(cfg["train"].get("weight_decay", 0.01))
    _decay_new_rows_only = (args.decay_scope == "new_rows" and args.method in ("neo", "pt"))
    _wd_opt = 0.0 if _decay_new_rows_only else _wd
    print(f"[optim] AdamW lr={lr} weight_decay={_wd_opt}" + (
        f" in the optimizer; decoupled decay {_wd} applied to the {len(all_new_ids)} new rows only after each step"
        if _decay_new_rows_only else " (applies to the full embedding matrix, masked rows included)"))
    if args.resume_ckpt:     # v1.210: the rows trained to step n (after init, so they overwrite it); v1.298: pt rows and LoRA adapters too
        if args.method in ("neo", "pt"):
            M.load_neologisms_by_name(model, os.path.join(args.resume_ckpt, "neo.pt"), name_to_ids)
        else:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file as _sft_load
            _res = set_peft_model_state_dict(model, _sft_load(os.path.join(args.resume_ckpt, "adapter_model.safetensors")))
            _unexp = list(getattr(_res, "unexpected_keys", []) or [])
            assert not _unexp, f"--resume_ckpt: unexpected adapter keys {_unexp[:5]}"
        print(f"[resume] {'rows' if args.method in ('neo', 'pt') else 'LoRA adapter'} loaded from {args.resume_ckpt}; "
              f"continuing at step {_start_step} of {steps}", flush=True)
    if _is_act and _act_ctrl is not None:
        opt = torch.optim.AdamW([{"params": [p for p in trainable if p is not _act_ctrl.delta], "lr": lr},
                                 {"params": [_act_ctrl.delta], "lr": (args.act_lr or lr)}], lr=lr, weight_decay=_wd_opt)
    else:
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=_wd_opt)
    # LR schedule: linear warmup -> cosine (or linear) decay to a floor.
    # Warmup tames the early phase where SFT and APO fight over shared tokens;
    # decay stabilizes the endpoint the checkpoints are evaluated at.
    sched_cfg = cfg["train"].get("lr_schedule", "cosine")
    schedule = args.lr_schedule or sched_cfg
    warmup = args.warmup if args.warmup is not None else int(cfg["train"].get("warmup", max(1, steps // 20)))
    min_frac = args.min_lr_frac if args.min_lr_frac is not None else float(cfg["train"].get("min_lr_frac", 0.1))

    def _lr_lambda(step_i):
        if schedule == "constant":
            return 1.0
        if step_i < warmup:
            return (step_i + 1) / max(warmup, 1)
        t = (step_i - warmup) / max(steps - warmup, 1)
        t = min(max(t, 0.0), 1.0)
        if schedule == "linear":
            return (1 - t) * (1 - min_frac) + min_frac
        return min_frac + 0.5 * (1 - min_frac) * (1 + math.cos(math.pi * t))  # cosine

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    for _ in range(_start_step):       # v1.210 resume: the schedule continues where the stopped run left it
        sched.step()
    _resume_state_loaded = False; _ost = None
    if args.resume_ckpt and args.resume_state == "auto":   # v1.300: full-state resume (advisor gate, 2026-10-07)
        _osp = os.path.join(args.resume_ckpt, "opt_state.pt")
        if os.path.exists(_osp):
            _ost = torch.load(_osp, map_location="cpu", weights_only=False)
            if args.method in ("neo", "pt"):
                for _p, _key in [(W, "W")] + ([(_pleP, "ple")] if _pleP is not None else []) + ([(_act_ctrl.delta, "act")] if _act_ctrl is not None else []):
                    if _key not in _ost["rows"]:
                        continue
                    _r = _ost["rows"][_key]
                    _ea = torch.zeros_like(_p); _es = torch.zeros_like(_p)
                    if _r.get("ids") is None:                                   # whole-tensor state (per-layer P)
                        _ea.copy_(_r["exp_avg"].to(_p.device, _p.dtype)); _es.copy_(_r["exp_avg_sq"].to(_p.device, _p.dtype))
                    else:
                        _ids = torch.as_tensor(_r["ids"], device=_p.device)
                        _ea[_ids] = _r["exp_avg"].to(_p.device, _p.dtype); _es[_ids] = _r["exp_avg_sq"].to(_p.device, _p.dtype)
                    opt.state[_p] = {"step": torch.tensor(float(_ost["opt_step"]), dtype=torch.float32), "exp_avg": _ea, "exp_avg_sq": _es}
            else:
                opt.load_state_dict(_ost["lora"])
            _resume_state_loaded = True
            print(f"[resume] optimizer moments restored from {_osp} (optimizer step {_ost['opt_step']}); RNG states follow the data skip", flush=True)
        else:
            print(f"[resume] WARNING: {_osp} missing -> fresh AdamW moments: this is a RESTARTED CONTINUATION", flush=True)
    elif args.resume_ckpt:
        print("[resume] --resume_state off -> fresh AdamW moments: RESTARTED CONTINUATION", flush=True)
    if args.resume_ckpt:
        with open(os.path.join(out, "resume_note.json"), "w") as _f:
            json.dump({"from_step": _start_step, "resume_ckpt": args.resume_ckpt,
                       "resume_state": "full" if _resume_state_loaded else "restarted"}, _f)
    if args.delim_loss_weight != 1.0:
        # delimiter ids: learned cite/gen rows if present, else the literal tags
        did = set()
        for w_ in ("<look>", "<read>", "<compare>"):
            did.update(processor.tokenizer(w_, add_special_tokens=False)["input_ids"])
        for nm, ids_ in (name_to_ids or {}).items():
            if nm in ("look", "read", "compare") or nm.endswith("_gen"):
                did.update(ids_)
        g["delim_w"] = args.delim_loss_weight
        g["delim_ids"] = sorted(did)
        print(f"[delim] CE weight {args.delim_loss_weight} on {len(did)} delimiter ids")
    _ops = None
    if args.ctrl_style == "compose" and args.method in ("neo", "pt"):
        # composed items on top of the single requests; LoRA is handled in ObjectiveDataset
        _ops = tuple(_ops or ("look", "read")) + ("compose",)
    if args.train_ops:
        _ops = tuple(w.strip() for w in args.train_ops.split(",") if w.strip())
        print(f"[ops] dataset ops set to {_ops}")
    elif args.extend_ckpt and args.add_word:
        _ops = tuple(w.strip() for w in args.add_word.split(",") if w.strip())
        print(f"[extend] dataset ops restricted to {_ops}")
    # ---- throughput (STATUS v1.99): records built on CPU by workers; batched margin terms on 80 GB cards
    _ds_dev = "cpu" if args.workers > 0 else device
    _dl_kw = (dict(num_workers=args.workers, persistent_workers=True, prefetch_factor=2, worker_init_fn=_worker_init)
              if args.workers > 0 else {})
    if args.workers > 0:
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")   # tensors via files, not /dev/shm
        except Exception:
            pass
    if str(args.batch_terms) == "auto":
        try:
            _big = torch.cuda.get_device_properties(0).total_memory >= 70e9
        except Exception:
            _big = False
        _bt = "1" if (_big and os.environ.get("SC_LOWMEM", "") != "1") else "0"
    else:
        _bt = "1" if str(args.batch_terms) == "1" else "0"
    os.environ["SC_BATCH_TERMS"] = _bt
    print(f"[throughput] workers={args.workers} (records on {_ds_dev}); batched margin terms={'on' if _bt == '1' else 'off'}")
    if task == "recheck":
        from scsrc.recheck import RecheckDataset
        dataset = RecheckDataset(data_dir or "data/vcf", "train", processor,
                                 device, args.n_tokens,
                                 slot_negative_frac=args.slot_negative_frac)
        loader = DataLoader(dataset, batch_size=1,
                            sampler=WeightedRandomSampler(dataset.weights, len(dataset)),
                            collate_fn=collate)
        print(f"[data] recheck train triples: {len(dataset)}")
    elif task == "mvg":
        from scsrc.mvg import MvgDataset
        dataset = MvgDataset(data_dir or "data/mvg", "train", processor,
                             _ds_dev, n_tokens=args.n_tokens,
                             method=args.method,
                             n_frames=args.n_frames,
                             ecc=(g.get("lam_ecc", 0) > 0),
                             compare_frac=args.compare_frac,
                             apo=(g.get("lam_apo", 0) > 0),
                             vgm=(g.get("lam_vgm", 0) > 0),
                             tc=(g.get("lam_tc", 0) > 0),
                             slot_negative_frac=args.slot_negative_frac,
                             ctm_pairs=(args.ctm_pairs
                                        if (g.get("lam_ctm", 0) > 0 or g.get("lam_sm", 0) > 0 or g.get("lam_sft_twin", 0) > 0 or g.get("lam_sft_stay", 0) > 0) else None),
                             sm=(g.get("lam_sm", 0) > 0 or g.get("lam_sft_stay", 0) > 0),
                             aat=(g.get("lam_aat", 0) > 0),
                             aat_max_area=args.aat_max_area,
                             cert_weight=args.cert_weight)
        sampler = WeightedRandomSampler(dataset.weights, len(dataset))
        loader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            collate_fn=collate, **_dl_kw)
        print(f"[data] mvg train items: {len(dataset)} (weighted sampling)")
    elif task == "mvgtemporal":
        from scsrc.temporal import TemporalDataset
        dataset = TemporalDataset(data_dir or "data/mvg_temporal", "train",
                                  processor, device, n_tokens=args.n_tokens,
                                  n_frames=args.n_frames,
                                  style=("pt" if args.method == "pt"
                                         else "instr" if args.method == "lora"
                                         else "neo"),
                                  apo=(g.get("lam_apo", 0) > 0),
                                  tc=(g.get("lam_tc", 0) > 0),
                                  ctm=(g.get("lam_ctm", 0) > 0 or g.get("lam_sft_twin", 0) > 0), ops=_ops)
        loader = DataLoader(dataset, batch_size=1,
                            sampler=WeightedRandomSampler(dataset.weights, len(dataset)),
                            collate_fn=collate)
        print(f"[data] mvgtemporal train items: {len(dataset)} (class-balanced)")
    elif task == "mvrawcorr":
        from scsrc.mvraw import MvrawCorrDataset
        dataset = MvrawCorrDataset(data_dir or "data/multivent_raw", "train",
                                   processor, device, n_tokens=args.n_tokens,
                                   n_frames=args.n_frames,
                                   ctm_pairs=((data_dir or "data/multivent_raw")
                                              + "/pairs.jsonl"
                                              if (g.get("lam_ctm", 0) > 0 or g.get("lam_sft_twin", 0) > 0) else None))
        loader = DataLoader(dataset, batch_size=1,
                            sampler=WeightedRandomSampler(dataset.weights, len(dataset)),
                            collate_fn=collate)
        print(f"[data] mvrawcorr train items: {len(dataset)} (class-balanced weights)")
    elif task == "mvraw":
        from scsrc.mvraw import MvrawDataset
        dataset = MvrawDataset(data_dir or "data/multivent_raw", "train",
                               processor, device, n_tokens=args.n_tokens,
                               n_frames=args.n_frames,
                               style=("pt" if args.method == "pt"
                                      else "instr" if args.method == "lora"
                                      else "neo"),
                               full_coverage_only=bool(args.full_coverage_only),
                               apo=(g.get("lam_apo", 0) > 0), sm=(g.get("lam_sm", 0) > 0 or g.get("lam_sft_stay", 0) > 0),
                               tc=(g.get("lam_tc", 0) > 0),
                               ctm_pairs=((data_dir or "data/multivent_raw") + "/pairs.jsonl"
                                          if (g.get("lam_ctm", 0) > 0 or g.get("lam_sft_twin", 0) > 0) else None),
                               ops=_ops)
        loader = DataLoader(dataset, batch_size=1,
                            sampler=WeightedRandomSampler(dataset.weights, len(dataset)),
                            collate_fn=collate)
        print(f"[data] mvraw train items: {len(dataset)} (weighted sampling)")
    elif task == "mvgreport":
        from scsrc.mvg import MvgReportDataset
        dataset = MvgReportDataset(processor, device, n_tokens=args.n_tokens,
                                   mode=("channel" if args.report_channel_words
                                         else "report"))
        loader = DataLoader(dataset, batch_size=1,
                            sampler=WeightedRandomSampler(dataset.weights, len(dataset)),
                            collate_fn=collate)
    elif task == "clevr3":
        from scsrc.clevr3 import Clevr3Dataset
        dataset = Clevr3Dataset(data_dir or "data/clevr3", "train", processor,
                                _ds_dev, n_tokens=args.n_tokens,
                                method=args.method, compose_frac=args.compose_frac,
                                mode=args.factor_mode, held_cells=held_cells,
                                slot_negative_frac=args.slot_negative_frac,
                                question_style=args.question_style,
                                aat=(g.get("lam_aat", 0) > 0),
                                apo=(g.get("lam_apo", 0) > 0),
                                vgm=(g.get("lam_vgm", 0) > 0),
                                ctm=(g.get("lam_ctm", 0) > 0 or g.get("lam_sft_twin", 0) > 0),
                                sm=(g.get("lam_sm", 0) > 0 or g.get("lam_sft_stay", 0) > 0),
                                tc=(g.get("lam_tc", 0) > 0),
                                image_twins=(args.clevr3_image_twins or None),
                                absence=(args.clevr3_absence or None),
                                absence_frac=args.clevr3_absence_frac)
        # lattice sampler: equalize (skill x source) cells so no cell dominates
        # 2026-09-23 (v1.126): the item ORDER comes from a generator of its own, seeded by --seed, so every
        # objective arm of a model draws the same items in the same order (before, any extra draw on the global
        # RNG -- the /dev/shm loader fallback's retry, prefix initialisation, ... -- shifted it).
        sampler = _SeededSampler(dataset.weights, len(dataset), seed=args.seed)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            collate_fn=collate, **_dl_kw)
        print(f"[data] clevr3 train items: {len(dataset)} over {len(dataset.cells)} "
              f"cells (mode={args.factor_mode}, weighted lattice sampling)")
    else:
        if args.ctrl_site == "decision":
            assert not any(g.get(k, 0) for k in ("lam_apo", "lam_vgm", "lam_tc", "lam_ctm")), \
                "--ctrl_site decision supports SFT-only for now (pair builders are user-site)"
        dataset = ObjectiveDataset(ds, specs, processor, args.method, args.arm,
                                   args.n_tokens, _ds_dev, g, per_op, task=task, ctrl_style=args.ctrl_style,
                                   absence_frac=args.absence_frac,
                                   cap_first_prob=args.cap_first_prob, slot_negative_frac=args.slot_negative_frac,
                                   ops=_ops, ctrl_site=args.ctrl_site,
                                   lookread_conflict_only=bool(args.lookread_conflict_only),
                                   disentangle_frac=args.disentangle_frac)
        # v1.126: seeded shuffle order, independent of other RNG consumers (see _SeededSampler)
        loader = DataLoader(dataset, batch_size=1, sampler=_SeededSampler(None, len(dataset), seed=args.seed),
                            collate_fn=collate, **_dl_kw)

    iit_it = None
    if args.lam_iit and args.lam_iit > 0:
        assert args.method == "neo" and args.arm == "ctrl" and task != "mvg", \
            "--lam_iit supports neo/ctrl synthetic tasks only"
        from scsrc.iit_train import IITPairDataset
        iit_ds = IITPairDataset(ds, specs, processor, args.n_tokens, device,
                                task, ctrl_style=args.ctrl_style)
        iit_loader = DataLoader(iit_ds, batch_size=1, shuffle=True,
                                collate_fn=collate)
        iit_it = iter(iit_loader)
        print(f"[iit] {len(iit_ds)} pair items, layer {args.iit_layer}, "
              f"lam {args.lam_iit} (every other micro-step)")

    try:
        import wandb
        wandb.init(project=cfg["wandb"]["project"], name=run_name, dir=paths.WANDB_DIR,
                   config={**cfg, **vars(args), "obj_global": g, "per_op": per_op},
                   mode=os.environ.get("WANDB_MODE", "offline")); use_wandb = True
        _ent = os.environ.get("WANDB_ENTITY", "<your-entity>")
        print(f"[wandb] run id {wandb.run.id} (offline). After `bash scripts/sync_wandb.sh` "
              f"on the login node, view at: https://wandb.ai/{_ent}/{cfg['wandb']['project']}/runs/{wandb.run.id}",
              flush=True)
    except Exception as e:
        print("[wandb] disabled:", e); use_wandb = False

    def _save_opt_state(d):
        """v1.300: AdamW moments of the trained rows (neo/pt; the frozen rows' moments are exactly zero) or of the
        adapter (LoRA) plus the RNG states, so --resume_ckpt continues the trajectory. Newest checkpoint only."""
        import glob as _glob
        try:
            _st = {"format": 1, "opt_step": None, "rows": {}, "lora": None,
                   "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                           "python": random.getstate(), "numpy": np.random.get_state()}}
            _step_of = lambda v: int(v.item()) if torch.is_tensor(v) else int(v)
            if args.method in ("neo", "pt"):
                for _p, _key in [(W, "W")] + ([(_pleP, "ple")] if _pleP is not None else []) + ([(_act_ctrl.delta, "act")] if _act_ctrl is not None else []):
                    _s = opt.state.get(_p)
                    if not _s:
                        continue
                    if _key == "W":          # the embedding matrix: only the new rows have non-zero moments
                        _ids = list(all_new_ids)
                        _st["rows"][_key] = {"ids": _ids, "exp_avg": _s["exp_avg"][_ids].detach().cpu(),
                                             "exp_avg_sq": _s["exp_avg_sq"][_ids].detach().cpu()}
                    else:                    # E4B per-layer parameter P (n_new x d_ple): whole tensor (indexing it with vocab ids was the 10-07/10-09 E4B CUDA assert)
                        _st["rows"][_key] = {"ids": None, "exp_avg": _s["exp_avg"].detach().cpu(), "exp_avg_sq": _s["exp_avg_sq"].detach().cpu()}
                    _st["opt_step"] = _step_of(_s["step"])
            else:
                _st["lora"] = opt.state_dict()
                _g0 = next(iter(_st["lora"]["state"].values()), None)
                if _g0 is not None and "step" in _g0:
                    _st["opt_step"] = _step_of(_g0["step"])
            torch.save(_st, os.path.join(d, "opt_state.pt"))
            _keep = sorted(_glob.glob(os.path.join(out, "ckpt", "step_*", "opt_state.pt")),
                           key=lambda q: int(os.path.basename(os.path.dirname(q)).split("_")[1]))[-2:]   # newest two (a resume test from the penultimate checkpoint, and a fallback)
            for _old in _glob.glob(os.path.join(out, "ckpt", "step_*", "opt_state.pt")):
                if _old not in _keep:
                    os.remove(_old)
        except Exception as _e:
            print(f"[ckpt] WARNING: optimizer state not saved ({type(_e).__name__}: {_e})", flush=True)

    def save_ckpt(step):
        d = os.path.join(out, "ckpt", f"step_{step}"); os.makedirs(d, exist_ok=True)
        if _ple_sync is not None:
            _ple_sync()          # v1.157: the trained per-layer rows go into the table, so neo.pt stores them
            _dp = (_pleP.data.float() - _ple_init).norm().item() / max(_ple_init.norm().item(), 1e-9)
            print(f"[ple] step {step}: per-layer rows moved {_dp:.4f} (relative to init)", flush=True)
            if step >= 100 and _dp == 0.0:
                raise SystemExit("[STOP] --train_per_layer: the per-layer rows did not move by step 100")
        if args.method in ("neo", "pt"):
            M.save_neologisms(model.get_input_embeddings().weight, all_new_ids, name_to_ids, os.path.join(d, "neo.pt"),
                              per_layer=M.per_layer_table(model))
            if _is_act and _act_ctrl is not None:
                import scsrc.act as A
                A.save(_act_ctrl, os.path.join(d, "act.pt"), args.act_layer, args.act_span)
        elif args.method == "lora":
            model.save_pretrained(d)
        _save_opt_state(d)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"step": step, "method": args.method, "arm": args.arm,
                       "resume_state": ("full" if _resume_state_loaded else "restarted") if args.resume_ckpt else "uninterrupted",
                       "act": ({"layer": args.act_layer, "span": args.act_span} if _is_act else None),
                       "gcf_ctrl_pos": args.gcf_ctrl_pos, "ladder_ctrl_pos": args.ladder_ctrl_pos,
                       "n_tokens": args.n_tokens, "seed": args.seed,
                       "model_id": args.model_id,
                       "add_word": args.add_word,
                       "extend_ckpt": args.extend_ckpt,
                       "lam_iit": args.lam_iit, "iit_layer": args.iit_layer,
                       "obj_tag": _obj_tag(g, per_op)}, f)
        print(f"[ckpt] step {step} -> {d}")
        # live samples at every checkpoint, ALL methods/arms: the model's actual
        # output next to the gold target and the distractor (the other source's
        # value), with a correctness parse. Printed to stdout (-> the .out file),
        # appended to runs/<name>/train_samples.jsonl, and logged as a wandb table.
        # MEMORY: this block calls model.generate(), which builds a KV cache and
        # sets use_cache on the way through. Under gradient checkpointing (mvraw)
        # that flag must go back to False or every subsequent training step
        # retains activations -- MVR LoRA grew to 136.8/139.8 GiB and OOM'd at
        # step ~114, i.e. ~14 steps AFTER the first checkpoint. E1 probe
        # (mvr_probe_lora_nosave, ckpt_every=900) ran 300/300 clean, which
        # implicates this path rather than the raw footprint. Save/restore the
        # flag, generate under no_grad, and hand the allocator its cache back.
        _uc = getattr(getattr(model, "config", None), "use_cache", None)
        try:
            from scsrc.infer import generate as _gen
            from scsrc import prompts as _P, metrics as _Me
            from scsrc.tokens import trigger as _trig
            model.eval()
            _ng = torch.no_grad()
            _ng.__enter__()
            rows = []
            conflicts = [sp for sp in specs if sp.cond == "conflict"][:2] or specs[:2]
            for sp in conflicts:
                img = get_image(ds, sp)
                if args.method == "neo" and args.arm == "gen":
                    user = _P.gen_template(task, 0).format(cap=sp.fed_caption)
                    txt = _gen(model, processor, img, user, device, 96)
                    gold = "IMG %s | TXT %s | %s" % (sp.image_key, sp.text_key, sp.relation)
                    dist = "collapse: both=%s" % sp.image_key
                    ok = -1  # multi-slot; judged at eval, shown here for eyeballing
                    rows.append(("gen", txt, gold, dist, ok))
                elif args.method == "lora":
                    user = ("Report each object's value according to the image and the "
                            "text, and whether they agree. Caption: \"%s\"." % sp.fed_caption)
                    txt = _gen(model, processor, img, user, device, 96)
                    rows.append(("lora", txt, "IMG %s | TXT %s | %s" % (sp.image_key, sp.text_key, sp.relation),
                                 "collapse: both=%s" % sp.image_key, -1))
                else:  # neo ctrl and pt: one row per operator with a real correctness bit
                    for op in ("look", "read", "compare"):
                        if args.method == "neo":
                            user = _P.render_ctrl(_P.CTRL_CANONICAL, _trig(f"{op}_ctrl", args.n_tokens), sp.fed_caption)
                        else:
                            user = _trig(f"pt_{op}", args.n_tokens) + " " + _P.pt_template(0).format(cap=sp.fed_caption)
                        _fb = (all_new_ids if args.method == "pt" or args.arm == "ctrl"
                               else ([i for nm, ids in name_to_ids.items()
                                      if nm.endswith("_ctrl") for i in ids]
                                     if args.arm == "link" else None))
                        txt = _gen(model, processor, img, user, device, 128, forbid_ids=_fb,
                                   stop_anchor="final answer:")
                        if op == "compare":
                            gold, dist = sp.relation, ("agree" if sp.relation == "conflict" else "conflict")
                            ok = int(_Me.parse_relation(txt) == sp.relation)
                        else:
                            gk = sp.image_key if op == "look" else sp.text_key
                            dk = sp.text_key if op == "look" else sp.image_key
                            gold, dist = str(gk), str(dk)
                            ok = int(_Me.object_accuracy(_Me.parse_pred_task(txt, task), gk))
                        rows.append((op, txt, gold, dist, ok))
            with open(os.path.join(out, "train_samples.jsonl"), "a") as fh:
                for op, txt, gold, dist, ok in rows:
                    print(f"[sample step {step} {op}] ok={ok} gold={gold} | distractor={dist} | out={txt[:120]!r}", flush=True)
                    fh.write(json.dumps({"step": step, "op": op, "ok": ok, "gold": gold,
                                         "distractor": dist, "out": txt[:400]}) + "\n")
            if use_wandb:
                import wandb as _wb
                tbl = _wb.Table(columns=["step", "op", "ok", "gold", "distractor", "output"],
                                data=[[step, op, ok, gold, dist, txt[:300]] for op, txt, gold, dist, ok in rows])
                _wb.log({"train_samples": tbl, "sample_acc": (sum(r[4] for r in rows if r[4] >= 0)
                         / max(sum(1 for r in rows if r[4] >= 0), 1)), "step": step})
            model.train()
        except Exception as e:
            print(f"[sample] skipped ({type(e).__name__}: {e})")
        finally:
            try:
                _ng.__exit__(None, None, None)
            except Exception:
                pass
            if _uc is not None:
                model.config.use_cache = _uc      # generate() may have flipped it
            model.train()
            torch.cuda.empty_cache()

    model.train()
    try:
        it = iter(loader)
    except OSError as _e:
        # 2026-09-22 (STATUS v1.103): worker start-up allocates POSIX semaphores in /dev/shm; a node whose shared
        # memory is exhausted raises "No space left on device" here and killed seven ladder runs before step 1.
        # Fall back to building records in this process: slower, identical records, and they still move to the GPU
        # because the dataset was constructed on CPU.
        print(f"[throughput] loader workers unavailable ({type(_e).__name__}: {_e}); "
              f"falling back to in-process record building", flush=True)
        loader.num_workers = 0
        it = iter(loader)
    if _start_step:                    # v1.210 resume: the records the first n steps consumed, so the data order continues
        import time as _time
        _t0 = _time.time()
        for _ in range(_start_step * accum):
            try:
                next(it)
            except StopIteration:
                it = iter(loader); next(it)
        print(f"[resume] skipped {_start_step * accum} records in {_time.time() - _t0:.0f}s; "
              f"lr now {sched.get_last_lr()[0]:.3g}", flush=True)
        if _resume_state_loaded and _ost.get("rng"):
            _r = _ost["rng"]; torch.set_rng_state(_r["torch"]); torch.cuda.set_rng_state_all(_r["cuda"])
            random.setstate(_r["python"]); np.random.set_state(_r["numpy"])
            print("[resume] RNG states restored", flush=True)
    if args.save_step0 and args.method in ("neo", "pt") and _start_step == 0:
        save_ckpt(0)        # the untrained controls (initialization rows) for the step-0 semantic probes
    _clamp_win_steps = _clamp_win_hits = _clamp_steps_total = _clamp_hit_steps = _clamp_rows_total = 0   # norm-clamp activation (2026-10-08)
    pbar = tqdm(range(_start_step, steps), desc=run_name)
    # one shared key set for initialization, accumulation, logging and reset (2026-10-01; the round-3 KeyError)
    LOG_KEYS = ("ce", "ce_twin", "ce_stay", "apo", "vgm", "tc", "ctm", "sm", "ecc", "ecc_n", "aat", "drift", "iit")
    agg = {**{k: 0.0 for k in LOG_KEYS}, "n": 0}
    pref_warmup = args.pref_warmup if args.pref_warmup is not None else int(cfg["objective"].get("pref_warmup", 100))
    if pref_warmup > 0 and (g.get("lam_apo", 0) or g.get("lam_vgm", 0) or g.get("lam_tc", 0) or g.get("lam_ctm", 0) or g.get("lam_sm", 0) or g.get("lam_sft_twin", 0) or g.get("lam_sft_stay", 0)):
        print(f"[obj] preference warmup: SFT-only for {pref_warmup} steps, "
              f"APO/VGM ramp to full by step {pref_warmup + max(pref_warmup // 2, 1)}", flush=True)

    for step in pbar:
        opt.zero_grad()
        for _mi in range(accum):
            if iit_it is not None and _mi % 2 == 1:
                from scsrc.iit_train import iit_step
                try:
                    irec = next(iit_it)
                except StopIteration:
                    iit_it = iter(iit_loader); irec = next(iit_it)
                _, icomps = iit_step(model, irec, args.lam_iit,
                                     args.iit_layer, accum)
                for k in ("ce", "iit"):
                    if k in icomps:
                        agg[k] = agg.get(k, 0.0) + icomps[k]
                agg["n"] += 1
                continue
            try:
                rec = next(it)
            except StopIteration:
                it = iter(loader); rec = next(it)
            if _ds_dev == "cpu":
                rec = _to_dev(rec, device)
            eff = resolve_obj(g, per_op, rec.get("op"))
            if pref_warmup > 0 and (eff.get("lam_apo", 0) or eff.get("lam_vgm", 0) or eff.get("lam_tc", 0) or eff.get("lam_ctm", 0) or eff.get("lam_sm", 0) or eff.get("lam_sft_twin", 0) or eff.get("lam_sft_stay", 0)):
                if step < pref_warmup:
                    scale = 0.0
                elif step < pref_warmup + max(pref_warmup // 2, 1):
                    scale = (step - pref_warmup) / max(pref_warmup // 2, 1)
                else:
                    scale = 1.0
                if scale < 1.0:
                    eff = {**eff, "lam_apo": eff.get("lam_apo", 0) * scale,
                           "lam_vgm": eff.get("lam_vgm", 0) * scale,
                           "lam_tc": eff.get("lam_tc", 0) * scale,
                           "lam_ctm": eff.get("lam_ctm", 0) * scale,
                           "lam_sm": eff.get("lam_sm", 0) * scale,
                           "lam_sft_twin": eff.get("lam_sft_twin", 0) * scale, "lam_sft_stay": eff.get("lam_sft_stay", 0) * scale}
            loss, comps = objective_step(model, rec, eff, all_new_ids, ref_rows, accum=accum)
            _w = float(rec.get("w", 1.0))
            if _w != 1.0 and torch.is_tensor(loss):
                loss = loss * _w
            if torch.is_tensor(loss):
                (loss / accum).backward()
            for k in LOG_KEYS:
                if k in comps:
                    agg[k] += comps[k]
            agg["n"] += 1
        _lr_used = opt.param_groups[0]["lr"]
        gn = torch.nn.utils.clip_grad_norm_(trainable, cfg["train"]["grad_clip"])
        if _decay_new_rows_only and _wd > 0:
            # --decay_scope new_rows: AdamW's decoupled decay p <- p - lr*wd*p on the trained rows only, applied BEFORE
            # the adaptive update exactly as torch AdamW orders it; the frozen rows are never touched
            with torch.no_grad():
                W.data[all_new_ids] *= (1.0 - _lr_used * _wd)
        opt.step(); sched.step()
        # tied input/output embeddings: if a new row's norm grows far past the
        # vocab average, its OUTPUT logit dominates and the model starts emitting
        # the token everywhere (the <tok><tok><tok> soup). Project new rows back
        # to at most `norm_clamp` x the mean old-row norm after every step.
        # (_norm_ref is None for methods without new rows, e.g. lora.)
        if os.environ.get("SC_LOG_OLDROW_NORM") and all_new_ids and step % 10 == 0:
            # F2 probe (2026-09-21, revised after review): does decoupled weight decay change the FROZEN
            # rows at all in the stored dtype? bf16 rounding can absorb p*(1-lr*wd) entirely, so the
            # exact-arithmetic shrink estimate is NOT an observation. Log dtype, tied-head storage,
            # max |delta| and the count of changed elements on a fixed 20k-row sample vs the first probe.
            with torch.no_grad():
                _Wd = model.get_input_embeddings().weight.data
                _mk = torch.ones(_Wd.shape[0], dtype=torch.bool, device=_Wd.device); _mk[all_new_ids] = False
                _smp = _Wd[_mk][:20000]
                if not hasattr(model, "_sc_oldrow_ref"):
                    model._sc_oldrow_ref = _smp.clone()
                    _oe = model.get_output_embeddings()
                    _tied = _oe is not None and _oe.weight.data_ptr() == _Wd.data_ptr()
                    print(f"[oldrow] dtype={_Wd.dtype} tied_lm_head={_tied} optimizer=AdamW wd={_wd} "
                          f"sample_rows={_smp.shape[0]}", flush=True)
                _dlt = (_smp.float() - model._sc_oldrow_ref.float()).abs()
                print(f"[oldrow] step {step} mean_norm {_smp.float().norm(dim=-1).mean().item():.6f} "
                      f"max_abs_delta {_dlt.max().item():.3e} changed_elems {(_dlt > 0).sum().item()} "
                      f"of {_dlt.numel()}", flush=True)
        if _norm_ref is not None and _nc > 0:
            with torch.no_grad():
                Wc = model.get_input_embeddings().weight
                lim = _nc * _norm_ref
                rows = Wc.data[all_new_ids]
                nrm = rows.norm(dim=-1, keepdim=True)
                Wc.data[all_new_ids] = torch.where(nrm > lim, rows * (lim / nrm), rows)
                _ch = int((nrm > lim).sum().item()); _clamp_win_steps += 1; _clamp_steps_total += 1   # 2026-10-08 diagnostics
                if _ch:
                    _clamp_win_hits += 1; _clamp_hit_steps += 1; _clamp_rows_total += _ch
        if args.tie_rows and name_to_ids:
            # TIE PROJECTION (capacity control): set each word's L rows to their
            # mean after every step = projected gradient descent onto the tied
            # subspace. Same positions / attention queries / repetition as
            # untied L, but ~one row of independent parameters per word. If
            # untied-L beats tied-L, contextual WIDTH is not the carrier --
            # parameter count is. If they match, width is real.
            with torch.no_grad():
                Wt = model.get_input_embeddings().weight
                for _nm, _ids in (name_to_ids or {}).items():
                    if len(_ids) > 1:
                        Wt.data[_ids] = Wt.data[_ids].mean(0, keepdim=True)
        if wd > 0:
            faulthandler.dump_traceback_later(wd)  # heartbeat: re-arm; cancels the pending timer
        if (step + 1) % 10 == 0:
            log = {k: agg.get(k, 0.0) / max(agg["n"], 1) for k in LOG_KEYS}
            log["grad_norm"] = float(gn); log["step"] = step; log["lr"] = float(opt.param_groups[0]["lr"])
            _show = {k: f"{log[k]:.3f}" for k in ("ce", "ce_twin", "ctm") }
            _show.update({k: f"{log[k]:.3f}" for k in ("ce_stay",) if abs(log.get(k, 0.0)) > 0})
            _show.update({k: f"{log[k]:.3f}" for k in ("apo", "sm", "ecc", "ecc_n", "vgm", "tc", "iit") if abs(log.get(k, 0.0)) > 0})   # 2026-10-02: every active term visible
            if _norm_ref is not None and _clamp_win_steps:
                _show["clamp"] = f"{_clamp_win_hits}/{_clamp_win_steps}"; _clamp_win_steps = _clamp_win_hits = 0
            pbar.set_postfix(**_show)
            if use_wandb:
                wandb.log(log)
            agg = {**{k: 0.0 for k in LOG_KEYS}, "n": 0}
        if (step + 1) % ckpt_every == 0 or (step + 1) == steps:
            save_ckpt(step + 1)
        if _STOP["flag"]:
            save_ckpt(step + 1); print("[signal] checkpointed; exiting 0"); break
    if args.selfcheck_items and task == "clevr3":
        _selfcheck_clevr3(model, processor, args, out, steps, args.selfcheck_items)
    if _norm_ref is not None and all_new_ids:        # 2026-10-08 (advisor): how often the norm clamp bit, and where the rows ended
        with torch.no_grad():
            _fn = model.get_input_embeddings().weight.data[all_new_ids].float().norm(dim=-1)
        json.dump({"norm_clamp": _nc, "norm_ref_oldrows": _norm_ref, "limit": _nc * _norm_ref, "lr": lr,
                   "steps_counted": _clamp_steps_total, "steps_clamped_any_row": _clamp_hit_steps, "clamped_row_events": _clamp_rows_total,
                   "n_new_rows": len(all_new_ids), "final_row_norm": {"min": _fn.min().item(), "mean": _fn.mean().item(), "max": _fn.max().item()},
                   "final_rows_at_limit": int((_fn >= 0.999 * _nc * _norm_ref).sum().item()) if _nc > 0 else 0,
                   **({"act": {**__import__("scsrc.act", fromlist=["diag"]).diag(), "delta_norm": [float(x) for x in _act_ctrl.delta.detach().float().norm(dim=-1)]}} if _act_ctrl is not None else {})},
                  open(os.path.join(out, "train_diag.json"), "w"), indent=1)
        print(f"[diag] norm clamp active on {_clamp_hit_steps}/{_clamp_steps_total} steps ({_clamp_rows_total} row events); "
              f"final new-row norms mean {_fn.mean().item():.3f} (limit {_nc * _norm_ref:.3f}, old-row mean {_norm_ref:.3f})", flush=True)
    if not _STOP["flag"] and not args.keep_opt_state:
        # 2026-10-22: a FINISHED run never resumes, so its optimizer state (up to ~240 MB for a LoRA run) is removed;
        # a run stopped at the wall keeps it for --resume_ckpt
        import glob as _g
        for _o in _g.glob(os.path.join(out, "ckpt", "step_*", "opt_state.pt")):
            os.remove(_o)
    print("[done]", run_name)


@torch.no_grad()
def _selfcheck_clevr3(model, processor, args, out, steps, k):
    """2026-09-29 (audit): score the first k Ground-CF training rows IN-PROCESS, both requests, all channel values as
    candidates under the active answer format, and store prompt text, candidates and log-likelihoods next to the final
    checkpoint. scripts/smoke_v2_reload.py reloads the checkpoint through evaluate.apply_checkpoint and must reproduce
    the prompt text exactly and the scores to ~1e-3."""
    import json as _json
    import scsrc.clevr3 as _C3
    from scsrc.clevr3 import clevr3_user_content, c3_cont
    from scsrc.collate import _encode, _prompt_len
    # render exactly as evaluate.apply_checkpoint's environment will: no training rotations, no per-row order coin,
    # the instruction pool from the run's --instr_rotate
    _saved = {_e: os.environ.get(_e) for _e in ("SC_Q_ROTATE", "SC_INSTR_ROTATE", "SC_INSTR_TRAIN_POOL", "SC_TRAIN_SOURCES")}
    for _e in ("SC_Q_ROTATE", "SC_INSTR_ROTATE", "SC_TRAIN_SOURCES"):
        os.environ.pop(_e, None)
    if getattr(args, "instr_rotate", ""):
        os.environ["SC_INSTR_TRAIN_POOL"] = args.instr_rotate
    _bits = (_C3._ORDER_BIT, _C3._POS_BIT); _C3._ORDER_BIT = None; _C3._POS_BIT = None
    rows = [_json.loads(l) for l in open(os.path.join(args.data_dir or "data/clevr3", "train.jsonl"))][:k]
    mode = args.factor_mode; meth = args.method if args.method in ("neo", "pt") else "lora"
    recs = []
    model.eval()
    for r in rows:
        for src in ("look", "read"):
            um = [{"role": "user", "content": clevr3_user_content(r, src, args.n_tokens, mode, 0, method=meth, question_style="explicit")}]
            txt = " ".join("[IMAGE]" if c.get("type") == "image" else c.get("text", "") for c in um[0]["content"])
            vals = sorted({str(v) for v in r["channels"].values()})
            sc = {}
            for v in vals:
                cont = c3_cont(v)
                fm = um + [{"role": "assistant", "content": [{"type": "text", "text": cont}]}]
                enc = _encode(processor, fm, "cuda"); plen = _prompt_len(processor, um, "cuda")
                ids = enc["input_ids"]; kw = {kk: vv for kk, vv in enc.items() if kk != "token_type_ids"}
                lg = model(**kw, use_cache=False).logits.float()
                lp = torch.log_softmax(lg[0, plen - 1:ids.shape[1] - 1, :], -1)
                sc[v] = float(lp.gather(-1, ids[0, plen:].unsqueeze(-1)).sum())
            recs.append({"id": r["id"], "source": src, "prompt": txt, "target": c3_cont(str(r["channels"][src])),
                         "cands": vals, "scores": sc, "pick": max(sc, key=sc.get)})
    p = os.path.join(out, "ckpt", f"step_{steps}", "selfcheck.json")
    _json.dump({"run": os.path.basename(out), "step": steps, "cont_format": os.environ.get("SC_GCF_CONT", "leadin"),
                "format": os.environ.get("SC_GCF_FORMAT", "legacy"), "records": recs}, open(p, "w"), indent=1)
    print(f"[selfcheck] {len(recs)} in-process scores written to {p}; first prompt:\n{recs[0]['prompt'][:400]}\n[selfcheck] target: {recs[0]['target']!r}")
    _C3._ORDER_BIT, _C3._POS_BIT = _bits
    for _e, _v in _saved.items():
        if _v is None: os.environ.pop(_e, None)
        else: os.environ[_e] = _v
    model.train()



class _SeededSampler(torch.utils.data.Sampler):
    """v1.126: WeightedRandomSampler (weights given) or a uniform shuffle (weights None) drawn from a PRIVATE
    torch.Generator seeded with --seed, so the item order is identical across runs that differ only in the
    objective. Logs the first 12 indices of the first pass so the claim can be checked from the .out files."""
    def __init__(self, weights, n, seed=0):
        self.w = None if weights is None else torch.as_tensor(weights, dtype=torch.double)
        self.n = int(n); self.g = torch.Generator().manual_seed(int(seed) + 1234567); self._logged = False

    def __len__(self):
        return self.n

    def __iter__(self):
        idx = (torch.multinomial(self.w, self.n, True, generator=self.g) if self.w is not None
               else torch.randperm(self.n, generator=self.g)).tolist()
        if not self._logged:
            print(f"[order] first sampled items: {idx[:12]} (n={self.n}, private generator)", flush=True); self._logged = True
        return iter(idx)


if __name__ == "__main__":
    main()
