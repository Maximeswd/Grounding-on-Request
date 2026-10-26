#!/usr/bin/env python
"""THE MASTER GRID — every routing/grounding number, one file.

Consolidates the scattered per-run eval JSONs into the full grid the paper's
appendix prints: dataset x model x method x L, with ROUTING and GROUNDING per
channel. Idempotent: re-run after any wave lands and the table fills in.

Sources (the two batteries the paper quotes, plus the video pair):
  ladder (vcf/color/coco): runs/*/eval/{ds}_test_id/flip_ll_step_800.json
      routing = look_acc/read_acc, grounding = look_flip/read_flip
  Ground-CF:               runs/*/eval/clevr3_test_id/flip_step_800.json
      (+ _instr/_noop mode files for the frozen baselines)
      routing = {ch}_acc, grounding = {ch}_flip, ch in look/read/ocr
  MVG:  eval/mvg_flip/flip_step_800_clean.json  grounding = vis/desc flip,
        ocr = deletion-ablation (marked *, not a flip)
  MVR:  eval/mvraw_test_ood/ablate*_step_800.json  grounding = per-channel
        JOINT = min(nec, spec)   (routing has no dense battery on MVR)

Method attribution comes from each run's config.json (never the filename),
except the frozen instruction/no-op rows, which are eval-time conditions
recognized by their mode/style tags. One row per (method, L); the parity cell
is L=5 seed 0.

Usage: python scripts/master_grid.py          -> results/master_grid.md + .json
"""
import re, glob, json, os, re, sys, collections

# split selector (2026-09-13): `--ood` harvests the *_test_ood eval dirs into
# results/master_grid_ood.{md,json} (+ _runs.json: which run fed each cell);
# the default stays the in-distribution grid, byte-identical to before.
SPLIT = "test_ood" if "--ood" in sys.argv else "test_id"
SUFFIX = "_ood" if SPLIT == "test_ood" else ""
RUNS = {}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None



def stay_c(d, ch):
    """Conjunctive stay for channel ch: the evaluator's summary field when present, else the mean of
    (acc_i AND stay_i) over the banked per-item records, else None (older Ground-CF files store no
    per-item stay and must be re-evaluated)."""
    v = d.get(f"{ch}_stay_c")
    if v is not None:
        return v
    r = d.get("records")
    if isinstance(r, dict) and r.get(f"{ch}_acc") and r.get(f"{ch}_stay") \
            and len(r[f"{ch}_acc"]) == len(r[f"{ch}_stay"]):
        return round(sum(int(a and s) for a, s in zip(r[f"{ch}_acc"], r[f"{ch}_stay"])) / len(r[f"{ch}_acc"]), 3)
    return None

def model_of(run, args):
    mid = (args or {}).get("model_id") or ""
    if "27" in mid: return "qwen27b" if "wen" in mid else "gemma27b"
    if "3.5-4B" in mid or "_qwen35" in run: return "qwen4b"
    if "3.5-2B" in mid or "_qwen2b" in run: return "qwen2b"
    if "E4B" in mid or "gemma4e4b" in run: return "e4b"
    return "gemma3"


def method_of(run, args):
    m, L = _method_of_proto(run, args)
    a = args or {}
    # 2026-09-23 (v1.124): fact captions and modality-specific adapters are their own recipes
    if m is not None and a.get("caption_style") == "fact":
        m = m + "+fc"
    # 2026-09-24 (v1.157): E4B runs whose new-token per-layer embedding rows were TRAINED (--train_per_layer 1)
    # are their own recipe; before, those rows were random and frozen
    if m is not None and a.get("train_per_layer"):
        m = m + "+ple"
    # 2026-09-24 (v1.165): control-position ablation -- the word trained BEFORE the caption; never the native word cell
    if m is not None and a.get("word_first"):
        m = m + "+wf"
    if m is not None and a.get("gcf_word_pos") in ("mix", "first"):
        m = m + ("+pmix" if a.get("gcf_word_pos") == "mix" else "+pfirst")   # v1.167/v1.184 placement ablations
    if m is not None and a.get("train_sources"):
        m = m + "+ad"          # one adapter per request; the Ground-CF loop merges the look/read halves into one cell
    # 2026-09-22 (STATUS v1.94): a Ground-CF run trained under the two-modality protocol is its own cell
    # 2026-09-24 (v1.147): Ground-CF only. Since OCR went off by default (v1.133) every new run records
    # c3_sources=look,read, so the 48 resubmitted ladder runs got +2s and fell outside cpgrid's ladder keys.
    if m is not None and a.get("c3_sources") == "look,read" and a.get("task") == "clevr3":
        m = m + "+2s"
    return m, L


def _method_of_proto(run, args):
    a = args or {}
    # compose-trained runs (2026-09-13) are their own cells: they must never
    # overwrite a parity cell through the newer-file-wins rule.
    if "_compose" in (a.get("run_name") or run or ""):
        _m, _L = _method_of_inner(run, a)
        return (_m + "+compose"), _L
    # weight-decay control (2026-09-21): identical args except --weight_decay 0, so without a suffix it
    # would land in the reference word+ctm cell and win through the newer-file-wins rule.
    if "_wd0" in (a.get("run_name") or run or ""):
        _m, _L = _method_of_inner(run, a)
        return (_m + "+wd0"), _L
    return _method_of_inner(run, a)


def _method_of_inner(run, a):
    m = a.get("method")
    if m == "lora":
        # LoRA trained with a counterfactual objective is its own cell (2026-09-14): without the
        # suffix a LoRA+CTM run would overwrite the SFT LoRA row through the newer-file-wins rule.
        lam = {k[4:] for k in a if k.startswith("lam_") and float(a.get(k) or 0) > 0}
        lam.discard("sft"); lam.discard("iit"); lam.discard("aat")
        base = f"lora_r{a.get('lora_r') or '?'}"
        if a.get("ctrl_style") == "request":
            base += "+req"      # request-conditioned recipe (2026-09-21), never the historical row
        rsfx = "+rot" if a.get("instr_rotate") else ""   # 2026-09-23 (v1.123): trained on rotated INSTR variants
        return ((base + "+" + "+".join(sorted(lam))) if lam else base) + rsfx, None
    if m == "pt":
        # wave O gave PT the causal arms: attribute them, or they'd collide
        # with (and overwrite) the plain-PT baseline row via the max-G keep.
        lam = {k[4:] for k in a if k.startswith("lam_") and float(a.get(k) or 0) > 0}
        lam.discard("sft"); lam.discard("iit"); lam.discard("aat")
        base = "pti" if "_pti_" in (a.get("run_name") or "") else "pt"
        # 2026-09-21: the same data-recipe axes as the word branch below. Without them the
        # Ground-CF PT run trained with 30% absence items (clevr3_f_pt_5tok_apoabs_seed0*) mapped to
        # the plain "pt+apo" key and, being newer, overwrote the plain APO run in the ID grid while
        # the OOD grid held the plain run -- Table 1's PT+APO Ground-CF cell was the abs recipe.
        v3 = float(a.get("slot_negative_frac") or 0) > 0
        img = bool(a.get("clevr3_image_twins")); ab = bool(a.get("clevr3_absence"))
        sfx = ("+v3" if v3 else "") + ("+img" if img else "") + ("+abs" if ab else "") + ("+cp" if a.get("cp_pairs") else "")
        if lam:
            return base + "+" + "+".join(sorted(lam)) + sfx, int(a.get("n_tokens") or 0) or None
        return base + sfx, int(a.get("n_tokens") or 0) or None
    if m == "neo":
        L = int(a.get("n_tokens") or 0) or None
        lam = {k[4:] for k in a if k.startswith("lam_") and float(a.get(k) or 0) > 0}
        lam.discard("sft"); lam.discard("iit"); lam.discard("aat")
        style = a.get("ctrl_style", "neutral")
        arm = a.get("arm", "ctrl")
        if arm != "ctrl" or style not in ("neutral",):
            return f"word[{arm}/{style}]" + ("+" + "+".join(sorted(lam)) if lam else ""), L
        # the v3 DATA recipe (slot negatives) is a separate axis from the
        # objective: before 2026-09-11 a run with both lam_ctm and slot
        # negatives collided with plain CTM in the same cell, and better()
        # then kept whichever eval file was newer, so Table 1 mixed recipes
        # across models (STATUS 2026-09-11 FLAG A). Keep them apart.
        v3 = float(a.get("slot_negative_frac") or 0) > 0
        # image-twin CTM on Ground-CF (2026-09-14) is a different data recipe from text-only CTM
        # and must not overwrite it in the cell.
        img = bool(a.get("clevr3_image_twins"))
        ab = bool(a.get("clevr3_absence"))
        cp = bool(a.get("cp_pairs"))      # conflict-preserving training pairs (2026-09-22): a separate recipe
        # 2026-09-23 (v1.118): the gamma-sensitivity runs differ from the reference ONLY in --gamma, which no
        # other part of the key records, so their evaluations would have replaced the reference
        # `word+ctm+img+cp+2s` cell through the newer-file-wins rule. Off-default gamma gets its own suffix.
        _gm = a.get("gamma")
        gsfx = f"+g{float(_gm):g}" if _gm is not None and abs(float(_gm) - 0.1) > 1e-9 else ""
        # 2026-09-23 (v1.123): Hewitt-style carrier templates are a different training prompt from the bare
        # `<question> <tokens>` form; they must never share a cell.
        tsfx = "+tpl" if a.get("neo_template") == "hewitt" else ""
        if lam:
            return "word+" + "+".join(sorted(lam)) + ("+v3" if v3 else "") + ("+img" if img else "") + ("+abs" if ab else "") + ("+cp" if cp else "") + gsfx + tsfx, L
        if v3:
            return "word+v3" + tsfx, L
        return "word_sft" + tsfx, L
    return None, None


CFG = {}
def cfg(run):
    if run not in CFG:
        d = load(f"{ROOT}/runs/{run}/config.json") or {}
        a = dict(d.get("args") or {})
        # 2026-09-22 (STATUS v1.96): the recipe key must name the objective the run APPLIED, not the one it was
        # asked for. train.py can force a term off (it silenced APO for every LoRA run until today, so LoRA+APO
        # trained as LoRA+SFT and the two Ground-CF adapters were byte-identical); the applied weights are in
        # obj_global. Any lam_* recorded there overrides the requested argument.
        for k, v in (d.get("obj_global") or {}).items():
            if k.startswith("lam_"):
                a[k] = v
        CFG[run] = a
    return CFG[run]


def fmt(v):
    return f"{v:.2f}".lstrip("0") if isinstance(v, float) else (v or "--")


_VOID_FILE = f"{ROOT}/results/void_run_patterns.txt"
_VOID = [re.compile(l.strip()) for l in open(_VOID_FILE) if l.strip() and not l.startswith("#")] \
    if os.path.exists(_VOID_FILE) else []


def _void(run):
    """Runs listed in results/void_run_patterns.txt (regex per line) never enter the grid. Added
    2026-09-21 for the ladder PT+CTM / LoRA+CTM runs whose CTM pairs carried no trainable request;
    their fixed reruns carry a _v2_ marker that the patterns do not match. Demotion, not deletion."""
    return any(p.search(run) for p in _VOID)


def better(newmeta, oldmeta):
    """PARITY SELECTION, not max-selection. The old keep-max rule silently
    picked the best-scoring file across seeds/run-variants -- a luckiest-file
    bias that inflated borderline cells (caught 2026-08-07 reconciling
    Table 1). Policy: seed 0 beats other seeds; among equals, the NEWER eval
    file wins (later vintages carry fixes/records, and re-evals supersede)."""
    ns, nm = newmeta
    os_, om = oldmeta
    if (ns == 0) != (os_ == 0):
        return ns == 0
    return nm > om


def seed_of(args):
    try:
        return int(args.get("seed", 0) or 0)
    except Exception:
        return 0


def main():
    grid = collections.defaultdict(dict)   # (ds, model, method, L) -> {metric: v}
    meta = {}                               # key -> (seed, file mtime)

    # ---------------- ladder
    # vcf3 = the third-colour VCF battery (STATUS v1.66), written by VCF-trained runs into eval/vcf3_<split>/
    # *3 = the conflict-preserving (third-value) battery of a ladder benchmark, written by that benchmark's
    # own runs into eval/<ds>3_<split>/ (STATUS v1.66 VCF; v1.81 ColorSwap and COCO-CF)
    for ds, dname in (("vcf", "vcf"), ("color", "color"), ("coco", "coco"),
                      ("vcf3", "vcf3"), ("color3", "color3"), ("coco3", "coco3")):
        _prefix = {"vcf3": "vcf", "color3": "color", "coco3": "coco"}.get(dname, dname)
        for f in (glob.glob(f"{ROOT}/runs/*/eval/{dname}_{SPLIT}/flip_ll_step_800.json")
                  + glob.glob(f"{ROOT}/runs/*/eval/{dname}_{SPLIT}/flip_ll_noop_step_800.json")
                  + glob.glob(f"{ROOT}/runs/*/eval/{dname}_{SPLIT}/flip_ll_iv[1-5]_step_800.json")
                  # 2026-09-23 (v1.120): HELD-OUT request template evaluations (SC_TEMPLATE=heldout) -- the
                  # paper's headline for every learned interface; keyed `+ho`, the in-template file stays as is
                  + glob.glob(f"{ROOT}/runs/*/eval/{dname}_{SPLIT}/flip_ll_heldout_step_800.json")
                  # 2026-09-24 (v1.153): the FIXED ladder protocol (SC_LADDER_PROTO=2: one candidate set per example in
                  # every version, one template per example). Its own cells (`+p2`), never merged with protocol 1.
                  + glob.glob(f"{ROOT}/runs/*/eval/{dname}_{SPLIT}/flip_ll_p2_step_800.json")
                  + glob.glob(f"{ROOT}/runs/*/eval/{dname}_{SPLIT}/flip_ll_p2_heldout_step_800.json")
                  + glob.glob(f"{ROOT}/runs/*/eval/{dname}_{SPLIT}/flip_ll_noop_p2_step_800.json")
                  + glob.glob(f"{ROOT}/runs/*/eval/{dname}_{SPLIT}/flip_ll_iv[1-5]_p2_step_800.json")):
            # 2026-09-17: the grid is the IN-DISTRIBUTION reference. A run trained on another task and
            # evaluated here (the word-aliasing transfer evals of 654686 write into foreign run dirs) is
            # a transfer measurement, not a grid cell, and must never compete for the same key.
            if f.split("/runs/")[1].split("/")[0].split("_")[0] != _prefix:
                continue
            run = f.split("/runs/")[1].split("/")[0]
            if _void(run):
                continue
            d = load(f)
            if not d or d.get("look_flip") is None:
                continue
            # 2026-09-21: the bank is bfloat16. A float32 evaluation (V100 pool) agrees on summary
            # metrics to <=.014 but only .96-.99 per item, which missed the pre-registered bar, so it
            # never enters the grid (STATUS v1.70). Files written before the dtype field exists carry
            # no key and are bf16 by construction.
            if d.get("weight_dtype") not in (None, "bfloat16"):
                continue
            # 2026-09-14: English phrasing files. iv5 is the paper's instruction row ("instr"); the
            # original phrasing is "instr_v0" and iv1-iv4 are "instr_iv1".."instr_iv4" (appendix table).
            # An iv-tagged file whose method field is not frozen is a neologism eval that happened to
            # run with the env var set (qwen2b/e4b neo dirs) and is NOT an instruction row.
            _ivm = re.search(r"flip_ll_iv(\d)(?:_p2)?_step_800", os.path.basename(f))
            if _ivm and str(d.get("method")) not in ("frozen", "instr"):
                continue
            args = cfg(run)
            # IN-DOMAIN ONLY: transfer evals (a vcf-trained run evaluated on
            # color data) live in the same eval-dir layout and were silently
            # colliding with in-domain rows (caught 2026-08-07; the ColorSwap
            # PT read=.21 cell was a SEED-2 + cross-run artifact).
            dd = os.path.basename(str(args.get("data_dir") or ""))
            if args and dd and dd != dname and dd != _prefix:
                continue
            # cfg-less frozen vehicles are in-domain by construction: their
            # eval dir name IS the dataset the eval ran on
            meth, L = method_of(run, args)
            if meth is None:                     # frozen rows carry method in the json
                meth = {"frozen": "instr", "instr": "instr", "noop": "noop"}.get(
                    str(d.get("method")), None)
                if meth is None:
                    continue
                if meth == "instr":
                    meth = ("instr" if _ivm and _ivm.group(1) == "5" else
                            f"instr_iv{_ivm.group(1)}" if _ivm else "instr_v0")
                L = None
            if "heldout" in os.path.basename(f):
                meth = meth + "+ho"
            if "_p2_" in os.path.basename(f):
                meth = meth + "+p2"          # v1.153: protocol 2, last suffix
            key = (ds, model_of(run, args), meth, L)
            row = {"R.look": d.get("look_acc"), "G.look": d.get("look_flip"),
                   "R.read": d.get("read_acc"), "G.read": d.get("read_flip"),
                   "S.look": d.get("look_stay"), "S.read": d.get("read_stay"),
                   # Sc = conjunctive stay (correct before AND unchanged under the competing edit),
                   # the mirror of flip; summary field since 2026-09-13, derived from the per-item
                   # records for older files (2026-09-14).
                   "Sc.look": stay_c(d, "look"), "Sc.read": stay_c(d, "read")}
            nm = (seed_of(args), os.path.getmtime(f))
            _ts = (args or {}).get("train_sources")
            if _ts:
                # 2026-09-23 (v1.125): one LoRA adapter per request -- keep this request's fields, merge halves
                _src = _ts.split(",")[0]
                row = {k: v for k, v in row.items() if k.endswith("." + _src)}
                if key in grid and RUNS.get(key) and run not in str(RUNS[key]):
                    merged = dict(grid[key]); merged.update(row); grid[key] = merged
                    RUNS[key] = f"{RUNS[key]}+{run}"; meta[key] = nm
                elif key not in grid or better(nm, meta[key]):
                    grid[key] = row; meta[key] = nm; RUNS[key] = run
                continue
            if key not in grid or better(nm, meta[key]):
                grid[key] = row; meta[key] = nm; RUNS[key] = run

    # ---------------- Ground-CF
    # 2026-09-22 (STATUS v1.86): SC_GCF_TAG=_cp4 restricts Ground-CF to the conflict-preserving four-skill
    # files; without it the pre-fix (agreement-edit) files would be harvested next to them.
    _gcf_tag = os.environ.get("SC_GCF_TAG", "")
    for f in glob.glob(f"{ROOT}/runs/*/eval/clevr3_{SPLIT}/flip_step_800*.json"):
        run = f.split("/runs/")[1].split("/")[0]
        if _void(run):
            continue
        base = os.path.basename(f)
        if _gcf_tag and _gcf_tag not in base:
            continue
        if "_probe_" in base:
            continue          # v1.130: interface-probe evaluations are read by the tracker, never grid cells
        if "_picks" in base or "_smoke" in base:
            continue
        if "_agree" in base:
            continue          # v1.161: old-vs-new (agreement-twin) diagnostic, never a grid cell
        if "_trcar" in base:
            continue          # v1.165: held-out question + TRAINING carriers (wording-shift diagnostic), never a grid cell
        if "_reloc" in base:
            continue          # v1.166: control relocated to an untrained position (SC_GCF_CTRL_POS), never a grid cell
        if "_ctw" in base or "_restwin" in base:
            continue          # v1.189/v1.191: clean-twin reruns are PROMOTED into the canonical name (scripts/promote_clean_twins.py);
                              # the _ctw source copy and the demoted residue-twin version (_restwin) are never grid cells
        # 2026-09-17: same guard. 654686 evaluated COCO- and ColorSwap-trained words on Ground-CF via
        # word aliasing and wrote flip_step_800.json into runs/coco_*/eval/clevr3_test_id/. method_of()
        # mapped those to the SAME key as the genuine Ground-CF word and the newer mtime WON, so Table 1
        # printed a COCO word's transfer score (.36/.17) as the Ground-CF word's own result (.88/1.0).
        if not run.startswith("clevr3"):
            continue
        d = load(f)
        if not d or d.get("look_flip") is None:
            continue
        # STATUS v1.95: a batched-scoring file must have reproduced the single-sequence picks on its checked items
        _bc = d.get("batched_check") or {}
        if d.get("batched_scoring") and _bc.get("n") and (_bc.get("pick_agreement") or 0) < 1.0:
            print(f"[skip] {base} in {run}: batched pick agreement {_bc.get('pick_agreement')} < 1.0 (v1.95)")
            continue
        args = cfg(run)
        if "_instr" in base:
            # 2026-09-23 (v1.114): the FILE records which instruction ran (`instr_variant`, added to
            # eval_clevr3_flip.py); prefer it. The filename regex stays as the fallback for older files.
            # Ground-CF's conflict-preserving baselines run INSTR_VARIANT=5 but are tagged `_cp4_instr`,
            # so the old filename-only rule filed them as v0 and they never joined the ladder's `instr` row.
            _iv = d.get("instr_variant")
            if _iv is None:
                _ivm = re.search(r"_instr_iv(\d)", base)
                _iv = int(_ivm.group(1)) if _ivm else 0
            meth = "instr" if int(_iv) == 5 else (f"instr_iv{_iv}" if _iv else "instr_v0")
            L = None
        elif "_noop" in base:
            meth, L = "noop", None
        else:
            meth, L = method_of(run, args)
        if meth is None:
            continue
        # 2026-09-23 (v1.119): a question-phrasing-shift evaluation (SC_QSTYLE=generic, tag _cp4_qgen) is its own
        # cell; without the suffix it would replace the in-phrasing reference cell through newer-file-wins.
        if "_qgen" in base:
            meth = meth + "+qgen"
        # 2026-09-24 (v1.143): answer-token-only decision score (SC_SCORE=answer, tag *_ans) is its own cell, never
        # merged with the banked whole-turn cells through newer-file-wins.
        if re.search(r"_ans(?=\.json$|_)", base):
            meth = meth.replace("+2s", "+ans+2s") if "+2s" in meth else meth + "+ans"
        # 2026-09-23 (v1.120): held-out question phrasing (SC_QSTYLE=heldout, tag _cp4_ho / _cp4xfer_ho) = the
        # headline cell for learned interfaces. `+ho` goes BEFORE `+2s` because cpgrid appends `+2s` last.
        if re.search(r"_ho(?=\.json$|_)", base):
            meth = meth.replace("+2s", "+ho+2s") if "+2s" in meth else meth + "+ho"
        # 2026-09-23 (v1.124): frozen baselines evaluated on FACT captions (SC_CAPTION_STYLE_OVERRIDE=fact, tag _fc)
        if "_fc" in base and "+fc" not in meth:
            meth = meth.replace("+2s", "+fc+2s") if "+2s" in meth else meth + "+fc"
        # 2026-09-22 (STATUS v1.98): the two-source suffix describes the EVALUATION that was run, which the
        # result file records in `sources`. A frozen baseline is hosted by whatever checkpoint happened to load
        # the backbone, so its run config need not carry the protocol; the file always does.
        _srcs = d.get("sources")
        _two = (_srcs == ["look", "read"]) or ((args or {}).get("c3_sources") == "look,read")
        if meth.startswith("noop") or meth.startswith("instr"):     # v1.139: "noop+fc" missed +2s (fact-caption tag came first)
            if _two:
                meth = meth + "+2s"
        # PT_INSERTED is only wired into the two-source request template; on
        # clevr3 the flag is a no-op, so pti runs are byte-identical duplicates
        # of pt (verified 2026-08-19: all 4 models, all 9 metrics identical).
        if meth == "pti":
            continue
        key = ("groundcf", model_of(run, args), meth, L)
        row = {}
        for ch in ("look", "read", "ocr"):
            row[f"Sc.{ch}"] = stay_c(d, ch)
            row[f"R.{ch}"] = d.get(f"{ch}_acc")
            row[f"G.{ch}"] = d.get(f"{ch}_flip")
            row[f"S.{ch}"] = d.get(f"{ch}_stay")
        # DENOMINATORS (2026-09-13): on Ground-CF look_acc is scored over ALL items while
        # look_flip is scored on the image-twin subset (colour and shape only; count and
        # spatial have no image twin). Comparing them is the different-denominator error the
        # ledger warns about, so the twin-subset accuracy is carried alongside as R.looktwin
        # and is the value any table that prints accuracy NEXT TO look flip must use.
        row["R.looktwin"] = d.get("look_twin_acc")
        # 2026-09-23 (STATUS v1.110): Ground-CF twins were extended from two skills (colour, shape) to four
        # (plus count and spatial). Both vintages are on disk under the same filename, and the old keep-newest
        # rule let a TWO-skill file win a cell while a sibling arm's cell came from a FOUR-skill file -- the
        # two are not comparable (a two-skill evaluation omits the harder half). A cell now prefers the file
        # with more skills; `n_skills` is carried into the row so any table can state it.
        _nsk = len(d.get("per_skill") or {})
        row["n_skills"] = _nsk
        nm = (seed_of(args), os.path.getmtime(f))
        _ts = (args or {}).get("train_sources")
        if _ts:
            # 2026-09-23 (v1.124): modality-specific adapter -- keep only THIS request's fields (the other
            # request was scored with the wrong adapter loaded) and merge with the other half; provenance = both runs.
            _src = _ts.split(",")[0]
            row = {k: v for k, v in row.items() if k.endswith("." + _src) or (k.endswith(".looktwin") and _src == "look") or k == "n_skills"}
            if key in grid and RUNS.get(key) and run not in str(RUNS[key]):
                merged = dict(grid[key]); merged.update(row); grid[key] = merged
                RUNS[key] = f"{RUNS[key]}+{run}"; meta[key] = nm
            elif key not in grid or better(nm, meta[key]):
                grid[key] = row; meta[key] = nm; RUNS[key] = run
            continue
        _prev = grid.get(key)
        if _prev is not None and (_prev.get("n_skills") or 0) != _nsk:
            if _nsk < (_prev.get("n_skills") or 0):
                continue                      # keep the richer evaluation
            grid[key] = row; meta[key] = nm; RUNS[key] = run; continue
        if key not in grid or better(nm, meta[key]):
            grid[key] = row; meta[key] = nm; RUNS[key] = run

    # ---------------- MVG (grounding battery; <ocr> is an ablation, starred)
    # 2026-09-21: the rebuilt-metadata evaluation (pairs_clean_rb, tag _rb) is banked under its OWN
    # dataset key "mvgrb", never mixed with the _clean cells: different pairs file, different numbers.
    for f in (glob.glob(f"{ROOT}/runs/*/eval/mvg_flip/flip_step_800_clean.json")
              + glob.glob(f"{ROOT}/runs/*/eval/mvg_flip/flip_step_800_rb.json")):
        run = f.split("/runs/")[1].split("/")[0]
        _dskey = "mvgrb" if f.endswith("_rb.json") else "mvg"
        d = load(f)
        if not d:
            continue
        vis = (d.get("vis_value_swap") or {}).get("flip")
        if vis is None:
            continue
        args = cfg(run)
        if "mvg" not in str(args.get("task") or "") and \
           "mvg" not in str(args.get("data_dir") or ""):
            continue                              # exclude transfer evals (in-domain only)
        meth, L = method_of(run, args)
        if meth is None:
            continue
        key = (_dskey, model_of(run, args), meth, L)
        row = {"G.look": vis, "G.read": (d.get("desc_swap") or {}).get("flip"),
               "G.ocr*": (d.get("ocr_ablate") or {}).get("drop_pos")}
        nm = (seed_of(args), os.path.getmtime(f))
        if key not in grid or better(nm, meta[key]):
            grid[key] = row; meta[key] = nm; RUNS[key] = run
    for style in ("instr", "noop"):
        # 2026-09-23 (v1.117): Qwen-2B/E4B frozen files are tagged `_clean_noop_noop.json` / `_clean_instr_instr.json`
        # (style AND tag appended), so the exact-name glob only ever matched Gemma-3 -- the MVG grid had baselines
        # for one model. `clean_{style}*` still excludes the crop files (`clean_crop0.25_...`).
        for f in glob.glob(f"{ROOT}/runs/_frozen_*/eval/mvg_flip/flip_step_800_clean_{style}*.json") + \
                 glob.glob(f"{ROOT}/runs/frozen_*/eval/mvg_flip/flip_step_800_clean_{style}*.json"):
            run = f.split("/runs/")[1].split("/")[0]
            d = load(f)
            if not d:
                continue
            model = ("qwen4b" if "qwen35" in run else "qwen2b" if "qwen2b" in run
                     else "e4b" if "e4b" in run or "gemma4" in run else "gemma3")
            vis = (d.get("vis_value_swap") or {}).get("flip")
            if vis is None:
                continue
            grid[("mvg", model, style, None)] = {
                "G.look": vis, "G.read": (d.get("desc_swap") or {}).get("flip"),
                "G.ocr*": (d.get("ocr_ablate") or {}).get("drop_pos")}

    # ---------------- MVR (JOINT per channel, ood)
    for f in glob.glob(f"{ROOT}/runs/*/eval/mvraw_test_ood/ablate*_step_800.json"):
        run = f.split("/runs/")[1].split("/")[0]
        if _void(run):
            continue
        base = os.path.basename(f)
        d = load(f)
        if not d:
            continue
        pc = d.get("per_channel") or {}
        row = {}
        for ch, lab in (("VIS", "J.look"), ("OCR", "J.ocr"), ("ASR", "J.asr")):
            v = pc.get(ch) or {}
            n, s = v.get("nec_pos"), v.get("spec_pos")
            if None not in (n, s):
                row[lab] = min(n, s)
        if not row:
            continue
        if run.startswith(("_frozen", "frozen")):
            meth = "instr" if "instr" in base else "noop" if "noop" in base else None
            if meth is None:
                continue
            model = ("qwen4b" if "qwen35" in run else "qwen2b" if "qwen2b" in run
                     else "e4b" if "e4b" in run or "gemma4" in run else "gemma3")
            L = None
        else:
            args = cfg(run)
            meth, L = method_of(run, args)
            model = model_of(run, args)
            if meth is None:
                continue
        key = ("mvr", model, meth, L)
        nm = (seed_of(cfg(run)), os.path.getmtime(f))
        if key not in grid or better(nm, meta[key]):
            grid[key] = row; meta[key] = nm; RUNS[key] = run

    # ---------------- render
    ORDER = ["noop", "instr", "pt", "lora_r1", "lora_r4", "lora_r16", "lora_r64",
             "word_sft", "word+v3", "word+apo", "word+ctm", "word+ctm+tc", "word+ecc"]
    def mkey(m):
        return (ORDER.index(m), ) if m in ORDER else (len(ORDER), m)

    out_json = {}
    lines = ["# MASTER GRID — routing (R) / grounding (G) per channel",
             "Generated by scripts/master_grid.py; '--' = not measured; L blank = "
             "not applicable (baselines) ; G.ocr* on MVG = deletion-ablation, not a flip.",
             ""]
    for ds, label, cols in (
            ("vcf", "VCF", ["R.look", "G.look", "R.read", "G.read"]),
            ("color", "ColorSwap", ["R.look", "G.look", "R.read", "G.read"]),
            ("coco", "COCO-CF", ["R.look", "G.look", "R.read", "G.read"]),
            # 2026-09-22 (STATUS v1.98): the conflict-preserving (third-value) ladder batteries. They were
            # harvested into `grid` from the first day (the loop above reads eval/<ds>3_<split>/) but this
            # output list did not name them, so every *3 row was silently dropped from master_grid.{md,json}
            # and no corrected ladder cell could ever reach a table or figure.
            ("vcf3", "VCF (conflict-preserving)", ["R.look", "G.look", "R.read", "G.read"]),
            ("color3", "ColorSwap (conflict-preserving)", ["R.look", "G.look", "R.read", "G.read"]),
            ("coco3", "COCO-CF (conflict-preserving)", ["R.look", "G.look", "R.read", "G.read"]),
            ("groundcf", "Ground-CF", ["R.look", "G.look", "R.read", "G.read", "R.ocr", "G.ocr"]),
            ("mvg", "MultiVENT-G", ["G.look", "G.read", "G.ocr*"]),
            ("mvr", "MultiVENT-Raw (JOINT, ood)", ["J.look", "J.ocr", "J.asr"])):
        lines += [f"\n## {label}", "",
                  "| model | method | L | " + " | ".join(cols) + " |",
                  "|---|---|---|" + "---|" * len(cols)]
        for model in ("gemma3", "qwen4b", "qwen2b", "e4b"):
            rows = sorted(((m, L, v) for (d2, mo, m, L), v in grid.items()
                           if d2 == ds and mo == model),
                          key=lambda x: (mkey(x[0]), x[1] or 0))
            for m, L, v in rows:
                out_json[f"{ds}|{model}|{m}|{L}"] = v
                lines.append(f"| {model} | {m} | {L or ''} | " +
                             " | ".join(fmt(v.get(c)) for c in cols) + " |")
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    open(f"{ROOT}/results/master_grid{SUFFIX}.md", "w").write("\n".join(lines) + "\n")
    json.dump(out_json, open(f"{ROOT}/results/master_grid{SUFFIX}.json", "w"), indent=1)
    json.dump({f"{d}|{mo}|{m}|{L}": r for (d, mo, m, L), r in RUNS.items()},
              open(f"{ROOT}/results/master_grid{SUFFIX}_runs.json", "w"), indent=1)
    n = len(out_json)
    per_ds = collections.Counter(k.split("|")[0] for k in out_json)
    print(f"[master_grid:{SPLIT}] {n} rows -> results/master_grid{SUFFIX}.md   " +
          " ".join(f"{k}={v}" for k, v in sorted(per_ds.items())))


if __name__ == "__main__":
    main()
