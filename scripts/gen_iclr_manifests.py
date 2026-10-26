#!/usr/bin/env python
"""Generate the ICLR-campaign manifests (configs/iclr/) from REFERENCE RUNS.

WHY REFERENCE-BASED: every new cell must be comparable to the banked grid, and
the recipes have sharp edges (clevr3 arms carry held_cells; mvg carries
iit_layer/aat_layer/cap_first_prob; mvr carries n_frames/pref_warmup). Hand-
typing 150 lines re-introduces the class of bug that cost the APO axis (§5).
So each line is the ACTUAL args dict of a step_800-verified run, with only
{model_id, n_tokens, seed, lora_r, run_name} overridden.

IDEMPOTENT: a cell whose run_name already has ckpt/step_800 is skipped, so the
script can be re-run after partial waves without duplicating work.

Waves (one manifest per GPU class -- placement IS the OOM plan, HANDOVER §12):
  A_synth_l40s   ladder + Ground-CF fills (new models, L=15, seeds, PT/LoRA)
  B_video_a100   MVG fills (q2b/e4b words, LoRA r4, L=30) -- 1 frame, 80GB
  C_mvr_h200     MVR fills (L-sweep, q2b/e4b words, qwen ctm retrain)
  D1_gen_l40s    coco gen words (prereq for link; deterministic ckpt path)
  D2_link_l40s   coco link + delimw variants + blockscmp (afterok: D1)
  E1_probe_h200  MVR-LoRA OOM isolation pair (save-trigger vs frame count)
  *_eval         guarded eval lines for every train line above

Usage: python scripts/gen_iclr_manifests.py   (writes configs/iclr/, prints counts)
"""
import json, os, sys, glob, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "configs", "iclr")
os.makedirs(OUT, exist_ok=True)

MODELS = {"g3": None, "q4b": "Qwen/Qwen3.5-4B",
          "q2b": "Qwen/Qwen3.5-2B", "e4b": "google/gemma-4-E4B-it"}
SFX = {"g3": "", "q4b": "_qwen35", "q2b": "_qwen2b", "e4b": "_gemma4e4b"}


def ref_args(run):
    a = json.load(open(f"{ROOT}/runs/{run}/config.json"))["args"]
    return {k: v for k, v in a.items()
            if v is not None and v is not False and v != ""}


def emit(args, **over):
    a = dict(args)
    a.update({k: v for k, v in over.items() if v is not None})
    for k in [k for k, v in over.items() if v is None]:
        a.pop(k, None)                      # explicit None removes a key
    parts = []
    for k, v in a.items():
        parts.append(f"--{k} {v}")
    return " ".join(parts)


def done(run_name):
    return os.path.isdir(f"{ROOT}/runs/{run_name}/ckpt/step_800")


class Wave:
    def __init__(self, name):
        self.name, self.lines, self.skipped = name, [], 0

    def add(self, line, run_name):
        if done(run_name):
            self.skipped += 1
            return False
        self.lines.append(line)
        return True

    def write(self):
        p = f"{OUT}/{self.name}.txt"
        open(p, "w").write("\n".join(self.lines) + ("\n" if self.lines else ""))
        print(f"  {self.name:16s} {len(self.lines):4d} lines  ({self.skipped} cells already done)")
        return p


GUARD = "[ -d runs/{r}/ckpt ] && {c} || true"


def ladder_evals(run, ds):
    c = []
    for sp in ("test_id", "test_ood"):
        c.append(GUARD.format(r=run, c=f"python scripts/eval_flip_ll.py --run_dir runs/{run} "
                                       f"--task color --data_dir data/{ds} --split {sp} --max_n 140"))
    c.append(GUARD.format(r=run, c=f"python scripts/eval_ladder.py --run_dir runs/{run} "
                                   f"--task color --data_dir data/{ds} --split test_id"))
    return c


def clevr3_evals(run):
    c = []
    for sp in ("test_id", "test_ood"):
        c.append(GUARD.format(r=run, c=f"python scripts/eval_clevr3.py --run_dir runs/{run} --split {sp} --step 800"))
        c.append(GUARD.format(r=run, c=f"python scripts/eval_clevr3_flip.py --run_dir runs/{run} "
                                       f"--data_dir data/clevr3 --split {sp} --step 800"))
    return c


def clevr3_frozen_evals(run):
    # instructed/noop ignore the learned word: any run_dir of the right model
    # serves as the frozen-baseline vehicle (the banked g3/q4b rows were made
    # exactly this way, configs/CLEVR3_BASELINE_FLIP.txt).
    c = []
    for mode, tag in (("instructed", "_instr"), ("noop", "_noop")):
        for sp in ("test_id", "test_ood"):
            c.append(GUARD.format(r=run, c=f"python scripts/eval_clevr3_flip.py --run_dir runs/{run} "
                                           f"--split {sp} --step 800 --mode {mode} --tag {tag}"))
    return c


def mvg_evals(run):
    pre = "QWEN_MAX_TOK=256 " if "qwen" in run else ""
    c = [GUARD.format(r=run, c=f"{pre}python scripts/eval_mvg3.py --run_dir runs/{run} --step 800")]
    for tag, extra in (("_clean", ""), ("_clean_crop", " --crop_bbox 0.25")):
        c.append(GUARD.format(r=run, c=f"{pre}python scripts/eval_mvg_flip.py --run_dir runs/{run} --step 800 "
                                       f"--pairs data/mvg_ctm/pairs_clean.jsonl --tag {tag} --max_per_type 300{extra}"))
    return c


def mvr_evals(run):
    pre = "QWEN_MAX_TOK=256 " if "qwen" in run else ""
    c = []
    for sp in ("test_id", "test_ood"):
        c.append(GUARD.format(r=run, c=f"{pre}python scripts/eval_mvraw_ablate.py --run_dir runs/{run} "
                                       f"--data_dir data/multivent_raw_clean --split {sp} --step 800 "
                                       f"--n_frames 2 --max_n 200"))
    c.append(GUARD.format(r=run, c=f"{pre}python scripts/eval_mvraw_swap.py --run_dir runs/{run} "
                                   f"--split test_ood --step 800 --n_frames 2 --max_n 400"))
    return c


def main():
    A, Ae = Wave("A_synth_l40s"), Wave("A_eval")
    B, Be = Wave("B_video_a100"), Wave("B_eval")
    C, Ce = Wave("C_mvr_h200"), Wave("C_eval")
    D1, D2, De = Wave("D1_gen_l40s"), Wave("D2_link_l40s"), Wave("D_eval")
    E1 = Wave("E1_probe_h200")

    # ---------------- references (all verified step_800 on disk)
    R = {a: ref_args(f"clevr3_f_5tok_{a}_seed0") for a in ("sft", "ctm", "ctmv3", "apo")}
    R["clevr3_pt"] = ref_args("clevr3_f_pt_5tok_seed0")
    R["clevr3_lora"] = ref_args("clevr3_f_lora_r4_seed0")
    R["v3"] = ref_args("coco_neo_ctrl_5tok_v3_seed0")            # ladder v3
    R["lad_sft"] = ref_args("coco_neo_ctrl_5tok_sft_seed0")
    R["lad_pt"] = ref_args("color_pt_na_5tok_sft_seed0")   # banked PT recipe (pt_inserted=0)
    R["lad_lora"] = ref_args("color_lora_r4_lr2e4_seed0")
    R["mvg_sftctm"] = ref_args("mvgV2_5tok_sftctm_hard_seed0")
    R["mvg_pt"] = ref_args("mvgV2_5tok_pt_seed0")
    R["mvg_lora"] = ref_args("mvgV2_lora_r16_seed0")
    R["mvr_sft"] = ref_args("mvr_g_5tok_sft_seed0c2h2")
    R["mvr_ctm"] = ref_args("mvr_g_5tok_ctm_seed0c2h2lm")
    R["mvr_lora"] = ref_args("mvr_g_lora_r4_seed0c2h2ck")
    R["link"] = ref_args("color_neo_link_5tok_sft_seed0_v2")
    R["gen"] = ref_args("color_neo_gen_4tok_sft_seed0")

    # ============ WAVE A: synthetic, L40S ============
    # A1. Ground-CF on the two NEW models: 4 arms + PT + LoRA r4/r16 at L=5,
    #     plus the ctm L-sweep {1,15,30} (ctm = the dev-JOINT winner).
    for m in ("q2b", "e4b"):
        mid, sfx = MODELS[m], SFX[m]
        for arm in ("sft", "ctm", "ctmv3", "apo"):
            rn = f"clevr3_f_5tok_{arm}_seed0{sfx}"
            if A.add(emit(R[arm], model_id=mid, run_name=rn), rn):
                Ae.lines += clevr3_evals(rn)
        for L in (1, 15, 30):
            rn = f"clevr3_f_{L}tok_ctm_seed0{sfx}"
            if A.add(emit(R["ctm"], model_id=mid, n_tokens=L, run_name=rn), rn):
                Ae.lines += clevr3_evals(rn)
        rn = f"clevr3_f_pt_5tok_seed0{sfx}"
        if A.add(emit(R["clevr3_pt"], model_id=mid, run_name=rn), rn):
            Ae.lines += clevr3_evals(rn)
        for r in (4, 16):
            rn = f"clevr3_f_lora_r{r}_seed0{sfx}"
            if A.add(emit(R["clevr3_lora"], model_id=mid, lora_r=r, run_name=rn), rn):
                Ae.lines += clevr3_evals(rn)
        # frozen instruction/no-op rows for the new model (eval-only, uses sft dir)
        Ae.lines += clevr3_frozen_evals(f"clevr3_f_5tok_sft_seed0{sfx}")

    # A2. Ground-CF L=15 fill on g3/q4b, all four arms (keeps the sweep
    #     arm-complete; existing grid is {1,2,5,10,20,30}).
    for m in ("g3", "q4b"):
        for arm in ("sft", "ctm", "ctmv3", "apo"):
            rn = f"clevr3_f_15tok_{arm}_seed0{SFX[m]}"
            if A.add(emit(R[arm], model_id=MODELS[m], n_tokens=15, run_name=rn), rn):
                Ae.lines += clevr3_evals(rn)

    # A3. coco on e4b (the only ladder x model hole): v3 L-sweep + sft
    for L in (1, 2, 5, 10, 15, 20, 25, 30):
        rn = f"coco_neo_ctrl_{L}tok_v3_seed0_gemma4e4b"
        if A.add(emit(R["v3"], model_id=MODELS["e4b"], n_tokens=L,
                      data_dir="data/coco", run_name=rn), rn):
            Ae.lines += ladder_evals(rn, "coco")
    rn = "coco_neo_ctrl_5tok_sft_seed0_gemma4e4b"
    if A.add(emit(R["lad_sft"], model_id=MODELS["e4b"], run_name=rn), rn):
        Ae.lines += ladder_evals(rn, "coco")
    for L in (25, 30):                                   # vcf e4b top-end fill
        rn = f"vcf_neo_ctrl_{L}tok_v3_seed0_gemma4e4b"
        if A.add(emit(R["v3"], model_id=MODELS["e4b"], n_tokens=L,
                      data_dir="data/vcf", run_name=rn), rn):
            Ae.lines += ladder_evals(rn, "vcf")

    # A4. PT + LoRA r4/r16 for every (model, ladder ds) hole
    LAD = {"color": "data/color", "vcf": "data/vcf", "coco": "data/coco"}
    have = set()
    for c in glob.glob(f"{ROOT}/runs/*/config.json"):
        run = c.split("/runs/")[1].split("/")[0]
        if not done(run):
            continue
        try:
            a = json.load(open(c))["args"]
        except Exception:
            continue
        mid = a.get("model_id") or ""
        m = ("q4b" if "3.5-4B" in mid else "q2b" if "3.5-2B" in mid
             else "e4b" if "E4B" in mid else "g3")
        dd = a.get("data_dir") or ""
        ds = next((d for d, p in LAD.items() if p in dd or f"/{d}" in dd), None)
        if ds is None and run.startswith("coco"):
            ds = "coco"
        if ds:
            have.add((ds, m, a.get("method"), a.get("lora_r")
                      or json.load(open(c)).get("lora", {}).get("r")))
    for ds, dd in LAD.items():
        for m in ("g3", "q4b", "q2b", "e4b"):
            mid, sfx = MODELS[m], SFX[m]
            if (ds, m, "pt", None) not in have and not any(
                    h[:3] == (ds, m, "pt") for h in have):
                rn = f"{ds}_pt_na_5tok_sft_seed0{sfx}"
                if A.add(emit(R["lad_pt"], model_id=mid, data_dir=dd, run_name=rn), rn):
                    Ae.lines += ladder_evals(rn, dd.split("/")[-1])
            for r in (4, 16):
                if not any(h[:3] == (ds, m, "lora") and h[3] == r for h in have):
                    rn = f"{ds}_lora_r{r}_lr2e4_seed0{sfx}"
                    if A.add(emit(R["lad_lora"], model_id=mid, data_dir=dd,
                                  lora_r=r, run_name=rn), rn):
                        Ae.lines += ladder_evals(rn, dd.split("/")[-1])

    # A5. SEEDS on the headline cells (watertight list #2): sft+pt seeds 1,2 on
    #     color+coco x g3/q4b (v3 seeds already banked), ctm+sft seeds on clevr3.
    for m in ("g3", "q4b"):
        mid, sfx = MODELS[m], SFX[m]
        for s in (1, 2):
            for ds, dd in (("color", "data/color"), ("coco", "data/coco")):
                rn = f"{ds}_neo_ctrl_5tok_sft_seed{s}{sfx}"
                if A.add(emit(R["lad_sft"], model_id=mid, data_dir=dd, seed=s, run_name=rn), rn):
                    Ae.lines += ladder_evals(rn, dd.split("/")[-1])
                rn = f"{ds}_pt_na_5tok_sft_seed{s}{sfx}"
                if A.add(emit(R["lad_pt"], model_id=mid, data_dir=dd, seed=s, run_name=rn), rn):
                    Ae.lines += ladder_evals(rn, dd.split("/")[-1])
            for arm in ("sft", "ctm"):
                rn = f"clevr3_f_5tok_{arm}_seed{s}{sfx}"
                if A.add(emit(R[arm], model_id=mid, seed=s, run_name=rn), rn):
                    Ae.lines += clevr3_evals(rn)

    # ============ WAVE B: MVG, A100-80GB (1 frame) ============
    for m in ("q2b", "e4b"):
        mid, sfx = MODELS[m], SFX[m]
        for arm, over in (("sftctm", {}), ("sft", {"lam_ctm": None})):
            rn = f"mvgV2_5tok_{arm}_hard_seed0{sfx}"
            if B.add(emit(R["mvg_sftctm"], model_id=mid, run_name=rn, **over), rn):
                Be.lines += mvg_evals(rn)
        if m == "q2b":                                  # full baseline set on 2B
            rn = f"mvgV2_5tok_pt_seed0{sfx}"
            if B.add(emit(R["mvg_pt"], model_id=mid, run_name=rn), rn):
                Be.lines += mvg_evals(rn)
            for r in (4, 16):
                rn = f"mvgV2_lora_r{r}_seed0{sfx}"
                if B.add(emit(R["mvg_lora"], model_id=mid, lora_r=r, run_name=rn), rn):
                    Be.lines += mvg_evals(rn)
    for m in ("g3", "q4b"):                             # LoRA r4 (paper wants r4+r16)
        rn = f"mvgV2_lora_r4_seed0{SFX[m]}"
        if B.add(emit(R["mvg_lora"], model_id=MODELS[m], lora_r=4, run_name=rn), rn):
            Be.lines += mvg_evals(rn)
    for m in ("g3", "q4b"):                             # L=30 completes the MVG axis
        rn = f"mvgV2_30tok_sftctm_hard_seed0{SFX[m]}"
        if B.add(emit(R["mvg_sftctm"], model_id=MODELS[m], n_tokens=30, run_name=rn), rn):
            Be.lines += mvg_evals(rn)
    # ecc qwen4b: trained (8 ckpts) but never evaluated -- eval-only lines
    Be.lines += mvg_evals("mvgV2_5tok_hard_ecc_seed0_qwen35")

    # ============ WAVE C: MVR, H200 ============
    for L in (1, 15, 30):                               # L-sweep on the sft arm
        rn = f"mvr_g_{L}tok_sft_seed0c2h2"
        if C.add(emit(R["mvr_sft"], n_tokens=L, run_name=rn), rn):
            Ce.lines += mvr_evals(rn)
    for m in ("q2b", "e4b"):                            # small/alt models
        mid, sfx = MODELS[m], SFX[m]
        for arm, ref in (("sft", "mvr_sft"), ("ctm", "mvr_ctm")):
            if m == "e4b" and arm == "ctm":
                continue                                # e4b ctm quad won't fit; sft first
            rn = f"mvr_g_5tok_{arm}_seed0{sfx}c2h2"
            if C.add(emit(R[ref], model_id=mid, run_name=rn), rn):
                Ce.lines += mvr_evals(rn)
    rn = "mvr_g_5tok_ctm_seed0_qwen35c2h2lm2"           # q4b ctm died at step_385
    if C.add(emit(R["mvr_ctm"], model_id=MODELS["q4b"], run_name=rn), rn):
        Ce.lines += mvr_evals(rn)

    # ============ WAVE D: generation words + link + composition ============
    for m in ("g3", "q4b"):
        rn = f"coco_neo_gen_5tok_sft_seed0{SFX[m]}_v2"
        D1.add(emit(R["gen"], model_id=MODELS[m], n_tokens=5,
                    data_dir="data/coco", run_name=rn), rn)
    for m in ("g3", "q4b"):
        gen_ck = f"runs/coco_neo_gen_5tok_sft_seed0{SFX[m]}_v2/ckpt/step_800/neo.pt"
        rn = f"coco_neo_link_5tok_sft_seed0{SFX[m]}_v2"
        if D2.add(emit(R["link"], model_id=MODELS[m], data_dir="data/coco",
                       gen_ckpt=gen_ck, run_name=rn), rn):
            De.lines += ladder_evals(rn, "coco")
            De.lines.append(GUARD.format(
                r=rn, c=f"python scripts/eval_order.py --run_dir runs/{rn} "
                        f"--task color --data_dir data/coco --split test_id"))
    # delim_w=0.2 (format pressure off the content CE) on vcf+coco, g3
    for ds, dd in (("vcf", "data/vcf"), ("coco", "data/coco")):
        gen_ck = ("runs/vcf_neo_gen_5tok_sft_seed0_v2/ckpt/step_800/neo.pt" if ds == "vcf"
                  else "runs/coco_neo_gen_5tok_sft_seed0_v2/ckpt/step_800/neo.pt")
        rn = f"{ds}_link_delimw_seed0"
        if D2.add(emit(R["link"], data_dir=dd, gen_ckpt=gen_ck,
                       delim_loss_weight=0.2, run_name=rn), rn):
            De.lines += ladder_evals(rn, dd.split("/")[-1])
    # ordered composition (blockscmp) on color+vcf, g3+q4b
    for m in ("g3", "q4b"):
        for ds, dd in (("color", "data/color"), ("vcf", "data/vcf")):
            rn = f"{ds}_blockscmp_5tok_seed0{SFX[m]}"
            if D2.add(emit(R["lad_sft"], model_id=MODELS[m], data_dir=dd,
                           ctrl_style="blockscmp", run_name=rn), rn):
                De.lines.append(GUARD.format(
                    r=rn, c=f"python scripts/eval_order.py --run_dir runs/{rn} "
                            f"--task color --data_dir {dd} --split test_id"))
                De.lines += ladder_evals(rn, dd.split("/")[-1])

    # ============ E1: MVR-LoRA OOM isolation (2 probes, 300 steps) ============
    # (a) same shape, NO mid-run save  -> if it passes step 114, the step-100
    #     checkpoint save is what triggers the growth.
    # (b) mid-run save kept, 1 frame   -> halves activation load; if only (b)
    #     passes, it is raw footprint, not the save.
    E1.add(emit(R["mvr_lora"], steps=300, config="configs/iclr/base_nosave.yaml",
                run_name="mvr_probe_lora_nosave"), "mvr_probe_lora_nosave")
    E1.add(emit(R["mvr_lora"], steps=300, n_frames=1,
                run_name="mvr_probe_lora_1frame"), "mvr_probe_lora_1frame")

    # ============ WAVE L: the L-completion grid (user request 2026-08-06) ====
    # L in {1,5,15,30} for words AND prompt tuning, all 4 models, every
    # supported arm, every dataset. Two kinds of hole, handled separately:
    #   - trained-but-unevaluated (q2b/e4b ladder sweeps!) -> eval-only
    #     backfill wave Lbf, submit immediately, no training;
    #   - genuinely missing cells -> train wave + chained evals.
    # Documented exclusions: PT+causal on Ground-CF (Clevr3 builders have no
    # method gating); PT+causal on VCF (PT-SFT is already 1.00/1.00 there);
    # pti L-sweep (placement shown minor at L=5); PT L-sweep on video (4-12h
    # per run for a control no reader compares at those widths); MVR ctm
    # L-sweep (causal arm helps only ASR; sft is the MVR word row).
    Lw = Wave("L_synth_l40s")
    Le, Lbf = Wave("L_eval"), Wave("L_evalfill")
    LB, LBe = Wave("LV_mvg_a100"), Wave("LV_mvg_eval")
    LC, LCe = Wave("LV_mvr_h200"), Wave("LV_mvr_eval")
    for k, r in (("lad_apo", "color_neo_ctrl_5tok_apo_seed0"),
                 ("lad_ctm", "color_neo_ctrl_5tok_ctm_seed0"),
                 ("lad_vgm", "color_neo_ctrl_5tok_vgm_seed0"),
                 ("pt_apo", "color_pt_na_5tok_apo_seed0"),
                 ("pt_ctm", "color_pt_na_5tok_ctm_seed0")):
        try:
            R[k] = ref_args(r)
        except Exception:
            print(f"  !! missing ref {r} -- wave-L cells needing it are SKIPPED")

    def lad_eval_done(rn, ds):
        return os.path.exists(f"{ROOT}/runs/{rn}/eval/{ds}_test_id/flip_ll_step_800.json")

    def c3_eval_done(rn):
        return os.path.exists(f"{ROOT}/runs/{rn}/eval/clevr3_test_id/flip_step_800.json")

    def mvg_eval_done(rn):
        return os.path.exists(f"{ROOT}/runs/{rn}/eval/mvg_flip/flip_step_800_clean.json")

    def mvr_eval_done(rn):
        return bool(glob.glob(f"{ROOT}/runs/{rn}/eval/mvraw_test_ood/ablate*_step_800.json"))

    def cell(wave, evwave, bfwave, ref, rn, evfn, evdone, **over):
        if ref not in R:
            return
        if done(rn):
            if not evdone(rn):
                bfwave.lines += evfn(rn)
            return
        if wave.add(emit(R[ref], run_name=rn, **over), rn):
            evwave.lines += evfn(rn)

    LS = (1, 5, 15, 30)
    ALLM = ("g3", "q4b", "q2b", "e4b")

    # L1. ladder words: sft/v3/apo/ctm x L x model x dataset
    for ds, dd in LAD.items():
        for m in ALLM:
            for arm, ref in (("sft", "lad_sft"), ("v3", "v3"),
                             ("apo", "lad_apo"), ("ctm", "lad_ctm")):
                for L in LS:
                    rn = f"{ds}_neo_ctrl_{L}tok_{arm}_seed0{SFX[m]}"
                    cell(Lw, Le, Lbf, ref, rn,
                         lambda r, _d=ds: ladder_evals(r, _d),
                         lambda r, _d=ds: lad_eval_done(r, _d),
                         model_id=MODELS[m], n_tokens=L, data_dir=dd)
    # VGM L-axis on the exposition cells only (the refuted-arm sweep)
    for m in ("g3", "q4b"):
        for L in (1, 15, 30):
            rn = f"color_neo_ctrl_{L}tok_vgm_seed0{SFX[m]}"
            cell(Lw, Le, Lbf, "lad_vgm", rn,
                 lambda r: ladder_evals(r, "color"),
                 lambda r: lad_eval_done(r, "color"),
                 model_id=MODELS[m], n_tokens=L, data_dir="data/color")

    # L2. ladder PT: sft at every L everywhere; +apo/+ctm at every L on
    #     color+coco (the datasets where the PT-rescue story lives)
    for ds, dd in LAD.items():
        for m in ALLM:
            for L in LS:
                rn = f"{ds}_pt_na_{L}tok_sft_seed0{SFX[m]}"
                cell(Lw, Le, Lbf, "lad_pt", rn,
                     lambda r, _d=ds: ladder_evals(r, _d),
                     lambda r, _d=ds: lad_eval_done(r, _d),
                     model_id=MODELS[m], n_tokens=L, data_dir=dd)
    for ds, dd in (("color", "data/color"), ("coco", "data/coco")):
        for m in ALLM:
            for arm, ref in (("apo", "pt_apo"), ("ctm", "pt_ctm")):
                for L in LS:
                    rn = f"{ds}_pt_na_{L}tok_{arm}_seed0{SFX[m]}"
                    cell(Lw, Le, Lbf, ref, rn,
                         lambda r, _d=ds: ladder_evals(r, _d),
                         lambda r, _d=ds: lad_eval_done(r, _d),
                         model_id=MODELS[m], n_tokens=L, data_dir=dd)

    # L3. Ground-CF: word arms x L x model + PT(sft) x L x model
    for m in ALLM:
        for arm in ("sft", "ctm", "ctmv3", "apo"):
            for L in LS:
                rn = f"clevr3_f_{L}tok_{arm}_seed0{SFX[m]}"
                cell(Lw, Le, Lbf, arm, rn, clevr3_evals, c3_eval_done,
                     model_id=MODELS[m], n_tokens=L)
        for L in LS:
            rn = f"clevr3_f_pt_{L}tok_seed0{SFX[m]}"
            cell(Lw, Le, Lbf, "clevr3_pt", rn, clevr3_evals, c3_eval_done,
                 model_id=MODELS[m], n_tokens=L)

    # L4. eval-backfill sweep: ANY trained ladder/Ground-CF word/PT run whose
    #     flip eval never landed (catches the odd-L sweep points 2/10/20/25).
    #     pti runs need the env prefix at EVAL time too (eval_flip_ll.py:107).
    import re as _re
    seen_bf = set(_re.search(r"runs/([^/]+)/", l).group(1)
                  for l in Lbf.lines if _re.search(r"runs/([^/]+)/", l))
    for c in sorted(glob.glob(f"{ROOT}/runs/*/ckpt/step_800")):
        run = c.split("/runs/")[1].split("/")[0]
        if run in seen_bf:
            continue
        mm = _re.match(r"^(vcf|color|coco)_(neo_ctrl|pti?_na)_\d+tok_", run)
        if mm and not lad_eval_done(run, mm.group(1)):
            ev = ladder_evals(run, mm.group(1))
            if "_pti_" in run:
                ev = [l.replace("python scripts/", "PT_INSERTED=1 python scripts/") for l in ev]
            Lbf.lines += ev
        elif _re.match(r"^clevr3_f_(\d+tok|pt_\d+tok)", run) and not c3_eval_done(run):
            Lbf.lines += clevr3_evals(run)

    # L5. video fills (words; PT parity row for the one missing MVG model)
    for m in ALLM:
        for L in LS:
            rn = f"mvgV2_{L}tok_sftctm_hard_seed0{SFX[m]}"
            cell(LB, LBe, LBe, "mvg_sftctm", rn, mvg_evals, mvg_eval_done,
                 model_id=MODELS[m], n_tokens=L)
    for m in ("g3", "q4b"):     # sft parity on the big models (q2b/e4b banked)
        rn = f"mvgV2_5tok_sft_hard_seed0{SFX[m]}"
        cell(LB, LBe, LBe, "mvg_sftctm", rn, mvg_evals, mvg_eval_done,
             model_id=MODELS[m], lam_ctm=None)
    rn = "mvgV2_5tok_pt_seed0_qwen35"
    cell(LB, LBe, LBe, "mvg_pt", rn, mvg_evals, mvg_eval_done,
         model_id=MODELS["q4b"])
    for m in ALLM:
        for L in LS:
            rn = f"mvr_g_{L}tok_sft_seed0{SFX[m]}c2h2"
            cell(LC, LCe, LCe, "mvr_sft", rn, mvr_evals, mvr_eval_done,
                 model_id=MODELS[m], n_tokens=L)

    # ============ WAVE M (2026-08-06): compare-CTM + both-causal gen->link ====
    # M1. The compare chapter, clean vintage: the banked cmpv2 runs are APO
    #     (constant-policy collapse: always-abstain vcf / always-answer color)
    #     and the ONE ctm attempt died at the 4h wall (step 612). The
    #     compare-CTM branch (twin-coupled verdict margin, preference.py:329)
    #     is in the code but NO banked run postdates it. Retrain the parity
    #     cells under the current vintage; flip_ll emits the v2 battery
    #     (cmp_joint = abstain-on-conflict AND answer-on-agree).
    # M2. Chain replication: the both-causal split-objective chain (APO on gen
    #     rows -> CTM on link ctrl) exists on ONE cell (color/gemma,
    #     .564/.943). Replicate coco x both models + color/qwen.
    Mw, Mg, Ml, Me = (Wave("M_cmp_l40s"), Wave("M_gen_l40s"),
                      Wave("M_link_l40s"), Wave("M_eval"))
    for ds, dd in LAD.items():
        for m in ("g3", "q4b"):
            rn = f"{ds}_cmpv2_5tok_ctm_seed0{SFX[m]}_v2"
            if "lad_ctm" in R and Mw.add(emit(R["lad_ctm"], model_id=MODELS[m],
                                              data_dir=dd, run_name=rn), rn):
                Me.lines += ladder_evals(rn, ds)
    for ds, dd, ms in (("coco", "data/coco", ("g3", "q4b")),
                       ("color", "data/color", ("q4b",))):
        for m in ms:
            gn = f"{ds}_neo_gen_5tok_apo_seed0{SFX[m]}_v3"
            if Mg.add(emit(R["gen"], model_id=MODELS[m], n_tokens=5,
                           lam_apo=1.0, data_dir=dd, run_name=gn), gn):
                Me.lines += ladder_evals(gn, ds)
            ln = f"{ds}_neo_link_5tok_ctm_seed0{SFX[m]}_v3"
            if Ml.add(emit(R["link"], model_id=MODELS[m], lam_ctm=1.0,
                           data_dir=dd, run_name=ln,
                           gen_ckpt=f"runs/{gn}/ckpt/step_800/neo.pt"), ln):
                Me.lines += ladder_evals(ln, ds)
                Me.lines.append(GUARD.format(
                    r=ln, c=f"python scripts/eval_order.py --run_dir runs/{ln} "
                            f"--task color --data_dir {dd} --split test_id"))
    # eval backfill: coco cmpv2-APO trained but never flip-evaluated
    Me.lines += ladder_evals("coco_cmpv2_5tok_apo_seed0", "coco")

    # ============ WAVE N_fill (2026-08-06): census holes + step-dynamics ====
    # N1. LoRA r1/r64 on the ladder (frontier ranks existed only on Ground-CF
    #     + ColorSwap/gemma) -- the census's genuine trainable holes.
    # N2. STEP SWEEP, eval-only: 474 runs saved ckpts every 100 steps but only
    #     3 runs have >1 eval step. Evaluate headline parity cells at steps
    #     100..700 -> the training-dynamics curve (routing early / grounding
    #     after the causal ramp at step 100-150) without any retraining.
    Nf, Nfe, Ns = Wave("N_lora_l40s"), Wave("N_lora_eval"), Wave("N_stepsweep")
    for ds, dd in LAD.items():
        for m in ALLM:
            for r in (1, 64):
                rn = f"{ds}_lora_r{r}_lr2e4_seed0{SFX[m]}"
                if Nf.add(emit(R["lad_lora"], model_id=MODELS[m], data_dir=dd,
                               lora_r=r, run_name=rn), rn):
                    Nfe.lines += ladder_evals(rn, ds)
    for ds in ("vcf", "color", "coco"):
        for sfx in ("", "_qwen35"):
            for rn in (f"{ds}_neo_ctrl_5tok_sft_seed0{sfx}",
                       f"{ds}_neo_ctrl_5tok_ctm_seed0{sfx}",
                       f"{ds}_pt_na_5tok_sft_seed0{sfx}",
                       f"{ds}_lora_r4_lr2e4_seed0{sfx}"):
                for st in (100, 200, 300, 400, 500, 600, 700):
                    Ns.lines.append(GUARD.format(
                        r=rn, c=f"python scripts/eval_flip_ll.py --run_dir runs/{rn} "
                                f"--task color --data_dir data/{ds} --split test_id "
                                f"--max_n 140 --step {st}"))
    for sfx in ("", "_qwen35"):
        for rn in (f"clevr3_f_5tok_sft_seed0{sfx}", f"clevr3_f_5tok_ctm_seed0{sfx}",
                   f"clevr3_f_pt_5tok_seed0{sfx}", f"clevr3_f_lora_r4_seed0{sfx}"):
            for st in (100, 200, 300, 400, 500, 600, 700):
                Ns.lines.append(GUARD.format(
                    r=rn, c=f"python scripts/eval_clevr3_flip.py --run_dir runs/{rn} "
                            f"--data_dir data/clevr3 --split test_id --step {st}"))

    SA, SB, SC, SD = (Wave("SMOKE_A_l40s"), Wave("SMOKE_B_a100"),
                      Wave("SMOKE_C_h200"), Wave("SMOKE_D_l40s"))
    SA.add(emit(R["ctm"], model_id=MODELS["q2b"], steps=25,
                run_name="sm_iclr_clevr3_q2b"), "sm_iclr_clevr3_q2b")
    SA.add(emit(R["ctm"], model_id=MODELS["e4b"], steps=25,
                run_name="sm_iclr_clevr3_e4b"), "sm_iclr_clevr3_e4b")
    SA.add(emit(R["lad_lora"], model_id=MODELS["e4b"], lora_r=16,
                steps=25, run_name="sm_iclr_lora_e4b"), "sm_iclr_lora_e4b")
    SB.add(emit(R["mvg_sftctm"], model_id=MODELS["e4b"], steps=25,
                run_name="sm_iclr_mvg_e4b"), "sm_iclr_mvg_e4b")
    SB.add(emit(R["mvg_sftctm"], model_id=MODELS["q2b"], steps=25,
                run_name="sm_iclr_mvg_q2b"), "sm_iclr_mvg_q2b")
    SC.add(emit(R["mvr_sft"], model_id=MODELS["q2b"], steps=25,
                run_name="sm_iclr_mvr_q2b"), "sm_iclr_mvr_q2b")
    SC.add(emit(R["mvr_sft"], model_id=MODELS["e4b"], steps=25,
                run_name="sm_iclr_mvr_e4b"), "sm_iclr_mvr_e4b")
    SD.add(emit(R["lad_sft"], data_dir="data/color", ctrl_style="blockscmp",
                steps=25, run_name="sm_iclr_blockscmp"), "sm_iclr_blockscmp")

    print("== ICLR manifests ==")
    for w in (SA, SB, SC, SD, A, Ae, B, Be, C, Ce, D1, D2, De, E1,
              Lw, Le, Lbf, LB, LBe, LC, LCe, Mw, Mg, Ml, Me, Nf, Nfe, Ns):
        w.write()


if __name__ == "__main__":
    main()
