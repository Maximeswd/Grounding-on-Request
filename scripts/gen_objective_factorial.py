#!/usr/bin/env python
"""Objective factorial (2026-10-22, author): plain SFT against counterfactual supervision with the Flip margin (CTM),
the Stay margin, or both; and the same three on top of paired SFT (cross-entropy on the twins).

The eight objectives form a 2 x 2 x 2 design over {twin cross-entropy} x {Flip margin} x {Stay margin}:
  sft        CE on the original context only                      --lam_sft 1
  sftctm     + Flip margin (CTM, the requested source's twin)      --lam_ctm 1
  sftsm      + Stay margin (the competing source's edit)           --lam_sm 1
  sftctmsm   + both margins
  psft       + CE on the twins (paired SFT)                        --lam_sft_twin 1
  psftctm / psftsm / psftctmsm   paired SFT + the margins
Interfaces use the fixed recipes (STATUS v1.307/v1.308): word lr .005 (L=1, L=5), prompt tuning lr .05 (L=1, L=5),
shared instruction-conditioned LoRA r16 lr 2e-4 (final checkpoint only, to save disk). Ground-CF, protocol v2, task holdout.

--part dev   seed 0, Gemma-3-4B, development partition (dev scenes excluded from training), one dev evaluation per run;
             word L=5, prompt tuning L=1 (the selected capacities) and LoRA. Judge: scripts/dev_select_objective.py.
--part test  seeds 0-2, reference + both shifts x ID/OOD; configs and models from --configs / --models.
Idempotent: runs with ckpt/step_800 that are not restarted continuations are skipped; evaluations that exist are skipped.
--eval_tag_suffix appends to every evaluation tag (e.g. _a100 to keep a comparison family on one GPU class).
Usage: python scripts/gen_objective_factorial.py --part dev|test [--configs word1,word5,pt1,pt5,lora] [--models gemma3,qwen35]
       [--objectives ...] [--eval_tag_suffix _a100] [--reeval_existing 1] --tag 20261022"""
import argparse, json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ = {"sft": (0, 0, 0), "sftctm": (0, 1, 0), "sftsm": (0, 0, 1), "sftctmsm": (0, 1, 1),
       "psft": (1, 0, 0), "psftctm": (1, 1, 0), "psftsm": (1, 0, 1), "psftctmsm": (1, 1, 1)}
MODEL = {"gemma3": ("", ""), "qwen35": (" --model_id Qwen/Qwen3.5-4B", "_qwen35"), "qwen2b": (" --model_id Qwen/Qwen3.5-2B", "_qwen2b"),
         "gemma4e4b": (" --model_id google/gemma-4-E4B-it --train_per_layer 1", "_gemma4e4b")}
rn = lambda l: re.search(r"--run_name (\S+)", l).group(1)

def lines(p): return [l.rstrip("\n") for l in open(os.path.join(ROOT, p)) if l.strip()]

def base_lines():
    g = lines("configs/l1_confirm_20261005_train_gemma.txt")
    word = [l for l in g if rn(l) == "clevr3_v2_1tok_psft_neo_holdcs_seed0"][0]
    pt = [l for l in g if rn(l) == "clevr3_v2_1tok_psft_pt_holdcs_seed0"][0]
    lora = [l for l in lines("configs/lora_ctmsm_20261005_train.txt") if rn(l) == "clevr3_v2_lora_r16_joint_psftctmsm_holdcs_seed0"][0]
    return word, pt, lora

def with_objective(line, obj):
    t, c, s = OBJ[obj]
    line = re.sub(r" --lam_(sft|sft_twin|ctm|sm|sft_stay) \S+", "", line)
    return line + f" --lam_sft 1 --lam_sft_twin {t} --lam_ctm {c} --lam_sm {s}"

def config_line(cfg, word, pt, lora):
    if cfg in ("word1", "word5"):
        return word.replace("--n_tokens 1 ", f"--n_tokens {cfg[-1]} ") + " --lr 0.005", f"clevr3_v2_{cfg[-1]}tok_{{obj}}_neo_holdcs{{dev}}_lr0p005_seed{{s}}{{m}}"
    if cfg in ("pt1", "pt5"):
        return pt.replace("--n_tokens 1 ", f"--n_tokens {cfg[-1]} "), f"clevr3_v2_{cfg[-1]}tok_{{obj}}_pt_holdcs{{dev}}_seed{{s}}{{m}}"
    if cfg == "lora":
        return lora + " --ckpt_every 800", "clevr3_v2_lora_r16_joint_{obj}_holdcs{dev}_seed{s}{m}"
    raise ValueError(cfg)

def clean_done(run):
    d = os.path.join(ROOT, "runs", run)
    if not os.path.isdir(os.path.join(d, "ckpt", "step_800")): return False, "untrained"
    rp = os.path.join(d, "resume_note.json")
    if os.path.exists(rp) and json.load(open(rp)).get("resume_state") == "restarted": return False, "RESTARTED"
    return True, "done"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["dev", "test"]); ap.add_argument("--tag", default="20261022")
    ap.add_argument("--configs", default=""); ap.add_argument("--models", default="gemma3")
    ap.add_argument("--objectives", default=",".join(OBJ)); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--eval_tag_suffix", default=""); ap.add_argument("--reeval_existing", type=int, default=0)
    ap.add_argument("--skip_names_from", default="", help="comma-separated manifests whose runs (and their evaluations) are owned by another wave still in progress")
    a = ap.parse_args()
    word, pt, lora = base_lines()
    owned = set()
    for f in [x for x in a.skip_names_from.split(",") if x]:
        owned |= {rn(l) for l in lines(f) if "--run_name" in l}
    configs = (a.configs or ("word5,pt1,lora" if a.part == "dev" else "word1,word5,pt1,pt5,lora")).split(",")
    seeds = [0] if a.part == "dev" else [int(x) for x in a.seeds.split(",")]
    models = ["gemma3"] if a.part == "dev" else a.models.split(",")
    train, evals, restarted = [], [], []
    for m in models:
        mflag, msfx = MODEL[m]
        for cfg in configs:
            line0, name_t = config_line(cfg, word, pt, lora)
            for obj in a.objectives.split(","):
                for s in seeds:
                    name = name_t.format(obj=obj, dev="_dev" if a.part == "dev" else "", s=s, m=msfx)
                    line = with_objective(re.sub(r"--seed \d+", f"--seed {s}", line0), obj)
                    line = re.sub(r"--run_name \S+", f"--run_name {name}", line) + mflag
                    if a.part == "dev":
                        line += " --exclude_scenes data/clevr3/dev_scenes.txt"
                    if "--workers" not in line: line += " --workers 2"
                    if name in owned:
                        continue                                   # trained and evaluated by its own wave
                    done, why = clean_done(name)
                    if why == "RESTARTED": restarted.append(name)
                    new = not done
                    if new: train.append(line)
                    if new or a.reeval_existing:
                        if a.part == "dev":
                            specs = [("explicit", "test_id", "dev", "pairs_look_all_dev.jsonl", "_v2ref")]
                        else:
                            specs = [(qs, sp, sp, f"pairs_look_all_{pf}.jsonl", tg) for sp, pf in (("test_id", "id"), ("test_ood", "ood"))
                                     for qs, tg in (("explicit", "_v2ref"), ("heldout", "_v2"))]
                        for qs, _, split, pairs, tg in specs:
                            tag = tg + a.eval_tag_suffix
                            if os.path.exists(os.path.join(ROOT, "runs", name, "eval", f"clevr3_{split}", f"flip_step_800{tag}.json")): continue
                            evals.append(f"SC_QSTYLE={qs} python scripts/eval_clevr3_flip.py --run_dir runs/{name} --data_dir data/clevr3 --split {split} "
                                         f"--step 800 --max_n 1000 --pairs_file data/clevr3/{pairs} --cp 1 --dump_picks 1 --tag {tag}")
    out_t = os.path.join(ROOT, "configs", f"objfact_{a.tag}_{a.part}_train.txt"); out_e = os.path.join(ROOT, "configs", f"objfact_{a.tag}_{a.part}_eval.txt")
    open(out_t, "w").write("\n".join(train) + ("\n" if train else "")); open(out_e, "w").write("\n".join(evals) + ("\n" if evals else ""))
    print(f"{a.part}: {len(train)} training runs -> {os.path.relpath(out_t, ROOT)}; {len(evals)} evaluations -> {os.path.relpath(out_e, ROOT)}")
    if restarted: print("  RESTARTED continuations (set aside before training them again):", restarted)

if __name__ == "__main__":
    main()
