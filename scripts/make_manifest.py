#!/usr/bin/env python
"""Write Slurm-array manifests (one CLI-arg line per independent single-GPU job).
The objective is L = lam_sft*CE + lam_apo*APO + lam_vgm*VGM; the objective-ablation
manifests run each term ALONE and in COMBINATION, one cell at a time."""
import os
HERE = os.path.dirname(__file__); CFG = os.path.join(HERE, "..", "configs")

# objective presets (appended as CLI flags)
SFT = "--lam_sft 1 --lam_apo 0 --lam_vgm 0"
APO = "--lam_sft 0 --lam_apo 1 --beta 0.1"
SFTAPO = "--lam_sft 1 --lam_apo 1 --beta 0.1"
VGM = "--lam_sft 0 --lam_vgm 1 --gamma 0.1"
SFTVGM = "--lam_sft 1 --lam_vgm 1 --gamma 0.1"
SFTAPOVGM = "--lam_sft 1 --lam_apo 1 --lam_vgm 1 --beta 0.1 --gamma 0.1"


def line(method, arm, n, seed, lr, steps, extra=""):
    s = f"--method {method} --arm {arm} --n_tokens {n} --seed {seed} --lr {lr} --steps {steps}"
    return s + (" " + extra if extra else "")


def write(name, lines):
    with open(os.path.join(CFG, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {name}: {len(lines)} jobs")


def main():
    # MAIN: three neologism arms + LoRA baseline, multi-seed, pure SFT (defaults).
    write("manifest_main.txt", [
        line(m, a, 4, s, lr, 400, SFT)
        for s in (0, 1, 2)
        for (m, a, lr) in [("neo", "ctrl", 0.05), ("neo", "gen", 0.05),
                           ("neo", "fuse", 0.05), ("lora", "na", 2e-4)]])

    # ARM A only (control tokens, SFT) -- run + sanity-check FIRST.
    write("manifest_arm_a.txt", [line("neo", "ctrl", 4, s, 0.05, 400, SFT) for s in (0, 1, 2)])

    # tokens-per-neologism sweep (SFT), seed 0.
    write("manifest_sweep.txt", [line("neo", a, n, 0, 0.05, 400, SFT)
                                 for n in (1, 2, 4, 8, 16) for a in ("ctrl", "gen")])

    # OBJECTIVE ABLATION on ctrl: SFT-only / APO-only / SFT+APO, each x3 seeds.
    # (Run AFTER you've seen Arm A's laundering numbers say a negative is worth it.)
    write("manifest_apo.txt", [line("neo", "ctrl", 4, s, 0.05, 400, preset)
                               for s in (0, 1, 2) for preset in (SFT, APO, SFTAPO)])

    # APO logistic-temperature sweep (SFT+APO), seed 0.
    write("manifest_obj_beta.txt", [line("neo", "ctrl", 4, 0, 0.05, 400,
                                         f"--lam_sft 1 --lam_apo 1 --beta {b}")
                                    for b in (0.05, 0.1, 0.5, 1.0)])

    # OPTIONAL multimodal grounding-margin arm (off by default; see README caveat).
    write("manifest_vgm.txt", [line("neo", "ctrl", 4, 0, 0.05, 400, p)
                               for p in (VGM, SFTVGM, SFTAPOVGM)])

    # SPLIT objective (lead candidate): VGM on <look>, APO on <read>/<compare>,
    # each on an SFT base. Compact JSON (no spaces) so it survives word-splitting.
    SPLIT = ('--lam_sft 1 --beta 0.1 --gamma 0.1 --per_op '
             '{"look":{"lam_apo":0,"lam_vgm":1},'
             '"read":{"lam_apo":1,"lam_vgm":0},'
             '"compare":{"lam_apo":1,"lam_vgm":0}}')
    write("manifest_split.txt", [line("neo", "ctrl", 4, s, 0.05, 400, SPLIT) for s in (0, 1, 2)])


if __name__ == "__main__":
    main()
