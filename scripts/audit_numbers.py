#!/usr/bin/env python
"""S3 number-audit: every 2-decimal number in the submission source must be
traceable to the grid, the stats pass, or a banked battery JSON.

Builds the reference set from:
  - results/master_grid.json          (all R./G./S./J. values, 2dp + 3dp)
  - results/stats_pass.json           (values + CI endpoints)
  - banked battery aggregates:        citefaith / attrib / verifier /
                                      genconsistency / relocation JSONs
  - generated tables (tables/*.tex)   (anything already rendered from the grid)

Then extracts every \.\d\d(\d)? decimal from the tex (comments stripped,
labels/refs ignored) and reports ORPHANS with surrounding context.

Usage: python scripts/audit_numbers.py paper/iclr2027/iclr2027_main9.tex
"""
import glob, json, os, re, sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tex_sources(path, root=None, seen=None):
    """Follow literal input/include commands and our resulttable wrapper.

    This closes the include visibility gap, not the correct-cell provenance gap.
    Generated tables remain weak reference material in this legacy membership check.
    """
    path = Path(path).resolve()
    root = path.parent if root is None else root
    seen = set() if seen is None else seen
    if path in seen:
        return []
    seen.add(path)
    text = re.sub(r"(?<!\\)%.*", "", path.read_text())
    found = [(str(path), text)]
    includes = re.findall(r"\\(?:input|include)\{([^{}#]+)\}", text)
    includes += ["tables/" + name for name in re.findall(r"\\resulttable\{([^{}#]+)\}", text)]
    for name in includes:
        target = root / (name if name.endswith(".tex") else name + ".tex")
        if not target.is_file():
            raise FileNotFoundError(f"{path}: missing included source {target}")
        found.extend(tex_sources(target, root, seen))
    return found


def add_num(ref, v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        for nd in (2, 3):
            r = round(float(v), nd)
            ref.add(f"{r:.{nd}f}".lstrip("0") or "0")
            ref.add(f"{r:.{nd}f}")


def walk(ref, o):
    if isinstance(o, dict):
        for v in o.values():
            walk(ref, v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            walk(ref, v)
    else:
        add_num(ref, o)


def main():
    tex_path = sys.argv[1] if len(sys.argv) > 1 else \
        f"{ROOT}/paper/iclr2027/iclr2027_main9.tex"
    ref = set()
    for p in (f"{ROOT}/results/master_grid.json",
              f"{ROOT}/results/stats_pass.json",
              f"{ROOT}/results/genlink.json",
              f"{ROOT}/results/detection_procedural.json",
              f"{ROOT}/results/generation_format.json",
              f"{ROOT}/results/objective_verdict.json",
              f"{ROOT}/results/modality_pooled.json",
              f"{ROOT}/results/scatter_region.json",
              f"{ROOT}/results/applications_groundcf.json",
              f"{ROOT}/results/attn_conditionality.json",
              f"{ROOT}/results/attn_vs_perf.json",
              f"{ROOT}/results/what_words_learn.json",
              f"{ROOT}/results/closed_form_bounds.json",
              f"{ROOT}/results/bootstrap_ci.json",
              f"{ROOT}/results/paired_tests.json",
              f"{ROOT}/results/attn_precision.json",
              f"{ROOT}/results/gap_summary.json",
              f"{ROOT}/results/pooled_table.json",
              f"{ROOT}/results/groundcf_difficulty.json",
              f"{ROOT}/results/quadrant.json",
              f"{ROOT}/results/big_table_winners.json",
              f"{ROOT}/results/master_grid_ood.json",
              f"{ROOT}/results/big_table_winners_ood.json",
              f"{ROOT}/results/width_compare.json",
              f"{ROOT}/results/ood_compare.json",
              f"{ROOT}/results/story_tables.json",
              f"{ROOT}/results/objectives_groundcf.json",
              f"{ROOT}/results/objective_deltas.json",
              f"{ROOT}/results/objective_heat.json",
              f"{ROOT}/results/objective_effect.json",
              f"{ROOT}/results/apps_panel.json",
              f"{ROOT}/results/fig_gain.json",
              f"{ROOT}/results/interface_control.json",
              f"{ROOT}/results/stay_conditioned.json",
              f"{ROOT}/results/instr_variants.json",
              f"{ROOT}/results/detect_roc.json",
              f"{ROOT}/results/detect_compare.json",
              f"{ROOT}/results/groundcf_skills.json",
              f"{ROOT}/results/generation_scored.json",
              f"{ROOT}/results/apps_tables.json",
              f"{ROOT}/results/attn_layer_compare.json",
              f"{ROOT}/results/selfverify_probe.json",
              f"{ROOT}/results/mvr_ablation.json",
              f"{ROOT}/results/fig_balance.json",
              f"{ROOT}/results/fig_video.json",
              f"{ROOT}/results/mechanism.json"):
        d = json.load(open(p)) if os.path.exists(p) else None
        if d:
            walk(ref, d)
    for pat in ("runs/*/eval/*/citefaith_step_800*.json",
                "runs/*/eval/*/attrib_step_800*.json",
                "runs/*/eval/*/verifier_step_800*.json",
                "runs/*/eval/*/relocation_step_800.json",
                "runs/*/eval/*/genconsistency_step_800*.json",
                "runs/*/eval/*/flip_ll_noop_step_800.json",
                "runs/*/eval/*/flip_ll_iv*_step_800.json",
                "runs/*/eval/*/flip_step_800_instr*.json",
                "runs/*/eval/mvg_flip/flip_step_800*crop*.json",
                "runs/*/eval/mvg_flip/flip_step_800_clean*.json",
                "runs/*/eval/mvraw_test_ood/ablate*_step_800.json"):
        for f in glob.glob(f"{ROOT}/{pat}"):
            d = json.load(open(f))
            walk(ref, {k: v for k, v in d.items() if k != "records"})
    # pinned poster-day numbers + battery CSVs (canonical MVG/MVR sources)
    pj = f"{ROOT}/poster/poster_data.json"
    if os.path.exists(pj):
        walk(ref, json.load(open(pj)))
    import csv as _csv
    for f in glob.glob(f"{ROOT}/results/spreadsheets/*.csv"):
        try:
            for row in _csv.reader(open(f)):
                for cell in row:
                    try:
                        add_num(ref, float(cell))
                    except ValueError:
                        pass
        except Exception:
            pass
    # numbers already rendered into generated tables are grid-traceable
    for f in glob.glob(f"{ROOT}/paper/iclr2027/tables/*.tex"):
        for m in re.finditer(r"(?<![\d.])(1?\.\d{2,3})", open(f).read()):
            ref.add(m.group(1))

    sources = tex_sources(tex_path)
    print(f"[audit] following {len(sources)} TeX sources, including generated tables")
    src = "\n".join(text for _, text in sources)
    src = re.sub(r"(?<!\\)%.*", "", src)                # comments
    src = re.sub(r"\\(label|ref|cite\w*)\{[^}]*\}", "", src)
    # Layout dimensions are not experimental measurements.
    src = re.sub(r"\\includegraphics\[[^\]]*\]", r"\\includegraphics", src)
    src = re.sub(r"\\resizebox\{[^}]*\}", r"\\resizebox", src)
    orphans, total = [], 0
    for m in re.finditer(r"(?<![\d.])(1?\.\d{2,3})(?!\d)", src):
        total += 1
        v = m.group(1)
        if v in ref or v.lstrip("0") in ref or ("0" + v) in ref:
            continue
        ctx = src[max(0, m.start() - 90):m.end() + 40].replace("\n", " ")
        orphans.append((v, ctx.strip()))
    print(f"[audit] {total} decimal mentions; {len(orphans)} ORPHANS")
    for v, ctx in orphans:
        print(f"  ORPHAN {v}: ...{ctx}...")
    if orphans:
        sys.exit(1)


if __name__ == "__main__":
    main()
