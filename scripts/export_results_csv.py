#!/usr/bin/env python
"""Export EVERY eval result to spreadsheets (CSV always; XLSX if openpyxl).

Walks runs/*/eval/**/*.json, attributes each file to (model, method, L) via
config.json (master_grid's rules -- never filenames), flattens scalar metrics
(one level of dict nesting, e.g. per_channel.VIS.nec_pos), and writes:

  results/spreadsheets/all_evals_long.csv    one row per (run, file, metric)
  results/spreadsheets/<battery>.csv         wide: one row per eval file,
                                             columns = that battery's metrics
  results/spreadsheets/master_grid.csv       the paper grid (from master_grid.json)
  results/spreadsheets/stats_comparisons.csv headline comparisons (stats_pass)
  results/spreadsheets/all_results.xlsx      one sheet per battery + grid

Idempotent: rerun after any wave; rows regenerate from disk.
Usage: python scripts/export_results_csv.py
"""
import csv, glob, json, os, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from master_grid import model_of, method_of, cfg          # attribution rules

OUT = os.path.join(ROOT, "results", "spreadsheets")
os.makedirs(OUT, exist_ok=True)

# Per-sheet experiment descriptions -> README sheet + DESCRIPTIONS.md.
# Every number's meaning, chance line, and canonical source in one place.
DESCRIPTIONS = {
    "master_grid": ("THE paper grid: routing (R) / grounding-flip (G) / MVR JOINT (J) "
                    "per channel, one row per dataset x model x method x L. "
                    "Selection: L=5 seed-0 parity, newest eval, in-domain only. "
                    "Chance: routing 1/2 or 1/3 by arity; blind flip <= 1/4; JOINT <= 1/2.",
                    "scripts/master_grid.py"),
    "ladder_flip": ("Ladder flip/stay battery (VCF/ColorSwap/COCO-CF): forced-choice LL "
                    "over the two source values; flip = tracks the twin edit, stay = "
                    "ignores the other source's edit. Includes compare-v2 verdict battery "
                    "(cmp_*: abstain-on-conflict AND answer-on-agree; cmp_joint gameproof). "
                    "records = per-item vectors (McNemar/function-space).",
                    "scripts/eval_flip_ll.py"),
    "clevr3_flip": ("Ground-CF 3-source battery (image/caption/OCR-in-image disagree by "
                    "construction): per-channel routing ({ch}_acc, 3-way, chance 1/3) and "
                    "flip ({ch}_flip, exact twins). _instr/_noop files = frozen baselines.",
                    "scripts/eval_clevr3_flip.py"),
    "clevr3_routing": ("Ground-CF routing-only eval (legacy battery).", "scripts/eval_clevr3.py"),
    "ladder_ctrl": ("Ladder routing-only control eval.", "scripts/eval_flip_ll.py"),
    "mvg_flip": ("MultiVENT-G grounding: vis_value_swap = mined-pair frame swap (flip; "
                 "mutual-exclusivity + single-attestation guards); desc_swap = true text "
                 "flip; ocr_ablate = DELETION-ablation (drop_pos), starred, never a flip.",
                 "scripts/eval_mvg_flip.py"),
    "mvg_routing": ("MVG routing/slot-filling evals.", "scripts/eval_mvg3.py"),
    "mvr_ablate": ("MultiVENT-Raw JOINT: do(channel absent) on human-attested claims; "
                   "nec = drop>eps(0.5 nats) when attesting channel removed, spec = no "
                   "drop otherwise, JOINT = min. Blind cap .5. *_pertok = length-"
                   "normalized (the LL length-bias answer).",
                   "scripts/eval_mvraw_ablate.py"),
    "mvr_swap": ("MVR length/topic/language-matched evidence swap: the honest raw-video "
                 "counterfactual; null for every method (.00-.08 vs .25) = the artifact "
                 "bound on ablation columns.", "scripts/eval_mvraw_swap.py"),
    "report_gen": ("Report generation (free decoding): role-tagged claim lists "
                   "([CH] role: value); P0/R0/F0 token-F vs gold claims, provenance_acc "
                   "= channel-tag accuracy. Assembler baseline F0 .031.",
                   "scripts/eval_mvg_report.py"),
    "genconsistency": ("Discrimination-vs-production audit: greedy generation vs "
                       "LL-argmax agreement; gen_*_flip = flip on generations; "
                       "unmappable = neither value produced. LL is conservative for "
                       "words; PT shows a true production deficit.",
                       "scripts/eval_genconsistency.py"),
    "order_compose": ("Composition: multi-word prompts executed in requested order "
                      "(exact-order, chance .25) + verdict binding.",
                      "scripts/eval_order.py"),
    "mechanism": ("Attention interventions: SAP-noise, span ablations "
                  "(ans2dir/dir2img/img/evid severing), pattern transplant. "
                  "ablate.tok.base_img -> ablate_img = flip before/after cut.",
                  "scripts/sap_transplant.py"),
    "verifier": ("Claim verifier: V(claim,ch)=m(claim|C)-m(claim|C\\ch) as classifier; "
                 "positives=attested claims, negatives=matched mate claims; AUROC + "
                 "channel attribution.", "scripts/eval_claim_verifier.py"),
    "relocation": ("Test-time relocation: trained rows moved to slot/prefix/postq/"
                   "postcap with filler-carrier guard; functional-site vs vector test.",
                   "scripts/eval_relocation.py"),
    "stats_comparisons": ("Headline causal-vs-sft comparisons with Wilson CIs and exact "
                          "McNemar where paired records exist.", "scripts/stats_pass.py"),
    "transfer": ("Zero-shot / cross-dataset transfer evals.", "scripts/eval_transfer.py"),
    "rag_stress": ("Retrieval stress: oracle vs top-1 retrieved captions; margin "
                   "collapse under distractors (failures visible).", "scripts/eval_mvg_rag.py"),
    "corroborate": ("Corroborate operator battery (negative result).", "scripts/eval_mvraw_corr.py"),
    "temporal": ("Temporal words battery (negative result).", "scripts/eval_temporal.py"),
    "verbalization": ("Self-verbalization + synonym plug-in controls.", "scripts/eval_verbalize.py"),
    "mvr_other": ("Misc MVR evals (register control, report variants).", "scripts/"),
    "other": ("Evals not matching a battery pattern (see file column).", "-"),
}

BATTERY = [  # (filename prefix -> battery name)
    ("flip_ll", "ladder_flip"), ("flip_step", "clevr3_flip"),
    ("ctrl_ll", "ladder_ctrl"), ("step_", "clevr3_routing"),
    ("genconsistency", "genconsistency"), ("order", "order_compose"),
    ("verifier", "verifier"), ("relocation", "relocation"),
    ("ablate", "mvr_ablate"), ("swap", "mvr_swap"),
    ("report", "report_gen"), ("mvg3", "mvg_routing"),
    ("transfer", "transfer"), ("rag", "rag_stress"),
    ("corr", "corroborate"), ("temporal", "temporal"),
    ("transplant", "mechanism"), ("sap", "mechanism"),
    ("probe", "mechanism"), ("verbalize", "verbalization"),
]


DIR_BATTERY = [("mvg_flip", "mvg_flip"), ("mvg_report", "report_gen"),
               ("mvraw_report", "report_gen"), ("mvg_rag", "rag_stress"),
               ("mvraw", "mvr")]


def battery_of(evaldir, base):
    for pre, name in DIR_BATTERY:          # dir wins over filename prefix
        if evaldir.startswith(pre):
            if name == "mvr":              # split mvr dir by file kind
                return ("mvr_swap" if base.startswith("swap")
                        else "mvr_ablate" if base.startswith("ablate")
                        else "report_gen" if "report" in base else "mvr_other")
            return name
    for pre, name in BATTERY:
        if base.startswith(pre):
            return name
    return "other"


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            yield f"{prefix}{k}", v
        elif isinstance(v, dict) and prefix == "":       # one level only
            yield from flatten(v, f"{k}.")


def main():
    long_rows, wide = [], collections.defaultdict(list)
    n_files = 0
    for f in sorted(glob.glob(f"{ROOT}/runs/*/eval/*/*.json")):
        parts = f.split("/")
        run, evaldir, base = parts[-4], parts[-2], parts[-1]
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        n_files += 1
        args = cfg(run)
        meth, L = method_of(run, args)
        if meth is None:                                  # frozen-vehicle files
            meth = str(d.get("method") or d.get("style") or "?")
            L = None
        model = model_of(run, args)
        bat = battery_of(evaldir, base)
        step = d.get("step", "")
        met = dict(flatten(d))
        row = {"run": run, "model": model, "method": meth, "L": L or "",
               "eval_dir": evaldir, "file": base, "battery": bat, "step": step}
        for k, v in sorted(met.items()):
            long_rows.append({**row, "metric": k, "value": v})
        wide[bat].append({**row, **met})

    with open(f"{OUT}/all_evals_long.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["run", "model", "method", "L", "eval_dir",
                                "file", "battery", "step", "metric", "value"])
        w.writeheader(); w.writerows(long_rows)

    sheets = {}
    for bat, rows in sorted(wide.items()):
        cols = ["run", "model", "method", "L", "eval_dir", "file", "step"]
        extra = sorted({k for r in rows for k in r
                        if k not in cols + ["battery"]})
        with open(f"{OUT}/{bat}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, cols + extra, restval="")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols + extra})
        sheets[bat] = (cols + extra, rows)

    # master grid + stats comparisons as sheets too
    g = json.load(open(f"{ROOT}/results/master_grid.json"))
    grows = []
    for k, v in g.items():
        ds, model, meth, L = k.split("|")
        grows.append({"dataset": ds, "model": model, "method": meth,
                      "L": "" if L == "None" else L, **v})
    gcols = ["dataset", "model", "method", "L"] + sorted(
        {k for r in grows for k in r} - {"dataset", "model", "method", "L"})
    with open(f"{OUT}/master_grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, gcols, restval="")
        w.writeheader(); w.writerows(grows)
    sheets["master_grid"] = (gcols, grows)

    sp = json.load(open(f"{ROOT}/results/stats_pass.json"))
    comps = sp.get("headline_comparisons", [])
    if comps:
        ccols = sorted({k for c in comps for k in c})
        with open(f"{OUT}/stats_comparisons.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, ccols, restval="")
            w.writeheader()
            for c in comps:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict))
                                else v) for k, v in c.items()})
        sheets["stats_comparisons"] = (ccols, [
            {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
             for k, v in c.items()} for c in comps])

    # DESCRIPTIONS.md + README rows (also indexes every file in the folder)
    md = ["# Results spreadsheets — what each sheet/file contains", ""]
    readme_rows = []
    order = ["master_grid"] + [b for b in sorted(sheets) if b != "master_grid"]
    for bat in order:
        desc, src = DESCRIPTIONS.get(bat, ("(no description registered)", "-"))
        n = len(sheets[bat][1])
        readme_rows.append([bat, n, desc, src])
        md += [f"## {bat}  ({n} rows — source: `{src}`)", desc, ""]
    for extra, desc in (
            ("all_evals_long.csv", "EVERY metric of EVERY eval json in long form: "
             "run, model, method, L, eval_dir, file, battery, step, metric, value."),
            ("coverage_census.csv / coverage_missing.csv", "every target grid cell "
             "classified BANKED/QUEUED/TRAINED/PARTIAL/MISSING."),
            ("steps_coverage.csv", "per run: which ckpt steps exist vs which steps "
             "have evals (the training-dynamics inventory)."),
            ("function_space_*.csv", "pairwise behavioural similarity between runs "
             "(per-item decision agreement + margin correlation)."),
            ("holes_status.md", "the filling-vs-not ledger with reasons."),
            ("generated_reports_samples.md", "raw generated WHAT/WHERE reports.")):
        md += [f"## {extra}", desc, ""]
    open(f"{OUT}/DESCRIPTIONS.md", "w").write("\n".join(md) + "\n")

    try:
        from openpyxl import Workbook
        wb = Workbook(); wb.remove(wb.active)
        ws = wb.create_sheet("README")
        ws.append(["sheet", "rows", "experiment / how to read it", "canonical source"])
        for row in readme_rows:
            ws.append(row)
        ws.freeze_panes = "A2"
        ws.column_dimensions["C"].width = 110
        for bat in order:
            cols, rows = sheets[bat]
            ws = wb.create_sheet(bat[:31])
            ws.append(cols)
            for r in rows:
                ws.append([r.get(c, "") for c in cols])
            ws.freeze_panes = "A2"
        wb.save(f"{OUT}/all_results.xlsx")
        xlsx = "all_results.xlsx (README sheet first)"
    except ImportError:
        xlsx = "(openpyxl not installed -- CSVs only)"

    print(f"[export] {n_files} eval files -> {len(long_rows)} metric rows")
    for bat, (cols, rows) in sorted(sheets.items()):
        print(f"  {bat:20s} {len(rows):5d} rows x {len(cols)} cols")
    print(f"[saved] {OUT}/  ({xlsx})")


if __name__ == "__main__":
    main()
