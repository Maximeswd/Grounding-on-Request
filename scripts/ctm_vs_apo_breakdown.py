"""CTM vs APO head to head, per model and per dataset (STATUS v1.62-v1.64).
Cells: ColorSwap, COCO-CF, Ground-CF x 4 models, L=5, from results/master_grid{,_ood}.json.
JOINT = min(image Flip, caption Flip, image Stay_c, caption Stay_c) per cell. Prints means per group and
per-cell win counts; writes results/ctm_vs_apo_breakdown.{txt,json}. Regenerate, never edit."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = {"ID": json.load(open(f"{ROOT}/results/master_grid.json")), "OOD": json.load(open(f"{ROOT}/results/master_grid_ood.json"))}
ARMS = {"neo+CTM": "word+ctm", "PT+APO": "pt+apo", "neo+APO": "word+apo", "neo+APO+CTM": "word+apo+ctm", "PT+CTM": "pt+ctm"}
TASKS = ["color", "coco", "groundcf"]; MODELS = ["gemma3", "qwen4b", "qwen2b", "e4b"]
MET = ("fl", "fr", "sl", "sr")
def cell(g, t, m, a):
    r = g.get(f"{t}|{m}|{a}|5")
    if not r: return None
    f = lambda k: None if r.get(k) is None else float(r[k])
    d = dict(fl=f("G.look"), fr=f("G.read"), sl=f("Sc.look"), sr=f("Sc.read"))
    if any(v is None for v in d.values()): return None
    d["J"] = min(d.values()); return d
out = {}
def show(title, by):
    print(f"\n{title}\n{'arm':12s} {'iFlip':>6s} {'cFlip':>6s} {'iStay':>6s} {'cStay':>6s} {'JOINT':>6s} {'worst':>6s}  n")
    res = {}
    for an, cs in by.items():
        if not cs: print(f"{an:12s}  (no cells)"); continue
        mu = {k: sum(c[k] for c in cs) / len(cs) for k in MET + ("J",)}; mu["worst"] = min(c["J"] for c in cs); mu["n"] = len(cs)
        res[an] = mu
        print(f"{an:12s} {mu['fl']:6.3f} {mu['fr']:6.3f} {mu['sl']:6.3f} {mu['sr']:6.3f} {mu['J']:6.3f} {mu['worst']:6.3f}  {len(cs)}")
    return res
for split in ("ID", "OOD"):
    g = G[split]; out[split] = {}
    matched = [(t, m) for t in TASKS for m in MODELS if all(cell(g, t, m, ARMS[a]) for a in ("neo+CTM", "PT+APO", "neo+APO"))]
    print(f"\n######## {split}: matched cells (neo+CTM, PT+APO, neo+APO present) = {len(matched)}: {matched}")
    out[split]["matched"] = matched
    out[split]["all"] = show(f"{split} all matched cells", {an: [cell(g, t, m, ak) for t, m in matched if cell(g, t, m, ak)] for an, ak in ARMS.items()})
    out[split]["by_model"] = {m: show(f"{split} model={m}", {an: [cell(g, t, mm, ak) for t, mm in matched if mm == m and cell(g, t, mm, ak)] for an, ak in ARMS.items()}) for m in MODELS}
    out[split]["by_dataset"] = {t: show(f"{split} dataset={t}", {an: [cell(g, tt, m, ak) for tt, m in matched if tt == t and cell(g, tt, m, ak)] for an, ak in ARMS.items()}) for t in TASKS}
    out[split]["wins"] = {}
    for opp in ("PT+APO", "neo+APO", "neo+APO+CTM", "PT+CTM"):
        W = {k: [0, 0] for k in ("J",) + MET}; rows = []
        for t, m in matched:
            a = cell(g, t, m, "word+ctm"); b = cell(g, t, m, ARMS[opp])
            if not b: continue
            for k in W: W[k][0] += a[k] > b[k]; W[k][1] += a[k] < b[k]
            rows.append(f"    {t:8s}/{m:6s}: JOINT {a['J']:.3f} vs {b['J']:.3f} | iFlip {a['fl']:.3f} vs {b['fl']:.3f} | cFlip {a['fr']:.3f} vs {b['fr']:.3f} | iStay {a['sl']:.3f} vs {b['sl']:.3f} | cStay {a['sr']:.3f} vs {b['sr']:.3f}")
        out[split]["wins"][opp] = W
        print(f"\n{split} neo+CTM vs {opp} (wins-losses, ties omitted): JOINT {W['J'][0]}-{W['J'][1]} | iFlip {W['fl'][0]}-{W['fl'][1]} | cFlip {W['fr'][0]}-{W['fr'][1]} | iStay {W['sl'][0]}-{W['sl'][1]} | cStay {W['sr'][0]}-{W['sr'][1]}")
        for r in rows: print(r)
json.dump(out, open(f"{ROOT}/results/ctm_vs_apo_breakdown.json", "w"), indent=1)
