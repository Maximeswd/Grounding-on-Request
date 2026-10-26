#!/usr/bin/env python
"""CANONICAL RESULTS INVENTORY (v1.192, 2026-09-25, author: "one canonical results inventory ... refuse to draw final
conclusions until every result we intend to use is represented there"). CPU only, read-only except its outputs.

One long-form row per result cell (family, model, dataset, split, attribute, method, objective, request, train_dataset,
train_position, eval_position, evidence_order, condition, metric, value, n, wilson_lo, wilson_hi, seed, checkpoint,
protocol_version, clean_twin, status, file, notes) for the 28 result families of the author's plan; a second table of
paired comparisons (exact McNemar on identical items); a Markdown overview with DONE / RUNNING / QUEUED / MISSING / N/A
per family, the queued work with ETAs, and what cannot finish before the deadline.

Status of a cell: DONE = final-protocol file present (Ground-CF: promoted corrected-twin rerun, see
scripts/promote_clean_twins.py); DONE* = value present on a superseded twin set (Ground-CF residue twins) while the
corrected rerun is RUNNING/QUEUED -- never final; RUNNING / QUEUED = the manifest line that writes the file belongs to a
live Slurm task (QUEUED-held: released by the priority gate or the keeper; PARKED: in a parked tier, will not run);
AUTO = submitted automatically when its training finishes; MISSING = neither file nor job; N/A = undefined by design.
Writes results/RESULTS_INVENTORY.{md,csv,json} and results/RESULTS_COMPARISONS.csv.
Usage: python scripts/results_inventory.py
"""
import collections, csv, datetime, glob, gzip, json, math, os, re, struct, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

MODELS = [("", "G3", "Gemma-3-4B"), ("_qwen35", "Q4", "Qwen3.5-4B"), ("_qwen2b", "Q2", "Qwen3.5-2B")]
LADDER = [("vcf", "VCF"), ("color", "ColorSwap"), ("coco", "COCO-CF")]
SPLITS = [("test_id", "ID"), ("test_ood", "OOD")]
REQS = ("look", "read")
SKILLS = [("color", "colour"), ("object", "shape"), ("count", "count"), ("spatial", "spatial")]
PROMO = json.load(open("results/clean_twin_promotion.json"))["promoted"] if os.path.exists("results/clean_twin_promotion.json") else {}
# v1.195: Ground-CF files written on the corrected twins under their canonical name (never _ctw, so not in the log)
CFB = set(json.load(open("results/clean_from_birth.json"))["clean"]) if os.path.exists("results/clean_from_birth.json") else set()
FIELDS = ["family", "tier", "model", "dataset", "split", "attribute", "method", "objective", "request", "train_dataset",
          "train_position", "eval_position", "evidence_order", "condition", "metric", "value", "n", "wilson_lo", "wilson_hi",
          "seed", "checkpoint", "protocol_version", "clean_twin", "status", "file", "notes"]
ROWS, CMP = [], []
from zoneinfo import ZoneInfo
NOW = datetime.datetime.now(ZoneInfo("Europe/Amsterdam"))   # log times are Amsterdam time


def J(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def wilson(p, n, z=1.96):
    if p is None or not n:
        return None, None
    d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 4), round((c + h) / d, 4)


# ================================================================== queue attribution
def _out_of(line):
    """The file a manifest line writes (None if unknown)."""
    env = dict(re.findall(r"(?:^|\s)([A-Z_][A-Z0-9_]*)=(\S*)", line.split(" python ")[0])) if " python " in line else {}
    arg = lambda k, d=None: (re.search(rf"--{k}\s+(\S+)", line) or [None, d])[1]
    run = arg("run_dir"); split = arg("split", "test_id"); step = arg("step", "800")
    if "train.py" in line:
        rn = arg("run_name")
        return f"runs/{rn}/ckpt/step_{arg('steps', '800')}" if rn else None
    if not run:
        return None
    run = run.rstrip("/")
    tag = arg("tag", "") or ""
    if "eval_clevr3_flip.py" in line:
        return f"{run}/eval/clevr3_{split}/flip_step_{step}{tag}.json"
    if "eval_compose_pair.py" in line:
        return f"{run}/eval/clevr3_{split}/compose_pair_step_{step}{tag}.json"
    if "eval_clevr3_switch.py" in line:
        return f"{run}/eval/clevr3_{split}/switch_step_{step}{tag}.json"
    if "eval_flip_ll.py" in line:
        iv = env.get("INSTR_VARIANT", "0"); t = f"_iv{iv}" if iv != "0" else ""
        if env.get("NOOP_MODE") == "1":
            t = "_noop"
        diag = (["lead"] if env.get("SC_LADDER_LEADIN") == "1" else []) + (["p2"] if env.get("SC_LADDER_PROTO") == "2" else [])
        if diag:
            t += "_" + "_".join(diag)
        tp = env.get("SC_TEMPLATE", "")
        t += "_heldout" if tp == "heldout" else (f"_{tp}" if tp.startswith("heldout") else "")
        ot = arg("out_task") or os.path.basename(arg("data_dir", "") or "")
        return f"{run}/eval/{ot}_{split}/flip_ll{t}_step_{step}.json"
    return None


def queue_index():
    """output path -> (status, job_task). Arrays are matched to manifests by exact task-index -> line mapping among the
    manifests written within 2 minutes before the submission (several are written in the same second by the keeper)."""
    known = {}
    for f in (".cleantwin_20260925",):
        for l in open(f) if os.path.exists(f) else []:
            k, j = l.split()
            known[j] = {"id": "configs/gcf_cleantwin_evals_id_v1189.txt", "ood": "configs/gcf_cleantwin_evals_ood_v1189.txt",
                        "extra_id": "configs/gcf_cleantwin_extra_id_v1191.txt", "extra_late": "configs/gcf_cleantwin_extra_late_v1191.txt"}[k]
    known.update({"708951": "configs/e5v2_compose_pair_v1188.txt", "708215": "configs/ladder_sft_evals_v1184.txt",
                  "708216": "configs/pt_transfer_v1184.txt", "708217": "configs/lora_transfer_v1184.txt"})
    parked = set()
    for f in (".held_parked_v1171", ".held_skip_e4b_v1160", ".held_superseded_v1157", ".held_protocol_v1_ladder"):
        if os.path.exists(f):
            parked |= {x.strip() for x in open(f) if x.strip()}
    gate = set()
    if os.path.exists(".prio_gate_groups"):
        gate = {x for l in open(".prio_gate_groups") for x in l.split() if x.isdigit()}
    try:
        sq = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%F|%K|%T|%r|%V|%j"],
                            capture_output=True, text=True, timeout=120).stdout.splitlines()
    except Exception:
        sq = []
    arr = collections.defaultdict(dict); sub = {}
    for l in sq:
        p = l.split("|")
        if len(p) < 6:
            continue
        arr[p[0]][p[1]] = (p[2], p[3]); sub[p[0]] = p[4]
    mans = {}
    for f in glob.glob("configs/*.txt") + glob.glob("configs/iclr/*.txt"):
        try:
            lines = [x.rstrip("\n") for x in open(f)]
        except Exception:
            continue
        mans[f] = (os.path.getmtime(f), [x for x in lines if x.strip()])
    idx = {}
    for j, tasks in arr.items():
        if j in known:
            cands = [known[j]]
        else:
            try:
                ts = datetime.datetime.strptime(sub[j], "%Y-%m-%dT%H:%M:%S").timestamp()
            except Exception:
                continue
            mx = max([int(t) for t in tasks if t.isdigit()] or [1])
            cands = [f for f, (m, ls) in mans.items() if -5 <= ts - m <= 120 and len(ls) >= mx]
        for t, (st, rs) in tasks.items():
            if not t.isdigit():
                continue
            if st == "RUNNING":
                s = "RUNNING"
            elif f"{j}_{t}" in parked or j in parked:
                s = "PARKED"
            elif "JobHeld" in rs:
                s = "QUEUED-held"
            else:
                s = "QUEUED"
            for f in cands:
                ls = mans.get(f, (0, []))[1]
                k = int(t) - 1
                if 0 <= k < len(ls):
                    o = _out_of(ls[k])
                    if o and (o not in idx or idx[o][0] == "PARKED"):
                        idx[o] = (s, f"{j}_{t}", j in gate)
    return idx


QI = queue_index()
# automatic submissions after training (pfirst watcher; keeper-registered joint LoRA evaluations)
AUTO = set()
for f in ("configs/gcf_pfirst_evals_v1184.txt", "configs/gcf_lora_joint_evals_v1176.txt"):
    for l in open(f) if os.path.exists(f) else []:
        o = _out_of(l)
        if o:
            AUTO.add(o)


def _qs(q):
    """QUEUED-held splits into GATE (held by the priority gate, released in order) and HELD-TIER (a keeper tier that
    releases only into idle lanes; ~1000 tier tasks are ahead of the late tiers, so these are not reached today)."""
    s, jt, in_gate = q
    if s == "QUEUED-held":
        return "QUEUED-gate" if in_gate else "HELD-tier"
    return s


def status_of(path, gcf=False):
    """(status, clean_twin) for an output path."""
    canon = os.path.relpath(path, ROOT) if os.path.isabs(path) else path
    if os.path.exists(canon):
        if not gcf:
            return "DONE", ""
        if canon in PROMO or canon in CFB:
            return "DONE", "true"
        ctw = canon[:-5] + "_ctw.json"
        q = QI.get(ctw)
        if q:
            return f"DONE* ({_qs(q)} clean)", "false"
        if ctw in AUTO:
            return "DONE* (AUTO clean)", "false"
        return "DONE* (no clean rerun queued)", "false"
    q = (QI.get(canon[:-5] + "_ctw.json") if gcf else None) or QI.get(canon)     # the corrected rerun wins
    if q:
        return _qs(q), ("true" if gcf and (canon[:-5] + "_ctw.json") in QI else "")
    if canon in AUTO or (gcf and canon[:-5] + "_ctw.json" in AUTO):
        return "AUTO", ""
    return "MISSING", ""


def add(**kw):
    r = {k: "" for k in FIELDS}; r.update({k: ("" if v is None else v) for k, v in kw.items()})
    if r["value"] != "" and r["n"] not in ("", None) and r["metric"] in ("acc", "flip", "stay", "rate", "fail_rate"):
        lo, hi = wilson(float(r["value"]), int(r["n"])); r["wilson_lo"], r["wilson_hi"] = lo, hi
    ROWS.append(r)


def cells_from(d, req, path, st, ct, **kw):
    """acc / flip / stay rows for one request of one eval file (ladder or Ground-CF)."""
    if d is None:
        for m in ("acc", "flip", "stay"):
            add(request=req, metric=m, status=st, clean_twin=ct, file=path, **kw)
        return
    n = d.get("n")
    nf = d.get("look_twin_n") if (req == "look" and "look_twin_n" in d) else n
    ns = d.get("read_stay_n") if (req == "read" and "read_stay_n" in d) else n
    for m, key, nn in (("acc", f"{req}_acc", n), ("flip", f"{req}_flip", nf), ("stay", f"{req}_stay_c", ns)):
        add(request=req, metric=m, value=d.get(key), n=nn, status=st, clean_twin=ct, file=path, **kw)


# ================================================================== 1. primary grid (+5, +18 derived later)
def ladder_path(ds, sfx, meth, split, req):
    e = f"eval/{ds}3_{split}"
    if meth == "No request":
        return f"runs/{ds}_base_frozen_0tok_none_seed0{sfx}/{e}/flip_ll_noop_lead_p2_step_800.json"
    if meth == "English":
        return f"runs/{ds}_base_frozen_0tok_none_seed0{sfx}/{e}/flip_ll_iv5_lead_p2_step_800.json"
    run = {"Word L1 CTM": f"{ds}_neo_ctrl_1tok_ctm_cp_seed0{sfx}", "Prefix L1 CTM": f"{ds}_pt_na_1tok_ctm_cp_seed0{sfx}",
           "Specialist LoRA r16 CTM": f"{ds}_lora_r16_lr2e4_ad_ctm_cp_{req}_seed0{sfx}",
           "Word L1 SFT": f"{ds}_neo_ctrl_1tok_sft_seed0{sfx}", "Prefix L1 SFT": f"{ds}_pt_na_1tok_sft_seed0{sfx}"}[meth]
    return f"runs/{run}/{e}/flip_ll_lead_p2_heldout_step_800.json"


def gcf_path(sfx, meth, split, req):
    e = f"eval/clevr3_{split}/flip_step_800"
    return {"No request": f"runs/clevr3_2s_5tok_sft_seed0{sfx}/{e}_cp4_ho_fc_noop.json",
            "English": f"runs/clevr3_2s_5tok_sft_seed0{sfx}/{e}_cp4_ho_fc_instr.json",
            "Word L1 CTM": f"runs/clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}/{e}_cp4_ho.json",
            "Prefix L1 CTM": f"runs/clevr3_2s_pt_1tok_ctm_cp_fc_seed0{sfx}/{e}_cp4_ho.json",
            "Specialist LoRA r16 CTM": f"runs/clevr3_2s_lora_r16_ctm_cp_fc_{req}_seed0{sfx}/{e}_cp4_ho.json",
            "Joint LoRA r16 CTM": f"runs/clevr3_2s_lora_r16_ctm_cp_fc_joint_seed0{sfx}/{e}_cp4_ho.json",
            "Word L1 SFT": f"runs/clevr3_2s_1tok_sft_fc_seed0{sfx}/{e}_cp4_ho.json",
            "Word L1 CTM+Stay": f"runs/clevr3_2s_1tok_ctmsm_cp_fc_seed0{sfx}/{e}_cp4_ho.json",
            "Word L1 pSFT": f"runs/clevr3_2s_1tok_psft_cp_fc_seed0{sfx}/{e}_cp4_ho.json",   # 2026-09-26 paired-SFT control (same twins, CE on both contexts)
            "Word L1 CTM, early": f"runs/clevr3_2s_1tok_ctm_cp_fc_pfirst_seed0{sfx}/{e}_cp4_ho_reloc_first.json",
            "Word L1 CTM, early, moved late": f"runs/clevr3_2s_1tok_ctm_cp_fc_pfirst_seed0{sfx}/{e}_cp4_ho.json",
            "Word L1 CTM, late, moved early": f"runs/clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}/{e}_cp4_ho_reloc_first.json",
            }[meth]


PRIMARY = ["No request", "English", "Word L1 CTM", "Prefix L1 CTM", "Specialist LoRA r16 CTM", "Joint LoRA r16 CTM"]
OBJ = {"No request": "frozen", "English": "frozen", "Word L1 CTM": "SFT+CTM", "Prefix L1 CTM": "SFT+CTM",
       "Specialist LoRA r16 CTM": "SFT+CTM", "Joint LoRA r16 CTM": "SFT+CTM", "Word L1 SFT": "SFT", "Prefix L1 SFT": "SFT",
       "Word L1 CTM+Stay": "SFT+CTM+Stay", "Word L1 pSFT": "SFT (paired)"}


def fam_primary():
    for sfx, mk, ml in MODELS:
        for split, sl in SPLITS:
            for ds, dn in LADDER + [("gcf", "Ground-CF")]:
                for meth in PRIMARY:
                    for req in REQS:
                        kw = dict(family="1 primary grid", tier="MAIN", model=mk, dataset=dn, split=sl, attribute="all", method=meth,
                                  objective=OBJ.get(meth, ""), train_dataset=dn, seed=0, checkpoint=800)
                        if ds != "gcf" and meth == "Joint LoRA r16 CTM":
                            add(request=req, metric="flip", status="N/A", notes="joint LoRA trained on Ground-CF and MVG only", **kw); continue
                        if ds == "gcf":
                            p = gcf_path(sfx, meth, split, req); st, ct = status_of(p, gcf=True)
                            kw.update(protocol_version="CP, held-out questions, fact captions", eval_position="native")
                        else:
                            p = ladder_path(ds, sfx, meth, split, req); st, ct = status_of(p)
                            kw.update(protocol_version="p2 + lead-in, 4 fixed candidates" + (", held-out templates" if meth not in ("No request", "English") else ""))
                        cells_from(J(p) if os.path.exists(p) else None, req, p, st, ct, **kw)


# ================================================================== 3. confidence (item level)
def fam_confidence():
    import correctness_vs_control as C
    C.CG.SPLIT = "test_id"
    for meth in ("English", "Prefix L1", "LoRA r16", "Word L1"):
        mlab = {"English": "English", "Prefix L1": "Prefix L1 CTM", "LoRA r16": "Specialist LoRA r16 CTM", "Word L1": "Word L1 CTM"}[meth]
        pools = collections.defaultdict(list)
        for sfx, mk, ml in MODELS:
            for ds, dn in C.DS:
                R = C.items(ds, dn, sfx, meth, "test_id").get("look") or []
                for r in R:
                    if r["ok0"] and r["flip"] is not None and r["margin"] is not None:
                        pools[("all", "all")].append(r); pools[(mk, "all")].append(r); pools[("all", dn)].append(r)
        for (mk, dn), rows in pools.items():
            for cond, sel in (("all correct", lambda r: True), ("p>=.90", lambda r: r["p_gold"] >= .90),
                              ("p>=.95", lambda r: r["p_gold"] >= .95), ("p>=.99", lambda r: r["p_gold"] >= .99)):
                S = [r for r in rows if sel(r)]
                add(family="3 confidence", tier="MAIN", model=mk, dataset=dn, split="ID", attribute="all", method=mlab, request="look",
                    condition=cond, metric="fail_rate", value=(round(sum(1 for r in S if not r["flip"]) / len(S), 4) if S else None),
                    n=len(S), status="DONE*" if dn in ("Ground-CF", "all") else "DONE",
                    notes="P(Flip fails | originally correct); Ground-CF items on residue twins until promotion")
            srt = sorted(rows, key=lambda r: r["margin"]); q = len(srt) // 4
            for i in range(4):
                S = srt[i * q:(i + 1) * q] if i < 3 else srt[3 * q:]
                add(family="3 confidence", tier="MAIN", model=mk, dataset=dn, split="ID", attribute="all", method=mlab, request="look",
                    condition=f"margin Q{i + 1}", metric="fail_rate", value=(round(sum(1 for r in S if not r["flip"]) / len(S), 4) if S else None),
                    n=len(S), status="DONE*" if dn in ("Ground-CF", "all") else "DONE",
                    notes=(f"margin {S[0]['margin']:.2f}..{S[-1]['margin']:.2f}" if S else ""))
    for ex in (("English", "G3", "id307_object", "diamond", "cross (caption)", "image diamond->circle", "diamond p=1.00", "cross p=.996",
                "competing caption value; Stay holds (caption->circle: diamond p=1.00)"),
               ("Word L1 CTM", "G3", "id957_object", "diamond", "circle (caption)", "image diamond->cross", "diamond p=1.00",
                "circle p=1.00", "competing caption value; Stay holds")):
        add(family="3 confidence (example)", tier="MAIN", model=ex[1], dataset="Ground-CF", split="ID", attribute="shape", method=ex[0],
            request="look", condition=ex[5], metric="example", value=ex[2], status="DONE", clean_twin="true",
            notes=f"requested {ex[3]}; competing {ex[4]}; original {ex[6]}; after edit {ex[7]}; {ex[8]}; twin verified by eye, residue 0 px")


# ================================================================== 4/18/19. Ground-CF per attribute
def fam_attributes():
    for sfx, mk, ml in MODELS:
        for split, sl in SPLITS:
            for meth in PRIMARY:
                for req in REQS:
                    p = gcf_path(sfx, meth, split, req); st, ct = status_of(p, gcf=True)
                    d = J(p) if os.path.exists(p) else None
                    P = ((d or {}).get("picks") or {}).get(req) or []
                    for sk, skl in SKILLS:
                        kw = dict(family="4 Ground-CF attributes", tier="MAIN", model=mk, dataset="Ground-CF", split=sl, attribute=skl,
                                  method=meth, objective=OBJ.get(meth, ""), request=req, status=st, clean_twin=ct, file=p, seed=0, checkpoint=800,
                                  notes=("held cell (never trained for this request)" if (req, sk) in (("look", "spatial"), ("read", "count")) and meth not in ("No request", "English") else ""))
                        R = [x for x in P if x.get("skill") == sk]
                        if not R:
                            for m in ("acc", "flip", "stay"):
                                add(metric=m, **kw)
                            continue
                        g = lambda x: x["gold"]
                        ok0 = [x.get("p0") == g(x)["base"] for x in R]
                        if req == "look":
                            fl = [o and x["p_twin"] == g(x)["twin"] for o, x in zip(ok0, R) if x.get("p_twin") is not None]
                            sv = [o and x["p_capedit"] == g(x)["capedit"] for o, x in zip(ok0, R) if x.get("p_capedit") is not None]
                        else:
                            fl = [o and x["p1"] == g(x)["capedit"] for o, x in zip(ok0, R) if x.get("p1") is not None]
                            sv = [o and x["p_twin"] == g(x)["twin"] for o, x in zip(ok0, R) if x.get("p_twin") is not None]
                        for m, v in (("acc", ok0), ("flip", fl), ("stay", sv)):
                            add(metric=m, value=(round(sum(v) / len(v), 4) if v else None), n=len(v), **kw)


# ================================================================== 6/7/8. objectives and position
def fam_objectives():
    for sfx, mk, ml in MODELS:
        for ds, dn in LADDER + [("gcf", "Ground-CF")]:
            for meth in ("Word L1 SFT", "Prefix L1 SFT"):
                for req in REQS:
                    kw = dict(family="6 CTM vs SFT", tier="MAIN", model=mk, dataset=dn, split="ID", attribute="all", method=meth,
                              objective="SFT", seed=0, checkpoint=800)
                    if ds == "gcf" and meth == "Prefix L1 SFT":
                        add(request=req, metric="flip", status="MISSING (needs training)", notes="Ground-CF prefix SFT L1 never trained", **kw); continue
                    if ds == "gcf":
                        p = gcf_path(sfx, meth, "test_id", req); st, ct = status_of(p, gcf=True)
                    else:
                        p = ladder_path(ds, sfx, meth, "test_id", req); st, ct = status_of(p)
                    cells_from(J(p) if os.path.exists(p) else None, req, p, st, ct, **kw)
        for req in REQS:
            add(family="6 CTM vs SFT", tier="MAIN", model=mk, dataset="all", split="ID", method="Specialist LoRA r16 SFT", request=req,
                metric="flip", status="N/A", notes="no LoRA SFT r16 runs under the final protocol (not run)")
        for split, sl in SPLITS:
            for req in REQS:
                p = gcf_path(sfx, "Word L1 CTM+Stay", split, req); st, ct = status_of(p, gcf=True)
                cells_from(J(p) if os.path.exists(p) else None, req, p, st, ct, family="7 CTM+Stay", tier="SUPPORT", model=mk,
                           dataset="Ground-CF", split=sl, attribute="all", method="Word L1 CTM+Stay", objective="SFT+CTM+Stay", seed=0, checkpoint=800,
                           notes="" if sl == "ID" else "optional")
                for meth, tp, ep in (("Word L1 CTM, early", "early", "early"), ("Word L1 CTM, early, moved late", "early", "late"),
                                     ("Word L1 CTM, late, moved early", "late", "early")):
                    p = gcf_path(sfx, meth, split, req); st, ct = status_of(p, gcf=True)
                    cells_from(J(p) if os.path.exists(p) else None, req, p, st, ct, family="8 position", tier="MAIN", model=mk,
                               dataset="Ground-CF", split=sl, attribute="all", method=meth, objective="SFT+CTM", train_position=tp,
                               eval_position=ep, seed=0, checkpoint=800)


# ================================================================== 9. evidence order
def fam_order():
    for sfx, mk, ml in MODELS:
        for meth in ("Word L1 CTM", "Prefix L1 CTM", "Word L1 CTM, early"):
            for req in REQS:
                p = gcf_path(sfx, meth, "test_id", req); st, ct = status_of(p, gcf=True)
                d = J(p) if os.path.exists(p) else None
                P = ((d or {}).get("picks") or {}).get(req) or []
                for order, cf in (("caption first", True), ("image first", False)):
                    R = [x for x in P if bool(x.get("cap_first")) == cf]
                    kw = dict(family="9 evidence order", tier="SUPPORT", model=mk, dataset="Ground-CF", split="ID", attribute="all", method=meth,
                              request=req, evidence_order=order, status=st, clean_twin=ct, file=p)
                    if not R:
                        add(metric="flip", **kw); continue
                    g = lambda x: x["gold"]; ok0 = [x.get("p0") == g(x)["base"] for x in R]
                    fk, sk = (("p_twin", "twin"), ("p_capedit", "capedit")) if req == "look" else (("p1", "capedit"), ("p_twin", "twin"))
                    fl = [o and x[fk[0]] == g(x)[fk[1]] for o, x in zip(ok0, R) if x.get(fk[0]) is not None]
                    sv = [o and x[sk[0]] == g(x)[sk[1]] for o, x in zip(ok0, R) if x.get(sk[0]) is not None]
                    for m, v in (("acc", ok0), ("flip", fl), ("stay", sv)):
                        add(metric=m, value=(round(sum(v) / len(v), 4) if v else None), n=len(v), **kw)


# ================================================================== 11. capacity (Qwen-4B Ground-CF)
def fam_capacity():
    specs = [("Word L1", "clevr3_2s_1tok_ctm_cp_fc_seed0_qwen35"), ("Word L20", "clevr3_2s_20tok_ctm_cp_fc_seed0_qwen35"),
             ("Prefix L1", "clevr3_2s_pt_1tok_ctm_cp_fc_seed0_qwen35"), ("Prefix L20", "clevr3_2s_pt_20tok_ctm_cp_fc_seed0_qwen35")]
    for lab, run in specs:
        for req in REQS:
            p = f"runs/{run}/eval/clevr3_test_id/flip_step_800_cp4_ho.json"; st, ct = status_of(p, gcf=True)
            cells_from(J(p) if os.path.exists(p) else None, req, p, st, ct, family="11 capacity", tier="DIAGNOSTIC", model="Q4",
                       dataset="Ground-CF", split="ID", attribute="all", method=lab, objective="SFT+CTM", seed=0, checkpoint=800)
    for r in (4, 16, 64):
        for req in REQS:
            run = f"clevr3_2s_lora_r{r}_ctm_cp_fc_{req}_seed0_qwen35"; p = f"runs/{run}/eval/clevr3_test_id/flip_step_800_cp4_ho.json"
            st, ct = status_of(p, gcf=True)
            if not os.path.isdir(f"runs/{run}"):
                st = "MISSING (not trained)"
            cells_from(J(p) if os.path.exists(p) else None, req, p, st, ct, family="11 capacity", tier="DIAGNOSTIC", model="Q4",
                       dataset="Ground-CF", split="ID", attribute="all", method=f"Specialist LoRA r{r}", objective="SFT+CTM", seed=0, checkpoint=800)


# ================================================================== 12. transfer
def fam_transfer():
    src = {"word": {"vcf": "vcf_neo_ctrl_1tok_ctm_cp_seed0{s}", "color": "color_neo_ctrl_1tok_ctm_cp_seed0{s}",
                    "coco": "coco_neo_ctrl_1tok_ctm_cp_seed0{s}", "gcf": "clevr3_2s_1tok_ctm_cp_fc_seed0{s}"},
           "prefix": {"vcf": "vcf_pt_na_1tok_ctm_cp_seed0{s}", "color": "color_pt_na_1tok_ctm_cp_seed0{s}",
                      "coco": "coco_pt_na_1tok_ctm_cp_seed0{s}", "gcf": "clevr3_2s_pt_1tok_ctm_cp_fc_seed0{s}"},
           "lora": {"vcf": "vcf_lora_r16_lr2e4_ad_ctm_cp_{r}_seed0{s}", "color": "color_lora_r16_lr2e4_ad_ctm_cp_{r}_seed0{s}",
                    "coco": "coco_lora_r16_lr2e4_ad_ctm_cp_{r}_seed0{s}", "gcf": "clevr3_2s_lora_r16_ctm_cp_fc_{r}_seed0{s}"}}
    names = dict(LADDER + [("gcf", "Ground-CF")])
    for sfx, mk, ml in MODELS:
        for meth, S in src.items():
            mlab = {"word": "Word L1 CTM", "prefix": "Prefix L1 CTM", "lora": "Specialist LoRA r16 CTM"}[meth]
            for s_ds, s_run in S.items():
                for t_ds in ("vcf", "color", "coco", "gcf"):
                    for req in REQS:
                        run = s_run.format(s=sfx, r=req)
                        kw = dict(family="12 transfer", tier="MAIN" if meth != "lora" else "SUPPORT", model=mk, dataset=names[t_ds], split="ID",
                                  attribute="all", method=mlab, objective="SFT+CTM", train_dataset=names[s_ds], seed=0, checkpoint=800,
                                  condition="in-domain" if s_ds == t_ds else "transfer")
                        if s_ds == t_ds:
                            p = gcf_path(sfx, mlab, "test_id", req) if t_ds == "gcf" else ladder_path(t_ds, sfx, mlab, "test_id", req)
                            st, ct = status_of(p, gcf=(t_ds == "gcf"))
                        elif meth == "lora" and "gcf" in (s_ds, t_ds):
                            add(request=req, metric="flip", status="N/A", notes="LoRA transfer planned ladder -> ladder only", **kw); continue
                        elif t_ds == "gcf":
                            p = f"runs/{run}/eval/clevr3_test_id/flip_step_800_cp4xfer_ho.json"; st, ct = status_of(p, gcf=True)
                        else:
                            p = f"runs/{run}/eval/{t_ds}3_test_id/flip_ll_lead_p2_heldout_step_800.json"; st, ct = status_of(p)
                        cells_from(J(p) if os.path.exists(p) else None, req, p, st, ct, **kw)


# ================================================================== 13-16. interface evaluations
def fam_interfaces():
    E4 = J("results/detect_two_prompts.json") or {}
    lab = {"clevr3_2s_1tok_ctm_cp_fc_seed0": "Word L1 CTM", "clevr3_2s_pt_1tok_ctm_cp_fc_seed0": "Prefix L1 CTM"}
    for r in E4.get("rows", []):
        run = r.get("run", ""); base = re.sub(r"(_qwen35|_qwen2b)$", "", run)
        mk = "Q4" if run.endswith("_qwen35") else "Q2" if run.endswith("_qwen2b") else "G3"
        meth = ("Specialist LoRA r16 CTM" if "lora" in run else lab.get(base) or
                ("English" if r.get("mode") == "instructed" else "No request" if r.get("mode") == "noop" else base))
        for key, m in (("ll_balanced", "balanced_acc"), ("ll_detect", "conflict_recall"), ("ll_false", "false_positive_rate"),
                       ("gen_balanced", "balanced_acc_generation")):
            if key in r:
                add(family="13 E4 two-pass detection", tier="SUPPORT", model=mk, dataset="Ground-CF", split="ID", method=meth,
                    condition="forced choice" if key.startswith("ll") else "parsed generation", metric=m, value=r[key],
                    n=r.get("ll_n" if key.startswith("ll") else "gen_n"), status="DONE*", file="results/detect_two_prompts.json",
                    notes=("Qwen-4B generation parsing broken: use forced choice" if mk == "Q4" and key.startswith("gen") else "") +
                          "; E2 generations on residue twins (agreement contexts are text edits)")
    for ds, dn in LADDER:
        add(family="13 E4 two-pass detection", tier="SUPPORT", dataset=dn, split="ID", metric="balanced_acc", status="N/A",
            notes="E4 run on Ground-CF only")
    for sfx, mk, ml in MODELS:
      for split, sl in SPLITS:
        for meth, p in (("Word L1 CTM", f"runs/clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}/eval/clevr3_{split}/switch_step_800_fc.json"),
                        ("Prefix L1 CTM", f"runs/clevr3_2s_pt_1tok_ctm_cp_fc_seed0{sfx}/eval/clevr3_{split}/switch_step_800_fc.json"),
                        ("English", f"runs/clevr3_2s_5tok_sft_seed0{sfx}/eval/clevr3_{split}/switch_step_800_fc_instr.json"),
                        ("No request", f"runs/clevr3_2s_5tok_sft_seed0{sfx}/eval/clevr3_{split}/switch_step_800_fc_noop.json")):
            st, _ = status_of(p); d = J(p) if os.path.exists(p) else None
            kw = dict(family="14 E6 switching", tier="MAIN" if sl == "ID" else "SUPPORT", model=mk, dataset="Ground-CF", split=sl,
                      method=meth, status=st, file=p, notes="chance both .111; image twins not used")
            if d:
                for asg, v in (d.get("assignments") or {}).items():
                    for m in ("q1", "q2", "both"):
                        add(condition=asg, metric=m, value=v.get(m), n=v.get("n"), **kw)
                add(condition="mixed (look|read, read|look)", metric="both", value=d.get("mixed_both"), n=d.get("n_pairs"), **kw)
            else:
                add(condition="mixed", metric="both", **kw)
        add(family="14 E6 switching", tier="MAIN", model=mk, dataset="Ground-CF", split="ID", method="Specialist LoRA r16 CTM",
            metric="both", status="N/A", notes="one adapter per request: cannot assign two requests in one forward pass")
        for split, sl in SPLITS:
            for meth, p in (("Word L1 CTM", f"runs/clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}/eval/clevr3_{split}/compose_pair_step_800_v2.json"),
                            ("Prefix L1 CTM", f"runs/clevr3_2s_pt_1tok_ctm_cp_fc_seed0{sfx}/eval/clevr3_{split}/compose_pair_step_800_v2.json"),
                            ("English", f"runs/clevr3_2s_5tok_sft_seed0{sfx}/eval/clevr3_{split}/compose_pair_step_800_v2_instr.json"),
                            ("No request", f"runs/clevr3_2s_5tok_sft_seed0{sfx}/eval/clevr3_{split}/compose_pair_step_800_v2_noop.json")):
                st, _ = status_of(p); d = J(p) if os.path.exists(p) else None
                for fam, keys in (("15 E5a composition", (("E5a", "composed_conflict"), ("E5a", "composed_agree"), ("E5a", "conflict_recall_derived"),
                                                          ("E5a", "agree_recall_derived"), ("E5a", "balanced_derived"))),
                                  ("16 E5b verdict", (("E5b.lead", "conflict_recall"), ("E5b.lead", "agree_recall"), ("E5b.lead", "balanced"),
                                                      ("E5b.bare", "balanced")))):
                    for grp, k in keys:
                        v = None
                        if d:
                            g = d
                            for part in grp.split("."):
                                g = (g or {}).get(part, {})
                            v = (g or {}).get(k)
                        add(family=fam, tier="MAIN" if fam.startswith("15") else "SUPPORT", model=mk, dataset="Ground-CF", split=sl,
                            method=meth, metric=k if grp == "E5a" else f"{grp.split('.')[1]}:{k}", value=v,
                            n=(d or {}).get("n_per_cell"), status=st, file=p,
                            notes="chance pair 1/9, balanced .5; image twins not used (text edits only)")
            add(family="15 E5a composition", tier="MAIN", model=mk, dataset="Ground-CF", split=sl, method="Specialist LoRA r16 CTM",
                metric="composed_conflict", status="N/A", notes="one adapter per request")


# ================================================================== 17. MVG
def fam_mvg():
    M = {"": "_frozen_gemma", "_qwen35": "_frozen_qwen35", "_qwen2b": "_frozen_qwen2b"}
    METH = [("No request", lambda s, f: (f, "_noop")), ("English", lambda s, f: (f, "_instr")),
            ("Inline soft tokens L5", lambda s, f: (f"mvgV2_5tok_ptctm_fix_seed0{s}", "")),
            ("Joint LoRA r16 CTM", lambda s, f: (f"mvgV2_lora_r16_ctm_seed0{s}", "")), ("Word L5 CTM", lambda s, f: (f"mvgV2_5tok_sftctm_hard_seed0{s}", ""))]
    for sfx, mk, ml in MODELS:
        for meth, fn in METH:
            for view, suf in (("full frame", ""), ("evidence crop +25%", "_crop0.25"), ("wider crop +100%", "_crop_crop1.0")):
                run, sty = fn(sfx, M[sfx]); p = f"runs/{run}/eval/mvg_flip/flip_step_800_rb{suf}{sty}.json"
                d = J(p) if os.path.exists(p) else None
                st = "DONE" if d else ("N/A" if view.startswith("wider") and meth not in ("Word L5 CTM", "Inline soft tokens L5") else "MISSING")
                kw = dict(family="17 MultiVENT-G", tier="MAIN", model=mk, dataset="MultiVENT-G", split="real-world", method=meth,
                          condition=view, status=st, file=p, notes=("position/trainer confound: not the controlled prefix baseline"
                                                                    if meth.startswith("Inline") else ""))
                v, t = ((d or {}).get("vis_value_swap") or {}), ((d or {}).get("desc_swap") or {})
                la = (v["acc_a"] + v["acc_b"]) / 2 if v.get("acc_a") is not None else None
                ra = (t["acc_a"] + t["acc_b"]) / 2 if t.get("acc_a") is not None else None
                for req, m, val, nn in (("look", "acc", la, v.get("n")), ("look", "flip", v.get("flip"), v.get("n")),
                                        ("look", "agreement (not Stay)", t.get("stay_other"), t.get("n")),
                                        ("read", "acc", ra, t.get("n")), ("read", "flip", t.get("flip"), t.get("n")),
                                        ("read", "agreement (not Stay)", v.get("stay_other"), v.get("n"))):
                    add(request=req, metric=m, value=None if val is None else round(val, 4), n=nn, **kw)
    add(family="17 MultiVENT-G", tier="MAIN", dataset="MultiVENT-G", split="OOD", metric="flip", status="N/A", notes="no defined OOD split")


# ================================================================== 20/21/22/23. design diagnostics
def fam_design():
    for split, sl in SPLITS:
        d = J(f"results/cp_blind_stub_{split}.json")
        for m, k, nk in (("flip agreement twins", "look_flip_agree", "look_agree_n"), ("flip conflict-preserving", "look_flip", "look_twin_n")):
            add(family="20 AR vs CP", tier="SUPPORT", dataset="Ground-CF", split=sl, method="image-blind scorer", request="look",
                metric=m, value=(d or {}).get(k), n=(d or {}).get(nk), status="DONE" if d else "MISSING", file=f"results/cp_blind_stub_{split}.json",
                notes="the old .172 (29 items) is obsolete: never print")
    for sfx, mk, ml in MODELS:
        for meth, run in (("Word L1 CTM", f"clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}"), ("Prefix L1 CTM", f"clevr3_2s_pt_1tok_ctm_cp_fc_seed0{sfx}"),
                          ("Specialist LoRA r16 CTM", f"clevr3_2s_lora_r16_ctm_cp_fc_look_seed0{sfx}")):
            p = f"runs/{run}/eval/clevr3_test_id/flip_step_800_cp4_ho_agree.json"; d = J(p) if os.path.exists(p) else None
            for m, k, nk in (("flip agreement twins", "look_flip_agree", "look_agree_n"), ("flip conflict-preserving", "look_flip", "look_twin_n")):
                add(family="20 AR vs CP", tier="SUPPORT", model=mk, dataset="Ground-CF", split="ID", method=meth, request="look", metric=m,
                    value=(d or {}).get(k), n=(d or {}).get(nk), status="DONE*" if d else "MISSING", clean_twin="false", file=p,
                    notes="identifiability claim only, never a ranking claim; residue twins in both designs")
    au = J("results/twin_residue_audit.json") or {}
    for k, v in au.items():
        split, sk = k.split("|")
        add(family="21 clean-twin audit", tier="SUPPORT", dataset="Ground-CF", split=split, attribute=sk, metric="twins with visible residue",
            value=v.get("visible"), n=v.get("n"), status="DONE", file="results/twin_residue_audit.json", notes=f"any residue {v.get('any')}")
    for key, v in PROMO.items():
        for m in ("look_flip", "read_stay_c", "look_acc"):
            add(family="21 clean-twin audit", tier="SUPPORT", dataset="Ground-CF", metric=f"{m} residue -> corrected", value=v["after"].get(m),
                status="DONE", clean_twin="true", file=key, notes=f"before {v['before'].get(m)}")
    for sfx, mk, ml in MODELS:
        for req in REQS:
            p = f"runs/clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}/eval/clevr3_test_id/flip_step_800_cp4_ho_trcar.json"; st, ct = status_of(p, gcf=True)
            cells_from(J(p) if os.path.exists(p) else None, req, p, st, ct, family="22 carrier generalization", tier="DIAGNOSTIC", model=mk,
                       dataset="Ground-CF", split="ID", method="Word L1 CTM", condition="training carriers + held-out question")
            for ds, dn in LADDER:
                p = f"runs/{ds}_neo_ctrl_1tok_ctm_cp_seed0{sfx}/eval/{ds}3_test_id/flip_ll_lead_p2_step_800.json"; st, _ = status_of(p)
                cells_from(J(p) if os.path.exists(p) else None, req, p, st, "", family="22 carrier generalization", tier="DIAGNOSTIC",
                           model=mk, dataset=dn, split="ID", method="Word L1 CTM", condition="training template")
        for meth, run, tag in (("Word L1 CTM", f"clevr3_2s_1tok_ctm_cp_fc_seed0{sfx}", "_gen_ho"), ("Prefix L1 CTM", f"clevr3_2s_pt_1tok_ctm_cp_fc_seed0{sfx}", "_gen_ho"),
                               ("English", f"clevr3_2s_5tok_sft_seed0{sfx}", "_gen_ho_fc_instr_hint"), ("No request", f"clevr3_2s_5tok_sft_seed0{sfx}", "_gen_ho_fc_noop_hint")):
            p = f"runs/{run}/eval/clevr3_test_id/gen_step_800{tag}.json"; d = J(p) if os.path.exists(p) else None
            for req in REQS:
                s = ((d or {}).get("summary") or {}).get(req) or {}
                for m in ("parse_cand_rate", "gen_acc", "gen_flip", "gen_stay_c", "gen_ntok_mean"):
                    add(family="23 free generation", tier="DIAGNOSTIC", model=mk, dataset="Ground-CF", split="ID", method=meth, request=req,
                        metric=m, value=s.get(m), n=s.get("n_items"), status="DONE*" if d else "MISSING", file=p,
                        notes="Qwen-4B generations are truncated reasoning: not a valid generation metric" if mk == "Q4" else "")


# ================================================================== 24/25. parameters and interface metadata
def fam_params():
    def count(f):
        with open(f, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]; hd = json.loads(h.read(n))
        return sum(math.prod(v["shape"]) for k, v in hd.items() if k != "__metadata__")
    hidden = {"G3": 2560, "Q4": 2560, "Q2": 2048}
    for sfx, mk, ml in MODELS:
        f = f"runs/clevr3_2s_lora_r16_ctm_cp_fc_look_seed0{sfx}/ckpt/step_800/adapter_model.safetensors"
        lora = count(f) if os.path.exists(f) else None
        h = hidden[mk]
        for meth, per_req, objs, note in (("Word L1", h, "2 rows in one frozen model", "one embedding row per request"),
                                          ("Prefix L1", h, "2 soft vectors in one frozen model", "one soft vector per request"),
                                          ("Specialist LoRA r16", lora, "2 adapters (one per request)", "q,k,v,o,gate,up,down; r=16"),
                                          ("Joint LoRA r16", lora, "1 adapter for both requests", "same adapter size as one specialist")):
            add(family="24 parameter counts", tier="MAIN", model=mk, method=meth, metric="trainable params per request-bearing object",
                value=per_req, status="DONE" if per_req else "MISSING", notes=f"{objs}; {note}; ratio to one word row: {per_req / h:.0f}x" if per_req else "")
    meta = [("Word", "1 row x hidden per request", "lexical token in a natural-language carrier", "yes", "yes (in text)", "no"),
            ("Prefix", "1 vector x hidden per request", "soft vector at the start of the turn (Ground-CF/ladder); inline on MVG", "yes", "yes (programmatic)", "no"),
            ("Specialist LoRA", "one r16 adapter per request", "adapter identity (global model state)", "no", "no (current implementation)", "N/A"),
            ("Joint LoRA", "one r16 adapter", "English request + shared adapter", "yes", "potentially", "no")]
    for m, params, loc, both, local, comp in meta:
        add(family="25 interface metadata", tier="MAIN", method=m, metric="interface", value=params, status="DONE",
            notes=f"control: {loc}; both requests in one model state: {both}; local insertion: {local}; composition trained: {comp}; "
                  f"base frozen: yes; trained with CTM: yes")


# ================================================================== 26. paired comparisons
def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y); n10 = sum(1 for x, y in zip(a, b) if y and not x); n = n01 + n10
    return n01, n10, (min(1.0, 2 * sum(math.comb(n, j) for j in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0)


def items_ladder(p, req):
    f = p.replace(".json", "_items.jsonl.gz")
    if not os.path.exists(f):
        return None
    by = collections.defaultdict(dict)
    for l in gzip.open(f, "rt"):
        a = json.loads(l)
        if "_meta" not in a and a.get("request") == req:
            by[a["item"]][a["ctx"]] = a["pick"]
    G = {"look": ("A", ("imgswap", "B"), ("capswap", "A")), "read": ("C", ("capswap", "D"), ("imgswap", "C"))}[req]
    return {i: {"flip": int(c.get("base") == G[0] and c.get(G[1][0]) == G[1][1]), "stay": int(c.get("base") == G[0] and c.get(G[2][0]) == G[2][1])}
            for i, c in by.items() if "base" in c}


def items_gcf(p, req):
    d = J(p) if os.path.exists(p) else None
    P = ((d or {}).get("picks") or {}).get(req)
    if not P:
        return None
    out = {}
    for x in P:
        g = x["gold"]; ok = x.get("p0") == g["base"]
        fk, sk = (("p_twin", "twin"), ("p_capedit", "capedit")) if req == "look" else (("p1", "capedit"), ("p_twin", "twin"))
        out[x["id"]] = {"flip": None if x.get(fk[0]) is None else int(ok and x[fk[0]] == g[fk[1]]),
                        "stay": None if x.get(sk[0]) is None else int(ok and x[sk[0]] == g[sk[1]])}
    return out


def fam_compare():
    pairs = [("Word L1 CTM", "Prefix L1 CTM"), ("Word L1 CTM", "Word L1 SFT"), ("Prefix L1 CTM", "Prefix L1 SFT"),
             ("Word L1 CTM, early", "Word L1 CTM")]
    for sfx, mk, ml in MODELS:
        for ds, dn in LADDER + [("gcf", "Ground-CF")]:
            for a_m, b_m in pairs:
                if (ds != "gcf" and "early" in a_m) or (ds == "gcf" and b_m == "Prefix L1 SFT"):
                    continue                              # early word: Ground-CF only; Ground-CF prefix SFT never trained
                for req in REQS:
                    if ds == "gcf":
                        pa, pb = gcf_path(sfx, a_m, "test_id", req), gcf_path(sfx, b_m, "test_id", req)
                        A, B = items_gcf(pa, req), items_gcf(pb, req)
                        ca, cb = (pa in PROMO or pa in CFB), (pb in PROMO or pb in CFB)
                        tw = "corrected" if (ca and cb) else ("mixed" if ca != cb else "residue")
                    else:
                        pa, pb = ladder_path(ds, sfx, a_m, "test_id", req), ladder_path(ds, sfx, b_m, "test_id", req)
                        A, B = items_ladder(pa, req), items_ladder(pb, req); tw = ""
                    for met in ("flip", "stay"):
                        row = {"comparison": f"{a_m} vs {b_m}", "subset": "all items", "model": mk, "dataset": dn, "split": "ID",
                               "request": req, "metric": met, "method_a": a_m, "method_b": b_m, "twins": tw}
                        if A is None or B is None or tw == "mixed":
                            row.update({"status": "pending" if tw != "mixed" else "pending (twin sets differ)"}); CMP.append(row); continue
                        ks = [k for k in A if k in B and A[k][met] is not None and B[k][met] is not None]
                        a = [A[k][met] for k in ks]; b = [B[k][met] for k in ks]
                        n01, n10, p = mcnemar(a, b)
                        row.update({"n": len(ks), "a_value": round(sum(a) / len(a), 4) if a else None, "b_value": round(sum(b) / len(b), 4) if b else None,
                                    "difference": round((sum(a) - sum(b)) / len(a), 4) if a else None, "a_only": n01, "b_only": n10,
                                    "p_value": float(f"{p:.3g}"), "status": "DONE" if tw != "residue" else "DONE* (residue twins)"})
                        CMP.append(row)
    CMP.append({"comparison": "MVG full frame vs evidence crop", "status": "not computed (per-item pairing not banked in the summary files)"})


# ================================================================== markdown overview
TIER = {"1": "MAIN", "2": "MAIN", "3": "MAIN", "4": "MAIN", "5": "MAIN", "6": "MAIN", "7": "SUPPORT", "8": "MAIN", "9": "SUPPORT",
        "10": "SUPPORT", "11": "DIAGNOSTIC", "12": "MAIN", "13": "SUPPORT", "14": "MAIN", "15": "MAIN", "16": "SUPPORT", "17": "MAIN",
        "18": "MAIN", "19": "SUPPORT", "20": "SUPPORT", "21": "SUPPORT", "22": "DIAGNOSTIC", "23": "DIAGNOSTIC", "24": "MAIN", "25": "MAIN",
        "26": "MAIN"}


def f3(x):
    return "-" if x in (None, "") else ("1.0" if round(float(x), 3) >= 1 else f"{float(x):.3f}"[1:] if float(x) >= 0 else f"{float(x):.3f}")


def macro(rows, method, split, req, metric, datasets=None, models=None):
    v = [float(r["value"]) for r in rows if r["family"] == "1 primary grid" and r["method"] == method and r["split"] == split
         and r["request"] == req and r["metric"] == metric and r["value"] != "" and (datasets is None or r["dataset"] in datasets)
         and (models is None or r["model"] in models)]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


def overview():
    md = [f"# Results inventory (generated {NOW:%Y-%m-%d %H:%M}, `scripts/results_inventory.py`)", "",
          "One canonical file set: `results/RESULTS_INVENTORY.csv` (one row per result cell, the author's columns), "
          "`results/RESULTS_INVENTORY.json` (same rows), `results/RESULTS_COMPARISONS.csv` (paired tests). This page is the overview.",
          "",
          "**Status legend.** DONE = final protocol (Ground-CF: corrected twins). **DONE\\*** = value exists on a superseded twin "
          "set (Ground-CF residue twins) and is provisional until its corrected rerun lands (in brackets). RUNNING / QUEUED "
          "(pending, FIFO) / QUEUED-gate (held by the priority gate, released in order) / AUTO (submitted when its training "
          "finishes) / HELD-tier (in a keeper tier that is released only into idle lanes; ~1000 tier tasks are ahead, so not "
          "reached today) / PARKED (parked tier; will not run) / MISSING (no file, no job) / N/A (undefined by design).", "",
          "**Scope.** Models G3 = Gemma-3-4B, Q4 = Qwen3.5-4B, Q2 = Qwen3.5-2B (Gemma-4-E4B dropped). Controlled datasets VCF, "
          "ColorSwap, COCO-CF, Ground-CF (ID + OOD; OOD = benchmark-specific held-out values/concepts; Ground-CF = held-out "
          "attribute values); MultiVENT-G real-world (no OOD split). Requests `<look>`/`<read>`; Accuracy / Flip / Stay "
          "(conjunctive). Seed 0, step 800 everywhere.", ""]
    # family status table
    fam = collections.OrderedDict()
    for r in ROWS:
        fam.setdefault(r["family"], collections.Counter())[r["status"].split(" (")[0] if r["status"].startswith("DONE*") else r["status"]] += 1
    md += ["## Status by family (cell counts)", "", "| family | tier | DONE | DONE* | RUNNING | QUEUED | QUEUED-gate | AUTO | HELD-tier | PARKED | MISSING | N/A |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f, c in fam.items():
        tier = TIER.get(f.split()[0], "")
        miss = sum(v for k, v in c.items() if k.startswith("MISSING"))
        md.append(f"| {f} | {tier} | {c.get('DONE', 0)} | {c.get('DONE*', 0)} | {c.get('RUNNING', 0)} | {c.get('QUEUED', 0)} | "
                  f"{c.get('QUEUED-gate', 0)} | {c.get('AUTO', 0)} | {c.get('HELD-tier', 0)} | {c.get('PARKED', 0)} | {miss} | {c.get('N/A', 0)} |")
    # 2. accuracy vs flip/stay (provisional)
    md += ["", "## 2. Accuracy vs Flip / Stay (ID macro over available model × dataset cells; provisional until Ground-CF is corrected)", "",
           "| method | request | cells | Acc | Flip | Stay | Acc−Flip | Acc−Stay | 1−Flip/Acc | 1−Stay/Acc |", "|---|---|---|---|---|---|---|---|---|---|"]
    for meth in PRIMARY:
        for req in REQS:
            a, n = macro(ROWS, meth, "ID", req, "acc"); f, _ = macro(ROWS, meth, "ID", req, "flip"); s, _ = macro(ROWS, meth, "ID", req, "stay")
            if a is None:
                md.append(f"| {meth} | {req} | 0 | pending | | | | | | |"); continue
            md.append(f"| {meth} | {req} | {n} | {f3(a)} | {f3(f)} | {f3(s)} | {f3(a - f)} | {f3(a - s)} | {f3(1 - f / a)} | {f3(1 - s / a)} |")
    # common coverage for LoRA
    common = [(r["model"], r["dataset"]) for r in ROWS if r["family"] == "1 primary grid" and r["method"] == "Specialist LoRA r16 CTM"
              and r["split"] == "ID" and r["request"] == "look" and r["metric"] == "flip" and r["value"] != ""]
    md += ["", f"Specialist LoRA covers {len(common)}/12 ID `<look>` cells; on exactly those cells (common coverage):", "",
           "| method | `<look>` Acc / Flip / Stay |", "|---|---|"]
    for meth in PRIMARY[:5]:
        vals = []
        for met in ("acc", "flip", "stay"):
            v = [float(r["value"]) for r in ROWS if r["family"] == "1 primary grid" and r["method"] == meth and r["split"] == "ID"
                 and r["request"] == "look" and r["metric"] == met and r["value"] != "" and (r["model"], r["dataset"]) in common]
            vals.append(sum(v) / len(v) if v else None)
        md.append(f"| {meth} | " + " / ".join(f3(x) for x in vals) + " |")
    # 5. ID -> OOD
    md += ["", "## 5. ID → OOD (`<look>` Flip, per dataset, mean over models with both splits)", "",
           "| method | " + " | ".join(dn for _, dn in LADDER + [("gcf", "Ground-CF")]) + " |", "|---|---|---|---|---|"]
    for meth in PRIMARY[:5]:
        cells = []
        for _, dn in LADDER + [("gcf", "Ground-CF")]:
            pr = []
            for sfx, mk, ml in MODELS:
                gi = [r for r in ROWS if r["family"] == "1 primary grid" and r["method"] == meth and r["dataset"] == dn and r["model"] == mk
                      and r["request"] == "look" and r["metric"] == "flip" and r["value"] != ""]
                i = [float(r["value"]) for r in gi if r["split"] == "ID"]; o = [float(r["value"]) for r in gi if r["split"] == "OOD"]
                if i and o:
                    pr.append((i[0], o[0]))
            if pr:
                ii = sum(x for x, _ in pr) / len(pr); oo = sum(y for _, y in pr) / len(pr)
                cells.append(f"{f3(ii)} → {f3(oo)} (R {oo / ii:.2f}, {len(pr)} m)" if ii else f"{f3(ii)} → {f3(oo)}")
            else:
                cells.append("pending")
        md.append(f"| {meth} | " + " | ".join(cells) + " |")
    # queue and deadline
    md += ["", "## Queue, ETAs and the deadline", "",
           "Deadline assumed: ICLR full paper **Fri Sep 25 23:59 AoE = Sat Sep 26 13:59 CEST** (EXECUTION_SEP.md gives the day, not "
           "the hour). Results needed before the rewrite, so the practical cut-off is **~18:00 CEST today**. Capacity: 16 GPU slots, "
           "of which 8 L40S are held by the older FIFO queue (ladder OOD / SFT / transfer evaluations). Priority on the fast "
           "cards (priority gate): early-word evals > corrected-twin core ID > extra ID > E5 v2 > corrected-twin core OOD > extra late.", "",
           "| work | jobs | status now | ETA (CEST) | in time? |", "|---|---|---|---|---|",
           "| early word training | 708193-5 | running on H | G3 ~07:25, Q2 ~09:35, Q4 ~10:45 | yes |",
           "| early word evals (ID old twins; ID+OOD corrected) | AUTO, 6 per model | after training | G3 ~08:45, Q2 ~11:00, Q4 ~12:15 | yes |",
           "| corrected-twin core ID (18) | 709321 | queued (top of gate) | ~11:00-12:00 | yes |",
           "| corrected-twin extra ID: word transfer to Ground-CF 9, SFT word 3 | 709370 | held (gate) | ~12:30 | yes |",
           "| joint LoRA Ground-CF training + 6 evals (corrected twins) | 702656-8 + AUTO | running | evals ~09:30-10:30 | yes |",
           "| E5a/E5b v2 (24) | 708951 | 2 running, 22 held | ~14:30 | yes |",
           "| corrected-twin core OOD (18) | 709322 | held (gate) | ~17:00 | borderline |",
           "| extra late: SFT word OOD 3, CTM+Stay word ID 3 | 709371 | held (gate) | ~18:00 | borderline |",
           "| ladder SFT evals (36) | 708215 | queued, L40S FIFO | ~10:00-11:00 | yes |",
           "| prefix transfer (36) | 708216 | queued, L40S FIFO | ~11:00-12:00 | yes |",
           "| ladder OOD (frozen + trained), Ground-CF baselines OOD: unheld part | 702609 / 702641 ... | running / queued, L40S | ~09:00-10:00 | yes |",
           "| ladder OOD: the HELD-tier part (see list below) | .held_tier2_evals etc. | behind ~1000 tier tasks | not today | NO unless released |",
           "| LoRA transfer (36) | 708217 | held, idle tier | only if slots idle | unlikely |",
           "| E6 Qwen-4B word ID (MAIN) | 700167_1 | PARKED (1183 park) | ~1 h once released | only if released |",
           "| E6 OOD, all methods | 700147 tasks | PARKED | - | only if released |", ""]
    # cannot finish
    stuck = collections.defaultdict(set)
    for r in ROWS:
        if r["tier"] == "MAIN" and r["status"] in ("HELD-tier", "PARKED") and r["metric"] in ("flip", "both", "composed_conflict"):
            q = QI.get(r["file"]) or QI.get(r["file"][:-5] + "_ctw.json") or ("", "?", False)
            stuck[(r["family"], r["status"])].add(f"{r['model']} {r['dataset']} {r['split']} {r['method']}"
                                                  + (f" (from {r['train_dataset']})" if r["family"].startswith("12") else "") + f" [{q[1]}]")
    md += ["## MAIN cells that will NOT run today unless released (HELD-tier / PARKED)", ""]
    for (fam_, st_), cells in sorted(stuck.items()):
        md.append(f"- **{fam_}** ({st_}, {len(cells)}): " + "; ".join(sorted(cells)))
    jobs = sorted({x.split("[")[1].rstrip("]") for v in stuck.values() for x in v if "[" in x})
    md += ["", "Note: the position tasks (701719, `configs/gcf_reloc_v1166.txt`) still point at the residue twins; switch those "
           "lines to `pairs_look_all_*_clean.jsonl` before releasing (tasks read the manifest when they start). The ladder tasks use "
           "no Ground-CF twins."]
    md += ["", f"Release all of them with one command (≈ {len(jobs)} tasks, ladder ones ~15 min each on L40S; E6 ~1 h): "
           f"`scontrol release {' '.join(jobs)}`" if jobs else ""]
    md += ["", "## Cannot finish in time (would need new training or were never queued)", "",
           "- **Ground-CF prefix SFT L1** (CTM vs SFT for the prefix on Ground-CF): never trained (~1.5 h training + evals on 3 models). Not in time unless started now.",
           "- **LoRA SFT r16** (LoRA CTM vs SFT): never trained under the final protocol; mark N/A.",
           "- **E6 Qwen-4B word (ID)** is PARKED with the 1183 tier tasks: E6 is a MAIN family and this is its only missing ID cell (~1 GPU-h). Needs an explicit release.",
           "- **E6 OOD** (parked) and **E4 on the ladder datasets** (never queued): N/A for the paper unless released.",
           "- **Capacity beyond Qwen-4B / Ground-CF** (and LoRA r64 `<read>`): not trained; the capacity result stays a one-model diagnostic.",
           "- **MVG full vs crop paired test**: per-item pairing not banked in the summary files; needs a re-run of the MVG evaluator with item dumps (or report unpaired Wilson intervals).",
           "- **Corrected-twin reruns of the appendix-only Ground-CF evaluations** (probes `_cp4_probe2_*`, relocation battery, capacity, γ, template, training-question `_cp4`): not queued; they stay residue-twin diagnostics unless dropped.",
           "- **CTM+Stay OOD**: optional; not queued.", "",
           "## DROP (never enters final aggregates)", "",
           "- Gemma-4-E4B everything; APO arms; the pre-lr-fix LoRA runs; MVR-trained words (leaderboard-contaminated); ladder protocol-1 / "
           "non-lead-in files; `mvgV2_5tok_pt*` (never saw their own request word) and the old MVG PT rows; the old E5 (`explicit2` "
           "semicolon, free generation) as a composition result; the image-blind **.172** (29-item smoke); Ground-CF residue-twin "
           "numbers once the corrected reruns are promoted (they remain on disk as `*_restwin.json`); mixed-placement and word-first "
           "runs as method cells (appendix ablations only).", "",
           "## Readiness checklist (author's 15 items)", ""]
    def fam_state(prefix, split=None, meth=None, ds=None):
        c = collections.Counter()
        for r in ROWS:
            if (r["family"].startswith(prefix) and (split is None or r["split"] == split) and (meth is None or meth in r["method"])
                    and (ds is None or r["dataset"] in ds)):
                c[r["status"].split(" (")[0]] += 1
        tot = sum(c.values())
        return f"{c.get('DONE', 0)} DONE, {c.get('DONE*', 0)} provisional, {tot - c.get('DONE', 0) - c.get('DONE*', 0) - c.get('N/A', 0)} open, {c.get('N/A', 0)} N/A"
    checks = [("Clean final ID primary grid", ("1 primary", "ID")), ("Final OOD primary grid", ("1 primary", "OOD")),
              ("Ground-CF per-attribute ID/OOD (clean)", "4 "),
              ("Word CTM vs SFT, 4 datasets", ("6 ", None, "Word")), ("Prefix CTM vs SFT, 3 ladder datasets", ("6 ", None, "Prefix", ("VCF", "ColorSwap", "COCO-CF"))),
              ("Early word ID/OOD + relocated", "8 "),
              ("Word / prefix transfer", "12 "), ("E6, 3 models", "14 "), ("E5a/E5b redesigned, 3 models", "15 "), ("E4 forced choice", "13 "),
              ("MVG all models/methods, full + crop", "17 "), ("Confidence vs control (final clean)", "3 "), ("Discordant word/prefix (final clean)", "10"),
              ("Parameter counts / interface metadata", "24 "), ("AR vs CP + blind scorer full set", "20 ")]
    for i, (lab, pre) in enumerate(checks, 1):
        md.append(f"{i}. **{lab}** — {fam_state(*pre) if isinstance(pre, tuple) else fam_state(pre)}")
    md += ["", "Discordant analysis (family 10) lives in `results/discordant_analysis.md` (residue twins; regenerated automatically "
           "after promotion). Paired comparisons (family 26): `results/RESULTS_COMPARISONS.csv`.", ""]
    return md


def main():
    for fn in (fam_primary, fam_confidence, fam_attributes, fam_objectives, fam_order, fam_capacity, fam_transfer, fam_interfaces,
               fam_mvg, fam_design, fam_params, fam_compare):
        try:
            fn()
        except Exception as e:                           # one broken family must not hide the others
            add(family=f"ERROR {fn.__name__}", status="MISSING", notes=repr(e)[:300])
    add(family="10 discordant word/prefix", tier="SUPPORT", dataset="Ground-CF", split="ID", request="look", metric="analysis",
        status="DONE*" if os.path.exists("results/discordant_analysis.json") else "MISSING", file="results/discordant_analysis.json",
        notes="regenerated by the promotion watcher once corrected twins are canonical")
    with open("results/RESULTS_INVENTORY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(ROWS)
    json.dump({"generated": f"{NOW:%Y-%m-%d %H:%M}", "rows": ROWS}, open("results/RESULTS_INVENTORY.json", "w"), indent=0, default=str)
    keys = sorted({k for r in CMP for k in r}, key=lambda k: ["comparison", "subset", "model", "dataset", "split", "request", "metric",
                                                              "method_a", "method_b", "a_value", "b_value", "difference", "a_only",
                                                              "b_only", "p_value", "n", "twins", "status"].index(k) if k in
                  ["comparison", "subset", "model", "dataset", "split", "request", "metric", "method_a", "method_b", "a_value", "b_value",
                   "difference", "a_only", "b_only", "p_value", "n", "twins", "status"] else 99)
    with open("results/RESULTS_COMPARISONS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(CMP)
    md = overview()
    open("results/RESULTS_INVENTORY.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md[:60])); print(f"[saved] {len(ROWS)} rows, {len(CMP)} comparisons")


if __name__ == "__main__":
    main()
