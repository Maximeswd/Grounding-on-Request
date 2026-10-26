#!/usr/bin/env python
"""Protocol-2 ladder outputs: pass criteria (independent of performance) + the compact diagnostic table (v1.154, 2026-09-24).

PASS CRITERIA, checked on the SAVED item logs of every protocol-2 file of the given runs:
  P1 invariants  per example and request: the prompt is identical across the three versions once the caption is masked;
                 the four candidate texts are identical across versions AND requests; four distinct values; the conflict
                 holds in every version (A != C, B != C, A != D)
  P2 finite      every candidate has a finite score and a positive token count
  P3 decisions   each saved pick is the argmax of the saved scores, and the summary Acc / Flip / Stay recomputed from the
                 saved picks equal the summary file
  P4 same items  every compared file has the same examples with the same values and the same candidate texts
  P5 matched     the matched formats differ only in the control: word-in-prefix-format == prefix own format, and
                 prefix-in-word-slot == word own format, once the control token is masked; the control-removed prompt is
                 the prefix's own prompt without its control
  P6 harvest     (--harvest) the cp2 grid cells of these runs resolve to +p2 keys whose values equal the _p2 files
A prefix win or a lower score than protocol 1 is NOT a failure: four-way selection is a different, harder test.
TABLE: per condition, image and caption Acc / Flip / Stay and the distractor-pick rate; then the pre-registered pooled
matched comparison (STATUS v1.151) on protocol 2, with the sign-flip test.
Usage: python scripts/verify_p2_outputs.py --ds coco --model _gemma4e4b [--runs_dir runs] [--harvest] [--out results/p2_diagnostic_coco_e4b]
"""
import argparse, collections, gzip, json, math, os, random, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTL = re.compile(r"<(?:look|read|compare)_ctrl>|<pt_(?:look|read|compare)>")
CX = ("base", "imgswap", "capswap")
GOLD = {"look": ("A", "B", "A"), "read": ("C", "C", "D")}     # correct pick per version (base, imgswap, capswap)


def load_items(path):
    if not os.path.exists(path):
        return None
    by = collections.defaultdict(dict)
    for l in gzip.open(path, "rt"):
        a = json.loads(l)
        if "_meta" not in a:
            by[(a["item"], a["request"])][a["ctx"]] = a
    return by


def outcomes(by, rq):
    acc = flip = stay = n = 0
    per = {}
    for (i, r), c in by.items():
        if r != rq or any(x not in c for x in CX):
            continue
        p = [c[x]["pick"] for x in CX]; g = GOLD[rq]
        a = p[0] == g[0]; f = a and (p[1] if rq == "look" else p[2]) == (g[1] if rq == "look" else g[2])
        s = a and (p[2] if rq == "look" else p[1]) == (g[2] if rq == "look" else g[1])
        per[i] = (int(a), int(f), int(s)); n += 1; acc += a; flip += f; stay += s
    return per, (round(acc / n, 3), round(flip / n, 3), round(stay / n, 3)) if n else (None,) * 3


def check_file(by, summ):
    bad = collections.Counter()
    for (i, rq), c in by.items():
        if any(x not in c for x in CX):
            bad["missing_version"] += 1; continue
        cap = {x: c[x]["caption"] for x in CX}
        masked = {x: c[x]["prompt"].replace(cap[x], "<CAP>") for x in CX}
        if not ("<CAP>" in masked["base"] and len(set(masked.values())) == 1):
            bad["prompt_not_fixed"] += 1
        texts = [tuple(sorted((k, v["text"]) for k, v in c[x]["candidates"].items())) for x in CX]
        if len(set(texts)) != 1 or len(texts[0]) != 4:
            bad["candidates_not_fixed"] += 1
        V = c["base"]["values"]
        if len(set(V.values())) != 4 or V["A"] == V["C"] or V["B"] == V["C"] or V["A"] == V["D"]:
            bad["conflict_or_distinct"] += 1
        for x in CX:
            cd = c[x]["candidates"]
            if not all(math.isfinite(v["sum"]) and v["ntok"] > 0 for v in cd.values()):
                bad["nonfinite_or_empty"] += 1
            best_turn = max(cd, key=lambda k: cd[k]["sum"]); best_ans = max(cd, key=lambda k: cd[k].get("answer_sum", cd[k]["sum"]))
            if c[x]["pick"] not in (best_turn, best_ans):
                bad["pick_not_argmax"] += 1
    # the same candidate texts across the two requests of one example
    for i in {k[0] for k in by}:
        if (i, "look") in by and (i, "read") in by:
            tl = sorted((k, v["text"]) for k, v in by[(i, "look")]["base"]["candidates"].items())
            tr = sorted((k, v["text"]) for k, v in by[(i, "read")]["base"]["candidates"].items())
            if tl != tr:
                bad["candidates_differ_across_requests"] += 1
    for rq in ("look", "read"):
        _, (a, f, s) = outcomes(by, rq)
        if summ and (abs(a - summ[f"{rq}_acc"]) > 1e-3 or abs(f - summ[f"{rq}_flip"]) > 1e-3 or abs(s - summ[f"{rq}_stay_c"]) > 1e-3):
            bad[f"summary_mismatch_{rq}"] += 1
    return bad


def signflip_p(d, draws=200000, seed=0):
    d = [x for x in d if x != 0]
    if not d:
        return 1.0
    obs = abs(sum(d)); rng = random.Random(seed)
    hit = sum(abs(sum(x if rng.random() < .5 else -x for x in d)) >= obs - 1e-12 for _ in range(draws))
    return round((hit + 1) / (draws + 1), 5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="coco"); ap.add_argument("--model", default="_gemma4e4b")
    ap.add_argument("--runs_dir", default=os.path.join(ROOT, "runs")); ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--out", default=None); a = ap.parse_args()
    ds, sfx, R = a.ds, a.model, a.runs_dir
    W, P, F = f"{ds}_neo_ctrl_1tok_ctm_cp_seed0{sfx}", f"{ds}_pt_na_1tok_ctm_cp_seed0{sfx}", f"{ds}_base_frozen_0tok_none_seed0{sfx}"
    LL, LR = f"{ds}_lora_r16_lr2e4_ad_ctm_cp_look_seed0{sfx}", f"{ds}_lora_r16_lr2e4_ad_ctm_cp_read_seed0{sfx}"
    COND = []   # (label, run, file stem, format note)
    for t, tl in (("", "training"), ("_heldout", "held-out")):
        COND += [(f"word, own format (after caption), {tl}", W, f"flip_ll_p2{t}"),
                 (f"word, before caption, {tl}", W, f"flip_ll_reqfirst_p2{t}"),
                 (f"word in prefix's slot + wording, {tl}", W, f"flip_ll_ptfmt_p2{t}"),
                 (f"prefix, own format (before caption), {tl}", P, f"flip_ll_p2{t}"),
                 (f"prefix in word's slot + wording, {tl}", P, f"flip_ll_ptslot_p2{t}"),
                 (f"prefix's instruction, no prefix, {tl}", P, f"flip_ll_noprefix_p2{t}"),
                 (f"LoRA image adapter, {tl}", LL, f"flip_ll_p2{t}"), (f"LoRA caption adapter, {tl}", LR, f"flip_ll_p2{t}")]
    COND += [("English instruction (variant 5)", F, "flip_ll_iv5_p2"), ("no request", F, "flip_ll_noop_p2")]
    d3 = f"{ds}3_test_id"; loaded = {}
    report = {"pass": {}, "rows": [], "files": {}}
    for lab, run, stem in COND:
        base = os.path.join(R, run, "eval", d3, stem + "_step_800")
        by = load_items(base + "_items.jsonl.gz")
        summ = json.load(open(base + ".json")) if os.path.exists(base + ".json") else None
        if by is None:
            report["rows"].append({"condition": lab, "status": "not evaluated yet"}); continue
        loaded[lab] = by; report["files"][lab] = os.path.relpath(base + ".json", ROOT)
        bad = check_file(by, summ)
        if summ and (summ.get("protocol") != 2 or any(v != summ["audit_p2"]["n"] for k, v in summ["audit_p2"].items() if k != "n")):
            bad["in_run_audit"] += 1
        row = {"condition": lab, "status": "ok" if not bad else "FAIL", "problems": dict(bad)}
        for rq in ("look", "read"):
            _, (ac, fl, st) = outcomes(by, rq); row[rq] = {"acc": ac, "flip": fl, "stay": st}
        row["distractor"] = (summ or {}).get("p2_distractor_rate"); report["rows"].append(row)
    # P4 same examples + candidates across files
    ref = None; p4 = collections.Counter()
    for lab, by in loaded.items():
        sig = {i: (tuple(sorted(c["base"]["values"].items())), tuple(sorted((k, v["text"]) for k, v in c["base"]["candidates"].items())))
               for (i, rq), c in by.items() if rq == "look" and "base" in c}
        if ref is None:
            ref = sig
        elif sig != ref:
            p4[lab] += 1
    # P5 matched formats differ only in the control
    def prompts(lab):
        by = loaded.get(lab); return {k: c["base"]["prompt"] for k, c in by.items()} if by else None
    p5 = collections.Counter()
    for t in ("training", "held-out"):
        for x, y in ((f"word in prefix's slot + wording, {t}", f"prefix, own format (before caption), {t}"),
                     (f"prefix in word's slot + wording, {t}", f"word, own format (after caption), {t}")):
            px, py = prompts(x), prompts(y)
            if px and py:
                p5[f"{x} vs {y}"] = sum(CTL.sub("<CTL>", px[k]) != CTL.sub("<CTL>", py[k]) for k in px if k in py)
        pn, pp = prompts(f"prefix's instruction, no prefix, {t}"), prompts(f"prefix, own format (before caption), {t}")
        if pn and pp:
            p5[f"no-prefix vs prefix, {t}"] = sum(CTL.sub("", pp[k]).strip() != pn[k].strip() for k in pn if k in pp)
    report["pass"] = {"P1-P3 per file": all(r.get("status") != "FAIL" for r in report["rows"]),
                      "P4 same examples and candidates": not p4, "P5 matched formats differ only in the control": not any(p5.values()),
                      "details": {"P4": dict(p4), "P5": dict(p5)}}
    # P6 harvest: cp2 cells of these runs are +p2 and equal the files
    if a.harvest:
        import sys; sys.path.insert(0, os.path.join(ROOT, "paper")); os.environ["SC_RECIPE_SET"] = "cp2"
        import cpgrid
        g = cpgrid.load_grid(); mk = {"": "gemma3", "_qwen35": "qwen4b", "_qwen2b": "qwen2b", "_gemma4e4b": "e4b"}[sfx]; p6 = {}
        for meth, L, lab in (("word+ctm", "1", "word, own format (after caption), held-out"), ("pt+ctm", "1", "prefix, own format (before caption), held-out"),
                             ("noop", "None", "no request"), ("instr", "None", "English instruction (variant 5)")):
            key = f"{ds}|{mk}|{meth}|{L}"; rk = g.remap(key); cell = g.get(key)
            row = next((r for r in report["rows"] if r["condition"] == lab and "look" in r), None)
            p6[key] = {"resolves_to": rk, "is_p2": rk.split("|")[2].endswith("+p2"), "matches_file": bool(cell and row and abs((cell.get("G.look") or -9) - row["look"]["flip"]) < 1e-3)}
        report["pass"]["P6 harvest selects +p2 only"] = all(v["is_p2"] and v["matches_file"] for v in p6.values())
        report["pass"]["details"]["P6"] = p6
    # pooled matched headline (pre-registered v1.151) on protocol 2
    pooled = {}
    need = ["word, own format (after caption), held-out", "word in prefix's slot + wording, held-out",
            "prefix in word's slot + wording, held-out", "prefix, own format (before caption), held-out"]
    if all(n in loaded for n in need):
        for rq in ("look", "read"):
            o = {n: outcomes(loaded[n], rq)[0] for n in need}
            ks = sorted(set.intersection(*[set(v) for v in o.values()]))
            for mi, m in ((1, "flip"), (2, "stay")):
                wv = [(o[need[0]][k][mi] + o[need[1]][k][mi]) / 2 for k in ks]; pv = [(o[need[2]][k][mi] + o[need[3]][k][mi]) / 2 for k in ks]
                pooled[f"{rq}_{m}"] = {"word": round(sum(wv) / len(ks), 3), "prefix": round(sum(pv) / len(ks), 3),
                                       "diff": round((sum(wv) - sum(pv)) / len(ks), 3), "p": signflip_p([x - y for x, y in zip(wv, pv)]), "n": len(ks)}
    report["pooled_matched_p2"] = pooled
    out = a.out or os.path.join("results", f"p2_diagnostic_{ds}{sfx}")
    json.dump(report, open(os.path.join(ROOT, out + ".json"), "w"), indent=1)
    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    md = [f"# Protocol-2 diagnostic — {ds} {sfx or '_gemma3'} (test_id, conflict items)", "",
          "Generated by `scripts/verify_p2_outputs.py`. Pass criteria are independent of performance.", "",
          "| check | result |", "|---|---|"] + [f"| {k} | {'pass' if v else '**FAIL**'} |" for k, v in report["pass"].items() if k != "details"]
    md += ["", "| condition | image Acc | image Flip | image Stay | caption Acc | caption Flip | caption Stay | distractor picks (img / cap) | checks |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        if "look" not in r:
            md.append(f"| {r['condition']} | – | – | – | – | – | – | – | {r['status']} |"); continue
        dr = r.get("distractor") or {}
        md.append(f"| {r['condition']} | {f3(r['look']['acc'])} | {f3(r['look']['flip'])} | {f3(r['look']['stay'])} | {f3(r['read']['acc'])} | "
                  f"{f3(r['read']['flip'])} | {f3(r['read']['stay'])} | {f3(dr.get('look'))} / {f3(dr.get('read'))} | {r['status']}{(' ' + str(r['problems'])) if r['problems'] else ''} |")
    if pooled:
        md += ["", "Pre-registered pooled matched comparison (STATUS v1.151), protocol 2, held-out texts:", "",
               "| request | metric | word (pooled) | prefix (pooled) | word − prefix | sign-flip p | n |", "|---|---|---|---|---|---|---|"]
        for k, v in pooled.items():
            rq, m = k.split("_")
            md.append(f"| {'image' if rq == 'look' else 'caption'} | {m} | {f3(v['word'])} | {f3(v['prefix'])} | {v['diff']:+.3f} | {v['p']} | {v['n']} |")
    open(os.path.join(ROOT, out + ".md"), "w").write("\n".join(md) + "\n")
    print("[saved]", out + ".{md,json}", {k: v for k, v in report["pass"].items() if k != "details"})


if __name__ == "__main__":
    main()
