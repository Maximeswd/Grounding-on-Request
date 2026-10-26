#!/usr/bin/env python
"""Conflict detection from TWO SEPARATE single-request prompts on Ground-CF (author, 2026-09-23, STATUS v1.131).

Reads the open-generation files written with --agree_ctx 1 (gen_step_<step><tag>.json) and computes, per interface and
model, detect / attrib / false / balanced for the forced choice and for the parsed generation
(scripts.eval_clevr3_generate.detect_two_prompts). Word, prefix, English and no request answer both requests from one
run; a LoRA adapter answers only its own request, so the <look> adapter's file is paired with the <read> adapter's file
of the same model and rank. Output: results/detect_two_prompts.json (one row per interface x model).
Usage: python scripts/detect_two_prompts.py [--tag _gen_ho] [--runs_dir runs]
"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detect_two_prompts(recs_look, recs_read):
    """v1.131: conflict detection from TWO SEPARATE single-request prompts. Per item, the <look> and <read> answers in
    the original (conflict) context and in the agree context (caption edited to the image value).
      detect = answers differ on the conflict item; attrib = both answers are their own source's value;
      false  = answers differ on the agree item; balanced = (detect + 1 - false) / 2 (chance .5 for a constant policy).
    Computed for the forced choice (ll_*) and for the parsed generation (gen_*). The two request lists may come from
    one run (word, prefix, English) or from two runs (one LoRA adapter per request)."""
    ix = lambda rs: {(r["id"], r["ctx"]): r for r in rs}
    L, Rd = ix(recs_look), ix(recs_read)
    out = {}
    for kind, val in (("ll", lambda r: r["pick_sum"]), ("gen", lambda r: r["gen_value"] if r["gen_label"] == "cand" else None)):
        det, att, fa = [], [], []
        for (iid, ctx), rl in L.items():
            if ctx != "base" or (iid, "base") not in Rd:
                continue
            rr = Rd[(iid, "base")]
            a, b = val(rl), val(rr)
            det.append(int(a is not None and b is not None and a != b))
            att.append(int(a == rl["gold"] and b == rr["gold"]))
            if (iid, "agree") in L and (iid, "agree") in Rd:
                a2, b2 = val(L[(iid, "agree")]), val(Rd[(iid, "agree")])
                fa.append(int(a2 is not None and b2 is not None and a2 != b2))
        m = lambda v: round(sum(v) / len(v), 4) if v else None
        out.update({f"{kind}_detect": m(det), f"{kind}_attrib": m(att), f"{kind}_false": m(fa),
                    f"{kind}_balanced": (round((m(det) + 1 - m(fa)) / 2, 4) if det and fa else None),
                    f"{kind}_n": len(det), f"{kind}_n_agree": len(fa)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs"); ap.add_argument("--split", default="test_id")
    ap.add_argument("--step", type=int, default=800); ap.add_argument("--tag", default="_gen_ho")
    ap.add_argument("--out", default="results/detect_two_prompts.json")
    a = ap.parse_args()
    rows = []
    pat = os.path.join(a.runs_dir, "clevr3_2s_*", "eval", f"clevr3_{a.split}", f"gen_step_{a.step}{a.tag}*.json")
    files = sorted(glob.glob(pat))
    lora = {}
    for fp in files:
        d = json.load(open(fp))
        if not d.get("agree_ctx"):
            continue
        run = d["run"]; rest = os.path.basename(fp)[len(f"gen_step_{a.step}{a.tag}"):-5]
        m = re.match(r"(clevr3_2s_lora_r\d+_\w+?)_(look|read)_(seed\d+.*)$", run)
        if m:
            lora.setdefault((m.group(1), m.group(3), rest), {})[m.group(2)] = d
            continue
        if d.get("two_prompt_detection"):
            rows.append({"run": run, "variant": rest, "method": d.get("method"), "mode": d.get("mode"),
                         "file": fp, **d["two_prompt_detection"]})
    for (stem, tail, rest), pair in sorted(lora.items()):
        if {"look", "read"} <= set(pair):
            L = [r for r in pair["look"]["records"] if r["source"] == "look"]
            R = [r for r in pair["read"]["records"] if r["source"] == "read"]
            rows.append({"run": f"{stem}_{{look,read}}_{tail}", "variant": rest, "method": "lora", "mode": "adapter pair",
                         **detect_two_prompts(L, R)})
        else:
            rows.append({"run": f"{stem}_*_{tail}", "variant": rest, "method": "lora", "missing": sorted({"look", "read"} - set(pair))})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"tag": a.tag, "split": a.split, "rows": rows,
               "note": "two separate single-request prompts per item; detect = answers differ on the conflict item, "
                       "false = answers differ on the agree item (caption edited to the image value); chance .5"},
              open(a.out, "w"), indent=1)
    for r in rows:
        print(f'{r["run"][:60]:60s} {r.get("variant",""):22s} ll_bal {r.get("ll_balanced")} gen_bal {r.get("gen_balanced")} n {r.get("ll_n")}')
    print("[saved]", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
