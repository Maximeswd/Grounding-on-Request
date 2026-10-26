#!/usr/bin/env python
"""Write docs/EVALUATIONS.md: every Ground-CF evaluation of the new protocol (v1.124+), with the EXACT inputs each
interface sees (rendered by the evaluators' own prompt functions), the scoring rule, the per-item fields that are logged
and one example record per evaluation (taken from a CPU dry run with a stub model: the fields are real, the NUMBERS in
the examples are placeholders). Rerun after any prompt or logging change.

Usage: python scripts/make_eval_catalog.py [--examples_from <dry-run dir>]   (copies trimmed example records to
       docs/eval_examples/; without the flag the copies already there are used)
"""
import argparse, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.update({"SC_C3_SOURCES": "look,read", "SC_CAPTION_STYLE": "fact"})
from scsrc import clevr3 as C                                            # noqa: E402
from scsrc.tokens import trigger                                          # noqa: E402
from scripts.eval_clevr3 import cand                                      # noqa: E402

EX_DIR = os.path.join(ROOT, "docs", "eval_examples")
IFACES = [  # label, method, mode, neo_template
    ("word (L=1)", "neo", "srconly", "hewitt"), ("prefix (L=1)", "pt", "srconly", "bare"),
    ("LoRA adapter (r=16, one per request)", "lora", "noop", "bare"), ("English", "neo", "instructed", "bare"),
    ("no request", "neo", "noop", "bare")]


def show(content):
    return " ".join("[IMAGE]" if c.get("type") == "image" else c["text"] for c in content)


def render(item, src, method, mode, tpl, qs="heldout", seed=0):
    os.environ["SC_NEO_TEMPLATE"] = tpl
    return show(C.clevr3_user_content(item, src, 1, mode, seed=seed, question_style=qs, method=method))


def block(txt):
    return "```text\n" + txt + "\n```"


def ex(name):
    fp = os.path.join(EX_DIR, name + ".json")
    if not os.path.exists(fp):
        return "_(example record not yet captured)_"
    return "```json\n" + json.dumps(json.load(open(fp)), indent=1, ensure_ascii=False)[:2600] + "\n```"


def first_record(fp):
    d = json.load(open(fp)) if fp.endswith(".json") else [json.loads(l) for l in open(fp)]
    if isinstance(d, list):
        return d[0]
    for k in ("records", "items", "picks"):
        v = d.get(k)
        if isinstance(v, list) and v:
            return v[0]
        if isinstance(v, dict):
            for kk in ("look", "read"):
                if v.get(kk):
                    return v[kk][0]
    return {k: v for k, v in d.items() if k != "note"}


def capture(src_dir):
    """Trimmed example records from a dry-run directory (scripts/dryrun_eval.py outputs)."""
    os.makedirs(EX_DIR, exist_ok=True)
    W, P, L = (os.path.join(src_dir, r, "eval", "clevr3_test_id") for r in
               ("clevr3_2s_1tok_ctm_cp_fc_seed0", "clevr3_2s_pt_1tok_ctm_cp_fc_seed0", "clevr3_2s_lora_r16_ctm_cp_fc_look_seed0"))
    want = {"headline_flip": f"{W}/flip_step_800_cp4_ho*.json", "gen": f"{W}/gen_step_800_gen_ho.json",
            "gen_mc": f"{W}/gen_step_800_gen_ho_mc.json", "compose": f"{W}/compose_detect_step_800_cp_fc_x2.json",
            "compose_lora_na": f"{L}/compose_detect_step_800_cp_fc_x2.json", "switch": f"{W}/switch_step_800_fc_items.jsonl",
            "switch_lora_na": f"{L}/switch_step_800_fc.json", "abstain": f"{W}/abstain*_step_800_cpneu*.json",
            "meta": f"{W}/meta_words_step_800_meta_fc2.json", "long_probe": f"{W}/flip_step_800_cp4_probe2_long_neutral*.json"}
    for k, pat in want.items():
        fs = sorted(glob.glob(pat))
        if not fs:
            print("[capture] missing", pat); continue
        rec = first_record(fs[0])
        if k == "meta":   # one record per family
            d = json.load(open(fs[0])); seen = {}
            for r in d["records"]:
                seen.setdefault(r["family"], r)
            rec = {"summary_keys": sorted(next(iter(d["summary"].values())).keys()), "one_record_per_family": seen}
        if k == "gen":
            d = json.load(open(fs[0]))
            rec = {"record": rec, "summary_keys_per_request": sorted(d["summary"]["look"].keys()),
                   "two_prompt_detection": d.get("two_prompt_detection")}
        json.dump(rec, open(os.path.join(EX_DIR, k + ".json"), "w"), indent=1, ensure_ascii=False)
        print("[capture]", k, "<-", os.path.relpath(fs[0], src_dir))


def lines_in(manifest, pat):
    try:
        return sum(1 for l in open(manifest) if pat in l)
    except FileNotFoundError:
        return 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--examples_from", default=None); a = ap.parse_args()
    if a.examples_from:
        capture(a.examples_from)
    rows = [json.loads(l) for l in open("data/clevr3/test_id.jsonl")]
    C.set_c3_vocab(rows)
    by_id = {r["id"]: r for r in rows}
    pr = next(json.loads(l) for l in open("data/clevr3/pairs_look_all_id.jsonl")
              if json.loads(l).get("skill") == "color" and json.loads(l)["id"] in by_id
              and str(by_id[json.loads(l)["id"]]["channels"]["look"]) != str(by_id[json.loads(l)["id"]]["channels"]["read"]))
    it = by_id[pr["id"]]
    A, B = str(it["channels"]["look"]), str(it["channels"]["read"])
    w = C.cp_text_edit_value(it, "read")[0]; tv = str(pr["gold_b"])
    cap_edit = C._edited_item(it, "read", w); agree = C._edited_item(it, "read", A)
    sp = next(r for r in rows if r["skill"] == "spatial" and str(r["channels"]["look"]) != str(r["channels"]["read"]))

    o = ["# Ground-CF evaluations — inputs, scoring, and what is logged (protocol v1.141)", "",
         "Generated by `scripts/make_eval_catalog.py` from the evaluators' own prompt functions. `[IMAGE]` marks the image.",
         "Example records come from a CPU dry run (`scripts/dryrun_eval.py`) with a stub model: **the fields are the real",
         "ones; the numbers in the examples are placeholders.** Interfaces: word L=1 (CTM = Flip-only, and CTM+Stay),",
         "prefix L=1 (CTM), LoRA r=16 as one adapter per request (CTM), English instruction, no request; four models",
         "(Gemma-3-4B, Qwen-3.5-4B, Qwen-3.5-2B, Gemma-4-E4B). Every prompt below differs from every training prompt",
         "(held-out question formulations, held-out carriers, unseen probe sentences).", "",
         "**Order and template design (v1.133).** Each item gets its block order and held-out forms from its rank r within",
         "its skill: caption before image when r is even (image first otherwise), held-out carrier (r//2)%2, held-out",
         "question (r//4)%2. Every skill x request therefore sees both orders and both held-out forms equally (measured",
         "0.49-0.51). For a given item, all methods and requests share the same block order, question-template assignment",
         "and held-out carrier index; only the request representation itself differs. Records carry",
         "`cap_first`. (Training: an independent 50/50 order coin per row, all six carriers for both requests.)", "",
         "## Running example", "",
         f"Item `{it['id']}` (skill color). Image value **{A}**, caption value **{B}** → a conflict item. Caption edit value",
         f"**{w}** (a value no shown source states). Image twin: the same scene re-rendered with value **{tv}**.",
         f"Candidate set for every context and both requests: {{{', '.join(dict.fromkeys([A, B, w, tv]))}}} (chance = 1/|Y|).", "",
         "Every forced choice scores the continuation below with teacher forcing (summed log-probability); the arg-max is",
         "the pick. The shared prefix is identical across candidates, so score differences are differences in log p(answer).",
         block(cand("<v>")), "",
         "## Summary of evaluations", "",
         "| # | evaluation | question it answers | interfaces | scoring | file |",
         "|---|---|---|---|---|---|",
         "| E1 | Headline Flip/Stay, held-out templates | does the answer track an edit of the requested source and ignore an edit of the other? | all 5 | forced choice | `flip_step_800_cp4_ho.json` |",
         "| E2 | Open generation + exact match | does free text say what the forced choice picks? | all 5 | greedy text, parsed after `final answer:`; LL of every candidate | `gen_step_800_gen_ho*.json` |",
         "| E3 | Multiple choice | lettered options, easy to check | all 5 | LL of every letter; generated letter parsed | `gen_step_800_gen_ho_mc*.json` |",
         "| E4 | Conflict detection, two separate prompts | do the two single-request answers differ exactly on conflict items? | all 5 (LoRA: look adapter + read adapter) | from E2 (LL and parsed text) | `results/detect_two_prompts.json` |",
         "| E5 | Conflict detection, one prompt (composition) | can two requests be composed zero-shot in one prompt? | word, prefix (inline, new regime), English; LoRA and no request N/A | LL of all 9 answer pairs; BothCorrect primary, conflict derived; generation | `compose_detect_step_800_cp_fc_x2*.json` |",
         "| E6 | Within-prompt switching | can two questions in one prompt carry different requests? | word, prefix, English, no request; LoRA N/A | LL of all 9 joint answers; generation | `switch_step_800_fc*.json` + `_items.jsonl` |",
         "| E7 | Zero-shot abstention (neutral policy) | does the request abstain when its source lacks the answer, without leaking? | word, prefix, LoRA, English (no request: N/A, the policy names a request) | LL (length-normalised) over {value, distractor, unknown}; generation | `abstain_twins_step_800_cpneu*.json` |",
         "| E8 | Metalinguistic access | can the model talk about `<look>` / `<read>`? | word, prefix (+ untrained-token controls), English; LoRA/no request: self-report only | LL of every option (anchored `final answer: <c>` primary); generation | `meta_words_step_800_meta_fc2*.json` |",
         "| E9 | Interface probes: format, position, long questions | does the request survive new sentence forms? | word, prefix, English (+LoRA/no request for fmt and long) | forced choice as E1 | `flip_step_800_cp4_probe_<name>*.json` |",
         "", "All files are under `runs/<run>/eval/clevr3_<split>/`. **Every evaluation runs on the ID and the OOD split** (v1.141).",
         "Chance and blind bounds are stated per evaluation below.", "",
         "**What is saved (v1.141).** For every scored item: the exact prompt text, the block order (`cap_first`), the",
         "request, the candidates with their summed log-likelihood, per-token mean and token count, the source values and",
         "gold; for generations the raw text verbatim before any parser, the parser label and value. E1/probes write",
         "`*_items.jsonl.gz` next to the summary file; the ladder evaluator and MVG do the same. Headline metrics rely on",
         "candidate log-likelihood; deterministic parsing of generations is the validation layer; no LLM judge is used.",
         "**Exact matching** (generation read after `final answer:`): Ground-CF E2/E3 (+E4), MVG (`_rbg`), ladder",
         "(`genconsistency_step_800_ho.json`, same held-out templates as the headline ladder evaluation).",
         "**Known limitation:** Ground-CF's count caption (\"There are 4 shapes in the image.\") and the standard count",
         "questions name the image; the probes and long questions use neutral wording (\"how many shapes there are\").", ""]

    # ---- E1
    o += ["## E1 — Headline Flip/Stay on held-out templates (`scripts/eval_clevr3_flip.py`, `SC_QSTYLE=heldout`)", "",
          "Three contexts per item and request. `<look>`: base, **flip** = image twin (image says " + tv + "), **stay** = caption",
          "edit (caption says " + w + "). `<read>`: base, **flip** = caption edit, **stay** = image twin.",
          "Accuracy = pick(base) = gold; Flip = Accuracy and pick(flip) = the edited value; Stay (conjunctive) = Accuracy and",
          "pick(stay) = pick(base). Flip and Stay are always reported separately; JOINT = min(Flip, Stay).", "",
          "**Base-context input, `<look>` request, per interface** (the `<read>` request swaps the word / prefix / adapter / phrase):", ""]
    for lab, meth, mode, tpl in IFACES:
        o += [f"*{lab}*", block(render(it, "look", meth, mode, tpl))]
    o += ["", "*word, `<look>`, stay context (caption edited)*", block(render(cap_edit, "look", "neo", "srconly", "hewitt")),
          "*word, `<look>`, flip context*: identical text; the image is the twin render.", "",
          "Logged per item (`--dump_picks 1`, one record per request): `id, skill, A` (image value) `, B` (caption value) `, w`",
          "(caption-edit value) `, twin` (twin value) `, Y` (candidates), `p0 / p_capedit / p_twin` (picks), `scores.{base,capedit,twin}.{candidate}`",
          "(summed log-prob of every candidate in every context), `values.{ctx}.{image,caption}` (what each source states in",
          "that context), `gold.{ctx}`, `lp_gold.{ctx}` (log-prob of the correct answer), `prompt_base` (first 50 items).", "",
          "Summary: `look_acc, look_flip, look_stay_c, read_acc, read_flip, read_stay_c`, per skill and macro over skills.", "",
          ex("headline_flip"), ""]
    # ---- E2
    os.environ["SC_NEO_TEMPLATE"] = "hewitt"
    from scripts.eval_clevr3_generate import FORMAT_HINT, FORMAT_HINT_MC, MC_INSTR, mc_options
    o += ["## E2 — Open generation with deterministic exact matching (`scripts/eval_clevr3_generate.py`)", "",
          "Same items, prompts and contexts as E1 (160 twin pairs, 40 per skill), plus an **agree** context (caption edited to",
          "the image value) for E4. In every context both are recorded: the forced choice over the candidates and a greedy",
          "generation (new-word rows forbidden, stop shortly after `final answer:`, max 192 new tokens). The generation is kept",
          "verbatim and parsed deterministically: the candidate after the last `final answer:`; without the anchor, a single",
          "candidate named in the reply; several candidates → `ambiguous`; refusal markers → `abstain`; else `other`.",
          "Trained interfaces learned the `final answer:` format. English and no request never did, so their lines add the",
          "format hint below to BOTH the forced choice and the generation (`--format_hint 1`, tag `_hint`):",
          block(FORMAT_HINT.strip()), "",
          "Logged per record (one per item × request × context): `id, skill, source, ctx` (base/flip/stay/agree) `, gold,",
          "values.{image,caption}, cands, cand_scores.{v}.{sum,mean,ntok}` (log-lik of every candidate), `lp_gold, pick_sum,",
          "pick_mean, text` (full generation), `gen_label, gen_value, gen_ntok, prompt, question_style, format_hint`.",
          "Summary per request: Accuracy/Flip/Stay under the forced choice and under generation, their item-level agreement",
          "(Cohen's kappa), parse-label histogram, P(genFlip | llFlip).", "", ex("gen"), ""]
    # ---- E3
    opts = mc_options(it["id"], tuple(dict.fromkeys([A, B, w, tv])))
    mc_txt = render(it, "look", "neo", "srconly", "hewitt") + MC_INSTR.format(opts="\n".join(f"{k}. {v}" for k, v in opts.items()))
    o += ["## E3 — Multiple choice (`eval_clevr3_generate.py --mc 1`)", "",
          "The item's candidate set as lettered options, in one per-item order that is fixed across contexts and requests",
          "(so a Flip is a change of letter). The forced choice scores `" + cand("<letter>") + "` for every letter;",
          "each letter maps back to its value, so Accuracy/Flip/Stay are computed exactly as in E1/E2. The generated letter",
          "is parsed after `final answer:`, else a lone letter (`B`, `(B)`, `B.`; the article \"A red …\" is not a letter), else",
          "the value named. 100 pairs (25 per skill). English/no request add: `" + FORMAT_HINT_MC.strip() + "`", "",
          "*word, `<look>`, base context*", block(mc_txt), "",
          "Logged: as E2, plus `mc_options` (letter → value), `cand_scores.{v}.letter`, `parse_route` (anchor/letter/value).",
          "Chance = 1/|options| (1/3 or 1/4 per item; reported per item).", "", ex("gen_mc"), ""]
    # ---- E4
    o += ["## E4 — Conflict detection from two separate prompts (`scripts/detect_two_prompts.py`)", "",
          "No new prompt: the `<look>` answer and the `<read>` answer to the SAME item, each asked in its own single-request",
          "prompt (E2 records, base and agree contexts). detect = the two answers differ on the conflict item; attrib = each",
          "answer is its own source's value; false = they differ on the agree item; balanced = (detect + 1 − false)/2, chance .5",
          "for any constant policy. Computed for the forced choice (`ll_*`) and for the parsed generations (`gen_*`). LoRA: the",
          "`<look>` adapter's file is paired with the `<read>` adapter's file of the same model.", "",
          "*agree context, word, `<read>`*", block(render(agree, "read", "neo", "srconly", "hewitt")), ""]
    # ---- E5
    def comp(c1, c2):
        s2, cf = C.eval_design(it, 0, "heldout"); q = C.heldout_question(it, s2)
        head = f"{C._blocks(it)['read']} [IMAGE]" if cf else f"[IMAGE] {C._blocks(it)['read']}"
        return (f"{head}\n{q}\nAnswer the question twice. First, answer according to {c1}. Second, answer according to {c2}. "
                "Return the two answers in that order, separated by a semicolon.")
    o += ["## E5 — Conflict detection in one prompt: zero-shot composition (`scripts/eval_compose_detect.py --template explicit2`)", "",
          "Both requests in one prompt with an explicit instruction (v1.141, reviewer); none of these runs was trained on",
          "composition, so this is zero-shot. All 9 answer pairs \"v1; v2\" over {image value, caption value, a third value}",
          "are scored (chance 1/9). **Primary: BothCorrect** (each answer is its own request's source). Conflict is DERIVED",
          "from the two answers (detect = they differ on the conflict item; false alarm = they differ on the agree twin, caption",
          "set to the image value); no verdict is asked. Both orders, 150 items, a parsed greedy reply as validation.", "",
          "*word*", block(comp(trigger("src_look", 1), trigger("src_read", 1))),
          "*prefix (its soft tokens placed inline: a new insertion regime relative to training)*", block(comp("<pt_look>", "<pt_read>")),
          "*English*", block(comp("the image", "the caption")),
          "*LoRA*: not expressible (one adapter per request is global model state) and *no request*: nothing to compose —",
          "both files hold explicit N/A records, never a zero.", "",
          "Logged per item × order × {conflict, agree}: the full prompt, `cap_first`, `gold`, `values.{image,caption}`,",
          "`candidate_scores` (all 9 pairs), `prediction`, `raw_generation`, `parse_label`, `parsed`. Summary per order:",
          "both_correct, q1, q2, detect, false, balanced (chance .5), gen_both_correct, gen_balanced, gen_parsed.", "",
          ex("compose"), "", ex("compose_lora_na"), ""]
    # ---- E6
    q1 = C.heldout_question(it, 0); q2 = C.heldout_question(sp, 1)
    sw = lambda c1, c2: ("Two questions follow, each with its own image and caption. Answer each question according to the request "
                         "given in that question. Return exactly the two answers, in order, separated by a semicolon. "
                         f"Question 1.\n{C._blocks(it)['read']} [IMAGE] {q1}" + (f" Answer according to {c1}." if c1 else "") +
                         f" Question 2.\n{C._blocks(sp)['read']} [IMAGE] {q2}" + (f" Answer according to {c2}." if c2 else ""))
    o += ["## E6 — Within-prompt switching (`scripts/eval_clevr3_switch.py`)", "",
          "Two independent conflict items (different skills) in one user turn, each question with its own request; all four",
          "assignments (look,look) (read,read) (look,read) (read,look). The mixed assignments are the scoping test. All 9",
          "joint answers \"a1; a2\" over cand(Q1) × cand(Q2) = {image value, caption value, third value} are scored",
          "(chance 1/9); a generation is kept and parsed as a pair. 150 pairs.", "",
          "*word, (look, read)*", block(sw(trigger("src_look", 1), trigger("src_read", 1))),
          "*prefix*: the soft tokens placed locally in the same slot. *English*: \"the image\" / \"the caption\". *no request*: no",
          "phrase. *LoRA*: N/A record (one adapter per forward pass).", "",
          "Logged per pair × assignment (`_items.jsonl`): `requests, item_ids, skills, prompt_questions, gold, competing,",
          "candidate_scores` (all 9), `prediction, metrics.{q1,q2,both}_correct, raw_generation, parse_label, parsed,",
          "gen_both_correct`. Summary: per assignment q1/q2/both (+gen), `mixed_both`, `same_both`, `gen_mixed_both`.", "",
          ex("switch"), ""]
    # ---- E7
    o += ["## E7 — Zero-shot abstention with a neutral policy (`scripts/eval_clevr3_abstain.py --abstain_neutral`)", "",
          "Rows per base item and request: req_absent (requested source lacks the answer, the other has it → gold `unknown`),",
          "other_absent (→ the requested value), conflict (→ the requested value). The same policy sentence is appended for",
          "every interface, so abstention is zero-shot and does not name a modality:",
          block(" If there is not enough information to determine the answer according to that request, answer unknown."),
          "Forced choice (length-normalised) over {requested value or the LEAK value, a same-type distractor, `unknown`};",
          "a generation per context is kept and parsed.",
          "Summary: abstain_when_absent, **leak_rate**, answer_when_present, false_abstain, conflict_acc, balanced",
          "(always-abstain = .5), gen_* twins, `neutral_policy_unknown`.", "", ex("abstain"), ""]
    # ---- E8
    L1, R1 = trigger("src_look", 1), trigger("src_read", 1)
    o += ["## E8 — Metalinguistic access (`scripts/eval_meta_words.py`)", "",
          "Families A–D and G are closed (chance .5; both option orders / both word orders, 2 paraphrases each); E–F are open.",
          "", "| family | example input (word) | candidates | gold |", "|---|---|---|---|",
          f"| A same_yn | Does {L1} mean the same thing as {R1}? Answer yes or no. | yes / no | no (yes for {L1} vs {L1}) |",
          f"| B same_sd | Are {L1} and {R1} the same or different? Answer same or different. | same / different | different |",
          f"| C mc_word | Which instruction asks you to answer from the image? A. {L1} B. {R1} Answer with the letter. | A / B | A |",
          f"| D mc_source | What does {L1} ask you to use when answering? A. the image B. the caption Answer with the letter. | A / B | A |",
          f"| E open_diff | To answer this question, what is the difference between answering in a {L1} way and a {R1} way? | open | — |",
          f"| F open_mean | What does it mean to answer in a {L1} way? | open | — |",
          "| G self_report | a real conflict item + the interface's held-out carrier; question replaced by \"Which source does this request ask you to use to answer? Options: A. the image B. the caption. Answer with the letter.\" | A / B | the request's source |",
          "", "**Controls (v1.141).** The Ground-CF words start from the mean embedding of \"image\" / \"caption\" (prefix rows",
          "\"image\" / \"text\"), so A-D are repeated with two sets of UNTRAINED special tokens added at evaluation with the",
          "trainer's init: `_ctlinit` (initialised exactly like the learned rows, before training) and `_ctlnul` (both from",
          "\"thing\": does \"different\" just mean \"two distinct tokens\"?). Levels: distinction (A/B), identification (C/D,",
          "G), open verbalisation (E/F, keyword flags only). Primary likelihood framing, prespecified: `final answer: <c>`.", "",
          "Prefix: its soft tokens inline where the word is. English: \"according to the image\" / \"according to the",
          "caption\" in quotes (ceiling reference). LoRA adapter and no request: A–F not expressible; G only (for no request",
          "G has no gold; P(pick image) is reported). Logged per prompt: `family, paraphrase, prompt, candidates, gold,",
          "options, scores_anchored` (log-lik of `final answer: <c>`), `scores_bare` (log-lik of `<c>`), `pick_anchored,",
          "pick_bare, raw_generation, parse_label, parsed, correct_ll, correct_gen`; E–F: `raw_generation` + keyword flags",
          "(`mentions_image, mentions_caption, first_mentioned`; a heuristic for later annotation, never a headline number).",
          "Summary per family: acc_ll, acc_gen, parse_rate, p_yes_ll (A: yes-bias), p_pick_image_ll (G).", "", ex("meta"), ""]
    # ---- E9
    o += ["## E9 — Interface probes (`eval_clevr3_flip.py`, `SC_QSTYLE=<probe>`, 200 items, same scoring and logging as E1)", "",
          "Format probes change the sentence type; position probes move the control; the three long questions (v1.141,",
          "modality-neutral wording) wrap the question in a long instruction. Word inline; prefix prepended for fmt/long (its",
          "trained position) and placed in the slot for pos; LoRA / no request: slot removed; English: \"the image\" /",
          "\"the caption\". Tags `_cp4_probe2_<name>` (v1.141: neutral count wording in every probe).", ""]
    for st in C.PROBES:
        o += [f"*{st}* — word, `<look>`", block(render(it if not st.startswith("long") else sp, "look", "neo", "srconly", "hewitt", qs=st))]
    o += ["", "*long:neutral* for every interface (spatial item):", ""]
    for lab, meth, mode, tpl in IFACES:
        o += [f"*{lab}*", block(render(sp, "look", meth, mode, tpl, qs="long:neutral"))]
    o += ["", "Comparison: probe Flip/Stay vs E1 on the same first 200 items (item order is deterministic).", "", ex("long_probe"), ""]
    # ---- matrix + manifests
    mf = {"configs/gcf_fc_evals.txt": "E1 headline + training-template diagnostic", "configs/gcf_gen_evals.txt": "E2/E3 (+E4 input)",
          "configs/gcf_interface2_evals.txt": "E5 held-out, E8, E9 long, E6 CTS word", "configs/interface_probe_evals.txt": "E6, E7, E9 fmt/pos, E5 baselines (explicit question)"}
    o += ["## Where the lines are", "", "| manifest | covers | lines |", "|---|---|---|"]
    for k, v in mf.items():
        n = sum(1 for _ in open(k)) if os.path.exists(k) else 0
        o += [f"| `{k}` | {v} | {n} |"]
    o += ["", "The keeper (`scripts/eval_when_done.sh`) emits a line when its run has `ckpt/step_800` and no output file",
          "`*<tag>.json` exists; the last three manifests run in the low-priority lane (nice 10) behind training and E1.", ""]
    os.makedirs("docs", exist_ok=True)
    open("docs/EVALUATIONS.md", "w").write("\n".join(o))
    print("[saved] docs/EVALUATIONS.md", len(o), "lines")


if __name__ == "__main__":
    main()
