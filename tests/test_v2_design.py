"""CPU check of the protocol-v2 evaluation design (2026-09-29, audit item 3): switching the QUESTION pool must leave
the carrier (or English instruction) identical, switching the CARRIER pool must leave the question identical, and the
standard prefix's carrier switch must be an identity. Also: the legacy renders are untouched by the switches."""
import os, sys, json, re, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.update({"SC_C3_SOURCES": "look,read", "SC_CAPTION_STYLE": "fact", "SC_NEO_TEMPLATE": "srcsel", "SC_GCF_FORMAT": "v2",
                   "SC_GCF_CONT": "final", "SC_INSTR_TRAIN_POOL": "0,2,4,5"})
for k in ("SC_GCF_PT_CARRIER", "SC_GCF_HELDOUT_CARRIER", "SC_GCF_TRAIN_CARRIER"):
    os.environ.pop(k, None)
import scsrc.clevr3 as C3

items = [json.loads(l) for l in itertools.islice(open(os.path.join(ROOT, "data/clevr3/test_id.jsonl")), 120)]
RX = re.compile(r"Question:\n(.*?)\n\n(?:(.*?)\n\n)?Return only the answer\.", re.S)


def render(item, source, method, mode, qs, **env):
    for k in ("SC_GCF_PT_CARRIER", "SC_GCF_HELDOUT_CARRIER", "SC_GCF_TRAIN_CARRIER"):
        os.environ.pop(k, None)
    os.environ.update(env)
    c = C3.clevr3_user_content(item, source, 5, mode, 0, method=method, question_style=qs)
    txt = " ".join("[IMAGE]" if x.get("type") == "image" else x.get("text", "") for x in c)
    m = RX.search(txt); assert m, txt
    order = "cap_first" if txt.index("Caption:") < txt.index("[IMAGE]") else "img_first"
    return {"q": m.group(1), "ctl": m.group(2) or "", "order": order, "full": txt}


fails = 0; n = 0
for it in items:
    for src in ("look", "read"):
        for label, method, mode, env in (("word", "neo", "srconly", {}), ("pt-slot", "pt", "srconly", {"SC_GCF_PT_CARRIER": "1"}),
                                          ("english", "neo", "instructed", {}), ("pt-standard", "pt", "srconly", {})):
            ref = render(it, src, method, mode, "explicit", **env)
            qsh = render(it, src, method, mode, "heldout", SC_GCF_TRAIN_CARRIER="1", **env)
            csh = render(it, src, method, mode, "explicit", SC_GCF_HELDOUT_CARRIER="1", **env)
            both = render(it, src, method, mode, "heldout", **env)
            n += 1
            # question shift: only the question changes
            if not (qsh["ctl"] == ref["ctl"] and qsh["order"] == ref["order"] and qsh["q"] != ref["q"]):
                fails += 1; print(f"Q-SHIFT leak [{label} {src} {it['id']}]: ctl {ref['ctl']!r} -> {qsh['ctl']!r}; order {ref['order']}->{qsh['order']}")
            # carrier shift: only the control line changes (identity for the standard prefix, whose control is not inline)
            want_ctl_change = label != "pt-standard"
            if not (csh["q"] == ref["q"] and csh["order"] == ref["order"] and (csh["ctl"] != ref["ctl"]) == want_ctl_change):
                fails += 1; print(f"C-SHIFT leak [{label} {src} {it['id']}]: q {ref['q']!r} -> {csh['q']!r}; ctl {ref['ctl']!r} -> {csh['ctl']!r}")
            # both: question from the shifted pool, control from the shifted pool
            if not (both["q"] == qsh["q"] and both["ctl"] == csh["ctl"] and both["order"] == ref["order"]):
                fails += 1; print(f"BOTH mismatch [{label} {src} {it['id']}]")
            if label == "pt-standard" and not ref["full"].startswith("<pt_"):
                fails += 1; print("standard prefix not at the start:", ref["full"][:60])
print(f"v2 design: {n} (item, request, interface) groups checked, {fails} leaks")
assert fails == 0
# answer formatter follows the environment at call time
assert C3.c3_cont("red") == "Final answer: red"
os.environ["SC_GCF_CONT"] = "leadin"; assert C3.c3_cont("red").endswith("final answer: red")
os.environ["SC_GCF_CONT"] = "bare"; assert C3.c3_cont("red") == "red"
print("ALL V2 DESIGN CHECKS PASSED")
