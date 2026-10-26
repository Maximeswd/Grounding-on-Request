"""Logic tests for the unified task pipeline. Runs in the sandbox: no torch,
no GPU, no real datasets (a stub `datasets` module is injected so data.py
imports). Images are string placeholders -- construction never decodes them.
What this canNOT test: real dataset iteration, the forward pass, the eval matrix
(all cluster-only). Run: python tests/test_tasks.py
"""
import sys, types, os, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# stub `datasets` so scsrc.data imports without the real package
_d = types.ModuleType("datasets")
_d.load_dataset = lambda *a, **k: None
_d.load_from_disk = lambda *a, **k: None
sys.modules.setdefault("datasets", _d)

from scsrc.tasks import (SlotItem, build_examples, split_by_class, caption_for,
                         conflict_value, caption_from_key, values_of)
from scsrc import task_adapters as A
from scsrc import metrics as Me
from scsrc.prompts import fmt_key
from scsrc.data import build_specs, get_image, keys_from_row

ok = 0


def check(name, cond):
    global ok
    assert cond, f"FAIL: {name}"
    ok += 1
    print(f"  ok  {name}")


print("== tasks: captions / conflict ==")
check("caption counting plural", caption_for("counting", "dog", "3") == "there are 3 dogs")
check("caption spatial pair", caption_for("spatial", "cat|table", "on") == "the cat is on the table")
check("caption attr", caption_for("attr", "apple", "shiny") == "the apple is shiny")
rng = random.Random(0)
cv = conflict_value("counting", "3", ["1", "2", "3", "4", "5"], rng)
check("counting conflict is near & different", cv in {"1", "2", "4", "5"})
check("spatial conflict different", conflict_value("spatial", "on", ["on", "under", "left of"], rng) in {"under", "left of"})

print("== tasks: build_examples swap matching (counting) ==")
items = [
    SlotItem("imgA", "dog", "3", "3", "dog", "counting"),
    SlotItem("imgB", "dog", "5", "5", "dog", "counting"),
    SlotItem("imgC", "cat", "2", "2", "cat", "counting"),   # lone group -> dropped (require_swap)
]
exs = build_examples(items, "counting", ["2", "3", "5"], random.Random(1), require_swap=True)
groups = {tuple(sorted(e.key_1.items()))[0][0] for e in exs}
check("only swap-able group kept", all(next(iter(e.key_1)) == "dog" for e in exs))
check("dog examples produced", len(exs) >= 2)
e0 = exs[0]
check("conflict is real (k1 != k2)", e0.key_1["dog"] != e0.key_2["dog"])
check("image_2 is the swap image", e0.image_2 in {"imgA", "imgB"} and e0.image_2 != e0.image_1)

print("== tasks: split_by_class (held-out classes) ==")
big = [SlotItem(f"i{n}{k}", "dog", str(n), str(n), "dog", "counting")
       for n in (1, 2, 3, 4) for k in range(5)]
sp = split_by_class(big, heldout={"4"}, id_frac=0.2, rng=random.Random(2))
check("held-out class only in test_ood", all(it.klass == "4" for it in sp["test_ood"]))
check("held-out absent from train", all(it.klass != "4" for it in sp["train"]))
check("train+test_id+ood == all", len(sp["train"]) + len(sp["test_id"]) + len(sp["test_ood"]) == len(big))

print("== data.build_specs reads STORED keys (value-agnostic) ==")
fake_ds = [
    {"image_1": "img1", "image_2": "img2",
     "caption_1": "there are 3 dogs", "caption_2": "there are 5 dogs",
     "key_1": '{"dog": "3"}', "key_2": '{"dog": "5"}', "id": 7},
]
okk, k1, k2 = keys_from_row(fake_ds[0])
check("keys_from_row decodes JSON", okk and k1 == {"dog": "3"} and k2 == {"dog": "5"})
specs = build_specs(fake_ds, "train", conditions=("agree", "conflict"))
conf = [s for s in specs if s.cond == "conflict"]
agree = [s for s in specs if s.cond == "agree"]
check("4 specs from 1 row", len(specs) == 4)
# conflict, image_1 shown: image truth dog:3, text asserts dog:5
c1 = [s for s in conf if s.img_idx == 1][0]
check("conflict img1 image_key=3", c1.image_key == {"dog": "3"})
check("conflict img1 text_key=5", c1.text_key == {"dog": "5"})
check("get_image returns the shown image", get_image(fake_ds, c1) == "img1")
check("agree img1 keys both 3", [s for s in agree if s.img_idx == 1][0].text_key == {"dog": "3"})

print("== data.build_specs held-out-class filter (ColorSwap path) ==")
cs_rows = [{"image_1": "i1", "image_2": "i2", "caption_1": "a red car",
            "caption_2": "a blue car", "id": 1},
           {"image_1": "i3", "image_2": "i4", "caption_1": "a green hat",
            "caption_2": "a yellow hat", "id": 2}]
tr = build_specs(cs_rows, "train", heldout_classes={"blue"}, class_mode="exclude")
check("exclude drops the red/blue example", all(s.example_id != 1 for s in tr))
oo = build_specs(cs_rows, "test", heldout_classes={"blue"}, class_mode="only")
check("only keeps the red/blue example", all(s.example_id == 1 for s in oo) and len(oo) > 0)

print("== metrics: task-aware value parsing ==")
check("counting digits", Me.first_value("there are 3 dogs", "counting") == "3")
check("counting number word", Me.first_value("I count five of them", "counting") == "5")
check("spatial longest match", Me.first_value("the cat is in front of the table", "spatial") == "in front of")
check("spatial single word", Me.first_value("it is under the chair", "spatial") == "under")
check("color unchanged", Me.first_value("a red car", "color") == "red")
check("parse_structured_any ints", Me.parse_structured_any("dog: 3, cat: 2") == {"dog": "3", "cat": "2"})
check("parse_pred_task counting", Me.parse_pred_task("dog: 4", "counting") == {"dog": "4"})

print("== prompts.fmt_key is value-agnostic ==")
check("fmt_key counting", fmt_key({"dog": "3"}) == "dog: 3")
check("fmt_key spatial", fmt_key({"cat|table": "on"}) == "cat|table: on")

print("== adapters: pure parsing helpers ==")
check("count_object men (irregular kept)", A.count_object("How many men are there?") in {"man", "men"})
check("count_object dogs->dog", A.count_object("How many dogs are in the image?") == "dog")
check("norm_count word", A.norm_count("two") == "2")
check("norm_count digit", A.norm_count("7") == "7")
check("canon_relation synonym", A.canon_relation("to the left of") == "left of")
check("canon_relation passthrough", A.canon_relation("under") == "under")
check("canon_relation reject", A.canon_relation("perpendicular to") is None)
pair = A.parse_vsr_pair("The teddy bear is in front of the person.", "in front of")
check("vsr pair parse", pair == ("teddy bear", "person"))
check("gqa_attr color", A.gqa_attr("What color is the car?") == ("car", "color"))

print("== metrics.classify_source + prompts.clean_q ==")
check("classify grounded", Me.classify_source("3", "3", "5") == "grounded")
check("classify laundered", Me.classify_source("5", "3", "5") == "laundered")
check("classify other", Me.classify_source("9", "3", "5") == "other")
check("classify none->other", Me.classify_source(None, "3", "5") == "other")
from scsrc.prompts import clean_q
check("clean_q counting has no caption slot", "{cap}" not in clean_q("counting") and "{obj}" in clean_q("counting"))
check("clean_q spatial", "relation" in clean_q("spatial"))

print(f"\nALL {ok} CHECKS PASSED")
