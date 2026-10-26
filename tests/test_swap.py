"""Logic test for the causal-swap pair construction (build_vgm_pair), the
symmetric core. Runs in the sandbox: torch + datasets are stubbed (build_vgm_pair
uses neither), and _seq is monkeypatched to capture (image, user_text,
continuation) instead of encoding. Verifies:
  look VGM swaps the IMAGE, continuation = image value;
  read VGM swaps the CAPTION (fed conflict caption vs shown image's own caption),
    continuation = text value  -> text-sensitivity, the symmetric twin.
"""
import sys, types, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# stub datasets (data.py) and torch (preference.py imports it but VGM doesn't use it)
for m in ("datasets",):
    mod = types.ModuleType(m); mod.load_dataset = lambda *a, **k: None
    mod.load_from_disk = lambda *a, **k: None; sys.modules.setdefault(m, mod)
_t = types.ModuleType("torch"); _tnn = types.ModuleType("torch.nn")
_tf = types.ModuleType("torch.nn.functional"); _tnn.functional = _tf
_t.nn = _tnn; _t.Tensor = object
sys.modules.setdefault("torch", _t); sys.modules.setdefault("torch.nn", _tnn)
sys.modules.setdefault("torch.nn.functional", _tf)

import scsrc.preference as P
from scsrc.prompts import fmt_key

# capture _seq args instead of encoding
captured = {}
P._seq = lambda processor, image, user, cont, device: {"image": image, "user": user, "cont": cont}

# fake ColorSwap-shaped row + a conflict spec on image_1
ds = [{"image_1": "IMG1", "image_2": "IMG2",
       "caption_1": "there are 3 dogs", "caption_2": "there are 5 dogs"}]
spec = types.SimpleNamespace(
    ex_index=0, img_idx=1, cond="conflict",
    fed_caption="there are 5 dogs",              # the OTHER image's caption -> text value
    image_key={"dog": "3"}, text_key={"dog": "5"}, relation="conflict")

ok = 0
def check(name, cond):
    global ok; assert cond, f"FAIL: {name}"; ok += 1; print(f"  ok  {name}")

print("== look VGM: image-axis sensitivity ==")
lv = P.build_vgm_pair(None, ds, spec, "ctrl", "look", 4, "cpu")
check("look pos uses shown image", lv["pos"]["image"] == "IMG1")
check("look neg uses swapped image", lv["neg"]["image"] == "IMG2")
check("look continuation carries image value", "dog: 3" in lv["pos"]["cont"])
check("look pos/neg differ only in image", lv["pos"]["user"] == lv["neg"]["user"])

print("== read VGM: caption-axis sensitivity (the symmetric twin) ==")
rv = P.build_vgm_pair(None, ds, spec, "ctrl", "read", 4, "cpu")
check("read continuation carries text value", "dog: 5" in rv["pos"]["cont"])
check("read same image in pos and neg", rv["pos"]["image"] == rv["neg"]["image"] == "IMG1")
check("read pos user embeds the fed (text-value) caption", "there are 5 dogs" in rv["pos"]["user"])
check("read neg user embeds the shown-image (image-value) caption", "there are 3 dogs" in rv["neg"]["user"])
check("read pos/neg differ (caption swapped)", rv["pos"]["user"] != rv["neg"]["user"])

print("== read VGM needs a real conflict (agree -> None) ==")
agree = types.SimpleNamespace(ex_index=0, img_idx=1, cond="agree",
                              fed_caption="there are 3 dogs",
                              image_key={"dog": "3"}, text_key={"dog": "3"}, relation="agree")
check("read VGM None on agree", P.build_vgm_pair(None, ds, agree, "ctrl", "read", 4, "cpu") is None)
check("compare VGM None", P.build_vgm_pair(None, ds, spec, "ctrl", "compare", 4, "cpu") is None)

print(f"\nALL {ok} CHECKS PASSED")
