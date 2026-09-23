import logging, os, sys
from pathlib import Path
from types import SimpleNamespace
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1])), str(REPO)]
import comfy.model_management as mm
from comfy.text_encoders.qwen35 import Qwen35
from mtp import MTPClip

transformer = Qwen35.__new__(Qwen35)  # structure only, no weights
object.__setattr__(transformer, "mtp", None)
calls = []
patcher = SimpleNamespace(load_device="cuda:0")
clip = SimpleNamespace(cond_stage_model=SimpleNamespace(clip="qwen35_9b", qwen35_9b=SimpleNamespace(transformer=transformer)),
                       generate=lambda tokens, **kw: calls.append(kw) or [1], patcher=patcher)
# before generating, every other model leaves the PE's device and the PE itself stays
freed = []
mm.free_memory = lambda required, device, keep_loaded=[], **kw: freed.append((device, keep_loaded, kw))
pe_entry, other = SimpleNamespace(model=patcher), SimpleNamespace(model=object())
mm.current_loaded_models[:] = [other, pe_entry]
records = []
logging.getLogger().addHandler(type("H", (logging.Handler,), {"emit": lambda self, r: records.append(r.getMessage())})())

MTPClip(clip).generate({"q": [[(1,)]]}, mtp=True, max_length=4)
assert calls == [{"mtp": True, "max_length": 4}]  # still delegates to the real generate
assert any("no MTP head" in m for m in records), records
assert freed == [("cuda:0", [pe_entry], {})], freed  # for_dynamic stays False, so dynamic models really unload

records.clear()
MTPClip(clip).generate({"q": [[(1,)]]}, mtp=False)
assert not records  # mtp off: nothing to warn about
print("ok")
