import importlib.util, os, sys
from pathlib import Path
from types import SimpleNamespace
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
spec = importlib.util.spec_from_file_location("qpe", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
node = importlib.util.module_from_spec(spec)
sys.modules["qpe"] = node
spec.loader.exec_module(node)
import torch
import qpe.llama_server as ls
from qpe.llama_setup import LlamaPE

# loader schema: backend is optional, ComfyUI is the first (default) option, llama.cpp has quant and kv_cache
schema = node.LoadQwenImage21PE.define_schema()
backend = [i for i in schema.inputs if i.id == "backend"][0]
assert backend.optional and [o.key for o in backend.options] == ["ComfyUI", "llama.cpp"]
llama_opt = backend.options[1]
assert [i.id for i in llama_opt.inputs] == ["quant", "kv_cache", "unload_model"]
assert list(llama_opt.inputs[0].options) == ["Q8_0", "Q4_K_M"] and list(llama_opt.inputs[1].options) == ["f16", "q8_0"]
assert llama_opt.inputs[2].default is True

# old workflows / API prompts without backend, and backend ComfyUI, take the ComfyUI path (Review Focus 1)
seen = []
node.ensure_model = lambda model: seen.append(model) or "x.safetensors"
node.comfy.sd.load_clip = lambda **kw: SimpleNamespace()
for b in (None, {"backend": "ComfyUI"}):
    out = node.LoadQwenImage21PE.execute("t2i", b) if b else node.LoadQwenImage21PE.execute("t2i")
    assert out.args[0].pe_task == "t2i"
assert seen == ["t2i", "t2i"]
# backend llama.cpp returns the handle from load_llama
node.load_llama = lambda model, quant, kv, unload: LlamaPE("i2i", "m", "v", quant, kv, "srv", unload)
h = node.LoadQwenImage21PE.execute("i2i", {"backend": "llama.cpp", "quant": "Q4_K_M", "kv_cache": "q8_0", "unload_model": False}).args[0]
assert isinstance(h, LlamaPE) and (h.quant, h.kv_cache, h.unload) == ("Q4_K_M", "q8_0", False)
# llama.cpp workflows saved before unload_model existed keep unloading
h = node.LoadQwenImage21PE.execute("i2i", {"backend": "llama.cpp", "quant": "Q4_K_M", "kv_cache": "q8_0"}).args[0]
assert h.unload is True

# generator: a LlamaPE goes to llama_server.generate with the PE prompt (0 image blocks) and the preset
calls = []
node.llama_generate = lambda handle, text, images, preset, seed, mtp: calls.append((handle, text, images, seed, mtp)) or \
    'thinking</think>{"rewritten_prompt": "a fox", "wh_ratio": "", "ratio_follow": "<image1>"}'
preset = {"preset": "Qwen-Image 2.1 PE (i2i)", "max_length": 8192, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
          "repetition_penalty": 1.0, "presence_penalty": 0.0, "thinking": True}
img = torch.zeros(1, 8, 8, 3)
out = node.TextGenerateQwen35MTP.execute(h, "make it a fox", preset, 7, {"image_1": img}, "3")
assert out.args[0] == "a fox" and out.args[2] == "thinking" and out.args[4] == "<image1>" and out.args[5] is True, out.args
(handle, text, images, seed, mtp), = calls
assert handle is h and seed == 7 and mtp == "3" and len(images) == 1
assert "<|vision_start|>" not in text and text.endswith("<|im_start|>assistant\n<think>\n") and "make it a fox" in text

# the loader/preset mismatch check still applies to the handle
t2i_preset = dict(preset, preset="Qwen-Image 2.1 PE (t2i)", presence_penalty=1.5)
try:
    node.TextGenerateQwen35MTP.execute(h, "x", t2i_preset, 1, {}, "auto")
    raise AssertionError("no error")
except ValueError as e:
    assert "loader is set to i2i" in str(e)

# the none preset refuses the llama.cpp backend with the spec's message
try:
    node.TextGenerateQwen35MTP.execute(h, "x", {"preset": "none", "max_length": 10, "sampling_mode": {}, "thinking": False}, 1, {}, "auto")
    raise AssertionError("no error")
except ValueError as e:
    assert str(e) == node.NONE_LLAMA_ERROR
print("ok")
