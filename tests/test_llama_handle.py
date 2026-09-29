import importlib.util, os, sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
spec = importlib.util.spec_from_file_location("qpe", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
node = importlib.util.module_from_spec(spec)
sys.modules["qpe"] = node
spec.loader.exec_module(node)
from qpe.llama_setup import gguf_names, ensure_gguf, LlamaPE, HANDLE_ERROR, GGUF_REPO

assert gguf_names("t2i", "Q8_0") == ("qwen3.5_9b_qwen_image_2.1_pe_t2i.mtp.Q8_0.gguf", None)
assert gguf_names("i2i", "Q4_K_M") == ("qwen3.5_9b_qwen_image_2.1_pe_i2i.mtp.Q4_K_M.gguf", "qwen3.5_9b_qwen_image_2.1_pe_i2i.mmproj.bf16.gguf")
assert gguf_names("t2i - heretic", "Q8_0") == ("qwen3.5_9b_qwen_image_2.1_pe_t2i_heretic.mtp.Q8_0.gguf", None)
# the heretic i2i model shares the i2i vision file
assert gguf_names("i2i - heretic", "Q4_K_M") == ("qwen3.5_9b_qwen_image_2.1_pe_i2i_heretic.mtp.Q4_K_M.gguf", "qwen3.5_9b_qwen_image_2.1_pe_i2i.mmproj.bf16.gguf")

# ensure_gguf asks the hub for exactly the files the model and quant need, into the given folder
with tempfile.TemporaryDirectory() as d:
    asked = []
    def fetch(repo, name, local_dir):
        asked.append((repo, name)); p = Path(local_dir) / name; p.write_bytes(b"gguf"); return str(p)
    model, mmproj = ensure_gguf("i2i", "Q8_0", fetch=fetch, folder=d)
    assert asked == [(GGUF_REPO, "qwen3.5_9b_qwen_image_2.1_pe_i2i.mtp.Q8_0.gguf"), (GGUF_REPO, "qwen3.5_9b_qwen_image_2.1_pe_i2i.mmproj.bf16.gguf")], asked
    assert Path(model).is_file() and Path(mmproj).is_file()
    asked.clear()
    assert ensure_gguf("t2i", "Q4_K_M", fetch=fetch, folder=d)[1] is None and len(asked) == 1

h = LlamaPE("i2i", "m.gguf", "v.gguf", "Q8_0", "q8_0", "llama-server.exe")
assert (h.pe_task, h.model, h.mmproj, h.quant, h.kv_cache, h.exe) == ("i2i", "m.gguf", "v.gguf", "Q8_0", "q8_0", "llama-server.exe")
assert getattr(h, "pe_task", None) == "i2i"
try:
    h.tokenize("x")  # what CLIPTextEncode would call
    raise AssertionError("no error")
except AttributeError as e:
    assert str(e) == HANDLE_ERROR and "only works with the Qwen-Image 2.1 Prompt Enhancer node" in HANDLE_ERROR
print("ok")
