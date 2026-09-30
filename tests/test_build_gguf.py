import importlib.util, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_gguf", REPO / "tools" / "build_gguf.py")
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)

assert bg.SOURCES == {"t2i": "Qwen/Qwen-Image-2.1-PE-T2I", "i2i": "Qwen/Qwen-Image-2.1-PE-I2I",
                      "t2i - heretic": "pottokao/Qwen-Image-2.1-PE-T2I-Heretic", "i2i - heretic": "darrellbest/Qwen-Image-2.1-PE-I2I-Heretic"}
cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"mtp_num_hidden_layers": None, "hidden_size": 4096}}
out = bg.patch_config(cfg)
assert out["text_config"]["mtp_num_hidden_layers"] == 1 and out["text_config"]["hidden_size"] == 4096
assert cfg["text_config"]["mtp_num_hidden_layers"] is None  # input left untouched
# a text-only export (no text_config) gets the field at the top level
out = bg.patch_config({"architectures": ["Qwen3_5ForCausalLM"], "mtp_num_hidden_layers": None})
assert out["mtp_num_hidden_layers"] == 1 and "text_config" not in out
idx = {"metadata": {"total_size": 1}, "weight_map": {"model.a": "model-00001.safetensors"}}
out = bg.add_mtp_to_index(idx, ["mtp.fc.weight", "mtp.norm.weight"])
assert out["weight_map"] == {"model.a": "model-00001.safetensors", "mtp.fc.weight": "model-mtp.safetensors", "mtp.norm.weight": "model-mtp.safetensors"}
assert "mtp.fc.weight" not in idx["weight_map"]
print("ok")
