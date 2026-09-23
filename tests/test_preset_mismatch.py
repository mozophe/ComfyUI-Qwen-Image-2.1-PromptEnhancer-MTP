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

class Reached(Exception): pass
def tokenize(*a, **k): raise Reached  # past the checks, into generation

def run(pe_task, preset_name, images):
    clip = SimpleNamespace(tokenize=tokenize) if pe_task is None else SimpleNamespace(tokenize=tokenize, pe_task=pe_task)
    preset = {"preset": preset_name, "thinking": True}
    try:
        node.TextGenerateQwen35MTP.generate_text(node.MTPClip(clip), "a fox", preset, 0, images, "auto")
    except Reached:
        return "ok"
    except ValueError as e:
        return str(e)

img = [torch.zeros(1, 8, 8, 3)]
# loader and preset disagree: refused, naming both
msg = run("t2i", "Qwen-Image 2.1 PE (i2i)", img)
assert "loader is set to t2i" in msg and "(i2i)" in msg, msg
msg = run("i2i", "Qwen-Image 2.1 PE (t2i)", [])
assert "loader is set to i2i" in msg, msg
# matching modes, and CLIPs from other loaders (no tag), go through
assert run("t2i", "Qwen-Image 2.1 PE (t2i)", []) == "ok"
assert run("i2i", "Qwen-Image 2.1 PE (i2i)", img) == "ok"
assert run(None, "Qwen-Image 2.1 PE (i2i)", img) == "ok"
print("ok")
