import importlib.util, os, sys
from pathlib import Path
from types import SimpleNamespace
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
spec = importlib.util.spec_from_file_location("qpe", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
node = importlib.util.module_from_spec(spec)
sys.modules["qpe"] = node
spec.loader.exec_module(node)

class Reached(Exception): pass
def tokenize(text, **k): raise Reached(text)  # the prompt that would be generated from

node.system_prompt = lambda task: f"OFFICIAL {task}"  # no download

def run(system):
    preset = {"preset": "Qwen-Image 2.1 PE (t2i)", "thinking": True}
    try:
        node.TextGenerateQwen35MTP.generate_text(node.MTPClip(SimpleNamespace(tokenize=tokenize)), "a fox", preset, 0, [], "auto", system)
    except Reached as e:
        return str(e)

# a connected system prompt replaces the official one
assert "<|im_start|>system\nMY RULES<|im_end|>" in run("  MY RULES\n"), run("  MY RULES\n")
# unconnected or blank keeps the official one
assert "<|im_start|>system\nOFFICIAL t2i<|im_end|>" in run(""), run("")
assert "<|im_start|>system\nOFFICIAL t2i<|im_end|>" in run("   "), run("   ")
# the node exposes core's input when core has it
parent = {i.id for i in node.TextGenerate.define_schema().inputs}
ours = {i.id for i in node.TextGenerateQwen35MTP.define_schema().inputs}
assert ("system_prompt" in ours) == ("system_prompt" in parent), (ours, parent)
print("ok", "system_prompt" in ours)
