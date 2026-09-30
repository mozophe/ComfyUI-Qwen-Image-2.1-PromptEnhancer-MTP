# Manual GPU check of the llama.cpp backend outside ComfyUI's graph: t2i and i2i for a quant, then a cancel.
# usage (GPU idle, ComfyUI not running): COMFYUI_PATH=D:/ai/ComfyUI D:/ai/ComfyUI/venv/Scripts/python.exe tools/llama_smoke.py Q8_0 f16
import importlib.util, os, sys, threading, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
spec = importlib.util.spec_from_file_location("qpe", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
node = importlib.util.module_from_spec(spec)
sys.modules["qpe"] = node
spec.loader.exec_module(node)
import torch
import comfy.model_management as mm
from qpe.llama_setup import load_llama
from qpe.llama_server import generate, SERVER
from qpe.mtp import pe_prompt, split_thinking
from qpe.pe import parse_answer, system_prompt

quant, kv = sys.argv[1], sys.argv[2]
preset = {"max_length": 8192, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0,
          "presence_penalty": 1.5, "thinking": True}
for task, images, idea in (("t2i", [], "a corgi playing guitar in the rain"),
                           ("i2i", [torch.rand(1, 1024, 1024, 3), torch.rand(1, 768, 1344, 3)], "Put the subject of <image1> into <image2>.")):
    h = load_llama(task, quant, kv)
    t = time.perf_counter()
    text = generate(h, pe_prompt(system_prompt(task), idea, 0, True), images, dict(preset, presence_penalty=1.5 if task == "t2i" else 0.0), 42, "auto")
    parsed = parse_answer(split_thinking(text)[1])
    print(f"{task} {quant} kv={kv}: {time.perf_counter() - t:.1f} s, parsed={parsed is not None}, asleep={SERVER.get('/props')['is_sleeping']}")
    assert parsed is not None

# cancel mid-generation: must raise the interrupt and leave the server asleep
threading.Timer(8, mm.interrupt_current_processing).start()
try:
    generate(load_llama("t2i", quant, kv), pe_prompt(system_prompt("t2i"), "a castle", 0, True), [], preset, 1, "auto")
    raise AssertionError("not interrupted")
except mm.InterruptProcessingException:
    mm.interrupt_current_processing(False)
    print("cancel ok, asleep:", SERVER.get("/props")["is_sleeping"])
SERVER.stop()
print("ok")
