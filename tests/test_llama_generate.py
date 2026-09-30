import contextlib, importlib.util, json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
spec = importlib.util.spec_from_file_location("qpe", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
node = importlib.util.module_from_spec(spec)
sys.modules["qpe"] = node
spec.loader.exec_module(node)
import qpe.llama_server as ls
from qpe.llama_setup import LlamaPE


class FakeServer:
    # records the order generate() talks to the server in
    marker = "<m>"

    def __init__(self, running=False):
        self.calls, self.running = [], running

    def ensure(self, handle, ctx, mtp, before_start=None):
        self.calls.append("ensure")
        if not self.running:
            if before_start:
                before_start()
            self.calls.append("start")
            self.running = True

    def post(self, path, body):
        self.calls.append(path)
        return {"tokens": [1, 2, 3]}

    def wait_asleep(self):
        self.calls.append("wait_asleep")

    @contextlib.contextmanager
    def stream(self, body):
        self.calls.append("stream")
        yield [f"data: {json.dumps({'content': 'ok', 'tokens': [1], 'stop': True})}\n".encode()]


mm = ls.comfy.model_management
mm.unload_all_models = lambda: ls.SERVER.calls.append("unload_comfy")
mm.soft_empty_cache = lambda *a, **k: None
ls.SERVER = FakeServer()
preset ={"max_length": 64, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0,
          "presence_penalty": 1.5, "thinking": True}
assert ls.generate(LlamaPE("t2i", "t.gguf", None, "Q8_0", "f16", "srv"), "P", [], preset, 1, "auto") == "ok"
calls = ls.SERVER.calls
# llama-server b11160 loses a request that arrives while it is falling asleep (the queue only wakes for a request that
# finds it already asleep), so the completion must wait until it sleeps
assert calls.index("wait_asleep") < calls.index("stream"), calls
assert calls[-1] == "wait_asleep", calls  # and the VRAM is freed again before the next node
# tokens are counted locally: /tokenize would wake the sleeping server, and the wait above would put it to sleep again,
# costing a second full model load per generation
assert "/tokenize" not in calls and calls.count("ensure") == 1, calls
assert calls.index("unload_comfy") < calls.index("start"), calls  # ComfyUI's models make room for the PE

# unload_model off, server already up: no sleep waits and ComfyUI's models stay loaded, so neither side reloads
kept = LlamaPE("t2i", "t.gguf", None, "Q8_0", "f16", "srv", unload=False)
ls.SERVER = FakeServer(running=True)
assert ls.generate(kept, "P", [], preset, 1, "auto") == "ok"
assert ls.SERVER.calls == ["ensure", "stream"], ls.SERVER.calls
# unload_model off, server (re)starting: ComfyUI's models still make room for it first
ls.SERVER = FakeServer()
assert ls.generate(kept, "P", [], preset, 1, "auto") == "ok"
assert ls.SERVER.calls == ["ensure", "unload_comfy", "start", "stream"], ls.SERVER.calls
print("ok")
