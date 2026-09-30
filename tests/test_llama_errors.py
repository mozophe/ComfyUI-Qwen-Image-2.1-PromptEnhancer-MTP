import importlib.util, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
spec = importlib.util.spec_from_file_location("qpe", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
node = importlib.util.module_from_spec(spec)
sys.modules["qpe"] = node
spec.loader.exec_module(node)
from qpe.llama_server import failure_message, LlamaServer

oom = failure_message("llama-server exited with code 1", "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000 MiB on device 0: cudaMalloc failed: out of memory", "x.log")
assert "Q4_K_M" in oom and "kv_cache: q8_0" in oom and "max_length" in oom and "cudaMalloc failed" in oom, oom
other = failure_message("llama-server did not answer within 600 s", "some log line", "C:/tmp/llama-server.log")
assert "did not answer" in other and "some log line" in other and "C:/tmp/llama-server.log" in other and "Q4_K_M" not in other

# a fresh manager has nothing running and does not crash on stop()
s = LlamaServer()
assert s.proc is None and s.ctx == 0
s.stop()
print("ok")
