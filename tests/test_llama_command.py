import importlib.util, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
spec = importlib.util.spec_from_file_location("qpe", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
node = importlib.util.module_from_spec(spec)
sys.modules["qpe"] = node
spec.loader.exec_module(node)
from qpe.llama_setup import LlamaPE
from qpe.llama_server import image_tokens, estimate_text_tokens, context_size, mtp_args, server_key, server_command, needs_restart

assert image_tokens(1024, 1024) == 32 * 32 + 2
assert image_tokens(700, 1000) == 22 * 32 + 2  # ceil(700/32)=22, ceil(1000/32)=32: never under-counts odd sizes
assert estimate_text_tokens("x" * 3000) == 1000 and estimate_text_tokens("ab") == 1
assert context_size(4550 + 2052, 8192) == 16384 and context_size(1, 1) == 4096 and context_size(4096, 4096) == 8192

assert mtp_args("off") == [] and mtp_args("auto") == ["--spec-type", "draft-mtp"]
assert mtp_args("4") == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]

t2i = LlamaPE("t2i", "t.gguf", None, "Q8_0", "f16", "srv.exe")
cmd = server_command(t2i, 12288, "auto", "dio", 8123)
assert cmd == ["srv.exe", "-m", "t.gguf", "-ngl", "99", "-fa", "on", "--fit", "off", "-np", "1", "-c", "12288",
               "-ctk", "f16", "-ctv", "f16", "--sleep-idle-seconds", "1", "--load-mode", "dio",
               "--host", "127.0.0.1", "--port", "8123", "--no-webui", "--spec-type", "draft-mtp"], cmd
i2i = LlamaPE("i2i", "i.gguf", "v.gguf", "Q4_K_M", "q8_0", "srv")
cmd = server_command(i2i, 16384, "off", "mmap", 9000)
assert cmd[cmd.index("--mmproj") + 1] == "v.gguf" and "--spec-type" not in cmd
assert cmd[cmd.index("-ctk") + 1] == "q8_0" and cmd[cmd.index("-ctv") + 1] == "q8_0" and cmd[cmd.index("--load-mode") + 1] == "mmap"

k = server_key(t2i, "auto")
assert not needs_restart(True, k, 16384, k, 12288)           # same everything, enough context: reuse
assert needs_restart(True, k, 12288, k, 16384)                # exact token count needs more context (Review Focus 2)
assert needs_restart(False, k, 16384, k, 12288)               # the process died between runs (Review Focus 3)
assert needs_restart(True, None, 0, k, 4096)                  # nothing running yet
assert needs_restart(True, k, 16384, server_key(i2i, "auto"), 4096)                                          # other model
assert needs_restart(True, k, 16384, server_key(t2i, "off"), 4096)                                           # MTP change
assert needs_restart(True, k, 16384, server_key(LlamaPE("t2i", "t.gguf", None, "Q8_0", "q8_0", "srv.exe"), "auto"), 4096)  # KV cache change
print("ok")
