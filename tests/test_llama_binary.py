import importlib.util, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
spec = importlib.util.spec_from_file_location("qpe", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
node = importlib.util.module_from_spec(spec)
sys.modules["qpe"] = node
spec.loader.exec_module(node)
from qpe.llama_setup import parse_driver_version, select_build, PLATFORM_ERROR

assert parse_driver_version("580.88\n") == (580, 88)
assert parse_driver_version("525.60.13") == (525, 60, 13)
assert parse_driver_version("") is None and parse_driver_version("N/A") is None

b = select_build("Windows", "AMD64", (581, 42))
assert b["cuda"] == "13.4" and b["dir"] == "b11160-cuda-13.4" and b["exe"] == "llama-server.exe"
assert b["assets"] == ["llama-b11160-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"], b
b = select_build("Windows", "AMD64", (552, 22))
assert b["cuda"] == "12.4" and b["assets"][0] == "llama-b11160-bin-win-cuda-12.4-x64.zip"
b = select_build("Linux", "x86_64", (570, 1))
assert b["cuda"] == "12.8" and b["exe"] == "llama-server"
assert b["assets"] == ["llama-b11160-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b11160-bin-ubuntu-cuda-12.8-x64.tar.gz"], b
assert select_build("Linux", "x86_64", (580,))["cuda"] == "13.4"
assert select_build("Linux", "x86_64", (525, 60, 13))["cuda"] == "12.8"

def raises(*args):
    try:
        select_build(*args)
    except RuntimeError as e:
        return str(e)
    raise AssertionError(f"no error for {args}")

assert raises("Darwin", "arm64", (580,)) == PLATFORM_ERROR
assert raises("Linux", "aarch64", (580,)) == PLATFORM_ERROR
assert raises("Windows", "AMD64", None) == PLATFORM_ERROR  # no nvidia-smi: no NVIDIA driver
msg = raises("Windows", "AMD64", (522, 6))
assert "522.6" in msg and "528.33" in msg, msg
msg = raises("Linux", "x86_64", (525, 60, 12))
assert "525.60.12" in msg and "525.60.13" in msg, msg
print("ok")
