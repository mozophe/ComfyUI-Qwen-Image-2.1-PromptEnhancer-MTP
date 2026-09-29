import importlib.util, io, os, sys, tarfile, tempfile, zipfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
spec = importlib.util.spec_from_file_location("qpe", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
node = importlib.util.module_from_spec(spec)
sys.modules["qpe"] = node
spec.loader.exec_module(node)
from qpe.llama_setup import install_archives, ensure_server

def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)

def make_tar(path, files):
    with tarfile.open(path, "w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            t.addfile(info, io.BytesIO(data))

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    # Windows layout: binary zip is flat, cudart zip is flat
    make_zip(d / "bin.zip", {"llama-server.exe": b"exe", "ggml-cuda.dll": b"dll"})
    make_zip(d / "rt.zip", {"cudart64_13.dll": b"rt"})
    exe = install_archives([d / "bin.zip", d / "rt.zip"], d / "out" / "b11160-cuda-13.4", "llama-server.exe", lambda p: None)
    assert exe == d / "out" / "b11160-cuda-13.4" / "llama-server.exe" and exe.read_bytes() == b"exe"
    assert (exe.parent / "cudart64_13.dll").is_file() and (exe.parent / "ggml-cuda.dll").is_file()
    assert not (d / "out" / "b11160-cuda-13.4.partial").exists()

    # Linux layout: binary in a subfolder, runtime libs elsewhere -> libs end up next to the exe
    make_tar(d / "bin.tgz", {"build/bin/llama-server": b"elf", "build/bin/libggml.so": b"so"})
    make_tar(d / "rt.tgz", {"lib/libcudart.so.13": b"rt"})
    exe = install_archives([d / "bin.tgz", d / "rt.tgz"], d / "lin" / "b11160-cuda-13.4", "llama-server", lambda p: None)
    assert exe.parent == d / "lin" / "b11160-cuda-13.4"
    assert {p.name for p in exe.parent.iterdir()} >= {"llama-server", "libggml.so", "libcudart.so.13"}

    # a failing check leaves nothing installed (Review Focus 4)
    def bad(p): raise RuntimeError("llama-server --version failed")
    try:
        install_archives([d / "bin.zip", d / "rt.zip"], d / "bad" / "b11160-cuda-13.4", "llama-server.exe", bad)
        raise AssertionError("no error")
    except RuntimeError:
        pass
    assert not (d / "bad" / "b11160-cuda-13.4").exists() and not (d / "bad" / "b11160-cuda-13.4.partial").exists()

    # an archive without the exe is an error, also with nothing left behind
    make_zip(d / "empty.zip", {"readme.txt": b"x"})
    try:
        install_archives([d / "empty.zip"], d / "none" / "x", "llama-server.exe", lambda p: None)
        raise AssertionError("no error")
    except RuntimeError as e:
        assert "llama-server.exe" in str(e)
    assert not (d / "none" / "x").exists()

    # ensure_server: downloads each asset once, then reuses the install
    build = {"assets": ["a.zip", "b.zip"], "dir": "b11160-cuda-13.4", "exe": "llama-server.exe"}
    fetched = []
    def fetch(url, dst):
        fetched.append(url)
        make_zip(dst, {"llama-server.exe": b"exe"} if url.endswith("a.zip") else {"x.dll": b"x"})
    exe = ensure_server(d / "models", build, fetch=fetch, check=lambda p: None)
    assert exe == d / "models" / "llama.cpp" / "b11160-cuda-13.4" / "llama-server.exe"
    assert [u.rsplit("/", 1)[1] for u in fetched] == ["a.zip", "b.zip"] and fetched[0].startswith("https://github.com/ggml-org/llama.cpp/releases/download/b11160/")
    assert ensure_server(d / "models", build, fetch=fetch, check=lambda p: None) == exe and len(fetched) == 2
print("ok")
