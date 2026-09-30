# llama.cpp backend setup: pick and install the pinned llama-server build for this NVIDIA driver, download the hosted
# GGUFs, and hand the loader a LlamaPE handle
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
import folder_paths
from huggingface_hub import hf_hub_download
from .pe import PE

LLAMA_TAG = "b11160"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/"
# SHA-256 of every asset select_build can pick, from the b11160 release's published digests: the archives hold executables
ASSET_SHA256 = {
    "llama-b11160-bin-win-cuda-13.4-x64.zip": "966b2b052a820d71ba1c2040a73c46afc772659a75395260a583746effb72cff",
    "cudart-llama-bin-win-cuda-13.4-x64.zip": "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
    "llama-b11160-bin-win-cuda-12.4-x64.zip": "e2bcd71b9a03e4ed7d8bdf77f129eb682d63bbff244ef24fbbe40f1003368763",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "llama-b11160-bin-ubuntu-cuda-13.4-x64.tar.gz": "76983c34644683614a0d4983b7948b503dac76bfacc9f96a811c0ca146f103d4",
    "cudart-llama-b11160-bin-ubuntu-cuda-13.4-x64.tar.gz": "14765bd08136c6838fbadf22f3484d2b9ad2be307ae0fa766ec42f4bb100b070",
    "llama-b11160-bin-ubuntu-cuda-12.8-x64.tar.gz": "8ed8d659383b6624ef2bd14ebd6984c8c1b1b2f13e4ba0c5c43f6038ca22591d",
    "cudart-llama-b11160-bin-ubuntu-cuda-12.8-x64.tar.gz": "54776c67e34b536f6123b1a0697931f86f846f9f71baa9d86f550485fbbc9df1",
}
CUDA13_MIN = (580,)
CUDA12_MIN = {"Windows": (528, 33), "Linux": (525, 60, 13)}
PLATFORM_ERROR = ("The llama.cpp backend needs an NVIDIA GPU on Windows or Linux (x64); "
                  "set the loader's backend to ComfyUI.")


def parse_driver_version(text):
    # "580.88" -> (580, 88); None when nvidia-smi printed nothing usable
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    try:
        return tuple(int(p) for p in first.split("."))
    except ValueError:
        return None


def driver_version(run=subprocess.run, sleep=time.sleep, tries=3):
    # None only when nvidia-smi doesn't exist (no NVIDIA driver). nvidia-smi can fail once under load (seen once in ~50
    # loads), so it is retried, and a lasting failure reports nvidia-smi's own output rather than "no NVIDIA GPU"
    for attempt in range(tries):
        try:
            out = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
            detail = f"exit {out.returncode}: {(out.stderr or out.stdout).strip()[-300:]}"
            version = parse_driver_version(out.stdout) if out.returncode == 0 else None
        except FileNotFoundError:
            return None
        except (OSError, subprocess.TimeoutExpired) as e:
            version, detail = None, str(e)
        if version is not None:
            return version
        if attempt < tries - 1:
            sleep(1)
    raise RuntimeError(f"nvidia-smi could not read the NVIDIA driver version ({detail}). Check the GPU driver, or set the "
                       "loader's backend to ComfyUI.")


def select_build(system, machine, driver):
    if system not in CUDA12_MIN or machine.lower() not in ("amd64", "x86_64") or driver is None:
        raise RuntimeError(PLATFORM_ERROR)
    if driver >= CUDA13_MIN:
        cuda = "13.4"
    elif driver >= CUDA12_MIN[system]:
        cuda = "12.4" if system == "Windows" else "12.8"
    else:
        found, need = ".".join(map(str, driver)), ".".join(map(str, CUDA12_MIN[system]))
        raise RuntimeError(f"The llama.cpp backend needs NVIDIA driver {need} or newer; this system has {found}. "
                           "Update the driver, or set the loader's backend to ComfyUI.")
    if system == "Windows":
        assets = [f"llama-{LLAMA_TAG}-bin-win-cuda-{cuda}-x64.zip", f"cudart-llama-bin-win-cuda-{cuda}-x64.zip"]
    else:
        assets = [f"llama-{LLAMA_TAG}-bin-ubuntu-cuda-{cuda}-x64.tar.gz", f"cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-{cuda}-x64.tar.gz"]
    return {"cuda": cuda, "assets": assets, "dir": f"{LLAMA_TAG}-cuda-{cuda}",
            "exe": "llama-server.exe" if system == "Windows" else "llama-server"}


def download(url, dst):
    # whole-file download to dst + ".partial", renamed when complete (the archives are 0.1-0.6 GB)
    part = Path(str(dst) + ".partial")
    logging.info(f"Qwen-Image 2.1 PE llama.cpp setup: downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    except OSError as e:
        raise RuntimeError(f"Downloading {url} failed ({e}); run the workflow again to retry.") from e
    os.replace(part, dst)


def check_server(exe):
    env = dict(os.environ, LD_LIBRARY_PATH=str(exe.parent)) if os.name != "nt" else None
    try:
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"{exe} does not run ({e}).") from e
    if out.returncode != 0:
        raise RuntimeError(f"{exe} --version failed (exit {out.returncode}): {(out.stderr or out.stdout).strip()[-500:]}")


def install_archives(archives, dest, exe_name, check):
    # extract into dest + ".partial", gather the exe and every shared library into one folder, check it runs, then move
    # it into place; any failure removes everything, so the next run starts clean
    part = dest.with_name(dest.name + ".partial")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    try:
        for a in archives:
            if a.name.endswith(".zip"):
                with zipfile.ZipFile(a) as z:
                    z.extractall(part)
            else:
                with tarfile.open(a) as t:
                    t.extractall(part, filter="data")
        exe = next(part.rglob(exe_name), None)
        if exe is None:
            raise RuntimeError(f"The llama.cpp download has no {exe_name}.")
        bindir = exe.parent
        for f in list(part.rglob("*")):
            if f.parent != bindir and (f.is_file() or f.is_symlink()) and (f.suffix == ".dll" or ".so" in f.name):
                shutil.move(str(f), bindir / f.name)
        if os.name != "nt":
            exe.chmod(0o755)
        check(exe)
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(bindir, dest)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    shutil.rmtree(part, ignore_errors=True)  # leftovers when the exe sat in a subfolder
    return dest / exe_name


def ensure_server(models_dir, build, fetch=download, check=check_server):
    dest = Path(models_dir) / "llama.cpp" / build["dir"]
    exe = dest / build["exe"]
    if exe.is_file():
        return exe
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        archives = []
        for asset in build["assets"]:
            archives.append(Path(tmp) / asset)
            fetch(RELEASE_URL + asset, archives[-1])
            with open(archives[-1], "rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            if digest != ASSET_SHA256.get(asset):
                raise RuntimeError(f"The llama.cpp download {asset} doesn't match its pinned SHA-256; nothing was installed. "
                                   "Run the workflow again; if it keeps failing, report it.")
        return install_archives(archives, dest, build["exe"], check)


GGUF_REPO = "mozophe/Qwen-Image-2.1-PE-MTP-GGUF"
QUANTS = ("Q8_0", "Q4_K_M")
KV_CACHE = ("f16", "q8_0")
# vision file per i2i model: one file, built from the i2i model; the standard Qwen3.5 vision part also works with heretic
MMPROJ = {"i2i": "qwen3.5_9b_qwen_image_2.1_pe_i2i.mmproj.bf16.gguf",
          "i2i - heretic": "qwen3.5_9b_qwen_image_2.1_pe_i2i.mmproj.bf16.gguf"}
HANDLE_ERROR = ("This is a llama.cpp prompt-enhancer handle; it only works with the Qwen-Image 2.1 Prompt Enhancer node.")


def gguf_names(model, quant):
    task = PE[model].get("task", model)
    base = f"qwen3.5_9b_qwen_image_2.1_pe_{task}{'_heretic' if PE[model].get('heretic') else ''}"
    return f"{base}.mtp.{quant}.gguf", MMPROJ.get(model)


def llm_roots():
    # models/LLM: where LLM nodes keep their models (ComfyUI core doesn't register it); a registered "LLM" folder wins
    if "LLM" in folder_paths.folder_names_and_paths:
        return list(folder_paths.get_folder_paths("LLM"))
    return [str(Path(folder_paths.models_dir) / "LLM")]


def gguf_folder():
    return Path(llm_roots()[0]) / "Qwen-Image-2.1-PE"


def find_file(name, roots):
    # a GGUF the user already has or moved anywhere under the LLM folders is used where it is
    return next((str(p) for r in roots if Path(r).is_dir() for p in Path(r).rglob(name) if p.is_file()), None)


def ensure_gguf(model, quant, fetch=None, folder=None, roots=None):
    fetch = fetch or (lambda repo, name, local_dir: hf_hub_download(repo, name, local_dir=local_dir))
    folder = Path(folder) if folder is not None else gguf_folder()
    roots = roots if roots is not None else llm_roots()
    paths = []
    for name in gguf_names(model, quant):
        if name is None:
            paths.append(None)
        elif (folder / name).is_file():
            paths.append(str(folder / name))
        elif (found := find_file(name, roots)) is not None:
            paths.append(found)
        else:
            folder.mkdir(parents=True, exist_ok=True)
            logging.info(f"Qwen-Image 2.1 PE llama.cpp setup: downloading {GGUF_REPO}/{name} -> {folder}")
            try:
                paths.append(str(fetch(GGUF_REPO, name, str(folder))))
            except Exception as e:
                raise RuntimeError(f"Downloading {GGUF_REPO}/{name} failed ({e}); run the workflow again to resume.") from e
    return tuple(paths)


class LlamaPE:
    # what the loader hands the generator through the CLIP socket for the llama.cpp backend
    __slots__ = ("pe_task", "model", "mmproj", "quant", "kv_cache", "exe", "unload", "vision_cpu")

    def __init__(self, pe_task, model, mmproj, quant, kv_cache, exe, unload=True, vision_cpu=False):
        self.pe_task, self.model, self.mmproj, self.quant, self.kv_cache, self.exe = pe_task, model, mmproj, quant, kv_cache, exe
        self.unload = unload  # free VRAM after each prompt; off keeps llama-server and ComfyUI's models loaded
        self.vision_cpu = vision_cpu  # keep the vision part in system RAM: about 1.2 GB less VRAM, slower image encoding

    def __getattr__(self, name):  # only reached for attributes a real CLIP has and this handle doesn't
        raise AttributeError(HANDLE_ERROR)


def load_llama(model, quant, kv_cache, unload=True, vision_cpu=False):
    build = select_build(platform.system(), platform.machine(), driver_version())
    exe = ensure_server(folder_paths.models_dir, build)
    gguf, mmproj = ensure_gguf(model, quant)
    return LlamaPE(PE[model].get("task", model), gguf, mmproj, quant, kv_cache, str(exe), unload, vision_cpu)
