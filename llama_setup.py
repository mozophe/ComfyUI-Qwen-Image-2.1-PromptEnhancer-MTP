# llama.cpp backend setup: pick and install the pinned llama-server build for this NVIDIA driver, download the hosted
# GGUFs, and hand the loader a LlamaPE handle
import subprocess

LLAMA_TAG = "b11160"
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


def driver_version():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_driver_version(out.stdout) if out.returncode == 0 else None


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
