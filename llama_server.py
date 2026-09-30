# llama.cpp backend runtime: one llama-server process that sleeps between generations (its VRAM freed, the process kept),
# restarted when the model, vision file, KV cache type, MTP setting or needed context changes
import atexit
import base64
import contextlib
import ctypes
import io
import json
import logging
import math
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import folder_paths
import comfy.model_management
import comfy.utils
from .pe import fit_image

OOM_MARKERS = ("out of memory", "cudamalloc failed", "failed to allocate")


def image_tokens(h, w):
    # Qwen3.5 vision: 16 px patches merged 2x2 -> one token per 32x32 px, plus <|vision_start|>/<|vision_end|>
    return math.ceil(h / 32) * math.ceil(w / 32) + 2


def estimate_text_tokens(text):
    # the PE system prompts run at ~4 chars/token; 3 leaves room, and /tokenize corrects it before generating
    return max(1, math.ceil(len(text) / 3))


def context_size(prompt_tokens, max_length):
    return math.ceil((prompt_tokens + max_length) / 4096) * 4096


def mtp_args(mtp):
    if mtp == "off":
        return []
    return ["--spec-type", "draft-mtp"] + ([] if mtp == "auto" else ["--spec-draft-n-max", str(int(mtp))])


def server_key(handle, mtp):
    return (handle.exe, handle.model, handle.mmproj, handle.kv_cache, mtp)


def server_command(handle, ctx, mtp, load_mode, port):
    cmd = [handle.exe, "-m", handle.model, "-ngl", "99", "-fa", "on", "--fit", "off", "-np", "1", "-c", str(ctx),
           "-ctk", handle.kv_cache, "-ctv", handle.kv_cache, "--sleep-idle-seconds", "1", "--load-mode", load_mode,
           "--host", "127.0.0.1", "--port", str(port), "--no-webui"]
    if handle.mmproj:
        cmd += ["--mmproj", handle.mmproj]
    return cmd + mtp_args(mtp)


def needs_restart(alive, running_key, running_ctx, key, ctx):
    return not alive or running_key != key or running_ctx < ctx


def insert_markers(text, marker, n):
    # the server's random media marker, once per image at the start of the user turn; mtmd wraps each in
    # <|vision_start|>...<|vision_end|> itself (pe_prompt is called with 0 images for this path)
    return text.replace("<|im_start|>user\n", "<|im_start|>user\n" + marker * n, 1) if n else text


def png_b64(image):
    arr = (image[0, ..., :3].clamp(0, 1) * 255).round().to(dtype=torch.uint8).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr), "RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def completion_body(prompt, preset, seed, media):
    return {"prompt": {"prompt_string": prompt, "multimodal_data": media} if media else prompt,
            # llama.cpp seeds are uint32 and 0xFFFFFFFF means random
            "n_predict": preset["max_length"], "seed": seed % 0xFFFFFFFF, "cache_prompt": False, "stream": True, "return_tokens": True,
            "temperature": preset["temperature"], "top_k": preset["top_k"], "top_p": preset["top_p"], "min_p": preset["min_p"],
            "repeat_penalty": preset["repetition_penalty"], "presence_penalty": preset["presence_penalty"]}


def read_stream(lines, on_tokens, interrupted):
    # llama-server SSE: "data: {json}" lines; the last chunk has "stop": true and the timings
    content = []
    for raw in lines:
        if interrupted():
            return None
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data: "):
            continue
        chunk = json.loads(line[6:])
        if "error" in chunk:
            err = chunk["error"]
            raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
        content.append(chunk.get("content", ""))
        on_tokens(len(chunk.get("tokens") or []))
        if chunk.get("stop"):
            return "".join(content), chunk
    raise RuntimeError("llama-server closed the stream before the answer ended.")


def failure_message(what, log_tail, log_path):
    advice = ("Out of VRAM: try quant Q4_K_M, kv_cache: q8_0, a lower max_length, or fewer images."
              if any(m in log_tail.lower() for m in OOM_MARKERS) else f"Full log: {log_path}")
    return f"llama.cpp backend: {what}. {advice}\nLast lines of the llama-server log:\n{log_tail}"


if os.name == "nt":
    from ctypes import wintypes

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _tie_to_comfyui(proc):
    # Windows: a job object that kills llama-server when ComfyUI's handle to it closes (ComfyUI exits or crashes).
    # Returns the job handle, which must stay open while the server runs. Linux uses PR_SET_PDEATHSIG instead.
    if os.name != "nt":
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = _ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))  # JobObjectExtendedLimitInformation
    k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(proc._handle))
    return job


def _pdeathsig():
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    def __init__(self):
        self.proc, self.job, self.key, self.ctx, self.port, self.marker = None, None, None, 0, None, None
        self.log_path = Path(folder_paths.get_temp_directory()) / "llama-server.log"

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, n=20):
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def fail(self, what):
        return RuntimeError(failure_message(what, self.log_tail(), self.log_path))

    def ensure(self, handle, ctx, mtp):
        key = server_key(handle, mtp)
        if needs_restart(self.alive(), self.key, self.ctx, key, ctx):
            self.stop()
            self.start(handle, ctx, mtp, key)

    def start(self, handle, ctx, mtp, key):
        try:  # ComfyUI's own fast-disk policy (--fast-disk / --disable-fast-disk included); older ComfyUI lacks comfy.storage
            import comfy.storage
            fast = comfy.storage.model_fast_disk([handle.model])
        except (ImportError, AttributeError):
            fast = False
        load_mode = "dio" if fast else "mmap"
        self.port = _free_port()
        cmd = server_command(handle, ctx, mtp, load_mode, self.port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        if os.name != "nt":
            env["LD_LIBRARY_PATH"] = str(Path(handle.exe).parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        logging.info(f"Qwen-Image 2.1 PE llama.cpp: starting {' '.join(cmd)}")
        with open(self.log_path, "w", encoding="utf-8") as log:
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                         preexec_fn=_pdeathsig if os.name != "nt" else None)
        self.job = _tie_to_comfyui(self.proc)
        deadline = time.monotonic() + 600
        while True:
            if self.proc.poll() is not None:
                raise self.fail(f"llama-server exited with code {self.proc.returncode} while starting")
            try:
                if self.get("/health").get("status") == "ok":
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                self.stop()
                raise self.fail("llama-server did not start within 600 s")
            time.sleep(0.25)
        self.key, self.ctx = key, ctx
        self.marker = self.get("/props").get("media_marker")

    def stop(self):
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.job is not None:
            ctypes.WinDLL("kernel32").CloseHandle(wintypes.HANDLE(self.job))
        self.proc, self.job, self.key, self.ctx, self.marker = None, None, None, 0, None

    def _request(self, path, body=None, timeout=600):
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data,
                                                             {"Content-Type": "application/json"}), timeout=timeout)

    def get(self, path):
        with self._request(path, timeout=5) as r:
            return json.load(r)

    def post(self, path, body):
        try:
            with self._request(path, body) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise self.fail(f"{path} failed: {e.read().decode('utf-8', 'replace')[:500]}") from e
        except OSError as e:
            raise self.fail(f"{path} failed ({e})") from e

    @contextlib.contextmanager
    def stream(self, body):
        try:
            with self._request("/completion", body) as r:
                yield r
        except urllib.error.HTTPError as e:
            raise self.fail(f"generation failed: {e.read().decode('utf-8', 'replace')[:500]}") from e
        except TimeoutError as e:
            raise self.fail("llama-server did not answer within 600 s") from e
        except OSError as e:
            if not self.alive():
                raise self.fail(f"llama-server exited with code {self.proc.returncode} while generating") from e
            raise self.fail(f"the connection to llama-server failed ({e})") from e

    def wait_asleep(self, timeout=60):
        # llama-server frees its VRAM 1 s after the last request; the next ComfyUI node needs that VRAM
        deadline = time.monotonic() + timeout
        while self.alive() and time.monotonic() < deadline:
            try:
                if self.get("/props").get("is_sleeping"):
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.1)


SERVER = LlamaServer()
atexit.register(SERVER.stop)


def generate(handle, text, images, preset, seed, mtp):
    mm = comfy.model_management
    fitted = [fit_image(im) for im in images]
    img_tokens = sum(image_tokens(im.shape[1], im.shape[2]) for im in fitted)
    mm.unload_all_models()  # llama-server's VRAM is outside ComfyUI's memory manager
    mm.soft_empty_cache()
    SERVER.ensure(handle, context_size(estimate_text_tokens(text) + img_tokens, preset["max_length"]), mtp)
    exact = len(SERVER.post("/tokenize", {"content": text})["tokens"]) + img_tokens
    SERVER.ensure(handle, context_size(exact, preset["max_length"]), mtp)  # restarts only if the estimate was short
    body = completion_body(insert_markers(text, SERVER.marker, len(fitted)), preset, seed, [png_b64(im) for im in fitted])
    pbar = comfy.utils.ProgressBar(preset["max_length"])
    done = [0]

    def on_tokens(n):
        done[0] += n
        pbar.update_absolute(min(done[0], preset["max_length"]))

    # b11160 loses a request that arrives while the server is falling asleep (its queue only wakes for a request that finds
    # it already asleep), so send only to a sleeping server: free after a generation, one extra load after a start
    SERVER.wait_asleep()
    try:
        with SERVER.stream(body) as lines:
            result = read_stream(lines, on_tokens, mm.processing_interrupted)
    except RuntimeError as e:
        if not str(e).startswith("llama.cpp backend:"):  # a server error chunk from read_stream
            raise SERVER.fail(str(e)) from e
        raise
    SERVER.wait_asleep()
    if result is None:  # closing the stream above made the server cancel the task
        raise mm.InterruptProcessingException()
    return result[0]
