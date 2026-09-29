# llama.cpp backend runtime: one llama-server process that sleeps between generations (its VRAM freed, the process kept),
# restarted when the model, vision file, KV cache type, MTP setting or needed context changes
import base64
import io
import json
import math
import numpy as np
import torch
from PIL import Image


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
            "n_predict": preset["max_length"], "seed": seed, "cache_prompt": False, "stream": True, "return_tokens": True,
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
