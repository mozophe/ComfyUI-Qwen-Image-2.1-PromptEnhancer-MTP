# llama.cpp backend runtime: one llama-server process that sleeps between generations (its VRAM freed, the process kept),
# restarted when the model, vision file, KV cache type, MTP setting or needed context changes
import math


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
