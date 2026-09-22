# Qwen-Image 2.1 prompt enhancer: first-use setup of the MTP checkpoint, the official system prompts,
# and the official prompt_rewrite handling of input images and the JSON answer.
import json
import logging
import shutil
from pathlib import Path
from huggingface_hub import HfFileSystem, hf_hub_download
import folder_paths
import comfy.utils
from .graft import CHUNK, graft, mtp_tensors

# official: json_repair fixes nearly valid JSON (a trailing comma, an unescaped quote); optional, as there
try:
    import json_repair
except ImportError:
    json_repair = None

PE_REPO = "Comfy-Org/Qwen-Image-2.1"
MTP_REPO = "Qwen/Qwen3.5-9B"
PE = {
    "t2i": {"file": "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors", "prompt_repo": "Qwen/Qwen-Image-2.1-PE-T2I"},
    "i2i": {"file": "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors", "prompt_repo": "Qwen/Qwen-Image-2.1-PE-I2I"},
}
SYSTEM_PROMPTS = Path(__file__).parent / "system_prompts"


def find_text_encoder(name):
    for rel in folder_paths.get_filename_list("text_encoders"):
        if Path(rel).name == name:
            return folder_paths.get_full_path("text_encoders", rel)
    return None


def ensure_model(task):
    # path of the PE checkpoint with the base model's MTP head; the first call builds it by streaming the PE weights
    # (a local copy if present, else Comfy-Org's download) straight into the grafted file
    src_name = PE[task]["file"]
    name = src_name.replace(".safetensors", ".mtp.safetensors")
    dst = Path(folder_paths.get_folder_paths("text_encoders")[0]) / "Qwen-Image-2.1-PE" / name
    if dst.is_file():
        return str(dst)
    existing = find_text_encoder(name)
    if existing is not None:
        return existing

    dst.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Qwen-Image 2.1 PE setup: fetching the MTP head from {MTP_REPO} (~0.5 GB)")
    extra = mtp_tensors(MTP_REPO)
    pbar = comfy.utils.ProgressBar(1000)
    progress = lambda done, total: pbar.update_absolute(done * 1000 // total)
    local = find_text_encoder(src_name)
    if local is not None:
        logging.info(f"Qwen-Image 2.1 PE setup: grafting {local} -> {dst}")
        with open(local, "rb") as src:
            graft(src, dst, extra, progress)
    else:
        logging.info(f"Qwen-Image 2.1 PE setup: downloading {src_name} (9.5 GB) from {PE_REPO} -> {dst}")
        with HfFileSystem().open(f"{PE_REPO}/text_encoders/{src_name}", "rb", block_size=CHUNK) as src:
            graft(src, dst, extra, progress)
    logging.info(f"Qwen-Image 2.1 PE setup: done, {dst}")
    return str(dst)


def system_prompt(task):
    # the official system prompt (Qwen Research License), downloaded once into system_prompts/
    path = SYSTEM_PROMPTS / f"qwen_image_2.1_pe_{task}.txt"
    if not path.is_file():
        SYSTEM_PROMPTS.mkdir(exist_ok=True)
        shutil.copyfile(hf_hub_download(PE[task]["prompt_repo"], "system_prompt.txt"), path)
    return path.read_text(encoding="utf-8")


def fit_image(image, max_pixels=1024 * 1024):
    # official load_image: shrink to at most 1 MP (training's IMAGE_MAX_PIXELS) keeping the aspect ratio, never enlarge
    image = image[..., :3]
    h, w = image.shape[1:3]
    if w * h <= max_pixels:
        return image
    s = (max_pixels / (w * h)) ** 0.5
    return comfy.utils.common_upscale(image.movedim(-1, 1), max(1, int(w * s)), max(1, int(h * s)), "lanczos", "disabled").movedim(1, -1)


def balanced_spans(answer):
    # official _balanced_spans: every balanced top-level {...} in order, skipping braces inside JSON strings
    spans = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(answer):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(answer[start:i + 1])
    return spans


def as_obj(candidate):
    # official _as_obj: strict json first, then json_repair if it is installed
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        if json_repair is None:
            return None
        obj = json_repair.repair_json(candidate, return_objects=True)
        if isinstance(obj, list):
            obj = obj[0] if obj else None
    return obj if isinstance(obj, dict) else None


def parse_answer(answer):
    # official parse_answer: the last JSON object in the answer that has a rewritten_prompt -> (prompt, wh_ratio, ratio_follow),
    # None when there is none (official then falls back to the raw answer with parse_ok false)
    answer = (answer or "").strip()
    for candidate in reversed(balanced_spans(answer)):
        obj = as_obj(candidate)
        if obj is None:
            continue
        rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")  # some training runs used the misspelling
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue
        return rewritten.strip(), str(obj.get("wh_ratio") or "").strip(), str(obj.get("ratio_follow") or "").strip()
    return None
