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


def parse_answer(answer):
    # official parse_answer: the last JSON object in the answer that has a rewritten_prompt -> (prompt, wh_ratio, ratio_follow),
    # None when there is none. Skips official's optional json_repair: near-JSON falls back to the raw answer.
    decoder = json.JSONDecoder()
    for i in reversed([i for i, c in enumerate(answer) if c == "{"]):
        try:
            obj = decoder.raw_decode(answer, i)[0]
        except ValueError:
            continue
        text = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")  # some training runs used the misspelling
        if isinstance(text, str) and text.strip():
            return text.strip(), str(obj.get("wh_ratio") or "").strip(), str(obj.get("ratio_follow") or "").strip()
    return None
