# Qwen-Image 2.1 prompt enhancer: first-use setup of the MTP checkpoint, the official system prompts,
# and the official prompt_rewrite handling of input images and the JSON answer.
import json
import logging
import shutil
from pathlib import Path
from huggingface_hub import HfFileSystem, hf_hub_download
import folder_paths
import comfy.model_management
import comfy.utils
import torch
from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight, quantize_int8_rowwise
from .graft import CHUNK, graft, mtp_tensors, read_header, write

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
    # abliterated bf16 fine-tunes, quantized on first use with the Comfy-Org checkpoint of the same task as the recipe
    "t2i - heretic": {"task": "t2i", "heretic": "pottokao/Qwen-Image-2.1-PE-T2I-Heretic"},
    "i2i - heretic": {"task": "i2i", "heretic": "darrellbest/Qwen-Image-2.1-PE-I2I-Heretic"},
}
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
SYSTEM_PROMPTS = Path(__file__).parent / "system_prompts"


def find_text_encoder(name):
    for rel in folder_paths.get_filename_list("text_encoders"):
        if Path(rel).name == name:
            return folder_paths.get_full_path("text_encoders", rel)
    return None


def ensure_model(model):
    # path of the PE checkpoint with the base model's MTP head; the first call builds it by streaming the PE weights
    # (a local copy if present, else Comfy-Org's download) straight into the grafted file. A heretic model streams its
    # bf16 weights instead, quantized to match the Comfy-Org checkpoint (only its header and quant markers are read)
    task = PE[model].get("task", model)
    heretic = PE[model].get("heretic")
    src_name = PE[task]["file"]
    name = (src_name.replace(f"_{task}.", f"_{task}_heretic.") if heretic else src_name).replace(".safetensors", ".mtp.safetensors")
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
    src = open(local, "rb") if local is not None else HfFileSystem().open(f"{PE_REPO}/text_encoders/{src_name}", "rb", block_size=CHUNK)
    with src:
        if heretic:
            logging.info(f"Qwen-Image 2.1 PE setup: downloading {heretic} (19 GB, bf16) and quantizing it like {src_name} -> {dst}")
            index = json.load(open(hf_hub_download(heretic, "model.safetensors.index.json")))["weight_map"]
            open_shard = lambda shard: HfFileSystem().open(f"{heretic}/{shard}", "rb", block_size=CHUNK)
            build_heretic(src, index, open_shard, dst, extra, progress, comfy.model_management.get_torch_device())
        elif local is not None:
            logging.info(f"Qwen-Image 2.1 PE setup: grafting {local} -> {dst}")
            graft(src, dst, extra, progress)
        else:
            logging.info(f"Qwen-Image 2.1 PE setup: downloading {src_name} (9.5 GB) from {PE_REPO} -> {dst}")
            graft(src, dst, extra, progress)
    logging.info(f"Qwen-Image 2.1 PE setup: done, {dst}")
    return str(dst)


def build_heretic(ref, index, open_shard, dst, extra, progress, device):
    # write the bf16 shards (`index`: tensor -> shard) as ref's int8 checkpoint: same tensors, dtypes and shapes, each
    # ref.comfy_quant layer quantized with its recipe by ComfyUI's own quantizer; resumable like graft
    n, ref_header = read_header(ref)
    ref_header.pop("__metadata__", None)
    size = lambda k: ref_header[k]["data_offsets"][1] - ref_header[k]["data_offsets"][0]
    markers = {}
    for k in sorted((k for k in ref_header if k.endswith(".comfy_quant")), key=lambda k: ref_header[k]["data_offsets"]):
        ref.seek(8 + n + ref_header[k]["data_offsets"][0])
        markers[k[:-len(".comfy_quant")]] = ref.read(size(k))
    wanted = set(ref_header) - {m + s for m in markers for s in (".weight_scale", ".comfy_quant")}
    if set(index) != wanted:
        raise ValueError(f"source and reference tensors differ: {sorted(set(index) ^ wanted)[:5]}")

    shards = {}
    for shard in sorted(set(index.values())):
        with open_shard(shard) as f:
            shards[shard] = read_header(f)
    # ref's tensors grouped per source tensor (a quantized layer's weight, scale and marker together), in source order
    groups = [[k] if k[:-len(".weight")] not in markers else [k, k + "_scale", k[:-len("weight")] + "comfy_quant"]
              for k in sorted(index, key=lambda k: (index[k], shards[index[k]][1][k]["data_offsets"]))]
    header, offset = {}, 0
    for k in (k for g in groups for k in g):
        header[k] = {"dtype": ref_header[k]["dtype"], "shape": ref_header[k]["shape"], "data_offsets": [offset, offset + size(k)]}
        offset += size(k)

    def data(done):
        for shard in sorted(shards):
            with open_shard(shard) as f:
                n, h = shards[shard]
                for g in groups:
                    k = g[0]
                    start, end = header[k]["data_offsets"][0], header[g[-1]]["data_offsets"][1]
                    if index[k] != shard or end <= done:
                        continue
                    f.seek(8 + n + h[k]["data_offsets"][0])
                    raw = f.read(h[k]["data_offsets"][1] - h[k]["data_offsets"][0])
                    if h[k]["shape"] != ref_header[k]["shape"]:
                        raise ValueError(f"{k}: shape {h[k]['shape']}, reference {ref_header[k]['shape']}")
                    if len(g) > 1:
                        conf = json.loads(markers[k[:-len(".weight")]])
                        w = torch.frombuffer(bytearray(raw), dtype=DTYPES[h[k]["dtype"]]).reshape(h[k]["shape"])
                        # ComfyUI's torch quantizer on whole fp32 tensors reproduces Comfy-Org's int8 bit for bit on CUDA;
                        # its fused kernel, the CPU, or row chunks each differ in about 1 value in 10M
                        w = w.to(device, torch.float32)
                        q, scale = quantize_int8_convrot_weight(w, conf["convrot_groupsize"]) if conf.get("convrot") else quantize_int8_rowwise(w)
                        raw = b"".join([q.cpu().numpy().tobytes(), scale.float().cpu().numpy().tobytes(), markers[k[:-len(".weight")]]])
                    elif h[k]["dtype"] != ref_header[k]["dtype"]:
                        raise ValueError(f"{k}: dtype {h[k]['dtype']}, reference {ref_header[k]['dtype']}")
                    if len(raw) != end - start:
                        raise ValueError(f"{k}: {len(raw)} bytes, expected {end - start}")
                    yield raw[max(0, done - start):]
    write(dst, header, extra, data, progress)


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
