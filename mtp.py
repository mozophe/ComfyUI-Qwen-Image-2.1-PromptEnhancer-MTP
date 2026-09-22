import logging
import torch
import comfy.model_management
import comfy.ops
from comfy.text_encoders.qwen35 import Qwen35
from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids


def pe_prompt(system_prompt, text, images, thinking):
    # Qwen3.5 chat_template.jinja layout: system turn, user turn with image blocks before the text,
    # then the generation prompt (thinking opens a <think> block, otherwise an empty closed one)
    vision = "<|vision_start|><|image_pad|><|vision_end|>" * images
    think = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{vision}{text}<|im_end|>\n<|im_start|>assistant\n{think}"


def strip_thinking(text):
    # the answer after the thinking block, as prompt_rewrite/pe_core.py split_thinking does
    if "</think>" in text:
        return text.partition("</think>")[2].strip()
    if "<think>" in text:
        return ""
    return text.strip()


def mrope_table(position_ids, cap):
    # sequence position -> MRoPE ids: the prompt keeps its 3D ids, generated tokens continue
    # text positions on all three axes from where the core non-MTP loop would (max of last column + 1)
    start = int(position_ids[:, -1].max()) + 1
    tail = torch.arange(start, start + cap - position_ids.shape[1], device=position_ids.device).expand(3, -1)
    return torch.cat([position_ids.to(tail.dtype), tail], dim=1)


def mrope_freqs(compute_freqs_cis, table):
    # 3-row MRoPE freqs come out as (1, seq, d); the decode paths slice the 1-row layout (1, 1, seq, d)
    return lambda position_ids, device: tuple(t.unsqueeze(1) for t in compute_freqs_cis(table[:, position_ids[0].long()], device))


def qwen35_encoder(clip):
    # the Qwen3.5 clip model inside a CLIP, else None
    cond = clip.cond_stage_model
    inner = getattr(cond, getattr(cond, "clip", ""), None)
    return inner if isinstance(getattr(inner, "transformer", None), Qwen35) else None


def has_image(tokens):
    return any(isinstance(t[0], dict) and t[0].get("type") == "image" for batch in tokens.values() for row in batch for t in row)


def generate_image_mtp(clip, inner, tokens, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.95, min_p=0.0,
                       repetition_penalty=1.0, seed=None, presence_penalty=0.0, mtp=True):
    # setup mirrors comfy.sd.CLIP.generate; sampling/depth mirror Qwen35.generate
    cond = clip.cond_stage_model
    cond.reset_clip_options()
    clip.load_model(tokens)
    device = clip.patcher.load_device
    cond.set_clip_options({"layer": None})
    cond.set_clip_options({"execution_device": device})
    sampling = None
    if do_sample and temperature != 0.0:
        sampling = {"temperature": temperature, "top_k": top_k, "top_p": top_p, "min_p": min_p, "repetition_penalty": repetition_penalty,
                    "presence_penalty": presence_penalty or 0.0, "seed": seed if seed is not None else 42}
    fixed_depth = None if mtp is True else max(2, min(5, int(mtp)))
    with comfy.model_management.cuda_device_context(device), comfy.ops.use_quantized_matmul(cond, device):
        rows = next(iter(tokens.values()))
        embeds, _, _, embeds_info = inner.process_tokens([[t[0] for t in row] for row in rows], inner.execution_device)
        position_ids = qwen2vl_mrope_position_ids(embeds_info, embeds.shape[1], embeds.device)
        model = inner.transformer
        table = mrope_table(position_ids, embeds.shape[1] + max_length + 16)  # _generate_mtp caps at L + max_length + 7
        model.model.compute_freqs_cis = mrope_freqs(model.model.compute_freqs_cis, table)
        try:
            return model._generate_mtp(embeds, max_length, None, sampling=sampling, fixed_depth=fixed_depth)
        finally:
            del model.model.compute_freqs_cis  # drop the instance override, back to the class method


class MTPClip:
    # CLIP proxy: image prompts on a Qwen3.5 MTP model take the MRoPE-aware MTP path, all else is the real CLIP
    def __init__(self, clip):
        self.clip = clip

    def __getattr__(self, name):
        return getattr(self.clip, name)

    def generate(self, tokens, mtp=True, **kwargs):
        inner = qwen35_encoder(self.clip) if mtp is not False else None
        if inner is not None and inner.transformer.mtp is None:
            logging.warning("mtp is on but this Qwen3.5 checkpoint has no MTP head, so it runs without MTP (slower). "
                            "For Qwen-Image 2.1 PE, load it with the 'Qwen-Image 2.1 PE Loader (MTP)' node.")
            inner = None
        if inner is None or not has_image(tokens):
            return self.clip.generate(tokens, mtp=mtp, **kwargs)
        return generate_image_mtp(self.clip, inner, tokens, mtp=mtp, **kwargs)
