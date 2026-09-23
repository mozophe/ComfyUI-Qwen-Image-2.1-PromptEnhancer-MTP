import logging
import torch
from typing_extensions import override
import folder_paths
import comfy.sd
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_textgen import TextGenerate
from .mtp import MTPClip, pe_prompt, split_thinking
from .pe import PE, ensure_model, fit_image, parse_answer, system_prompt

# official prompt_rewrite/pe_core.py profiles: shared sampling, per-task presence penalty and token cap
PRESETS = {
    "Qwen-Image 2.1 PE (t2i)": {"task": "t2i", "max_length": 16256, "presence_penalty": 1.5},
    "Qwen-Image 2.1 PE (i2i)": {"task": "i2i", "max_length": 24000, "presence_penalty": 0.0},
}


def preset_inputs(max_length, presence_penalty):
    return [
        io.Int.Input("max_length", default=max_length, min=1, max=32768),
        io.Float.Input("temperature", default=1.0, min=0.01, max=2.0, step=0.000001),
        io.Int.Input("top_k", default=20, min=0, max=1000),
        io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
        io.Float.Input("min_p", default=0.0, min=0.0, max=1.0, step=0.01),
        io.Float.Input("repetition_penalty", default=1.0, min=0.0, max=5.0, step=0.01),
        io.Float.Input("presence_penalty", default=presence_penalty, min=0.0, max=5.0, step=0.01),
        io.Boolean.Input("thinking", default=True, tooltip="The PE models were trained with thinking on and degrade without it."),
    ]


class TextGenerateQwen35MTP(TextGenerate):
    @classmethod
    def define_schema(cls):
        parent = super().define_schema()
        inp = {i.id: i for i in parent.inputs}
        presets = [io.DynamicCombo.Option(name, preset_inputs(p["max_length"], p["presence_penalty"])) for name, p in PRESETS.items()]
        presets.append(io.DynamicCombo.Option("none", [inp["max_length"], inp["sampling_mode"], inp["thinking"], inp["use_default_template"]]))
        return io.Schema(
            node_id="TextGenerateQwen35MTP",
            display_name="Qwen-Image 2.1 Prompt Enhancer (MTP)",
            category=parent.category,
            description="Generate Text with MTP speculative decoding also for Qwen3.5 image prompts, plus official Qwen-Image 2.1 PE presets.",
            search_aliases=["LLM", "qwen", "mtp", "speculative", "prompt enhance"],
            inputs=[inp["clip"], inp["prompt"],
                    io.Autogrow.Input("images", template=io.Autogrow.TemplateNames(io.Image.Input("image"), names=[f"image_{i}" for i in range(1, 11)], min=0),
                                      tooltip="Input images, in order: the model calls them <image1>, <image2>, ... Sizes may differ; each is shrunk to at most 1 MP."),
                    io.DynamicCombo.Input("preset", options=presets, tooltip="PE presets add the official system prompt and sampling defaults."),
                    # outside the preset: the frontend duplicates a control_after_generate widget nested in a DynamicCombo on
                    # every rebuild, so saved widget values shift on reload
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True,
                                 tooltip="Sampling seed for the PE presets. The 'none' preset uses its own seed."),
                    inp["mtp"]],
            # the official prompt_rewrite output record's answer fields
            outputs=[io.String.Output("positive_prompt", display_name="positive_prompt", tooltip="The rewritten prompt from the answer's JSON, for the text encoder. The whole answer if it has none."),
                     io.String.Output("negative_prompt", display_name="negative_prompt", tooltip="Always empty: neither PE model writes one. Kept for workflows that expect the slot."),
                     io.String.Output("thinking", display_name="thinking", tooltip="The reasoning, without the answer."),
                     io.String.Output("wh_ratio", display_name="wh_ratio", tooltip="The aspect ratio the model chose, e.g. 16:9. Empty when it follows an input image."),
                     io.String.Output("ratio_follow", display_name="ratio_follow", tooltip="i2i: the input image whose aspect ratio the output keeps, e.g. <image1>. Empty otherwise."),
                     io.Boolean.Output("parse_ok", display_name="parse_ok", tooltip="False when the answer had no valid JSON; positive_prompt is then the whole answer.")],
        )

    @classmethod
    def execute(cls, clip, prompt, preset, seed=0, images=None, mtp="auto") -> io.NodeOutput:
        # connected inputs in socket order, each batch split into single images
        images = [im[i:i + 1] for _, im in sorted((images or {}).items(), key=lambda kv: int(kv[0].rsplit("_", 1)[1])) if im is not None
                  for i in range(im.shape[0])]
        text = cls.generate_text(MTPClip(clip), prompt, preset, seed, images, mtp)
        thinking, answer = split_thinking(text)
        parsed = parse_answer(answer)
        if parsed is None:
            logging.warning("Qwen-Image 2.1 PE: the answer has no JSON rewritten_prompt; positive_prompt is the whole answer.")
        positive, wh_ratio, ratio_follow = parsed or (answer, "", "")
        if PRESETS.get(preset["preset"], {}).get("task") == "t2i":
            ratio_follow = ""  # official: t2i has no ratio_follow
        return io.NodeOutput(positive, "", thinking, wh_ratio, ratio_follow, parsed is not None)

    @classmethod
    def generate_text(cls, clip, prompt, preset, seed, images, mtp):
        name = preset["preset"]
        if name == "none":
            if len({im.shape[1:] for im in images}) > 1:
                raise ValueError("The 'none' preset takes images of one size; use a PE preset for differently sized images.")
            return super().execute(clip, prompt, preset["max_length"], preset["sampling_mode"], image=torch.cat(images) if images else None,
                                   thinking=preset.get("thinking", False), use_default_template=preset.get("use_default_template", True),
                                   mtp=mtp).args[0]

        task = PRESETS[name]["task"]
        # official resolve_image_paths refuses these rather than run the wrong experiment
        if task == "t2i" and images:
            raise ValueError(f"{name} takes no images; disconnect them or use the i2i preset and model.")
        if task == "i2i" and not images:
            raise ValueError(f"{name} needs at least one image.")
        if not preset["thinking"]:
            logging.warning(f"{name}: thinking is off; the PE models were trained with thinking on and degrade without it.")
        text = pe_prompt(system_prompt(task), prompt, len(images), preset["thinking"])
        tokens = clip.tokenize(text, images=[fit_image(im) for im in images], min_length=1)
        ids = clip.generate(tokens, do_sample=True, max_length=preset["max_length"], temperature=preset["temperature"], top_k=preset["top_k"],
                            top_p=preset["top_p"], min_p=preset["min_p"], repetition_penalty=preset["repetition_penalty"], seed=seed,
                            presence_penalty=preset["presence_penalty"], mtp=False if mtp == "off" else (True if mtp == "auto" else int(mtp)))
        return clip.decode(ids)


class LoadQwenImage21PE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadQwenImage21PEMTP",
            display_name="Qwen-Image 2.1 PE Loader (MTP)",
            category="loaders",
            description="Loads the Qwen-Image 2.1 prompt enhancer with an MTP head. The first run downloads it (9.5 GB) and prepares it; later runs load it directly.",
            inputs=[io.Combo.Input("model", options=list(PE), tooltip="t2i: text-to-image prompt enhancer. i2i: image-edit prompt enhancer (use with an image).")],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        clip = comfy.sd.load_clip(ckpt_paths=[ensure_model(model)], embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                  clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
        return io.NodeOutput(clip)


class Qwen35MTPExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TextGenerateQwen35MTP, LoadQwenImage21PE]


async def comfy_entrypoint() -> Qwen35MTPExtension:
    return Qwen35MTPExtension()
