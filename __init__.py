import logging
from typing_extensions import override
import folder_paths
import comfy.sd
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_textgen import TextGenerate
from .mtp import MTPClip, pe_prompt
from .pe import PE, ensure_model, system_prompt

# official prompt_rewrite/pe_core.py profiles: shared sampling, per-task presence penalty and token cap
PRESETS = {
    "Qwen-Image 2.1 PE (edit)": {"task": "edit", "max_length": 24000, "presence_penalty": 0.0},
    "Qwen-Image 2.1 PE (t2i)": {"task": "t2i", "max_length": 16256, "presence_penalty": 1.5},
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
        io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
        io.Boolean.Input("thinking", default=True, tooltip="The PE models were trained with thinking on and degrade without it."),
    ]


class TextGenerateQwen35MTP(TextGenerate):
    @classmethod
    def define_schema(cls):
        parent = super().define_schema()
        inp = {i.id: i for i in parent.inputs}
        presets = [io.DynamicCombo.Option("none", [inp["max_length"], inp["sampling_mode"], inp["thinking"], inp["use_default_template"]])]
        presets += [io.DynamicCombo.Option(name, preset_inputs(p["max_length"], p["presence_penalty"])) for name, p in PRESETS.items()]
        return io.Schema(
            node_id="TextGenerateQwen35MTP",
            display_name="Generate Text (Qwen3.5 MTP)",
            category=parent.category,
            description="Generate Text with MTP speculative decoding also for Qwen3.5 image prompts, plus official Qwen-Image 2.1 PE presets.",
            search_aliases=["LLM", "qwen", "mtp", "speculative", "prompt enhance"],
            inputs=[inp["clip"], inp["prompt"], inp["image"], inp["video"], inp["audio"],
                    io.DynamicCombo.Input("preset", options=presets, tooltip="PE presets add the official system prompt and sampling defaults."),
                    inp["mtp"]],
            outputs=parent.outputs,
        )

    @classmethod
    def execute(cls, clip, prompt, preset, image=None, video=None, audio=None, mtp="auto") -> io.NodeOutput:
        clip = MTPClip(clip)
        name = preset["preset"]
        if name == "none":
            return super().execute(clip, prompt, preset["max_length"], preset["sampling_mode"], image=image, thinking=preset.get("thinking", False),
                                   use_default_template=preset.get("use_default_template", True), video=video, audio=audio, mtp=mtp)

        if not preset["thinking"]:
            logging.warning(f"{name}: thinking is off; the PE models were trained with thinking on and degrade without it.")
        text = pe_prompt(system_prompt(PRESETS[name]["task"]), prompt, 0 if image is None else image.shape[0], preset["thinking"])
        tokens = clip.tokenize(text, image=image, min_length=1, video=video, audio=audio)
        ids = clip.generate(tokens, do_sample=True, max_length=preset["max_length"], temperature=preset["temperature"], top_k=preset["top_k"],
                            top_p=preset["top_p"], min_p=preset["min_p"], repetition_penalty=preset["repetition_penalty"], seed=preset["seed"],
                            presence_penalty=preset["presence_penalty"], mtp=False if mtp == "off" else (True if mtp == "auto" else int(mtp)))
        return io.NodeOutput(clip.decode(ids))


class LoadQwenImage21PE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadQwenImage21PEMTP",
            display_name="Load Qwen-Image 2.1 PE (MTP)",
            category="loaders",
            description="Loads the Qwen-Image 2.1 prompt enhancer with an MTP head. The first run downloads it (9.5 GB) and prepares it; later runs load it directly.",
            inputs=[io.Combo.Input("model", options=list(PE), tooltip="edit: image-edit prompt enhancer (use with an image). t2i: text-to-image prompt enhancer.")],
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
