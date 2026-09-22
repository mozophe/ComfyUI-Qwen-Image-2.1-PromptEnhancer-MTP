from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_textgen import TextGenerate
from .mtp import MTPClip


class TextGenerateQwen35MTP(TextGenerate):
    @classmethod
    def define_schema(cls):
        parent = super().define_schema()
        return io.Schema(
            node_id="TextGenerateQwen35MTP",
            display_name="Generate Text (Qwen3.5 MTP)",
            category=parent.category,
            description="Generate Text with MTP speculative decoding also for Qwen3.5 image prompts.",
            search_aliases=["LLM", "qwen", "mtp", "speculative"],
            inputs=parent.inputs,
            outputs=parent.outputs,
        )

    @classmethod
    def execute(cls, clip, **kwargs) -> io.NodeOutput:
        return super().execute(MTPClip(clip), **kwargs)


class Qwen35MTPExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TextGenerateQwen35MTP]


async def comfy_entrypoint() -> Qwen35MTPExtension:
    return Qwen35MTPExtension()
