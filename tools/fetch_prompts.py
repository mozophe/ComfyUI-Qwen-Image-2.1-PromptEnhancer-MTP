# Download the official Qwen-Image 2.1 PE system prompts into ../system_prompts (not redistributed with this repo).
# usage: python fetch_prompts.py
from pathlib import Path
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parents[1] / "system_prompts"
OUT.mkdir(exist_ok=True)
for repo, name in (("Qwen/Qwen-Image-2.1-PE-I2I", "qwen_image_2.1_pe_edit.txt"), ("Qwen/Qwen-Image-2.1-PE-T2I", "qwen_image_2.1_pe_t2i.txt")):
    text = Path(hf_hub_download(repo, "system_prompt.txt")).read_text(encoding="utf-8")
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"{repo} -> {OUT / name} ({len(text)} chars)")
print("These prompts are under the Qwen Research License (non-commercial): https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/blob/main/LICENSE")
