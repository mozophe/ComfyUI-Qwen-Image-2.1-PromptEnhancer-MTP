import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1])), str(REPO)]
from mtp import pe_prompt

# Qwen3.5 chat_template.jinja: system turn, user turn with images before text, generation prompt
V = "<|vision_start|><|image_pad|><|vision_end|>"
assert pe_prompt("SYS", "edit it", 1, True) == \
    f"<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\n{V}edit it<|im_end|>\n<|im_start|>assistant\n<think>\n"
assert pe_prompt("SYS", "a cat", 0, False) == \
    "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
assert pe_prompt("S", "t", 2, True).count(V) == 2
print("ok")
