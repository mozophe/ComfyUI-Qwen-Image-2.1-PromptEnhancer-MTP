import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1])), str(REPO)]
from mtp import strip_thinking

# cases follow the official prompt_rewrite/pe_core.py split_thinking
assert strip_thinking('reasoning...\n</think>\n\n{"rewritten_prompt": "x"}') == '{"rewritten_prompt": "x"}'  # generation started inside <think>
assert strip_thinking("<think>\nr\n</think>\n\nanswer") == "answer"
assert strip_thinking("a</think>b</think>c") == "b</think>c"  # only the first close ends the thinking
assert strip_thinking("<think>\nunfinished reasoning") == ""  # cut off while thinking: no answer yet
assert strip_thinking("  plain answer\n") == "plain answer"
print("ok")
