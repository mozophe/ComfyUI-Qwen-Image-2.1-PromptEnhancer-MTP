import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1])), str(REPO)]
from mtp import split_thinking

# cases follow the official prompt_rewrite/pe_core.py split_thinking
assert split_thinking('reasoning...\n</think>\n\n{"rewritten_prompt": "x"}') == ("reasoning...", '{"rewritten_prompt": "x"}')  # generation started inside <think>
assert split_thinking("<think>\nr\n</think>\n\nanswer") == ("r", "answer")
assert split_thinking("a</think>b</think>c") == ("a", "b</think>c")  # only the first close ends the thinking
assert split_thinking("<think>\nunfinished reasoning") == ("unfinished reasoning", "")  # cut off while thinking: no answer yet
assert split_thinking("  plain answer\n") == ("", "plain answer")
print("ok")
