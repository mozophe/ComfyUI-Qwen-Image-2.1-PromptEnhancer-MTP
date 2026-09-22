import os, sys, types
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
pkg = types.ModuleType("qpe")  # pe.py imports .graft: load it as a package module without running the node's __init__
pkg.__path__ = [str(REPO)]
sys.modules["qpe"] = pkg
import torch
from qpe.pe import fit_image, parse_answer

# answers as the official pe_core.parse_answer reads them
assert parse_answer('{"rewritten_prompt": "a fox", "wh_ratio": "16:9"}') == ("a fox", "16:9", "")
assert parse_answer('Here it is:\n```json\n{"rewritten_prompt": " edit ", "wh_ratio": "", "ratio_follow": "<image2>"}\n```') == ("edit", "", "<image2>")
assert parse_answer('{"rewrited_prompt": "typo key"}') == ("typo key", "", "")
assert parse_answer('{"note": "draft"} then {"rewritten_prompt": "last wins", "extra": {"x": 1}}') == ("last wins", "", "")
assert parse_answer('{"rewritten_prompt": "has {braces} and \\"quotes\\""}') == ('has {braces} and "quotes"', "", "")
assert parse_answer('{"rewritten_prompt": "cut off') is None
assert parse_answer("plain text, no JSON") is None
assert parse_answer('{"rewritten_prompt": ""}') is None

# shrink to at most 1 MP keeping the aspect ratio, never enlarge, drop alpha
big = fit_image(torch.rand(1, 3000, 4000, 4))
assert big.shape == (1, 886, 1182, 3), big.shape
assert big.shape[1] * big.shape[2] <= 1024 * 1024
small = torch.rand(1, 300, 400, 3)
assert torch.equal(fit_image(small), small)
print("ok")
