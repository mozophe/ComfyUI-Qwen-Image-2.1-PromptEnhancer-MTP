import os, sys, types
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
pkg = types.ModuleType("qpe")  # pe.py imports .graft: load it as a package module without running the node's __init__
pkg.__path__ = [str(REPO)]
sys.modules["qpe"] = pkg
import torch
from qpe.pe import balanced_spans, fit_image, json_repair, parse_answer

# answers as the official pe_core.parse_answer reads them
assert parse_answer('{"rewritten_prompt": "a fox", "wh_ratio": "16:9"}') == ("a fox", "16:9", "")
assert parse_answer('Here it is:\n```json\n{"rewritten_prompt": " edit ", "wh_ratio": "", "ratio_follow": "<image2>"}\n```') == ("edit", "", "<image2>")
assert parse_answer('{"rewrited_prompt": "typo key"}') == ("typo key", "", "")
assert parse_answer('{"note": "draft"} then {"rewritten_prompt": "last wins", "extra": {"x": 1}}') == ("last wins", "", "")
assert parse_answer('{"rewritten_prompt": "has {braces} and \\"quotes\\""}') == ('has {braces} and "quotes"', "", "")
assert parse_answer('{"rewritten_prompt": "cut off') is None
assert parse_answer("plain text, no JSON") is None
assert parse_answer('{"rewritten_prompt": ""}') is None
assert parse_answer('{"rewritten_prompt": "outer", "x": {"rewritten_prompt": "inner"}}') == ("outer", "", "")  # top-level objects only

# official _balanced_spans: top-level objects in order, braces inside strings skipped
assert balanced_spans('a {"k": "}{"} b {"n": {"m": 1}} {unclosed') == ['{"k": "}{"}', '{"n": {"m": 1}}']

# nearly valid JSON: repaired when json_repair is installed (as the official requirements do), else no parse
broken = '{"rewritten_prompt": "a fox", "wh_ratio": "3:2",}'
assert parse_answer(broken) == (("a fox", "3:2", "") if json_repair else None), parse_answer(broken)

# shrink to at most 1 MP keeping the aspect ratio, never enlarge, drop alpha
big = fit_image(torch.rand(1, 3000, 4000, 4))
assert big.shape == (1, 886, 1182, 3), big.shape
assert big.shape[1] * big.shape[2] <= 1024 * 1024
small = torch.rand(1, 300, 400, 3)
assert torch.equal(fit_image(small), small)
print("ok")
