import base64, importlib.util, io, json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
spec = importlib.util.spec_from_file_location("qpe", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
node = importlib.util.module_from_spec(spec)
sys.modules["qpe"] = node
spec.loader.exec_module(node)
import torch
from PIL import Image
from qpe.mtp import pe_prompt
from qpe.llama_server import insert_markers, png_b64, completion_body, read_stream

text = pe_prompt("SYS", "put <image1> into <image2>", 0, True)
out = insert_markers(text, "<__media_x__>", 2)
assert out.count("<__media_x__>") == 2 and "<|im_start|>user\n<__media_x__><__media_x__>put <image1>" in out, out
assert "<|image_pad|>" not in out and insert_markers(text, "<m>", 0) == text

img = torch.zeros(1, 6, 10, 3); img[0, :, :, 0] = 1.0
decoded = Image.open(io.BytesIO(base64.b64decode(png_b64(img))))
assert decoded.size == (10, 6) and decoded.mode == "RGB" and decoded.getpixel((0, 0)) == (255, 0, 0)

preset = {"max_length": 8192, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
          "repetition_penalty": 1.0, "presence_penalty": 1.5, "thinking": True, "preset": "Qwen-Image 2.1 PE (t2i)"}
b = completion_body("P", preset, 42, [])
assert b == {"prompt": "P", "n_predict": 8192, "seed": 42, "cache_prompt": False, "stream": True, "return_tokens": True,
             "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "repeat_penalty": 1.0, "presence_penalty": 1.5}, b
b = completion_body("P", preset, 7, ["AAA", "BBB"])
assert b["prompt"] == {"prompt_string": "P", "multimodal_data": ["AAA", "BBB"]} and b["seed"] == 7

def sse(*chunks):
    return [f"data: {json.dumps(c)}\n".encode() for c in chunks] + [b"\n"]

counted = []
lines = sse({"content": "think", "tokens": [1, 2], "stop": False}, {"content": "</think>{}", "tokens": [3], "stop": False},
            {"content": "", "tokens": [], "stop": True, "stop_type": "eos", "timings": {"predicted_n": 3}})
content, final = read_stream(lines, counted.append, lambda: False)
assert content == "think</think>{}" and final["stop_type"] == "eos" and sum(counted) == 3

# cancel (Review Focus 5): stops reading and reports None, never a partial answer
calls = iter([False, True])
assert read_stream(sse({"content": "a", "tokens": [1], "stop": False}, {"content": "b", "tokens": [2], "stop": False}), lambda n: None, lambda: next(calls)) is None

for bad, needle in ((sse({"error": {"code": 500, "message": "failed to allocate CUDA buffer"}}), "failed to allocate"),
                    (sse({"content": "half", "tokens": [1], "stop": False}), "before the answer ended")):
    try:
        read_stream(bad, lambda n: None, lambda: False)
        raise AssertionError("no error")
    except RuntimeError as e:
        assert needle in str(e), e
print("ok")
