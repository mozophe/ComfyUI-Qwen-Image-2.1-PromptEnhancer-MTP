import importlib.util, io, json, os, sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1]))]
spec = importlib.util.spec_from_file_location("qpe", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
node = importlib.util.module_from_spec(spec)
sys.modules["qpe"] = node
spec.loader.exec_module(node)
import torch
from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight
from safetensors.torch import save, load_file
from qpe.pe import build_heretic

conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}
w = torch.randn(64, 32, dtype=torch.bfloat16)
norm = torch.randn(32, dtype=torch.bfloat16)
q, scale = quantize_int8_convrot_weight(w.float(), 16)
ref = save({"model.n.weight": torch.zeros(32, dtype=torch.bfloat16), "model.l.weight": torch.zeros_like(q),
            "model.l.weight_scale": torch.zeros_like(scale), "model.l.comfy_quant": torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)})
shards = {"a.safetensors": save({"model.n.weight": norm}), "b.safetensors": save({"model.l.weight": w})}
index = {"model.n.weight": "a.safetensors", "model.l.weight": "b.safetensors"}
open_shard = lambda s: io.BytesIO(shards[s])
extra = {"mtp.fc.weight": ("BF16", [2], torch.ones(2, dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes())}

with tempfile.TemporaryDirectory() as d:
    dst = Path(d) / "out.safetensors"
    build_heretic(io.BytesIO(ref), index, open_shard, dst, extra, None, "cpu")
    out = load_file(dst)
    assert torch.equal(out["model.l.weight"], q) and torch.equal(out["model.l.weight_scale"], scale)
    assert torch.equal(out["model.n.weight"], norm) and torch.equal(out["mtp.fc.weight"], torch.ones(2, dtype=torch.bfloat16))
    assert json.loads(bytes(out["model.l.comfy_quant"].tolist())) == conf
    full = dst.read_bytes()

    # interrupted run: a truncated .partial resumes to the identical file
    part = dst.with_name(dst.name + ".partial")
    part.write_bytes(full[:len(full) // 2])
    dst.unlink()
    build_heretic(io.BytesIO(ref), index, open_shard, dst, extra, None, "cpu")
    assert dst.read_bytes() == full

    # a source that doesn't match the reference is refused
    try:
        build_heretic(io.BytesIO(ref), {"model.n.weight": "a.safetensors"}, open_shard, Path(d) / "x.safetensors", extra, None, "cpu")
        raise AssertionError("mismatch not refused")
    except ValueError:
        pass
print("ok")
