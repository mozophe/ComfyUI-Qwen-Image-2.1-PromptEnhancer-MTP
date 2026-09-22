import io, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import save, load_file
from graft import graft, read_header

src_sd = {"model.a.weight": torch.randn(64, 32, dtype=torch.bfloat16), "model.b.weight_scale": torch.randn(7, dtype=torch.float32),
          "model.c.comfy_quant": torch.tensor([1, 2, 3], dtype=torch.uint8)}
extra_sd = {"mtp.fc.weight": torch.randn(16, 8, dtype=torch.bfloat16), "mtp.norm.weight": torch.randn(8, dtype=torch.bfloat16)}
src_bytes = save(src_sd, metadata={"format": "pt"})
n, header = read_header(io.BytesIO(save(extra_sd)))
blob = save(extra_sd)
extra = {k: (v["dtype"], v["shape"], blob[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]]) for k, v in header.items() if k != "__metadata__"}

with tempfile.TemporaryDirectory() as d:
    dst = Path(d) / "out.safetensors"
    graft(io.BytesIO(src_bytes), dst, extra, chunk=100)
    out = load_file(dst)
    assert set(out) == set(src_sd) | set(extra_sd)
    for k, v in {**src_sd, **extra_sd}.items():
        assert torch.equal(out[k], v), k
    full = dst.read_bytes()

    # interrupted run: a truncated .partial resumes to the identical file
    part = dst.with_name(dst.name + ".partial")
    part.write_bytes(full[:len(full) // 2])
    dst.unlink()
    graft(io.BytesIO(src_bytes), dst, extra, chunk=100)
    assert dst.read_bytes() == full and not part.exists()

    # a key collision is refused
    try:
        graft(io.BytesIO(src_bytes), Path(d) / "x.safetensors", {"model.a.weight": extra["mtp.fc.weight"]})
        raise AssertionError("collision not refused")
    except ValueError:
        pass
print("ok")
