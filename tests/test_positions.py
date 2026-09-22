import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get("COMFYUI_PATH", str(REPO.parents[1])), str(REPO)]
import torch
from comfy.text_encoders.llama import precompute_freqs_cis
from mtp import mrope_table, mrope_freqs


def rope(ids, device):  # stand-in for Qwen35Transformer.compute_freqs_cis
    return precompute_freqs_cis(64, ids, 10000000.0, rope_dims=[11, 11, 10], interleaved_mrope=True, device=device)


# prompt: 4 text tokens, a 2x3 image grid, 2 text tokens
pid = torch.tensor([[0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 7, 8],
                    [0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 7, 8],
                    [0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8]])
L, cap = pid.shape[1], 20
table = mrope_table(pid, cap)
assert table.shape == (3, cap)
assert torch.equal(table[:, :L], pid)
assert torch.equal(table[:, L:], torch.arange(9, 9 + cap - L).expand(3, -1))  # continues from max(last column) + 1

f = mrope_freqs(rope, table)
prefix = f(torch.arange(L).unsqueeze(0), "cpu")
for got, ref in zip(prefix, rope(pid, "cpu")):
    assert got.shape[:3] == (1, 1, L)
    assert torch.allclose(got, ref.unsqueeze(1))
tail = f(torch.arange(L, cap).unsqueeze(0), "cpu")
for got, ref in zip(tail, rope(torch.arange(9, 9 + cap - L).unsqueeze(0), "cpu")):  # 1-row reference
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-6)
print("ok")
