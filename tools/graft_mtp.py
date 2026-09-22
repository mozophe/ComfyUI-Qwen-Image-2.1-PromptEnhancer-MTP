# Add a Qwen3.5 base model's MTP head to a fine-tuned checkpoint of the same size.
# usage: python graft_mtp.py <checkpoint.safetensors> <base repo, e.g. Qwen/Qwen3.5-4B> <out.safetensors>
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graft import graft, mtp_tensors

src, repo, dst = sys.argv[1], sys.argv[2], Path(sys.argv[3])
with open(src, "rb") as f:
    graft(f, dst, mtp_tensors(repo), lambda done, total: print(f"\r{100 * done // total}%", end=""))
print(f"\nwrote {dst}")
