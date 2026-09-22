# Fetch only the mtp.* tensors of a Qwen3.5 base repo via HTTP range reads.
# usage: python fetch_mtp.py Qwen/Qwen3.5-9B qwen35_9b_mtp.safetensors
import json, struct, sys
from collections import defaultdict
import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from safetensors.torch import save_file

repo, out_path = sys.argv[1], sys.argv[2]
index = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
by_shard = defaultdict(list)
for k, shard in index.items():
    if k.startswith("mtp."):
        by_shard[shard].append(k)
if not by_shard:
    sys.exit(f"{repo} has no mtp.* tensors")

dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
fs = HfFileSystem()
out = {}
for shard, keys in by_shard.items():
    with fs.open(f"{repo}/{shard}", "rb", block_size=8 * 2**20) as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        for k in keys:
            m = header[k]
            a, b = m["data_offsets"]
            f.seek(8 + n + a)
            out[k] = torch.frombuffer(bytearray(f.read(b - a)), dtype=dtypes[m["dtype"]]).reshape(m["shape"])
save_file(out, out_path)
print(f"saved {len(out)} tensors to {out_path}")
