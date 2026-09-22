# Copy a Qwen3.5 checkpoint and add mtp.* tensors to it.
# usage: python graft.py <checkpoint.safetensors> <mtp.safetensors> <out.safetensors>
import sys
from safetensors import safe_open
from safetensors.torch import save_file

src, mtp_path, dst = sys.argv[1:4]
sd = {}
with safe_open(src, "pt") as f:
    metadata = f.metadata()
    for k in f.keys():
        sd[k] = f.get_tensor(k)
with safe_open(mtp_path, "pt") as f:
    for k in f.keys():
        if k in sd:
            sys.exit(f"{src} already has {k}")
        sd[k] = f.get_tensor(k)
save_file(sd, dst, metadata=metadata)
print(f"wrote {dst} ({len(sd)} tensors)")
