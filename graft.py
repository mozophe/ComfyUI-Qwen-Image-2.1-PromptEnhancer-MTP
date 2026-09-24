# Streamed safetensors grafting: no ComfyUI imports, shared by the loader node and tools/graft_mtp.py.
import json, os, struct
from huggingface_hub import HfFileSystem, hf_hub_download

CHUNK = 64 * 2**20


def read_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return n, json.loads(f.read(n))


def mtp_tensors(repo):
    # {name: (dtype, shape, raw bytes)} of a Qwen3.5 repo's mtp.* tensors, fetched with range reads (~0.5 GB for 9B)
    index = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
    fs = HfFileSystem()
    out = {}
    for shard in sorted({s for k, s in index.items() if k.startswith("mtp.")}):
        with fs.open(f"{repo}/{shard}", "rb", block_size=CHUNK) as f:
            n, header = read_header(f)
            for k in sorted(k for k in header if k.startswith("mtp.")):
                a, b = header[k]["data_offsets"]
                f.seek(8 + n + a)
                out[k] = (header[k]["dtype"], header[k]["shape"], f.read(b - a))
    if not out:
        raise ValueError(f"{repo} has no mtp.* tensors")
    return out


def graft(src, dst, extra, progress=None, chunk=CHUNK):
    # write src (a seekable binary file) plus `extra` tensors to dst, streaming the tensor data so neither
    # file is held in memory
    n, header = read_header(src)
    data_len = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)

    def data(done):
        src.seek(8 + n + done)
        while done < data_len:
            buf = src.read(min(chunk, data_len - done))
            if not buf:
                raise IOError(f"source ended at {done} of {data_len} bytes")
            yield buf
            done += len(buf)
    write(dst, header, extra, data, progress)


def write(dst, header, extra, data, progress=None):
    # write a safetensors file of `header` (data_offsets included) plus `extra` tensors; data(done) yields the
    # header's tensor bytes from offset `done` on. Builds dst + ".partial" and resumes it when a previous run was interrupted
    header = dict(header)
    data_len = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
    offset = data_len
    for k, (dtype, shape, raw) in extra.items():
        if k in header:
            raise ValueError(f"source already has {k}")
        header[k] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    raw = json.dumps(header, separators=(",", ":")).encode()
    head = struct.pack("<Q", len(raw) + (-len(raw) % 8)) + raw + b" " * (-len(raw) % 8)

    part = dst.with_name(dst.name + ".partial")
    done = 0
    if part.exists():
        with open(part, "rb") as f:
            if f.read(len(head)) == head:
                done = min(part.stat().st_size - len(head), data_len)
    with open(part, "r+b" if done else "wb") as out:
        out.seek(len(head) + done)
        out.truncate()
        if not done:
            out.seek(0)
            out.write(head)
        for buf in data(done):
            out.write(buf)
            done += len(buf)
            if progress is not None:
                progress(done, data_len)
        if done != data_len:
            raise IOError(f"wrote {done} of {data_len} bytes")
        for _, _, blob in extra.values():
            out.write(blob)
    os.replace(part, dst)
