# Maintainer tool: build the llama.cpp backend's GGUFs for one PE model and optionally upload them.
# bf16 source -> MTP head grafted -> bf16 GGUF -> Q8_0 + Q4_K_M (+ bf16 vision file for i2i) -> KL divergence + smoke -> upload
# usage: python tools/build_gguf.py <model> <llama.cpp dir> <work dir> <eval text> [--upload]
#   <llama.cpp dir>: llama.cpp source at tag b11160 with its binaries in bin/ (llama-quantize, llama-perplexity, llama-server)
#   run with a Python that has transformers >= 5 and torch (e.g. ComfyUI's venv), PYTHONPATH is set to <llama.cpp dir>/gguf-py
import copy, json, os, subprocess, sys
from pathlib import Path

SOURCES = {"t2i": "Qwen/Qwen-Image-2.1-PE-T2I", "i2i": "Qwen/Qwen-Image-2.1-PE-I2I",
           "t2i - heretic": "pottokao/Qwen-Image-2.1-PE-T2I-Heretic", "i2i - heretic": "darrellbest/Qwen-Image-2.1-PE-I2I-Heretic"}
GGUF_REPO = "mozophe/Qwen-Image-2.1-PE-MTP-GGUF"
KLD_BAR = {"Q8_0": 0.01, "Q4_K_M": 0.05}


def patch_config(cfg):
    out = copy.deepcopy(cfg)
    # the PE configs have null, which the converter rejects; text-only exports keep it at the top level
    (out["text_config"] if "text_config" in out else out)["mtp_num_hidden_layers"] = 1
    return out


def add_mtp_to_index(index, names):
    out = copy.deepcopy(index)
    out["weight_map"].update({n: "model-mtp.safetensors" for n in names})
    return out


def run(cmd, **kw):
    print(">", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def kld(ll, gguf, base_logits, eval_text):
    out = subprocess.run([str(ll / "bin" / "llama-perplexity"), "-m", gguf, "-f", eval_text, "-c", "4096", "-b", "4096", "-fa", "on",
                          "--kl-divergence-base", base_logits, "--kl-divergence"], capture_output=True, text=True)
    line = next(l for l in (out.stdout + out.stderr).splitlines() if "Mean    KLD" in l)
    return float(line.split(":")[1].split("±")[0])


def main(model, ll, work, eval_text, upload):
    from huggingface_hub import snapshot_download, HfApi
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graft import write, mtp_tensors
    ll, work = Path(ll), Path(work)
    task = "i2i" if model.startswith("i2i") else "t2i"
    base = f"qwen3.5_9b_qwen_image_2.1_pe_{task}{'_heretic' if 'heretic' in model else ''}"
    src = work / base
    snapshot_download(SOURCES[model], local_dir=src)
    extra = mtp_tensors("Qwen/Qwen3.5-9B")
    write(src / "model-mtp.safetensors", {}, extra, lambda done: iter(()))
    cfg = src / "config.json"
    json.dump(patch_config(json.load(open(cfg))), open(cfg, "w"), indent=2)
    idx = src / "model.safetensors.index.json"
    json.dump(add_mtp_to_index(json.load(open(idx)), sorted(extra)), open(idx, "w"), indent=2)

    env = dict(os.environ, PYTHONPATH=str(ll / "gguf-py"))
    out = work / "out"
    out.mkdir(exist_ok=True)
    bf16 = work / f"{base}.mtp.bf16.gguf"
    run([sys.executable, ll / "convert_hf_to_gguf.py", src, "--outtype", "bf16", "--outfile", bf16], env=env)
    files = []
    for quant in ("Q8_0", "Q4_K_M"):
        files.append(out / f"{base}.mtp.{quant}.gguf")
        run([ll / "bin" / "llama-quantize", bf16, files[-1], quant])
    if model == "i2i":  # the one vision file; the heretic i2i model reuses it
        files.append(out / f"{base}.mmproj.bf16.gguf")
        run([sys.executable, ll / "convert_hf_to_gguf.py", src, "--mmproj", "--outtype", "bf16", "--outfile", files[-1]], env=env)

    logits = work / f"{base}.kld_base.bin"
    run([ll / "bin" / "llama-perplexity", "-m", bf16, "-f", eval_text, "-c", "4096", "-b", "4096", "-fa", "on", "--kl-divergence-base", logits])
    report = {}
    for f in files[:2]:
        quant = f.name.split(".")[-2]
        report[quant] = kld(ll, f, logits, eval_text)
        print(f"{f.name}: mean KLD {report[quant]:.4f} (bar {KLD_BAR[quant]})")
        if report[quant] > KLD_BAR[quant]:
            sys.exit(f"{f.name} fails the KLD bar; not uploading")
    json.dump({"model": model, "source": SOURCES[model], "llama.cpp": "b11160", "mean_kld": report},
              open(out / f"{base}.build.json", "w"), indent=2)
    if upload:
        HfApi().upload_large_folder(repo_id=GGUF_REPO, repo_type="model", folder_path=str(out),
                                    allow_patterns=[f"{base}.*"])


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--upload"]
    main(a[0], a[1], a[2], a[3], "--upload" in sys.argv)
