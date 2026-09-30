<div align="center">

# ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP

**ComfyUI nodes for Qwen-Image 2.1's official prompt enhancer, with MTP speculative decoding for faster generation.**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-%E2%89%A5%200.37.0-blue)](https://github.com/Comfy-Org/ComfyUI)
[![MTP speedup](https://img.shields.io/badge/MTP-1.35%E2%80%931.65%C3%97%20faster-orange)](#performance)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)

[Installation](#installation) •
[Quick start](#quick-start) •
[Nodes](#nodes) •
[llama.cpp backend](#llamacpp-backend-faster-nvidia) •
[Performance](#performance) •
[Troubleshooting](#troubleshooting) •
[Changelog](#changelog)

</div>

---

## Overview

Qwen-Image 2.1 works best with long, detailed prompts. Its official prompt enhancer (PE) writes them for you: give it a short idea or edit instruction, and it returns a detailed prompt.

This extension runs the PE inside ComfyUI with Qwen's own system prompts and settings, and makes it faster with **multi-token prediction (MTP)**: about **1.35×** for text-to-image and **1.65×** for image editing, in tokens generated per second.

- **Automatic setup.** The model downloads and prepares itself on first use.
- **Official settings.** System prompts and sampling values match Qwen's reference code.
- **Ready-to-use output.** The rewritten prompt comes out on its own, separate from the model's reasoning.
- **Multi-image editing.** Up to 10 input images, any size.
- **Heretic versions.** Community abliterated fine-tunes that refuse less, for both t2i and i2i.
- **Optional llama.cpp backend.** On NVIDIA GPUs (Windows or Linux), runs the PE at about 1.4× (text-to-image) to 2.1× (editing) the tok/s of the ComfyUI backend, both with MTP on. See [llama.cpp backend](#llamacpp-backend-faster-nvidia).

## Requirements

- **ComfyUI:** v0.37.0 or newer
- **VRAM:** 16 GB recommended. Smaller GPUs work, but much more slowly.
- **System RAM:** 32 GB recommended
- **Disk space:** about 10 GB per model (t2i for text-to-image, i2i for editing): the 9.5 GB enhancer plus the 0.5 GB MTP head

## Installation

### ComfyUI Manager

> [!NOTE]
> The node is published to the Comfy Registry but is still waiting for review, so it may not show up in the Manager yet. Until then, use the [manual install](#manual).

Open **Manager → Custom Nodes Manager**, search for **Qwen-Image 2.1 Prompt Enhancer (MTP)**, click **Install**, and restart ComfyUI. The Manager installs the requirements too. To update, use **Update** in the same place.

### Manual

Clone the repository into your ComfyUI custom_nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP
```

Everything the node needs already comes with ComfyUI. Optionally, install the requirements (json-repair, which fixes the rare answer that is almost valid JSON). Run the command for your ComfyUI version:

| ComfyUI version | Command |
|---|---|
| Windows portable | From the ComfyUI_windows_portable folder:<br>`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP\requirements.txt` |
| Desktop app | In the app's terminal panel, from the extension's folder:<br>`pip install -r requirements.txt` |
| Manual install | With ComfyUI's virtual environment activated, from the extension's folder:<br>`pip install -r requirements.txt` |

Restart ComfyUI. The first run downloads the model (progress is shown on the node). If the download is interrupted, run the workflow again and it resumes.

To update, run git pull in the extension's folder.

## Quick start

### Sample workflows

The easiest way to start is to drag one of these images into ComfyUI. Each image carries its workflow, so dropping it loads the whole graph. The same workflows are also in the [workflows](workflows) folder as JSON.

**Text-to-image** ([JSON](workflows/qwen_image_2.1_t2i_prompt_enhancer.json))

![Text-to-image workflow](workflows/qwen_image_2.1_t2i_prompt_enhancer.png)

**Editing with two input images** ([JSON](workflows/qwen_image_2.1_edit_prompt_enhancer.json))

![Edit workflow](workflows/qwen_image_2.1_edit_prompt_enhancer.png)

Both are ComfyUI's official Qwen-Image 2.1 templates with the enhancer added in front. They use the int8 models from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).

### Adding it to your own workflow

```
Qwen-Image 2.1 PE Loader (MTP) ──clip──► Qwen-Image 2.1 Prompt Enhancer (MTP) ──positive_prompt──► your text encoder
                           Load Image ──image_1──┘   (editing only)
```

1. Choose the same mode on both nodes: a t2i loader model (t2i or t2i - heretic) with the t2i preset, or an i2i one with the i2i preset.
2. Write a short instruction in prompt:
   - **Text-to-image:** *"a fox reading a book in a snowy forest, watercolor"*
   - **Editing:** *"Put the woman from &lt;image2&gt; into the street in &lt;image1&gt;"*. Images are numbered by the input they are connected to, and can be any size.
3. Use positive_prompt as your prompt.

## Nodes

### Qwen-Image 2.1 PE Loader (MTP)

Loads the prompt enhancer, ready for fast MTP generation.

| Input | Description |
|---|---|
| model | t2i for text-to-image, i2i for editing. The heretic versions are community abliterated fine-tunes that refuse less. |
| backend | ComfyUI (default) runs the PE inside ComfyUI. llama.cpp runs it in a separate llama-server process, faster; NVIDIA only. See [llama.cpp backend](#llamacpp-backend-faster-nvidia). |
| quant | llama.cpp only: Q8_0 (9.8 GB, for 16 GB GPUs and up) or Q4_K_M (6.0 GB, for 8–12 GB GPUs) |
| kv_cache | llama.cpp only: f16 (default) or q8_0, which stores the KV cache at 8 bits to save memory |

With the ComfyUI backend, on first use it downloads the prompt enhancer from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/text_encoders) (about 9.5 GB) and the MTP head from [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (about 0.5 GB), then combines them into a prepared copy in ComfyUI/models/text_encoders/Qwen-Image-2.1-PE/: the int8 convrot enhancer with the MTP head added. If you already have the Comfy-Org PE checkpoint in a text_encoders folder, it is used instead of downloading.

The heretic versions download from [pottokao/Qwen-Image-2.1-PE-T2I-Heretic](https://huggingface.co/pottokao/Qwen-Image-2.1-PE-T2I-Heretic) and [darrellbest/Qwen-Image-2.1-PE-I2I-Heretic](https://huggingface.co/darrellbest/Qwen-Image-2.1-PE-I2I-Heretic) instead. Those are bf16, so the download is about 19 GB, but they are quantized as they stream in to the same int8 convrot format as the Comfy-Org checkpoint. Nothing else is kept on disk, and the prepared copy is the same 10 GB. Use them with the matching t2i or i2i preset.

After setup, the prepared file (the one ending in `.mtp.safetensors`) is an ordinary int8 convrot checkpoint with an MTP head. You can also load it with the stock **Load CLIP** node (type qwen_image), and MTP still works. You only lose the check that the loader and preset modes match.

### Qwen-Image 2.1 Prompt Enhancer (MTP)

ComfyUI's **Generate Text** node with the official PE presets built in.

| Input | Description |
|---|---|
| clip | The enhancer from the loader |
| prompt | Your short instruction |
| image_1 … image_10 | Input images for editing, referred to as &lt;image1&gt;, &lt;image2&gt;… |
| preset | Qwen-Image 2.1 PE (t2i), Qwen-Image 2.1 PE (i2i), or none to use it as a plain Generate Text node |
| seed | Change it for a different result |
| mtp | auto (recommended), off, or a fixed draft depth |

| Output | Description |
|---|---|
| positive_prompt | The detailed prompt, ready for your text encoder |
| negative_prompt | Always empty; the enhancer doesn't write one |
| thinking | The model's reasoning |
| wh_ratio | The chosen aspect ratio, for example 16:9 |
| ratio_follow | Editing only: the image whose shape to keep, for example &lt;image1&gt; |
| parse_ok | false if the answer couldn't be read; positive_prompt then contains the full answer |

The presets use the official values from Qwen's [prompt_rewrite](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite) code, except max_length: it defaults to 8192 instead of 16256 (t2i) and 24000 (i2i), because a higher max_length is slower (see [Performance](#performance)). Every value can be changed on the node.

> [!IMPORTANT]
> Keep **thinking** on. Both models were trained to reason before answering and give worse prompts without it.

## llama.cpp backend (faster, NVIDIA)

The loader can run the prompt enhancer with [llama.cpp](https://github.com/ggml-org/llama.cpp) instead of inside ComfyUI: set **backend** to `llama.cpp`. The generator node, its presets and its outputs stay the same.

**Why it was added:** on NVIDIA GPUs llama.cpp generates the enhanced prompt much faster: on an RTX 4090 Laptop with MTP on, 47 vs 34 tok/s for text-to-image (1.4×) and 68 vs 33 tok/s for editing with two images (2.1×). Its speed also doesn't drop with a higher max_length, so the official lengths cost nothing extra. Its Q8_0 model matches the quality of the int8 model the ComfyUI backend uses. See [Performance](#performance) for the measurements.

**Requirements:** an NVIDIA GPU with driver 528.33 or newer on Windows, or 525.60.13 or newer on Linux (x64). Other GPUs use the ComfyUI backend.

**Downloads (first use):**

- llama-server, a pinned prebuilt llama.cpp release (about 0.7 GB, once), into `ComfyUI/models/llama.cpp/`. The node picks the CUDA 13 or CUDA 12 build that matches your driver.
- The model from [mozophe/Qwen-Image-2.1-PE-MTP-GGUF](https://huggingface.co/mozophe/Qwen-Image-2.1-PE-MTP-GGUF): Q8_0 9.8 GB or Q4_K_M 6.0 GB, plus 0.9 GB for the i2i vision part, into `ComfyUI/models/LLM/Qwen-Image-2.1-PE/`. If you already have the file anywhere under `models/LLM`, it is used where it is.

The GGUFs include the MTP head, so MTP works here too. The model card lists how they were built and their quality measurements.

**Choosing settings:**

- **quant:** Q8_0 for 16 GB GPUs and up. Q4_K_M for 8–12 GB; it is also faster, with a small quality cost.
- **kv_cache:** f16 by default. q8_0 saves some memory for Q4_K_M on 8 GB GPUs, at almost no quality cost.
- **max_length:** with llama.cpp you can raise it to the official values (16256 for text-to-image, 24000 for editing) without slowing down; it only reserves a little more memory.

llama-server runs as a separate process. The node unloads ComfyUI's models before each prompt, and llama-server frees its VRAM about a second after answering, so the rest of the workflow gets the GPU back. It stops when ComfyUI exits. After ComfyUI starts, or when you change the model, quant, kv_cache or mtp setting, the first prompt takes a few seconds longer while llama-server starts.

The `none` preset isn't supported with the llama.cpp backend.

## Performance

Measured on an RTX 4090 Laptop GPU (16 GB), with one input image for editing. Speed-ups compare tokens generated per second (tok/s), not total time: the PE writes answers of different lengths from run to run, so time alone would mix length with speed.

| Mode | max_length | MTP off | MTP on | Speed-up |
|---|---|---|---|---|
| Text-to-image | 8192 (default) | 35 tok/s | 47 tok/s | 1.36× |
| Text-to-image | 16256 (official) | 27 tok/s | 38 tok/s | 1.38× |
| Editing | 8192 (default) | 31 tok/s | 52 tok/s | 1.65× |
| Editing | 24000 (official) | 21 tok/s | 36 tok/s | 1.67× |

A typical answer is 2,000–4,000 tokens, so 8192 leaves plenty of room. If an answer is ever cut off, raise max_length.

Peak VRAM use was about 14 GB for text-to-image and 16 GB for editing. With MTP on, quality is unchanged, but the same seed gives different text than with MTP off.

### ComfyUI vs llama.cpp backend

End to end through the nodes, at the official max_length (16256 for text-to-image, 24000 for editing with two images), on the same RTX 4090 Laptop. Each row is 6 prompts, each followed by the Qwen-Image 2.1 diffusion workflow (25 steps, 1024²). The PE time includes everything the backend does, such as llama.cpp waking up and freeing VRAM. Tokens/s is the answer's tokens divided by that PE time, and Speed-up is Tokens/s relative to the ComfyUI backend with MTP off.

| Mode | Backend | MTP | PE time | Tokens/s | Speed-up |
|---|---|---|---|---|---|
| Text-to-image | ComfyUI (int8) | off | 75 s | 24 | 1.00× (baseline) |
| Text-to-image | ComfyUI (int8) | on | 52 s | 34 | 1.42× |
| Text-to-image | llama.cpp (Q8_0) | off | 40 s | 42 | 1.75× |
| Text-to-image | **llama.cpp (Q8_0)** | **on** | **32 s** | **47** | **1.96×** |
| Editing | ComfyUI (int8) | off | 170 s | 19 | 1.00× (baseline) |
| Editing | ComfyUI (int8) | on | 105 s | 33 | 1.74× |
| Editing | llama.cpp (Q8_0) | off | 79 s | 47 | 2.47× |
| Editing | **llama.cpp (Q8_0)** | **on** | **55 s** | **68** | **3.58×** |

The rest of the workflow took the same time with either backend (about 18 s for text-to-image, 43 s for editing), so only the PE time differs. With llama.cpp, max_length 8192, 16256 and 24000 gave the same speed.

## Troubleshooting

<details>
<summary><b>"mtp is on but this Qwen3.5 checkpoint has no MTP head"</b></summary>

The checkpoint has no MTP head. This happens with the original Comfy-Org file, for example. Use **Qwen-Image 2.1 PE Loader (MTP)**, or in **Load CLIP** pick the file whose name ends in `.mtp.safetensors`, which the loader creates on first use.
</details>

<details>
<summary><b>"The PE loader is set to t2i but the preset is … (i2i)"</b></summary>

The loader and the preset are set to different modes. Choose t2i on both for text-to-image, or i2i on both for editing. The heretic models count as their t2i or i2i mode.
</details>

<details>
<summary><b>parse_ok is false</b></summary>

The node couldn't find the rewritten prompt in the answer, so positive_prompt holds the whole answer instead. Check the thinking and positive_prompt outputs to see which case applies:

- **The answer was cut off** (it ends mid-sentence, or positive_prompt is empty): generation reached max_length. Increase it on the node.
- **The JSON is slightly broken**: install the requirements (see [Installation](#installation)) so json-repair can fix it, or run again with a different seed.
- **Thinking is off**: turn it back on; the model follows the answer format less reliably without it.
- **The preset is none**: expected, since that preset doesn't ask for the PE's answer format.
</details>

<details>
<summary><b>llama.cpp backend: "needs an NVIDIA GPU" or "nvidia-smi could not read the NVIDIA driver version"</b></summary>

The llama.cpp backend only runs on NVIDIA GPUs under Windows or Linux (x64), and uses `nvidia-smi` to pick its build. Update the NVIDIA driver, or set the loader's backend to ComfyUI.
</details>

<details>
<summary><b>llama.cpp backend: out of VRAM</b></summary>

Use quant Q4_K_M, set kv_cache to q8_0, lower max_length, or use fewer input images. The error shows the last lines of llama-server's log, which is saved as `llama-server.log` in ComfyUI's temp folder.
</details>

<details>
<summary><b>llama.cpp backend: a download failed</b></summary>

Run the workflow again; downloads resume. The error names the file and URL.
</details>

<details>
<summary><b>The node stopped working after a ComfyUI update</b></summary>

This extension depends on parts of ComfyUI that can change between versions. Please [open an issue](https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP/issues) with the error message and your ComfyUI version.
</details>

## Changelog

- **1.2.0** (unreleased): optional llama.cpp backend for NVIDIA GPUs (Windows/Linux) with Q8_0 and Q4_K_M GGUFs and MTP: about 1.4× (text-to-image) to 2.1× (editing) the tok/s of the ComfyUI backend.
- **1.1.2** (2026-09-24): max_length defaults to 8192 instead of 16256/24000, which is faster, and answers fit well within it.
- **1.1.1** (2026-09-24): the PE seed defaults to a fixed 42; refreshed sample workflows.
- **1.1.0** (2026-09-24): heretic t2i and i2i models.
- **1.0.0** (2026-09-23): first release on the Comfy Registry.

## License

The code in this repository is released under the [MIT License](LICENSE).

The prompt enhancer weights and system prompts are released under the non-commercial [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/blob/main/LICENSE). They are not included in this repository; they are downloaded on first use. The GGUFs for the llama.cpp backend are derived from them and use the same license.

## Acknowledgements

- [Qwen team](https://github.com/QwenLM/Qwen-Image-2.1) for Qwen-Image 2.1, the prompt enhancers and the reference implementation
- [Comfy-Org](https://github.com/Comfy-Org/ComfyUI) for ComfyUI, its Qwen3.5 text encoder and MTP decoding, and the [int8 checkpoints](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
- [ggml-org](https://github.com/ggml-org/llama.cpp) for llama.cpp, its Qwen3.5 support and MTP speculative decoding
