<div align="center">

# ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP

**The official Qwen-Image 2.1 prompt enhancer for ComfyUI, accelerated with MTP speculative decoding.**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP)](https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP/commits/main)
[![Issues](https://img.shields.io/github/issues/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP)](https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP/issues)

[Installation](#installation) •
[Quick start](#quick-start) •
[Nodes](#nodes) •
[Presets](#presets) •
[Performance](#performance) •
[Troubleshooting](#troubleshooting)

</div>

---

## Overview

Qwen-Image 2.1 ships two prompt enhancers (PE), Qwen3.5-9B fine-tunes that rewrite a short instruction into the detailed prompt Qwen-Image 2.1 was trained on. This extension runs them inside ComfyUI with the official system prompts and sampling settings. It also adds **multi-token prediction (MTP)**, which speeds up generation by about 1.35× for t2i and 1.65× for i2i.

### Features

- **One-click setup.** The model downloads and prepares itself on first use. Downloads resume after an interruption, and existing local copies are reused.
- **Official presets.** The system prompt, thinking mode and sampling values come from Qwen's reference implementation.
- **MTP for image prompts.** MTP speculative decoding also works when an image is attached; core ComfyUI falls back to regular decoding in that case.
- **Ready-to-use outputs.** The node parses the model's JSON answer the way the official code does and outputs each field separately: the rewritten prompt, the aspect ratio and the thinking.
- **Multiple input images.** Up to 10 images for editing, each any size, handled as the official pipeline does: in order as `<image1>`, `<image2>`…, each shrunk to at most 1 MP.

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Nodes](#nodes)
- [Presets](#presets)
- [Model files and disk usage](#model-files-and-disk-usage)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Requirements

- A recent ComfyUI build that includes the Qwen3.5 text encoder with MTP support (`comfy/text_encoders/qwen35.py`)
- A GPU with **16 GB of VRAM** (tested on NVIDIA), which keeps the whole model on the GPU at the default settings. Cards with less VRAM also work, because ComfyUI streams part of the weights from system RAM, but generation is much slower.
- **32 GB of system RAM** recommended. ComfyUI reserves up to about 20 GB of system memory (RAM plus page file) while the model runs, although with 16 GB of VRAM only about 2 GB of it is actually in use during generation.
- About **9.3 GB** of free disk space per model (`t2i`, `i2i`)
- An internet connection for the first run only

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP
```

Optionally, install `json-repair` with ComfyUI's Python, as the official code does. It repairs answers that are nearly valid JSON; without it, every well-formed answer still parses. ComfyUI Manager installs it automatically.

```bash
pip install -r ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP/requirements.txt
```

Restart ComfyUI.

To update:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP
git pull
```

## Quick start

### Image editing (i2i)

```
Qwen-Image 2.1 PE Loader (MTP) [i2i] ──clip──► Qwen-Image 2.1 Prompt Enhancer (MTP) [preset: Qwen-Image 2.1 PE (i2i)] ──positive_prompt──► your Qwen-Image 2.1 workflow
Load Image ──image_1──┘
```

Connect more images to `image_2`, `image_3`… for multi-image edits, and refer to them in your instruction as `<image1>`, `<image2>`…, for example *"Put the woman from `<image2>` into the street in `<image1>`"*. There's no need to resize them first.

### Text-to-image (t2i)

Select `t2i` on both nodes and leave the image inputs unconnected.

Write your instruction as plain text, for example *"Make this image a realistic photo"*. Use the `positive_prompt` output as the prompt, and size the latent to `wh_ratio`. The first run takes a while because it downloads and prepares the model; progress is shown on the node.

## Nodes

### Qwen-Image 2.1 PE Loader (MTP)

Loads the prompt enhancer with an MTP head attached.

| Input | Description |
|---|---|
| `model` | `t2i` for text-to-image, `i2i` for image editing (use with an image) |

| Output | Description |
|---|---|
| `CLIP` | The prompt enhancer, ready for the Prompt Enhancer node |

On first use it downloads the checkpoint from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/text_encoders) (9.5 GB) and grafts on the MTP head from `Qwen/Qwen3.5-9B` (about 0.5 GB) in a single streaming pass. Later runs load the prepared file directly.

### Qwen-Image 2.1 Prompt Enhancer (MTP)

ComfyUI's **Generate Text** node, extended with PE presets and MTP support for image prompts.

| Input | Description |
|---|---|
| `clip` | From the loader above, or any Qwen3.5 text encoder |
| `prompt` | Your instruction, as plain text |
| `image_1` … `image_10` | Input images, in order; the model refers to them as `<image1>`, `<image2>`…. Each can be a different size; any image over 1 MP is shrunk to 1 MP, as in the official pipeline. The i2i preset needs at least one image, and the t2i preset takes none. |
| `preset` | A PE preset, or `none` for Generate Text's own inputs |
| `mtp` | `auto`, `off`, or a fixed draft depth |

| Output | Description |
|---|---|
| `positive_prompt` | The `rewritten_prompt` from the answer, ready for the text encoder. If the answer has no valid JSON, this is the whole answer and a warning is logged. |
| `negative_prompt` | Always empty, because neither PE model writes one. It's there for workflows that expect the slot. |
| `thinking` | The model's reasoning |
| `wh_ratio` | The aspect ratio the model chose, such as `16:9`. Empty when the output follows an input image. |
| `ratio_follow` | i2i only: the input image whose aspect ratio the output keeps, such as `<image1>`. Empty otherwise. |
| `parse_ok` | `false` when the answer had no valid JSON |

These are the answer fields of the official `prompt_rewrite` output record.

**Image size.** Like the official PE, the node gives an aspect ratio rather than pixel dimensions; you set the size in your latent.

- **t2i:** size the latent to `wh_ratio`. It's part of the rewrite, because a prompt written for a wide composition gives a different picture on a square canvas.
- **i2i:** `ratio_follow` names the canvas image, the one whose framing the edit keeps. Connect that image as `image_1` of ComfyUI's **Text Encode Qwen Image 2.1** node, whose `latent` output matches the first reference image's size. When `ratio_follow` is empty, the model chose a new shape in `wh_ratio` instead.

## Presets

| Preset | max_length | temp | top_k | top_p | min_p | repetition | presence | thinking |
|---|---|---|---|---|---|---|---|---|
| Qwen-Image 2.1 PE (t2i) | 16256 | 1.0 | 20 | 0.95 | 0 | 1.0 | 1.5 | on |
| Qwen-Image 2.1 PE (i2i) | 24000 | 1.0 | 20 | 0.95 | 0 | 1.0 | 0 | on |
| none | *Generate Text's own inputs* | | | | | | | |

These are the official values from [`prompt_rewrite/pe_core.py`](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite), and every value can be edited on the node. The official system prompt is downloaded on first use.

> [!IMPORTANT]
> Keep **thinking** on. Both models were trained with a `<think>` block and degrade without it. The node logs a warning if it is turned off.

## Model files and disk usage

| What | Location | Size |
|---|---|---|
| Prepared model | `ComfyUI/models/text_encoders/Qwen-Image-2.1-PE/qwen3.5_9b_qwen_image_2.1_pe_{i2i,t2i}.int8_convrot.mtp.safetensors` | 9.3 GB each |
| In-progress download | Same folder, as `….mtp.safetensors.partial`, renamed when complete | up to 9.3 GB |
| System prompts | `ComfyUI/custom_nodes/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP/system_prompts/` | ~28 KB |
| Hugging Face cache | `~/.cache/huggingface/hub/` (index and prompt files only) | < 1 MB |

- The original checkpoint is streamed straight into the prepared file and never stored on its own. The MTP head is held in memory only during setup.
- If `qwen3.5_9b_qwen_image_2.1_pe_*.int8_convrot.safetensors` is already anywhere under your `text_encoders` folders, it is used as the source and nothing is downloaded.
- A prepared `.mtp.safetensors` file anywhere under your `text_encoders` folders is found and reused.
- New files go to the first `text_encoders` path ComfyUI knows about.
- To free the space, delete the `.mtp.safetensors` file.

## Performance

Generation speed, excluding prompt processing, on an RTX 4090 Laptop (16 GB) with ComfyUI's default dynamic VRAM. Everything else uses the preset defaults; i2i runs use one image scaled to 1 MP.

| Preset | max_length | MTP off | MTP on | Speed-up |
|---|---|---|---|---|
| t2i | 16256 (default) | 27.4 tok/s | 37.9 tok/s | 1.38× |
| t2i | 8192 | 34.8 tok/s | 47.4 tok/s | 1.36× |
| t2i | 4096 | 40.9 tok/s | 54.8 tok/s | 1.34× |
| i2i | 24000 (default) | 21.3 tok/s | 35.6 tok/s | 1.67× |
| i2i | 8192 | 31.3 tok/s | 51.5 tok/s | 1.65× |
| i2i | 4096 | 36.0 tok/s | 58.8 tok/s | 1.63× |

Peak VRAM was 12.8–13.9 GiB for t2i and 14.6–15.7 GiB for i2i.

Decoding slows as `max_length` grows, because ComfyUI attends over the whole allocated cache. Outputs including the reasoning were 1.7k–3.7k tokens, so lowering `max_length` to 8192 is much faster while leaving headroom. At 4096 a long answer can be cut off; raise `max_length` again if the JSON is ever incomplete.

With sampling on, MTP keeps the same output quality, but a given seed produces different text than with MTP off.

## Troubleshooting

<details>
<summary><b>"mtp is on but this Qwen3.5 checkpoint has no MTP head"</b></summary>

The checkpoint was loaded with a regular CLIP loader, so it runs without MTP and is slower. Load it with **Qwen-Image 2.1 PE Loader (MTP)** instead.
</details>

<details>
<summary><b>The JSON output is cut off</b></summary>

Generation hit `max_length`. Increase it on the node.
</details>

<details>
<summary><b>The download was interrupted</b></summary>

Run the workflow again. The `.partial` file is picked up and the download resumes where it stopped.
</details>

<details>
<summary><b>The node stopped working after a ComfyUI update</b></summary>

This extension relies on ComfyUI internals (`Qwen35._generate_mtp`, `process_tokens`, `compute_freqs_cis`). An update to those can break it; it fails with an error rather than producing wrong output silently. Please [open an issue](https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP/issues) with the error and your ComfyUI version.
</details>

## Development

The tests are plain Python scripts that print `ok` on success. Run them with ComfyUI's Python and point `COMFYUI_PATH` at your ComfyUI checkout (it defaults to two levels above this repo, which is correct when the repo sits in `custom_nodes`):

```bash
COMFYUI_PATH=/path/to/ComfyUI python tests/test_positions.py
```

| Module | Purpose |
|---|---|
| `__init__.py` | Node definitions and presets |
| `mtp.py` | MTP decoding for image prompts (MRoPE-aware position table) |
| `pe.py` | First-run model setup, official system prompts, image resizing and answer parsing |
| `graft.py` | Streaming safetensors grafting, free of ComfyUI imports |
| `tools/graft_mtp.py` | Command-line wrapper around `graft.py` |

## Contributing

Bug reports and pull requests are welcome. For anything beyond a small fix, please open an issue first to discuss the change. Before submitting, run the test scripts and describe how you tested it in ComfyUI.

## License

The code in this repository is released under the [MIT License](LICENSE).

The prompt enhancer weights and system prompts are released under the non-commercial [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/blob/main/LICENSE). This repository contains neither; both are downloaded to your machine on first use.

## Acknowledgements

- [Qwen team](https://github.com/QwenLM/Qwen-Image-2.1) for Qwen-Image 2.1, the prompt enhancers and the reference implementation
- [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) for the int8 checkpoints
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the Qwen3.5 text encoder and MTP decoding this extension builds on
