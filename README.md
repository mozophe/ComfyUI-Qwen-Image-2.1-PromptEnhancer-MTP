<div align="center">

# ComfyUI-QwenImage-2.1-PromptEnhancer

**The official Qwen-Image 2.1 prompt enhancer for ComfyUI, accelerated with MTP speculative decoding.**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/mozophe/ComfyUI-QwenImage-2.1-PromptEnhancer)](https://github.com/mozophe/ComfyUI-QwenImage-2.1-PromptEnhancer/commits/main)
[![Issues](https://img.shields.io/github/issues/mozophe/ComfyUI-QwenImage-2.1-PromptEnhancer)](https://github.com/mozophe/ComfyUI-QwenImage-2.1-PromptEnhancer/issues)

[Installation](#installation) •
[Quick start](#quick-start) •
[Nodes](#nodes) •
[Presets](#presets) •
[Performance](#performance) •
[Troubleshooting](#troubleshooting)

</div>

---

## Overview

Qwen-Image 2.1 ships two prompt enhancers (PE), Qwen3.5-9B fine-tunes that rewrite a short instruction into the detailed prompt Qwen-Image 2.1 was trained on. This extension runs them inside ComfyUI with the official system prompts and sampling settings. It also adds **multi-token prediction (MTP)**, which roughly doubles decoding speed at long lengths.

### Features

- **One-click setup.** The model downloads and prepares itself on first use. Downloads resume after an interruption, and existing local copies are reused.
- **Official presets.** The system prompt, thinking mode and sampling values come from Qwen's reference implementation.
- **MTP for image prompts.** MTP speculative decoding also works when an image is attached; core ComfyUI falls back to regular decoding in that case.
- **Clean outputs.** The node returns the answer with the reasoning removed, and separately the full output with the reasoning included.
- **Reusable tooling.** A standalone script adds an MTP head to any Qwen3.5 fine-tune.

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Nodes](#nodes)
- [Presets](#presets)
- [Model files and disk usage](#model-files-and-disk-usage)
- [Performance](#performance)
- [Other Qwen3.5 fine-tunes](#other-qwen35-fine-tunes)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Requirements

- A recent ComfyUI build that includes the Qwen3.5 text encoder with MTP support (`comfy/text_encoders/qwen35.py`)
- About **9.3 GB** of free disk space per model (`t2i`, `i2i`)
- An internet connection for the first run only

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mozophe/ComfyUI-QwenImage-2.1-PromptEnhancer
```

Restart ComfyUI. No extra Python packages are needed.

To update:

```bash
cd ComfyUI/custom_nodes/ComfyUI-QwenImage-2.1-PromptEnhancer
git pull
```

## Quick start

### Image editing (i2i)

```
Qwen-Image 2.1 PE Loader (MTP) [i2i] ──clip──► Qwen-Image 2.1 Prompt Enhancer [preset: Qwen-Image 2.1 PE (i2i)]
Load Image ─► ImageScaleToTotalPixels (1.0 MP, lanczos) ──image──┘
```

### Text-to-image (t2i)

Select `t2i` on both nodes and leave the image input unconnected.

Write your instruction as plain text, for example *"Make this image a realistic photo"*. The first run takes a while because it downloads and prepares the model; progress is shown on the node.

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

### Qwen-Image 2.1 Prompt Enhancer

ComfyUI's **Generate Text** node, extended with PE presets and MTP support for image prompts.

| Input | Description |
|---|---|
| `clip` | From the loader above, or any Qwen3.5 text encoder |
| `prompt` | Your instruction, as plain text |
| `image` / `video` / `audio` | Optional media inputs |
| `preset` | A PE preset, or `none` for Generate Text's own inputs |
| `mtp` | `auto`, `off`, or a fixed draft depth |

| Output | Description |
|---|---|
| `generated_text` | The answer without the reasoning. With a PE preset this is a JSON object with `rewritten_prompt` and `wh_ratio`, plus `ratio_follow` for i2i. |
| `generated_text_with_thinking` | The full output, reasoning included |

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
| System prompts | `ComfyUI/custom_nodes/ComfyUI-QwenImage-2.1-PromptEnhancer/system_prompts/` | ~28 KB |
| Hugging Face cache | `~/.cache/huggingface/hub/` (index and prompt files only) | < 1 MB |

- The original checkpoint is streamed straight into the prepared file and never stored on its own. The MTP head is held in memory only during setup.
- If `qwen3.5_9b_qwen_image_2.1_pe_*.int8_convrot.safetensors` is already anywhere under your `text_encoders` folders, it is used as the source and nothing is downloaded.
- A prepared `.mtp.safetensors` file anywhere under your `text_encoders` folders is found and reused.
- New files go to the first `text_encoders` path ComfyUI knows about.
- To free the space, delete the `.mtp.safetensors` file.

## Performance

RTX 4090 Laptop, i2i preset, one image:

| max_length | MTP off | MTP on |
|---|---|---|
| 24000 (default) | 21.3 tok/s | 36.7 tok/s |
| 8192 | – | 52.9 tok/s |
| 2048 | 47.8 tok/s | 72.7 tok/s |

Decoding slows as `max_length` grows, because ComfyUI attends over the whole allocated cache. Outputs including the reasoning were 1.9k–3.3k tokens, so lowering `max_length` to about 8192 is much faster. Raise it again if the JSON is ever cut off.

With sampling on, MTP keeps the same output quality, but a given seed produces different text than with MTP off.

## Other Qwen3.5 fine-tunes

The `mtp` option works with any Qwen3.5 checkpoint that contains `mtp.*` tensors. To add them to another fine-tune, graft them from the base model of the same size:

```bash
python tools/graft_mtp.py <finetune.safetensors> Qwen/Qwen3.5-4B <finetune.mtp.safetensors>
```

## Troubleshooting

<details>
<summary><b>"mtp is on but this Qwen3.5 checkpoint has no MTP head"</b></summary>

The checkpoint was loaded with a regular CLIP loader, so it runs without MTP and is slower. Load it with **Qwen-Image 2.1 PE Loader (MTP)** instead, or add a head with `tools/graft_mtp.py`.
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

This extension relies on ComfyUI internals (`Qwen35._generate_mtp`, `process_tokens`, `compute_freqs_cis`). An update to those can break it; it fails with an error rather than producing wrong output silently. Please [open an issue](https://github.com/mozophe/ComfyUI-QwenImage-2.1-PromptEnhancer/issues) with the error and your ComfyUI version.
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
| `pe.py` | First-run model setup and official system prompts |
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
