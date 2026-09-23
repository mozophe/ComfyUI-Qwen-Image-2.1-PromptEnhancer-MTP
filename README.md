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
[Performance](#performance) •
[Troubleshooting](#troubleshooting)

</div>

---

## Overview

Qwen-Image 2.1 works best with long, detailed prompts. Its official prompt enhancer (PE) writes them for you: give it a short idea or edit instruction, and it returns a detailed prompt along with the aspect ratio to use.

This extension runs the PE inside ComfyUI with Qwen's own system prompts and settings, and makes it faster with **multi-token prediction (MTP)**: about **1.35×** for text-to-image and **1.65×** for image editing.

- **Automatic setup.** The model downloads and prepares itself on first use.
- **Official settings.** System prompts and sampling values match Qwen's reference code.
- **Ready-to-use outputs.** The rewritten prompt and aspect ratio come out as separate outputs.
- **Multi-image editing.** Up to 10 input images, any size.

## Requirements

- A recent version of ComfyUI (with Qwen3.5 MTP support)
- **16 GB of VRAM** recommended. Smaller GPUs work, but much more slowly.
- **32 GB of system RAM** recommended
- About **9.3 GB** of disk space per model (`t2i` for text-to-image, `i2i` for editing)

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP
pip install -r ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP/requirements.txt
```

Restart ComfyUI. The first run downloads the model (progress is shown on the node). If the download is interrupted, run the workflow again and it resumes.

To update, run `git pull` in the extension's folder.

## Quick start

### Sample workflows

The easiest way to start is to drag a sample workflow from the [`workflows`](workflows) folder into ComfyUI.

| Workflow | Use it for |
|---|---|
| [`qwen_image_2.1_t2i_prompt_enhancer.json`](workflows/qwen_image_2.1_t2i_prompt_enhancer.json) | Text-to-image |
| [`qwen_image_2.1_edit_prompt_enhancer.json`](workflows/qwen_image_2.1_edit_prompt_enhancer.json) | Editing with two input images |

Both are ComfyUI's official Qwen-Image 2.1 templates with the enhancer added in front. They use the int8 models from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).

### Adding it to your own workflow

```
Qwen-Image 2.1 PE Loader (MTP) ──clip──► Qwen-Image 2.1 Prompt Enhancer (MTP) ──positive_prompt──► your text encoder
                           Load Image ──image_1──┘   (editing only)
```

1. Choose the same mode on both nodes: the `t2i` loader with the t2i preset, or the `i2i` loader with the i2i preset.
2. Write a short instruction in `prompt`:
   - **Text-to-image:** *"a fox reading a book in a snowy forest, watercolor"*
   - **Editing:** *"Put the woman from `<image2>` into the street in `<image1>`"*. Images are numbered by the input they are connected to, and can be any size.
3. Use `positive_prompt` as your prompt, and size the image as described below.

### Sizing the image

The enhancer chooses an aspect ratio rather than exact pixel sizes, and the prompt it writes is composed for that shape.

- **Text-to-image:** set your latent to the ratio in `wh_ratio`, for example `16:9`.
- **Editing:** `ratio_follow` names the image whose framing to keep, for example `<image1>`. Connect that image as `image_1` of **Text Encode Qwen Image 2.1** and use its `latent` output. If `ratio_follow` is empty, use `wh_ratio` instead.

## Nodes

### Qwen-Image 2.1 PE Loader (MTP)

Loads the prompt enhancer, ready for fast MTP generation.

| Input | Description |
|---|---|
| `model` | `t2i` for text-to-image, `i2i` for editing |

On first use it downloads the model (about 9.5 GB) and saves a prepared copy in `ComfyUI/models/text_encoders/Qwen-Image-2.1-PE/`. If you already have the Comfy-Org PE checkpoint in a `text_encoders` folder, it is used instead of downloading. To free the space, delete the `.mtp.safetensors` file.

### Qwen-Image 2.1 Prompt Enhancer (MTP)

ComfyUI's **Generate Text** node with the official PE presets built in.

| Input | Description |
|---|---|
| `clip` | The enhancer from the loader |
| `prompt` | Your short instruction |
| `image_1` … `image_10` | Input images for editing, referred to as `<image1>`, `<image2>`… |
| `preset` | `Qwen-Image 2.1 PE (t2i)`, `Qwen-Image 2.1 PE (i2i)`, or `none` to use it as a plain Generate Text node |
| `seed` | Change it for a different result |
| `mtp` | `auto` (recommended), `off`, or a fixed draft depth |

| Output | Description |
|---|---|
| `positive_prompt` | The detailed prompt, ready for your text encoder |
| `negative_prompt` | Always empty; the enhancer doesn't write one |
| `thinking` | The model's reasoning |
| `wh_ratio` | The chosen aspect ratio, for example `16:9` |
| `ratio_follow` | Editing only: the image whose shape to keep, for example `<image1>` |
| `parse_ok` | `false` if the answer couldn't be read; `positive_prompt` then contains the full answer |

The presets use the official values from Qwen's [`prompt_rewrite`](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite) code, and every value can be changed on the node.

> [!IMPORTANT]
> Keep **thinking** on. Both models were trained to reason before answering and give worse prompts without it.

## Performance

Measured on an RTX 4090 Laptop GPU (16 GB), with one input image for editing.

| Mode | `max_length` | MTP off | MTP on | Speed-up |
|---|---|---|---|---|
| Text-to-image | 16256 (default) | 27 tok/s | 38 tok/s | 1.38× |
| Text-to-image | 8192 | 35 tok/s | 47 tok/s | 1.36× |
| Editing | 24000 (default) | 21 tok/s | 36 tok/s | 1.67× |
| Editing | 8192 | 31 tok/s | 52 tok/s | 1.65× |

**Tip:** set `max_length` to **8192** for faster results. A typical answer is 2,000–4,000 tokens, so this leaves plenty of room. If an answer is ever cut off, raise it again.

Peak VRAM use was about 14 GB for text-to-image and 16 GB for editing. With MTP on, quality is unchanged, but the same seed gives different text than with MTP off.

## Troubleshooting

<details>
<summary><b>"mtp is on but this Qwen3.5 checkpoint has no MTP head"</b></summary>

The model was loaded with a regular CLIP loader. Use **Qwen-Image 2.1 PE Loader (MTP)** instead.
</details>

<details>
<summary><b>The output is cut off, or <code>parse_ok</code> is false</b></summary>

The answer reached `max_length`. Increase it on the node.
</details>

<details>
<summary><b>The node stopped working after a ComfyUI update</b></summary>

This extension depends on parts of ComfyUI that can change between versions. Please [open an issue](https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP/issues) with the error message and your ComfyUI version.
</details>

## License

The code in this repository is released under the [MIT License](LICENSE).

The prompt enhancer weights and system prompts are released under the non-commercial [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/blob/main/LICENSE). They are not included in this repository; they are downloaded on first use.

## Acknowledgements

- [Qwen team](https://github.com/QwenLM/Qwen-Image-2.1) for Qwen-Image 2.1, the prompt enhancers and the reference implementation
- [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) for the int8 checkpoints
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the Qwen3.5 text encoder and MTP decoding this extension builds on
