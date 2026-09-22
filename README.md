# ComfyUI-Qwen35-MTP

A ready-to-use **Qwen-Image 2.1 prompt enhancer** for ComfyUI, sped up with MTP speculative decoding.

- **Load Qwen-Image 2.1 PE (MTP)** — pick `t2i` or `i2i`. On first run it downloads the prompt enhancer (9.5 GB, from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/text_encoders)) and adds the MTP head from `Qwen/Qwen3.5-9B` (~0.5 GB) in one pass. Later runs just load it.
- **Prompt Enhancer (Qwen3.5 MTP)** — ComfyUI's Generate Text with presets for the prompt enhancer (official system prompt, thinking and sampling settings) and MTP that also works when an image is attached.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/mozophe/ComfyUI-Qwen35-MTP
```

Restart ComfyUI.

## Use

```
Load Qwen-Image 2.1 PE (MTP) [i2i] ──clip──► Prompt Enhancer (Qwen3.5 MTP) [preset: Qwen-Image 2.1 PE (i2i)]
Load Image ─► ImageScaleToTotalPixels (1.0 MP, lanczos) ──image──┘
```

Write your instruction as plain text ("Make this image a realistic photo"). For text-to-image, use `t2i` in both nodes and no image.

The first run takes a while because it downloads and prepares the model (progress shows on the node). If the download is interrupted, it resumes. If `qwen3.5_9b_qwen_image_2.1_pe_*.int8_convrot.safetensors` is already anywhere under your `text_encoders` folders, that file is used and nothing is downloaded.

### Where files go

| What | Where | Size |
|---|---|---|
| Prepared model | `ComfyUI/models/text_encoders/Qwen-Image-2.1-PE/qwen3.5_9b_qwen_image_2.1_pe_{i2i,t2i}.int8_convrot.mtp.safetensors` | 9.3 GB each |
| During the download | the same folder, as `….mtp.safetensors.partial`, renamed when complete | up to 9.3 GB |
| System prompts | `ComfyUI/custom_nodes/ComfyUI-Qwen35-MTP/system_prompts/` | ~28 KB |
| Hugging Face cache | `~/.cache/huggingface/hub/` (small index and prompt files only) | < 1 MB |

The original checkpoint is streamed straight into the prepared file and never saved on its own. The MTP head is only held in memory during setup, so each model needs about 9.3 GB of disk. The folder is the first `text_encoders` path ComfyUI knows about. A prepared model already present anywhere under your `text_encoders` folders is found and reused. To free the space, delete the `.mtp.safetensors` file.

The node has two outputs:

- **generated_text** — the answer with the thinking removed. With a PE preset that is the JSON object (`rewritten_prompt`, `wh_ratio`, and for i2i `ratio_follow`).
- **generated_text_with_thinking** — the full output, reasoning included, for when you want to see how the model got there.

## Presets

| preset | max_length | temp | top_k | top_p | min_p | repetition | presence | thinking |
|---|---|---|---|---|---|---|---|---|
| Qwen-Image 2.1 PE (t2i) | 16256 | 1.0 | 20 | 0.95 | 0 | 1.0 | 1.5 | on |
| Qwen-Image 2.1 PE (i2i) | 24000 | 1.0 | 20 | 0.95 | 0 | 1.0 | 0 | on |
| none | Generate Text's own inputs | | | | | | | |

Values are the official ones from [`prompt_rewrite/pe_core.py`](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite) and are editable. The official system prompt is downloaded on first use. Thinking should stay on: "both models were trained with a `<think>` block and degrade without it."

## Speed

RTX 4090 Laptop, i2i preset, one image:

| max_length | MTP off | MTP on |
|---|---|---|
| 24000 (default) | 21.3 t/s | 36.7 t/s |
| 8192 | – | 52.9 t/s |
| 2048 | 47.8 t/s | 72.7 t/s |

Decoding gets slower the higher `max_length` is, because ComfyUI attends over the whole allocated cache. Outputs including the reasoning were 1.9k–3.3k tokens, so lowering `max_length` to ~8192 is much faster; raise it if the JSON is ever cut off. With sampling on, MTP keeps the same output quality, but a seed gives different text than with MTP off.

## Other Qwen3.5 fine-tunes

`mtp` works with any Qwen3.5 checkpoint that has `mtp.*` tensors. To add them to another fine-tune, use the base model of the same size:

```
python tools/graft_mtp.py <finetune.safetensors> Qwen/Qwen3.5-4B <finetune.mtp.safetensors>
```

## Notes

- Relies on ComfyUI internals (`Qwen35._generate_mtp`, `process_tokens`, `compute_freqs_cis`). A ComfyUI update may break it; it fails with an error rather than silently.
- The prompt enhancer weights and system prompts are under the non-commercial [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/blob/main/LICENSE). This repo contains neither; they are downloaded to your machine.
