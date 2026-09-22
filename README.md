# ComfyUI-Qwen35-MTP

One node, **Generate Text (Qwen3.5 MTP)**: ComfyUI's Generate Text with two additions for Qwen3.5 text encoders.

- **MTP speculative decoding for image prompts.** Core Generate Text turns MTP off whenever an image is attached. This node keeps it on, feeding the model the correct MRoPE positions.
- **Qwen-Image 2.1 prompt enhancer presets.** Uses the official system prompt, the official chat layout (thinking on) and the official sampling defaults.

Any other model, and any prompt without an image, goes through ComfyUI's normal `generate`.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/mozophe/ComfyUI-Qwen35-MTP
```

This repo ships no weights and no system prompts. The tools below fetch them locally.

## MTP weights

MTP needs the `mtp.*` tensors in the checkpoint. The Qwen-Image 2.1 PE checkpoints don't include them, so graft them in from the base model **of the same size**:

```
cd ComfyUI/custom_nodes/ComfyUI-Qwen35-MTP/tools
python fetch_mtp.py Qwen/Qwen3.5-9B qwen35_9b_mtp.safetensors          # ~0.5 GB, only the mtp.* tensors
python graft.py <pe_i2i.safetensors> qwen35_9b_mtp.safetensors <pe_i2i.mtp.safetensors>
```

Put the output in `models/text_encoders` and load it with `Load CLIP` (type `qwen_image`). ComfyUI detects the head automatically. Checkpoints without `mtp.*` tensors still work, just without MTP.

## Presets

```
python tools/fetch_prompts.py      # official system prompts -> system_prompts/
```

| preset | system prompt | max_length | temp | top_k | top_p | min_p | repetition | presence | thinking |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-Image 2.1 PE (edit) | PE-I2I | 24000 | 1.0 | 20 | 0.95 | 0 | 1.0 | 0 | on |
| Qwen-Image 2.1 PE (t2i) | PE-T2I | 16256 | 1.0 | 20 | 0.95 | 0 | 1.0 | 1.5 | on |
| none | – | core Generate Text inputs | | | | | | | |

The values come from the official [`prompt_rewrite/pe_core.py`](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite) and can all be edited. The official README says: "Thinking is required: both models were trained with a `<think>` block and degrade without it." The node warns if you turn thinking off.

Pair each preset with its own checkpoint: the edit preset with the `pe_i2i` checkpoint, the t2i preset with `pe_t2i`.

The output is the reasoning followed by `</think>` and a JSON object (`rewritten_prompt`, `wh_ratio`, and `ratio_follow` for edit). To pull out the prompt, strip everything up to `</think>` with `Replace Text (Regex)` (pattern `(?s).*</think>\s*`, empty replacement), then use `Extract Text from JSON` with key `rewritten_prompt`.

The official pipeline downscales input images to at most 1 MP (LANCZOS). ComfyUI doesn't, so put `ImageScaleToTotalPixels` (1.0 megapixels, lanczos) in front of the node for large images.

## Speed

Measured on an RTX 4090 Laptop with the Qwen3.5-9B PE (int8) and the edit preset, one image:

| max_length | MTP off | MTP on |
|---|---|---|
| 24000 (official) | 21.3 t/s | 36.7 t/s |
| 2048 | 47.8 t/s | 72.7 t/s |

Decode cost grows with `max_length`, because ComfyUI attends over the whole allocated KV cache. Outputs, reasoning included, were 1.9k–3.3k tokens, so setting `max_length` to around 8192 is much faster. Raise it again if the JSON ever comes out truncated.

With sampling on, MTP keeps the output distribution the same, but a given seed produces different text than with MTP off.

## Limitations

- Relies on ComfyUI internals (`Qwen35._generate_mtp`, `process_tokens`, `compute_freqs_cis`). A ComfyUI update can break it, and it fails loudly rather than silently falling back.
- The official system prompts and the PE checkpoints are under the non-commercial [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/blob/main/LICENSE). Grafted checkpoints are derived from them.
