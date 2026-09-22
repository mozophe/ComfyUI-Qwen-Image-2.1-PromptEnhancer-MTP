# ComfyUI-Qwen35-MTP — design

## Problem

ComfyUI's `Generate Text` supports MTP speculative decoding for Qwen3.5, but:

1. It is disabled whenever the prompt contains an image. Image prompts carry 3D MRoPE
   `position_ids`, and `Qwen35.generate` (`comfy/text_encoders/qwen35.py`) falls back to the
   plain one-token loop when `position_ids` is set; `_generate_mtp` only knows 1D positions.
2. Many fine-tunes (e.g. the Qwen-Image 2.1 prompt enhancers) ship without the `mtp.*` weights.

ComfyUI core must not be modified.

## Measured (spike, RTX 4090 Laptop, Qwen3.5-9B PE int8, top_k 20 / top_p 0.95 / min_p 0.05 / rep 1.05)

| case | max_length | MTP off | MTP on |
|---|---|---|---|
| i2i | 2048 | 47.8 t/s | 72.7 t/s (2.08 tok/step) |
| i2i | 24000 | 24.1 t/s | 38.4 t/s |
| t2i | 2048 | ~48.5 t/s | 65–68 t/s (1.84–1.93 tok/step) |

The base model's MTP head (`Qwen/Qwen3.5-9B`) drafts well for the fine-tune. Decode cost also
scales with `max_length` (core allocates and attends over the full KV capacity), so the README
advises a `max_length` just above the expected output length.

## Components

### Node `Generate Text (Qwen3.5 MTP)` (`__init__.py`)

- Subclasses core `TextGenerate` (same pattern as core `TextGenerateLTX2Prompt`): same inputs and
  output, new `node_id` `TextGenerateQwen35MTP`, display name `Generate Text (Qwen3.5 MTP)`.
- `execute` calls the parent `execute` with the CLIP wrapped in a thin proxy. The proxy forwards
  everything to the real CLIP except `generate`.
- Proxy `generate`:
  - Model is not Qwen3.5, has no MTP head, `mtp` is off, or the prompt has no image →
    delegate to the real `clip.generate` unchanged (core already does text-only MTP).
  - Otherwise: same setup as `comfy.sd.CLIP.generate` (load model, clip options, device context,
    quantized matmul), build embeds with `process_tokens`, compute MRoPE ids with
    `qwen2vl_mrope_position_ids`, then call `transformer._generate_mtp` while
    `compute_freqs_cis` on that model instance is overridden for the duration of the call.
- Position mapping: a table indexed by sequence position. Columns `[0, L)` are the prompt's real
  MRoPE ids; column `L + i` is `max(ids) + 1 + i` on all three axes (what the core non-MTP loop
  feeds). The override returns freqs in the 1-row layout `(1, 1, seq, d)` that the decode paths
  slice. The instance attribute is removed in a `finally`.
- No silent fallback: if the private internals it relies on change, the error surfaces.

### Tools (`tools/`)

- `fetch_mtp.py <base repo> <out.safetensors>`: reads the base repo's index and fetches only the
  `mtp.*` tensors via HTTP range reads (~0.5 GB for 9B instead of ~15 GB of shards).
- `graft.py <checkpoint> <mtp.safetensors> <out>`: copies a checkpoint and appends the `mtp.*`
  tensors (bf16), preserving metadata; refuses key collisions. ComfyUI's loader detects the head
  via `mtp.fc.weight`.
- The base model must be the same size as the fine-tune. Weights are never committed.

## Scope

- Works with any Qwen3.5 size that has MTP weights (official or grafted). Other models pass
  through to core `generate`.
- Not included: runtime attachment of an MTP head to a non-MTP checkpoint, changes to the
  full-capacity attention cost, the unrelated core crash on a second `generate` in one process.

## Testing

- `tests/test_positions.py`: asserts the position table (prefix equals MRoPE ids, tail continues
  from `max + 1`, all three axes equal in the tail) and that the override's output layout matches
  the 1-row path — CPU only, no model.
- Manual: benchmark script (MTP on/off, image/no image) against the spike numbers; output
  coherence check. Greedy token equality is not a valid check: core text-only MTP already diverges
  from non-MTP greedy (bf16 batched verify vs single-token decode).

## Distribution

Private GitHub repo `ComfyUI-Qwen35-MTP` until tested; installed by cloning into
`ComfyUI/custom_nodes`. README: install, graft steps, `max_length` advice.
