# Qwen Export Utilities

This workspace contains a collection of scripts I use to inspect and debug the ExecuTorch exports of Qwen2.5-Omni. They fall roughly into three categories:

- **Runtime drivers** – run the full audio→ExecuTorch→LLM pipeline and compare against HuggingFace.
- **Validation / diff tools** – check individual components (audio tower, prefill, decode) against HF baselines.
- **Export helpers** – generate the `.pte` artifacts (whisper preprocess, audio tower, prefill, decode).

Below is a quick guide to the key files.

## Runtime / Debug drivers

| Script | Purpose |
| ------ | ------- |
| `run_full_pipeline.py` | Main end-to-end driver: loads a WAV, runs Whisper preprocess + audio tower + ExecuTorch prefill/decode, and (optionally) runs the HF reference model for comparison or force-decoding tokens. This is the script behind typical pipeline runs. |
| `full_pipeline_debug.py` | Heavyweight debug runner. It executes the same pipeline but dumps **all** intermediate artifacts (prompt ids, mel, audio_ctx, every KV tensor, kv_vis, logits, generated tokens) and immediately replays the identical state through the HF model to compute per-step MAE/cosine. Use when you need to pinpoint where ET vs HF diverge. |
| `run_pte_audio_tower.py` | Runs Whisper preprocess + audio tower through ExecuTorch and HF, compares mel/audio embeddings, and saves “golden” artifacts (mel.npy, audio_emb_exec.npy, audio_emb_ref.npy). Great for verifying the audio front-end independent of the LLM. |
| `run_pte_prefill_decode.py` | Locks ExecuTorch and HF into the exact same token sequence and compares logits step-by-step (MAE ≈ 1e‑5 when healthy). Provides the `et_logits*.npy` / `hf_logits*.npy` dumps that `logits_debug.py` reads. |
| `logits_debug.py` | Convenience script to compare two `.npy` logits tensors (et vs hf). Prints MAE/MSE/relative error and decodes the top differing tokens, so you know which IDs drifted. |

## Validation scripts

| Script | Purpose |
| ------ | ------- |
| `layer_diff.py` / `layer_diff_fp16.py` / `layer_diff_fp32.py` | Compare per-layer hidden states between ExecuTorch prefill exports and HuggingFace under controlled inputs. Useful for confirming a specific export (fp16/bf16/fp32) matches the HF backbone layer-by-layer. |
| `before_LLM.py` | Quick prompt-id checker. Generates ExecuTorch vs HF token sequences (with optional chat-template differences) and raises a warning if they diverge. Keeps you from blaming the LLM when the tokenizer setup is the culprit. |
| `layer_diff_bf16.py`, `layer_diff_bf16.txt`, `layer_diff_fp16.txt`, etc. | Saved output logs/results for the various layer-diff runs. |

## Export helpers

| Script | Purpose |
| ------ | ------- |
| `export_prefill_decode.py` | Torch.export wrapper for the **fp16** prefill + decode graphs. Runs HF Qwen text-only, traces Prefill/Decode wrappers, and emits `.pte` artifacts (`llm_prefill_tokens_64p_750a_fp16.pte`, `llm_decode_cacheT1024_fp16.pte`). |
| `export_prefill_decode_32.py` | Same as above but in fp32. Use this when you need matching fp32 exports; the decode PTE and prefill PTE must be generated together to ensure they agree on KV tensor counts. |
| `export_audio_tower.py` | Exports the Whisper preprocess PTE (`whisper_preprocess.pte`) and audio tower PTE (`qwen_audio_tower_30s_static.pte`) using torch.export + ExecuTorch lowering. |
| `mel_spectrogram.py` | Standalone Whisper audio processor (PyTorch implementation) used during the export. |

## Miscellaneous

- `run_new.py`, `metal_prefill.py`, `prefill.py`: experimental scripts for alternative runtimes (Metal, PyTorch CPU reference) during export and debugging.
- `results.txt`, `pipeline_output.txt`, `run_pte_prefill_decode_output.txt`: sample logs from previous runs (ignored in `.gitignore`).
- `.gitignore`: excludes generated artifacts (`artifacts/`, `*.npy`, `*.pte`, logs, etc.) so the repo tracks only source scripts/configs.

## Usage Notes

1. **Keep PTE pairs in sync**: whenever you regenerate a prefill PTE, re-export the matching decode PTE so they agree on KV tensor counts and dtypes (fp16 vs fp32).
2. **Dump artifacts only when debugging**: `full_pipeline_debug.py` writes dozens of large `.npy` files. Stick to `run_full_pipeline.py` for day-to-day inference and switch to the debug runner only when you need deep visibility.
3. **HF compare options**: most runtime scripts accept `--hf_compare`, `--hf_tokens`, `--hf_dump_dir` to control reference runs. Check `run_full_pipeline.py` for the full argument list.

Feel free to extend this README as new utilities are added. The goal is to keep the executable scripts self-contained while treating model weights and run artifacts as build outputs (see `.gitignore`). Happy debugging!
