# Qwen Export Utilities

This workspace contains a collection of scripts I use to export, validate, and debug the ExecuTorch versions of Qwen2.5-Omni. To keep things tidy the repo is now split into two top-level trees:

- `src/` – source code that generates the ExecuTorch artifacts (grouped by component).
- `tests/` – end-to-end and component-level validation scripts (mirrors the same component layout).

All run-time dumps, `.npy` blobs, `.pte` binaries, and logs stay in the repo root (and are ignored via `.gitignore`).

## Directory layout

```
src/
  preprocessor/   # Whisper/mel export utilities
  audio_tower/    # Qwen audio tower export utilities
  prefill/        # Prefill export pipeline (fp16)
  decode/         # Prefill+decode export pipeline (fp32 variant)
tests/
  preprocessor/   # Prompt/token validation
  audio_tower/    # Audio tower / mel comparisons
  prefill/        # Prefill layer diffs + KV checks
  decode/         # Decode/logit comparisons
  integration/    # Full ExecuTorch vs HF pipeline runners
```

## Export helpers (`src/…`)

| Path | Description |
| ---- | ----------- |
| `src/preprocessor/mel_spectrogram.py` | Standalone Whisper audio processor in PyTorch; used when exporting the whisper preprocess PTE. |
| `src/audio_tower/export_audio_tower.py` | Torch.export script that produces `whisper_preprocess.pte` + `qwen_audio_tower_30s_static.pte`. |
| `src/prefill/export_prefill_decode.py` | Exports the **fp16** prefill + decode graphs (Qwen text-only). Emits `llm_prefill_tokens_64p_750a_fp16.pte` and `llm_decode_cacheT1024_fp16.pte`. |
| `src/decode/export_prefill_decode_32.py` | FP32 variant of the above. Run this whenever you need matching fp32 prefill/decode PTEs; the pair must be generated together. |

## Validation & debug scripts (`tests/…`)

| Path | Description |
| ---- | ----------- |
| `tests/preprocessor/before_LLM.py` | Prompt tokenizer checker. Generates ExecuTorch vs HF `input_ids` and reports mismatches (plus dumps the arrays under `artifacts/prompt_check/`). |
| `tests/audio_tower/run_pte_audio_tower.py` | Runs Whisper preprocess + audio tower through ExecuTorch and HF, compares mel/audio embeddings, and writes “golden” `.npy` artifacts. |
| `tests/audio_tower/run_new.py` | Ad-hoc comparison utility that feeds alternate audio embeddings to pinpoint tower differences. |
| `tests/prefill/layer_diff*.py` | Per-layer diff scripts (bf16/fp32) to confirm prefill exports match the HF backbone. Logs sit in `tests/prefill/logs/`. |
| `tests/prefill/run_pte_prefill_decode.py` | Locks ExecuTorch and HF into the same forced tokens and checks MAE step-by-step. Generates the logits `.npy` that other tools consume. |
| `tests/decode/logits_debug.py` | Compares two logits dumps (`et_logitsX.npy` vs `hf_logitsX.npy`), prints MAE/MSE/relative error, and decodes the top divergent tokens. |
| `tests/decode/metal_prefill.py` | Experimental Metal / PyTorch reference runner for decode experiments. |
| `tests/integration/run_full_pipeline.py` | Main end-to-end driver: WAV → ExecuTorch whisper/audio tower → prefill/decode, with optional HF reference run and forced tokens. |
| `tests/integration/full_pipeline_debug.py` | Heavyweight debugger: runs the full pipeline, dumps every artifact (prompt ids, mel, audio_ctx, per-layer KV, kv_vis, logits, tokens), then replays the identical state through HF to compute MAE/cosine per step. |

Supporting logs such as `pipeline_output.txt`, `layer_diff_fp32.txt`, and `run_pte_prefill_decode_output.txt` live inside the corresponding `tests/*/logs/` folders and remain ignored by git.

## Usage Notes

1. **Keep PTE pairs in sync** – whenever you regenerate a prefill export, re-run the matching decode export (fp16 or fp32) so their KV tensor counts/dtypes match. Mixing old/new `.pte` files will cause runtime mismatches.
2. **Use the right runner for the job** – stick to `tests/integration/run_full_pipeline.py` for everyday inference. Switch to `full_pipeline_debug.py` only when you need to dump/replay every tensor.
3. **HF reference options** – most runners accept flags such as `--hf_compare`, `--hf_tokens`, `--hf_dump_dir`, and `--hf_force_tokens`. Check each script’s `--help` for details; all paths are relative to the repo root.

Feel free to extend this README as new utilities appear. The goal is to keep export code under `src/`, validation scripts under `tests/`, and treat `.npy`/`.pte`/log outputs as build artifacts (ignored via `.gitignore`). Happy debugging!
