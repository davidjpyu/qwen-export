#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the ExecuTorch full pipeline with real audio/prompt, dump every
prefill/decode artifact (input_ids, audio_ctx, KV tensors, kv_vis, logits,
generated tokens), and immediately replay the same state through the HuggingFace
reference model for comparison.
"""

import argparse
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, Qwen2_5OmniForConditionalGeneration
from transformers.cache_utils import DynamicCache

SR = 16000
DURATION_S = 30


# ----------------- ExecuTorch helpers -----------------
def et_load_forward(pte_path: str):
    from executorch.runtime import Runtime

    rt = Runtime.get()
    prog = rt.load_program(pte_path)
    return prog.load_method("forward")


def to_torch_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    try:
        return x.to_torch()
    except Exception:
        return torch.tensor(x)


def et_call(method, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    out_list = method.execute(list(inputs))
    return [to_torch_tensor(x) for x in out_list]


def torch_dtype_short_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.int32:
        return "int32"
    return str(dtype)


def infer_method_input_dtype(method, index: int) -> Optional[torch.dtype]:
    meta = getattr(method, "metadata", None)
    if meta is None:
        return None
    try:
        info = meta.input_tensor_meta(index)
    except Exception:
        return None
    info_str = str(info)
    if "dtype=Half" in info_str:
        return torch.float16
    if "dtype=Float" in info_str:
        return torch.float32
    if "dtype=BFloat16" in info_str:
        return torch.bfloat16
    return None


# ----------------- Audio utilities -----------------
def load_wave(wav_path: Optional[str], silence: bool, duration_s: int = DURATION_S, sr: int = SR) -> torch.Tensor:
    need = sr * duration_s
    if wav_path:
        try:
            import soundfile as sf
        except Exception as exc:
            raise RuntimeError("Please install soundfile to read wav inputs (pip install soundfile)") from exc
        wave_np, actual_sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if actual_sr != sr:
            raise ValueError(f"Expected {sr}Hz audio, got {actual_sr}")
        if wave_np.ndim == 2:
            wave_np = wave_np.mean(axis=1).astype(np.float32)
        if len(wave_np) >= need:
            wave_np = wave_np[:need]
        else:
            pad = np.zeros(need - len(wave_np), dtype=np.float32)
            wave_np = np.concatenate([wave_np, pad], axis=0)
        return torch.from_numpy(wave_np)
    if silence:
        return torch.zeros(need, dtype=torch.float32)
    torch.manual_seed(0)
    return torch.randn(need, dtype=torch.float32)


def run_whisper_preprocess(proc_method, wave: torch.Tensor) -> torch.Tensor:
    mel = et_call(proc_method, [wave.to(torch.float32)])[0]
    mel = to_torch_tensor(mel).to(torch.float32).contiguous()
    if mel.dim() != 3 or mel.shape[1] != 128:
        raise RuntimeError(f"Unexpected mel shape {tuple(mel.shape)}")
    return mel


def run_audio_tower(enc_method, mel: torch.Tensor) -> torch.Tensor:
    feats_CT = mel.squeeze(0)
    out = et_call(enc_method, [feats_CT.contiguous()])[0]
    return to_torch_tensor(out).to(torch.float32).contiguous()


def align_audio_ctx(audio_ctx: torch.Tensor, n_audio: int) -> torch.Tensor:
    cur = audio_ctx.shape[0]
    if cur == n_audio:
        return audio_ctx
    d = audio_ctx.shape[1]
    if cur > n_audio:
        return audio_ctx[:n_audio]
    pad = torch.zeros(n_audio - cur, d, dtype=audio_ctx.dtype)
    return torch.cat([audio_ctx, pad], dim=0)


# ----------------- KV helpers -----------------
def left_pad_kv_to_cap(kv_flat: List[torch.Tensor], kv_cap: int) -> Tuple[List[torch.Tensor], int]:
    out = []
    T0 = None
    for t in kv_flat:
        if t.dim() != 4:
            # non-KV tensors (e.g., extra head outputs) are left untouched
            out.append(t.contiguous())
            continue
        B, H, T, Dh = t.shape
        if T0 is None:
            T0 = T
        elif T0 != T:
            raise RuntimeError("Mismatched KV time lengths")
        if T > kv_cap:
            t = t[:, :, -kv_cap:, :]
            T = kv_cap
        if T < kv_cap:
            pad = torch.zeros(B, H, kv_cap - T, Dh, dtype=t.dtype)
            t = torch.cat([pad, t], dim=2)
        out.append(t.contiguous())
    return out, int(min(T0 or kv_cap, kv_cap))


def build_kv_vis(T_visible: int, kv_cap: int) -> torch.Tensor:
    kv_vis = torch.zeros(1, kv_cap + 1, dtype=torch.bool)
    if T_visible > 0:
        kv_vis[0, (kv_cap - T_visible): kv_cap] = True
    kv_vis[0, -1] = True
    return kv_vis


def unflatten_to_legacy(kv_flat: Sequence[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if len(kv_flat) % 2 != 0:
        raise RuntimeError("KV list must contain pairs of key/value tensors.")
    legacy = []
    for i in range(0, len(kv_flat), 2):
        legacy.append((kv_flat[i], kv_flat[i + 1]))
    return legacy


def kv_flat_to_hf_cache(kv_flat: Sequence[torch.Tensor], seen_tokens: int) -> DynamicCache:
    legacy = unflatten_to_legacy(kv_flat)
    cache = DynamicCache()
    cache.key_cache = [k for k, _ in legacy]
    cache.value_cache = [v for _, v in legacy]
    cache.seen_tokens = int(seen_tokens)
    return cache


# ----------------- Misc helpers -----------------
def build_prompt_tokens(
    tokenizer: AutoTokenizer, prompt: str, l_text: int, use_chat_template: bool
) -> Tuple[torch.Tensor, List[int]]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        full_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )[0].tolist()
    else:
        full_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0].tolist()
    if len(full_ids) > l_text:
        used = full_ids[-l_text:]
    elif len(full_ids) < l_text:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        used = full_ids + [pad_id] * (l_text - len(full_ids))
    else:
        used = full_ids
    tensor = torch.tensor([used], dtype=torch.int32)
    return tensor, used


def greedy_top1(logits: torch.Tensor) -> int:
    flat = logits.view(-1).float()
    return int(torch.argmax(flat, dim=-1).item())


def compare_logits(a: torch.Tensor, b: torch.Tensor, name: str) -> dict:
    a = a.float().view(-1)
    b = b.float().view(-1)
    diff = a - b
    mae = diff.abs().mean().item()
    mse = (diff * diff).mean().item()
    rel = (diff.norm() / (b.norm() + 1e-12)).item()
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
    max_abs = diff.abs().max().item()
    print(f"[COMPARE {name}] MAE={mae:.3e}  MSE={mse:.3e}  REL={rel:.3e}  MAX={max_abs:.3e}  COS={cos:.6f}")
    return {"mae": mae, "mse": mse, "rel": rel, "cos": cos, "max_abs": max_abs}


def dump_tensor(path: str, tensor: torch.Tensor):
    np.save(path, tensor.cpu().numpy())
    print(f"[DUMP] saved {path}")
    return path


def dump_kv_list(kv_list: Sequence[torch.Tensor], outdir: str, prefix: str) -> List[str]:
    paths = []
    for idx, tensor in enumerate(kv_list):
        path = os.path.join(outdir, f"{prefix}_layer{idx}.npy")
        dump_tensor(path, tensor)
        paths.append(path)
    return paths


# ----------------- HF helpers -----------------
def load_hf_text_only(model_id: str, revision: str):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        config=cfg,
        trust_remote_code=True,
        revision=revision,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).eval()
    thinker = getattr(omni, "thinker", getattr(omni, "model", omni))
    for name in ["audio_tower", "vision_tower", "audio_projector", "visual_projector"]:
        if hasattr(thinker, name):
            try:
                setattr(thinker, name, None)
                delattr(thinker, name)
            except Exception:
                pass
    if hasattr(omni, "perception"):
        try:
            omni.perception = None
            delattr(omni, "perception")
        except Exception:
            pass

    lm_backbone = getattr(thinker, "model", thinker)
    lm_head = getattr(thinker, "lm_head", None) or getattr(thinker, "output", None) or getattr(omni, "lm_head", None)
    embed = getattr(lm_backbone, "embed_tokens", None) or getattr(thinker, "embed_tokens", None) \
        or getattr(lm_backbone, "tok_embeddings", None)
    if not isinstance(lm_head, nn.Module) or not isinstance(embed, nn.Embedding):
        raise RuntimeError("HF model missing lm_head or embed_tokens.")
    lm_backbone = lm_backbone.float().eval()
    lm_head = lm_head.float().eval()
    embed = embed.float().eval()
    return lm_backbone, lm_head, embed


# ----------------- Main debug runner -----------------
def main():
    parser = argparse.ArgumentParser(description="ExecuTorch vs HF pipeline debugger")
    parser.add_argument("--wav", type=str, default="audio_sample1.wav")
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--prompt", type=str, default="User: transcribe the following audio. Assistant:")
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--whisper_pte", type=str, default="whisper_preprocess.pte")
    parser.add_argument("--audio_tower_pte", type=str, default="qwen_audio_tower_30s_static.pte")
    parser.add_argument("--prefill_pte", type=str, default="llm_prefill_tokens_64p_750a_fp16.pte")
    parser.add_argument("--decode_pte", type=str, default="llm_decode_cacheT1024_fp16.pte")
    parser.add_argument("--prefill_dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--l_text", type=int, default=64)
    parser.add_argument("--n_audio", type=int, default=750)
    parser.add_argument("--kv_cap", type=int, default=1024)
    parser.add_argument("--decode_steps", type=int, default=1, help="How many decode steps to run/dump")
    parser.add_argument("--dump_dir", type=str, default="artifacts/full_debug")
    args = parser.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)
    dtype = torch.float32 if args.prefill_dtype == "fp32" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, revision=args.revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("[INFO] loading audio + ExecuTorch programs...")
    wave = load_wave(args.wav, args.silence)
    proc_method = et_load_forward(args.whisper_pte)
    enc_method = et_load_forward(args.audio_tower_pte)
    prefill_method = et_load_forward(args.prefill_pte)
    decode_method = et_load_forward(args.decode_pte)

    inferred_dtype = infer_method_input_dtype(prefill_method, 1)
    if inferred_dtype is not None and inferred_dtype != dtype:
        print(
            f"[WARN] Prefill PTE expects {torch_dtype_short_name(inferred_dtype)} audio_ctx; "
            f"--prefill_dtype was {args.prefill_dtype}, overriding."
        )
        dtype = inferred_dtype

    mel = run_whisper_preprocess(proc_method, wave)
    audio_ctx = run_audio_tower(enc_method, mel).to(dtype)
    audio_ctx = align_audio_ctx(audio_ctx, args.n_audio)
    dump_tensor(os.path.join(args.dump_dir, "audio_ctx.npy"), audio_ctx)
    dump_tensor(os.path.join(args.dump_dir, "mel.npy"), mel)

    input_ids, used_tokens = build_prompt_tokens(
        tokenizer, args.prompt, args.l_text, use_chat_template=(not args.no_chat_template)
    )
    dump_tensor(os.path.join(args.dump_dir, "input_ids.npy"), input_ids)
    with open(os.path.join(args.dump_dir, "prompt_preview.txt"), "w", encoding="utf-8") as f:
        f.write(tokenizer.decode(used_tokens, skip_special_tokens=False))

    print("[INFO] running ExecuTorch prefill...")
    out_list = et_call(prefill_method, [input_ids, audio_ctx])
    et_prefill_logits = out_list[0].float()
    raw_outputs = [t.contiguous() for t in out_list[1:]]
    kv_outputs = raw_outputs[:74]  # match decode expectation
    extra_outputs = raw_outputs[74:]
    kv_flat, T_visible = left_pad_kv_to_cap(kv_outputs, args.kv_cap)
    kv_flat_prefill = [t.clone() for t in kv_flat]
    kv_vis = build_kv_vis(T_visible, args.kv_cap)
    dump_tensor(os.path.join(args.dump_dir, "et_logits0.npy"), et_prefill_logits)
    dump_tensor(os.path.join(args.dump_dir, "kv_vis_step0.npy"), kv_vis)
    kv_paths = dump_kv_list(kv_flat_prefill, args.dump_dir, "kv_step0")
    for idx, tensor in enumerate(extra_outputs):
        dump_tensor(os.path.join(args.dump_dir, f"prefill_extra_{idx}.npy"), tensor)

    print(f"[INFO] prefill KV tensors: {len(kv_flat)}, visible tokens: {T_visible}")

    decode_logits: List[torch.Tensor] = []
    generated_tokens: List[int] = []
    cur_token_id = greedy_top1(et_prefill_logits)
    generated_tokens.append(cur_token_id)
    cur_token = torch.tensor([cur_token_id], dtype=torch.int32)
    cur_T = T_visible

    kv_flat_decode = [t.clone() for t in kv_flat_prefill]

    kv_inputs_count = decode_method.metadata.num_inputs() - 3
    if kv_inputs_count <= 0 or kv_inputs_count % 2 != 0:
        raise RuntimeError(f"Unexpected decode input count {kv_inputs_count+3}")
    kv_layers_needed = kv_inputs_count // 2
    kv_flat_decode = [t.clone() for t in kv_flat_prefill[: kv_layers_needed * 2]]

    for step in range(args.decode_steps):
        print(f"[INFO] ExecuTorch decode step {step}, cur_T={cur_T}")
        past_seen = torch.tensor([cur_T], dtype=torch.int64)
        et_inputs = [cur_token, past_seen, kv_vis] + kv_flat_decode
        et_out = et_call(decode_method, et_inputs)
        logits = et_out[0].float()
        decode_logits.append(logits.clone())
        kv_flat_decode = [t.contiguous() for t in et_out[1:]]
        dump_tensor(os.path.join(args.dump_dir, f"et_logits_step{step+1}.npy"), logits)
        dump_tensor(os.path.join(args.dump_dir, f"kv_vis_step{step+1}.npy"), kv_vis)
        dump_kv_list(kv_flat_decode, args.dump_dir, f"kv_step{step+1}")

        if cur_T < args.kv_cap:
            cur_T += 1
        kv_vis = build_kv_vis(cur_T, args.kv_cap)

        cur_token_id = greedy_top1(logits)
        generated_tokens.append(cur_token_id)
        cur_token = torch.tensor([cur_token_id], dtype=torch.int32)

    with open(os.path.join(args.dump_dir, "et_tokens.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(str(t) for t in generated_tokens))
    print(f"[INFO] ExecuTorch generated tokens: {generated_tokens}")

    # ----------------- HF replay -----------------
    print("[INFO] loading HF text-only model for comparison...")
    hf_backbone, hf_head, hf_embed = load_hf_text_only(args.model_id, args.revision)
    hf_input_ids = input_ids.to(torch.long)
    audio_ctx_hf = audio_ctx.to(hf_embed.weight.dtype if hasattr(hf_embed, "weight") else torch.float32)

    with torch.no_grad():
        embeds = hf_embed(hf_input_ids)
        hidden0 = torch.cat([embeds, audio_ctx_hf.unsqueeze(0)], dim=1)
        out0 = hf_backbone(inputs_embeds=hidden0, use_cache=True)
        hf_prefill_logits = hf_head(out0.last_hidden_state[:, -1, :]).squeeze(0).float()
    prefill_metrics = compare_logits(et_prefill_logits, hf_prefill_logits, name="prefill")
    dump_tensor(os.path.join(args.dump_dir, "hf_logits0.npy"), hf_prefill_logits)

    # Replay decode using ExecuTorch KV/state
    cur_cache = kv_flat_to_hf_cache(kv_flat_prefill, T_visible)
    cur_vis = build_kv_vis(T_visible, args.kv_cap)
    decode_metrics = []
    cache = cur_cache
    kv_vis_hf = cur_vis
    seen = T_visible

    for step, token_id in enumerate(generated_tokens):
        cur_tok = torch.tensor([[token_id]], dtype=torch.long)
        with torch.no_grad():
            hf_out = hf_backbone(
                input_ids=cur_tok,
                past_key_values=cache,
                attention_mask=kv_vis_hf,
                use_cache=True,
            )
            hf_logits = hf_head(hf_out.last_hidden_state[:, -1, :]).squeeze(0).float()
        dump_tensor(os.path.join(args.dump_dir, f"hf_logits_step{step}.npy"), hf_logits)
        et_logits = et_prefill_logits if step == 0 else decode_logits[step - 1]
        metrics = compare_logits(et_logits, hf_logits, name=f"step{step}")
        decode_metrics.append({"step": step, **metrics})
        cache = hf_out.past_key_values
        seen = min(seen + 1, args.kv_cap)
        kv_vis_hf = build_kv_vis(seen, args.kv_cap)

    summary = {
        "prompt": args.prompt,
        "tokens": generated_tokens,
        "prefill_metrics": prefill_metrics,
        "decode_metrics": decode_metrics,
        "kv_files": kv_paths,
    }
    with open(os.path.join(args.dump_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Debug artifacts ready in {args.dump_dir}")


if __name__ == "__main__":
    main()
