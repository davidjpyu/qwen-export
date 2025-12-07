#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end ExecuTorch driver that turns (prompt text + 30s audio) into text
using the Whisper preprocess PTE, the Qwen audio tower, and the exported
prefill/decode LLM programs. This script keeps tokenization on the host but
executes all heavy neural nets through ExecuTorch runtimes.
"""
import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, Qwen2_5OmniForConditionalGeneration
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
    """Best-effort read of the ExecuTorch Method metadata to detect dtype."""
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
def load_wave(wav_path: str, silence: bool, duration_s: int = DURATION_S, sr: int = SR) -> torch.Tensor:
    """Return float32 mono waveform of length sr * duration_s."""
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
    """Executes whisper_preprocess.pte and returns [1, 128, T] mel tensor."""
    mel = et_call(proc_method, [wave.to(torch.float32)])[0]
    mel = to_torch_tensor(mel).to(torch.float32).contiguous()
    if mel.dim() != 3 or mel.shape[1] != 128:
        raise RuntimeError(f"Unexpected mel shape {tuple(mel.shape)}")
    return mel


def run_audio_tower(enc_method, mel: torch.Tensor) -> torch.Tensor:
    """Executes qwen audio tower on mel features and returns [N_audio, D]."""
    feats_CT = mel.squeeze(0)  # [128, T]
    out = et_call(enc_method, [feats_CT.contiguous()])[0]
    return to_torch_tensor(out).to(torch.float32).contiguous()


def align_audio_ctx(audio_ctx: torch.Tensor, n_audio: int) -> torch.Tensor:
    """Pad/crop audio embeddings to the static n_audio expected by prefill."""
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
    """Left-pad KV tensors along time so each has length kv_cap."""
    out = []
    T0 = None
    for t in kv_flat:
        if t.dim() != 4:
            raise RuntimeError(f"KV tensor should be rank 4, got {tuple(t.shape)}")
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
    """Creates [1, kv_cap + 1] bool mask used by the ET decode program."""
    kv_vis = torch.zeros(1, kv_cap + 1, dtype=torch.bool)
    if T_visible > 0:
        kv_vis[0, (kv_cap - T_visible): kv_cap] = True
    kv_vis[0, -1] = True
    return kv_vis


def unflatten_to_legacy(kv_flat: Sequence[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert [k0, v0, k1, v1, ...] into legacy [(k0, v0), ...] format."""
    if len(kv_flat) % 2 != 0:
        raise RuntimeError("KV list must contain pairs of key/value tensors.")
    legacy: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(kv_flat), 2):
        legacy.append((kv_flat[i], kv_flat[i + 1]))
    return legacy


def kv_flat_to_hf_cache(kv_flat: Sequence[torch.Tensor], seen_tokens: int) -> DynamicCache:
    """Re-create a HF DynamicCache from ET-style left-padded KV tensors."""
    legacy = unflatten_to_legacy(kv_flat)
    cache = DynamicCache()
    cache.key_cache = [k for k, _ in legacy]
    cache.value_cache = [v for _, v in legacy]
    cache.seen_tokens = int(seen_tokens)
    return cache


def compare_logits(a: torch.Tensor, b: torch.Tensor, topk: int = 5):
    """Return MAE/rel/cosine + top-k overlap statistics."""
    a = a.float().view(-1)
    b = b.float().view(-1)
    diff = a - b
    mae = diff.abs().mean().item()
    mse = (diff * diff).mean().item()
    rel = (diff.norm() / (b.norm() + 1e-12)).item()
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
    ak = torch.topk(a, k=min(topk, a.shape[0])).indices
    bk = torch.topk(b, k=min(topk, b.shape[0])).indices
    top1_match = int(ak[0].item() == bk[0].item())
    overlap = len(set(ak.tolist()) & set(bk.tolist()))
    return {
        "mae": mae,
        "mse": mse,
        "rel": rel,
        "cos": cos,
        "top1_match": top1_match,
        f"topk_overlap": overlap,
    }


# ----------------- Tokenizer helpers -----------------
def build_prompt_tokens(
    tokenizer: AutoTokenizer, prompt: str, l_text: int, use_chat_template: bool
) -> Tuple[torch.Tensor, List[int]]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        full_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )[0].tolist()
    else:
        full_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
            "input_ids"
        ][0].tolist()

    if len(full_ids) > l_text:
        print(
            f"[WARN] prompt tokens {len(full_ids)} exceed l_text={l_text}; keeping last {l_text} tokens"
        )
        used = full_ids[-l_text:]
    elif len(full_ids) < l_text:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        used = full_ids + [pad_id] * (l_text - len(full_ids))
    else:
        used = full_ids

    tensor = torch.tensor([used], dtype=torch.int32)
    return tensor, used

def load_token_list(path: str) -> List[int]:
    tokens: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.replace(",", " ")
            if line.strip():
                tokens.extend(int(tok) for tok in line.split())
    return tokens


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    flat = logits.view(-1).float()
    if temperature <= 0.0:
        # The argmax of the full logits tensor is the token ID.
        # Do not treat it as an index into another tensor.
        # The logits tensor shape is (vocab_size,).
        return int(torch.argmax(flat, dim=-1).item())
    probs = torch.softmax(flat / max(temperature, 1e-5), dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        if mask.any():
            mask[..., 0] = False  # ensure at least one token remains
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
        probs = probs / probs.sum()
    next_token = torch.multinomial(probs, num_samples=1).item()
    return int(next_token)


# ----------------- HF reference helpers -----------------
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
    lm_backbone.half()
    lm_head.half()
    embed.half()
    return lm_backbone.eval(), lm_head.eval(), embed.eval()


def run_hf_reference_generation(
    input_ids: torch.Tensor,
    audio_ctx: torch.Tensor,
    tokenizer: AutoTokenizer,
    args,
    forced_tokens: Optional[List[int]] = None,
    hf_modules: Optional[Tuple[nn.Module, nn.Module, nn.Module]] = None,
    dump_dir: Optional[str] = None,
) -> str:
    if hf_modules is None:
        lm_backbone, lm_head, embed = load_hf_text_only(args.model_id, args.revision)
    else:
        lm_backbone, lm_head, embed = hf_modules
    generated: List[int] = []
    cache = None
    audio_ctx_hf = audio_ctx.to(embed.weight.dtype if hasattr(embed, "weight") else torch.float16)
    forced_idx = 0

    with torch.no_grad():
        embeds = embed(input_ids.to(torch.long))
        hidden0 = torch.cat([embeds, audio_ctx_hf.unsqueeze(0)], dim=1)
        out0 = lm_backbone(inputs_embeds=hidden0, use_cache=True)
        logits = lm_head(out0.last_hidden_state[:, -1, :]).squeeze(0).float()
        cache = out0.past_key_values
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            np.save(os.path.join(dump_dir, "hf_logits0_free.npy"), logits.cpu().numpy())
            top_val, top_idx = torch.topk(logits, k=min(10, logits.numel()))
            decoded = tokenizer.decode(top_idx.tolist(), skip_special_tokens=False)
            print(f"[HF-FREE prefill] top ids={top_idx.tolist()} top_vals={top_val.tolist()}")

    if forced_tokens is not None and forced_idx < len(forced_tokens):
        cur_token_id = forced_tokens[forced_idx]
        forced_idx += 1
        forced_note = "forced"
    else:
        cur_token_id = sample_next_token(logits, args.temperature, args.top_p)
        forced_note = "sampled"
    decoded_tok = tokenizer.decode([cur_token_id], skip_special_tokens=False)
    print(f"[HF-DECODE] step -1 token={cur_token_id} text={decoded_tok!r} ({forced_note})")
    generated.append(cur_token_id)
    cur_token = torch.tensor([[cur_token_id]], dtype=torch.long)

    for step in range(args.max_new_tokens - 1):
        with torch.no_grad():
            out = lm_backbone(input_ids=cur_token, past_key_values=cache, use_cache=True)
            logits = lm_head(out.last_hidden_state[:, -1, :]).squeeze(0).float()
            cache = out.past_key_values
            if dump_dir:
                np.save(os.path.join(dump_dir, f"hf_logits_free_step{step+1}.npy"), logits.cpu().numpy())

        if forced_tokens is not None and forced_idx < len(forced_tokens):
            cur_token_id = forced_tokens[forced_idx]
            forced_idx += 1
            forced_note = "forced"
        else:
            cur_token_id = sample_next_token(logits, args.temperature, args.top_p)
            forced_note = "sampled"
        decoded_tok = tokenizer.decode([cur_token_id], skip_special_tokens=False)
        print(f"[HF-DECODE] step {step} token={cur_token_id} text={decoded_tok!r} ({forced_note})")
        generated.append(cur_token_id)
        cur_token = torch.tensor([[cur_token_id]], dtype=torch.long)

        if args.stop_on_eos and tokenizer.eos_token_id is not None and cur_token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


# ----------------- Main pipeline -----------------
def run_pipeline(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, revision=args.revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cli_prefill_dtype = args.prefill_dtype
    prefill_dtype = torch.float32 if cli_prefill_dtype == "fp32" else torch.float16

    print("[INFO] loading audio...")
    wave = load_wave(args.wav, args.silence)

    print("[INFO] loading ExecuTorch programs...")
    proc_method = et_load_forward(args.whisper_pte)
    enc_method = et_load_forward(args.audio_tower_pte)
    prefill_method = decode_method = None
    if not args.hf_only:
        prefill_method = et_load_forward(args.prefill_pte)
        decode_method = et_load_forward(args.decode_pte)
        inferred_dtype = infer_method_input_dtype(prefill_method, 1)
        if inferred_dtype is not None:
            print(
                f"[INFO] Prefill program '{args.prefill_pte}' expects "
                f"{torch_dtype_short_name(inferred_dtype)} audio_ctx inputs"
            )
        if inferred_dtype is not None and inferred_dtype != prefill_dtype:
            inferred_name = torch_dtype_short_name(inferred_dtype)
            print(
                f"[WARN] Prefill export expects {inferred_name} audio_ctx inputs but --prefill_dtype={cli_prefill_dtype}; "
                f"overriding to {inferred_name}."
            )
            prefill_dtype = inferred_dtype
            cli_prefill_dtype = inferred_name

    print("[INFO] running whisper preprocess...")
    mel = run_whisper_preprocess(proc_method, wave)
    print(f"[INFO] mel shape: {tuple(mel.shape)}")

    print("[INFO] running audio tower...")
    audio_ctx = run_audio_tower(enc_method, mel).to(prefill_dtype).contiguous()
    print(f"[INFO] audio_ctx shape: {tuple(audio_ctx.shape)}")
    audio_ctx = align_audio_ctx(audio_ctx, args.n_audio)
    if args.dump_audio_ctx:
        np.save(args.dump_audio_ctx, audio_ctx.cpu().numpy())
        print(f"[INFO] dumped audio_ctx to {args.dump_audio_ctx}")
    head_audio = audio_ctx[:5, :8].cpu().tolist()
    print(f"[FULL] audio_ctx_head: {head_audio}")

    input_ids, used_prompt_tokens = build_prompt_tokens(
        tokenizer, args.prompt, args.l_text, use_chat_template=(not args.no_chat_template)
    )
    prompt_preview = tokenizer.decode(used_prompt_tokens, skip_special_tokens=False)
    print(f"[INFO] prompt tokenized to shape {tuple(input_ids.shape)}")
    print(f"[INFO] prompt preview (after trunc/pad): {prompt_preview}")
    print(f"[FULL] input_ids: {input_ids[0].tolist()}")

    et_forced_tokens = None
    et_forced_idx = 0
    if args.hf_tokens:
        et_forced_tokens = load_token_list(args.hf_tokens)
        print(f"[INFO] ExecuTorch force-decode enabled with {len(et_forced_tokens)} tokens from {args.hf_tokens}")

    hf_forced_tokens = None
    if args.hf_force_tokens:
        hf_forced_tokens = load_token_list(args.hf_force_tokens)
        print(f"[INFO] HF force-decode enabled with {len(hf_forced_tokens)} tokens from {args.hf_force_tokens}")

    hf_needed = args.hf_only or getattr(args, "hf_compare", False)
    hf_modules: Optional[Tuple[nn.Module, nn.Module, nn.Module]] = None
    hf_backbone = hf_head = hf_embed = None
    if hf_needed:
        hf_modules = load_hf_text_only(args.model_id, args.revision)
        hf_backbone, hf_head, hf_embed = hf_modules
        if prefill_dtype == torch.float32:
            hf_backbone = hf_backbone.float()
            hf_head = hf_head.float()
            hf_embed = hf_embed.float()
            hf_modules = (hf_backbone, hf_head, hf_embed)

    hf_text = None

    if not args.hf_only:
        print("[INFO] running prefill...")
        out_list = et_call(prefill_method, [input_ids, audio_ctx])
        logits = out_list[0].float()
        kv_flat = [t.contiguous() for t in out_list[1:]]
        print(f"[INFO] prefill done, logits shape {tuple(logits.shape)}, KV tensors {len(kv_flat)}")

        if not kv_flat:
            raise RuntimeError("Prefill did not return KV tensors.")

        kv_flat, T_visible = left_pad_kv_to_cap(kv_flat, args.kv_cap)
        kv_vis = build_kv_vis(T_visible, args.kv_cap)
        cur_T = T_visible

        if hf_modules is not None:
            audio_ctx_hf = audio_ctx.to(hf_embed.weight.dtype if hasattr(hf_embed, "weight") else torch.float16)
            with torch.no_grad():
                embeds = hf_embed(input_ids.to(torch.long))
                hidden0 = torch.cat([embeds, audio_ctx_hf.unsqueeze(0)], dim=1)
                out0 = hf_backbone(inputs_embeds=hidden0, use_cache=True)
                hf_logits0 = hf_head(out0.last_hidden_state[:, -1, :]).squeeze(0).float()
            cmp0 = compare_logits(logits, hf_logits0, topk=5)
            et_top0 = int(torch.argmax(logits).item())
            hf_top0 = int(torch.argmax(hf_logits0).item())
            et_tok0 = tokenizer.decode([et_top0], skip_special_tokens=False)
            hf_tok0 = tokenizer.decode([hf_top0], skip_special_tokens=False)
            print(
                "[HF-COMPARE prefill] "
                f"MAE={cmp0['mae']:.3e} Cos={cmp0['cos']:.6f} Top1Match={bool(cmp0['top1_match'])} TopkOverlap={cmp0['topk_overlap']} "
                f"ET_top={et_top0}:{et_tok0!r} HF_top={hf_top0}:{hf_tok0!r}"
            )

        generated: List[int] = []
        print("[INFO] starting decode loop...")
        if et_forced_tokens is not None and et_forced_idx < len(et_forced_tokens):
            cur_token_id = et_forced_tokens[et_forced_idx]
            et_forced_idx += 1
            forced_note = "forced"
        else:
            cur_token_id = sample_next_token(logits, args.temperature, args.top_p)
            forced_note = "sampled"
        decoded_tok = tokenizer.decode([cur_token_id], skip_special_tokens=False)
        print(f"[DECODE] step -1 token={cur_token_id} text={decoded_tok!r} ({forced_note})")
        generated.append(cur_token_id)
        cur_token = torch.tensor([cur_token_id], dtype=torch.int32)

        for step in range(args.max_new_tokens - 1):
            print(f"[DECODE] step {step}, cur_T={cur_T}")
            past_seen = torch.tensor([cur_T], dtype=torch.int64)
            et_inputs = [cur_token, past_seen, kv_vis] + kv_flat
            et_out = et_call(decode_method, et_inputs)
            logits = et_out[0].float()
            kv_flat = [t.contiguous() for t in et_out[1:]]

            if hf_modules is not None:
                hf_cache = kv_flat_to_hf_cache(kv_flat, cur_T)
                cur_token_hf = cur_token.view(1, 1).to(torch.long)
                with torch.no_grad():
                    hf_out = hf_backbone(
                        input_ids=cur_token_hf,
                        past_key_values=hf_cache,
                        attention_mask=kv_vis,
                        use_cache=True,
                    )
                    hf_logits = hf_head(hf_out.last_hidden_state[:, -1, :]).squeeze(0).float()
                cmp = compare_logits(logits, hf_logits, topk=5)
                et_top = int(torch.argmax(logits).item())
                hf_top = int(torch.argmax(hf_logits).item())
                et_tok = tokenizer.decode([et_top], skip_special_tokens=False)
                hf_tok = tokenizer.decode([hf_top], skip_special_tokens=False)
                print(
                    "[HF-COMPARE step "
                    f"{step}] MAE={cmp['mae']:.3e} Cos={cmp['cos']:.6f} Top1Match={bool(cmp['top1_match'])} "
                    f"ET_top={et_top}:{et_tok!r} HF_top={hf_top}:{hf_tok!r}"
                )

            if cur_T < args.kv_cap:
                cur_T += 1
            kv_vis = build_kv_vis(cur_T, args.kv_cap)

            if et_forced_tokens is not None and et_forced_idx < len(et_forced_tokens):
                cur_token_id = et_forced_tokens[et_forced_idx]
                et_forced_idx += 1
                forced_note = "forced"
            else:
                cur_token_id = sample_next_token(logits, args.temperature, args.top_p)
                forced_note = "sampled"
            decoded_tok = tokenizer.decode([cur_token_id], skip_special_tokens=False)
            print(f"[DECODE] step {step} token={cur_token_id} text={decoded_tok!r} ({forced_note})")
            generated.append(cur_token_id)
            cur_token = torch.tensor([cur_token_id], dtype=torch.int32)

            if args.stop_on_eos and tokenizer.eos_token_id is not None and cur_token_id == tokenizer.eos_token_id:
                print(f"[DECODE] hit EOS at step {step}")
                break

        # Decode only the generated tokens, skipping special tokens for a clean output.
        gen_text = tokenizer.decode(generated, skip_special_tokens=True)

        # For returning the full conversation, we can construct it if needed,
        # but for display, the generated text is most important.
        full_text = args.prompt + gen_text
        print(f"\n=== Generated Text ===\n{gen_text}")

        if args.dump_tokens:
            with open(args.dump_tokens, "w", encoding="utf-8") as f:
                f.write(" ".join(str(t) for t in generated))
            print(f"[INFO] dumped ExecuTorch tokens to {args.dump_tokens}")

    if args.hf_only:
        hf_text = run_hf_reference_generation(
            input_ids,
            audio_ctx,
            tokenizer,
            args,
            forced_tokens=hf_forced_tokens,
            hf_modules=hf_modules,
            dump_dir=args.hf_dump_dir,
        )
        print(f"\n=== HF Reference Text ===\n{hf_text}")
        return args.prompt + hf_text, hf_text, hf_text

    hf_text = None
    if getattr(args, "hf_compare", False):
        print("[INFO] running HF reference generation (this may take a while)...")
        hf_text = run_hf_reference_generation(
            input_ids,
            audio_ctx,
            tokenizer,
            args,
            forced_tokens=hf_forced_tokens,
            hf_modules=hf_modules,
            dump_dir=args.hf_dump_dir,
        )
        print(f"\n=== HF Reference Text ===\n{hf_text}")

    if hf_text is None:
        return full_text, gen_text
    return full_text, gen_text, hf_text


def build_arg_parser():
    ap = argparse.ArgumentParser(description="ExecuTorch Qwen Omni audio+text driver")
    ap.add_argument("--wav", type=str, default="audio_sample1.wav",
                    help="Path to 16kHz mono WAV file (default: audio_sample1.wav)")
    ap.add_argument("--silence", action="store_true", help="Use 30s of zeros instead of WAV/random audio")
    ap.add_argument("--prompt", type=str, default="User: transcribe the following audio. Assistant:")
    ap.add_argument("--whisper_pte", type=str, default="whisper_preprocess.pte")
    ap.add_argument("--audio_tower_pte", type=str, default="qwen_audio_tower_30s_static.pte")
    ap.add_argument("--prefill_pte", type=str, default="llm_prefill_tokens_64p_750a_fp16.pte")
    ap.add_argument("--decode_pte", type=str, default="llm_decode_cacheT1024_fp16.pte")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--revision", type=str, default="main")
    ap.add_argument("--l_text", type=int, default=64, help="Static prompt token length expected by prefill export")
    ap.add_argument("--n_audio", type=int, default=750, help="Static audio embedding length expected by prefill export")
    ap.add_argument("--kv_cap", type=int, default=1024, help="Static KV cache capacity for decode export")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 for greedy argmax decoding")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--stop_on_eos", action="store_true", help="Stop once EOS token is generated")
    ap.add_argument("--prefill_dtype", choices=["fp16", "fp32"], default="fp16",
                    help="dtype expected by prefill/decode PTE inputs")
    ap.add_argument("--dump_audio_ctx", type=str, default=None,
                    help="Optional path to save ExecuTorch audio embeddings (.npy)")
    ap.add_argument("--hf_tokens", type=str, default=None,
                    help="Optional path to whitespace/comma separated token ids to force ExecuTorch decode")
    ap.add_argument("--dump_tokens", type=str, default=None,
                    help="Optional path to dump ExecuTorch generated token ids")
    ap.add_argument("--hf_force_tokens", type=str, default=None,
                    help="Optional path to token ids to force HF reference decode")
    ap.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable tokenizer chat template wrapping and treat --prompt as raw text",
    )
    ap.add_argument(
        "--hf_compare",
        action="store_true",
        help="Also run HuggingFace Qwen model on the same prompt/audio for comparison",
    )
    ap.add_argument(
        "--hf_only",
        action="store_true",
        help="Skip ExecuTorch decode and only run the HuggingFace reference model",
    )
    ap.add_argument(
        "--hf_dump_dir",
        type=str,
        default=None,
        help="Optional directory to dump HF reference logits per step",
    )
    return ap


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    for path in [args.whisper_pte, args.audio_tower_pte, args.prefill_pte, args.decode_pte]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required PTE file: {path}")

    run_pipeline(args)


if __name__ == "__main__":
    main()
