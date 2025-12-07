#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare ET vs HF logits under different audio embeddings to pinpoint whether the
ExecuTorch audio tower export drifts from the HuggingFace reference.
"""

import argparse
import os
from typing import List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, Qwen2_5OmniForConditionalGeneration


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


# ----------------- Utilities -----------------
def align_audio_ctx(audio_ctx: torch.Tensor, n_audio: int) -> torch.Tensor:
    cur = audio_ctx.shape[0]
    if cur == n_audio:
        return audio_ctx
    dim = audio_ctx.shape[1]
    if cur > n_audio:
        return audio_ctx[:n_audio].contiguous()
    pad = torch.zeros(n_audio - cur, dim, dtype=audio_ctx.dtype)
    return torch.cat([audio_ctx, pad], dim=0)


def build_prompt_tokens(
    tokenizer: AutoTokenizer, prompt: str, l_text: int, use_chat_template: bool
) -> torch.Tensor:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )[0].tolist()
    else:
        token_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0].tolist()
    if len(token_ids) > l_text:
        token_ids = token_ids[-l_text:]
    elif len(token_ids) < l_text:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        token_ids = token_ids + [pad_id] * (l_text - len(token_ids))
    return torch.tensor([token_ids], dtype=torch.int32)


def compare_logits(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    a = a.float().detach()
    b = b.float().detach()
    diff = a - b
    mae = diff.abs().mean().item()
    mse = (diff * diff).mean().item()
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
    return mae, mse, cos


def load_hf_modules(model_id: str, revision: str, dtype: torch.dtype):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        config=cfg,
        trust_remote_code=True,
        revision=revision,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    thinker = getattr(omni, "thinker", getattr(omni, "model", omni))
    lm_backbone = getattr(thinker, "model", thinker)
    lm_head = getattr(thinker, "lm_head", None) or getattr(thinker, "output", None) or getattr(omni, "lm_head", None)
    embed = getattr(lm_backbone, "embed_tokens", None) or getattr(thinker, "embed_tokens", None) \
        or getattr(lm_backbone, "tok_embeddings", None)
    if not isinstance(lm_head, torch.nn.Module) or not isinstance(embed, torch.nn.Module):
        raise RuntimeError("Missing lm_head or embedding module in HF checkpoint")
    if dtype == torch.float32:
        lm_backbone.float()
        lm_head.float()
        embed.float()
    else:
        lm_backbone.half()
        lm_head.half()
        embed.half()
    return lm_backbone.eval(), lm_head.eval(), embed.eval(), omni.eval()


class PrefillTokensWrapper(torch.nn.Module):
    """Copy of the wrapper used during export for direct PyTorch comparisons."""

    def __init__(self, lm_backbone, lm_head, embed, audio_dtype: torch.dtype):
        super().__init__()
        self.backbone = lm_backbone
        self.lm_head = lm_head
        self.embed = embed
        self.audio_dtype = audio_dtype

    def forward(self, input_ids: torch.Tensor, audio_ctx: torch.Tensor):
        x = self.embed(input_ids.to(torch.long))
        hidden = torch.cat([x, audio_ctx.to(self.audio_dtype).unsqueeze(0)], dim=1)
        out = self.backbone(inputs_embeds=hidden, use_cache=True)
        logits = self.lm_head(out.last_hidden_state[:, -1, :]).squeeze(0)
        return logits, out


def run_hf_prefill(
    lm_backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    embed: torch.nn.Module,
    input_ids: torch.Tensor,
    audio_ctx: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        prompt_embeds = embed(input_ids.to(torch.long))
        hidden = torch.cat([prompt_embeds, audio_ctx.unsqueeze(0)], dim=1)
        out = lm_backbone(inputs_embeds=hidden, use_cache=False)
        logits = lm_head(out.last_hidden_state[:, -1, :]).squeeze(0).float()
    return logits


def run_full_model_prefill(
    full_model: Qwen2_5OmniForConditionalGeneration,
    embed: torch.nn.Module,
    input_ids: torch.Tensor,
    audio_ctx: torch.Tensor,
) -> Optional[torch.Tensor]:
    if full_model is None:
        return None
    try:
        with torch.no_grad():
            prompt_embeds = embed(input_ids.to(torch.long))
            hidden = torch.cat([prompt_embeds, audio_ctx.unsqueeze(0)], dim=1)
            outputs = full_model(inputs_embeds=hidden, use_cache=False)
            logits = outputs.logits[:, -1, :].squeeze(0).float()
        return logits
    except TypeError as exc:
        print(f"[WARN] Full HF model forward lacks inputs_embeds support ({exc}); skipping.")
        return None


def maybe_load(path: Optional[str], dtype: torch.dtype) -> Optional[torch.Tensor]:
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.from_numpy(np.load(path)).to(dtype).contiguous()


def main():
    ap = argparse.ArgumentParser(description="Diagnose audio embedding drift (ExecuTorch vs HF)")
    ap.add_argument("--exec_audio", default="artifacts/golden_30s/audio_emb_exec.npy")
    ap.add_argument("--ref_audio", default="artifacts/golden_30s/audio_emb_ref.npy")
    ap.add_argument("--prefill_pte", default="llm_prefill_tokens_64p_750a_fp16.pte")
    ap.add_argument("--prefill_dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prompt", default="User: transcribe the following audio. Assistant:")
    ap.add_argument("--l_text", type=int, default=64)
    ap.add_argument("--n_audio", type=int, default=750)
    ap.add_argument("--use_chat_template", action="store_true")
    args = ap.parse_args()

    prefill_dtype = torch.float32 if args.prefill_dtype == "fp32" else torch.float16

    exec_audio = maybe_load(args.exec_audio, prefill_dtype)
    ref_audio = maybe_load(args.ref_audio, prefill_dtype)
    if exec_audio is None and ref_audio is None:
        raise RuntimeError("Need at least one of --exec_audio or --ref_audio")
    print(f"[INFO] exec_audio: {None if exec_audio is None else tuple(exec_audio.shape)}")
    print(f"[INFO] ref_audio : {None if ref_audio is None else tuple(ref_audio.shape)}")

    lm_backbone, lm_head, embed, full_model = load_hf_modules(args.model_id, args.revision, prefill_dtype)
    prefill_pt = PrefillTokensWrapper(lm_backbone, lm_head, embed, prefill_dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, revision=args.revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    input_ids = build_prompt_tokens(tokenizer, args.prompt, args.l_text, args.use_chat_template)

    def prepare(audio_tensor: torch.Tensor) -> torch.Tensor:
        return align_audio_ctx(audio_tensor.to(prefill_dtype).contiguous(), args.n_audio)

    hf_logits_ref = hf_logits_exec = None
    pt_logits_ref = pt_logits_exec = None
    full_logits_ref = full_logits_exec = None
    if ref_audio is not None:
        ref_ctx = prepare(ref_audio)
        hf_logits_ref = run_hf_prefill(lm_backbone, lm_head, embed, input_ids, ref_ctx)
        print("[INFO] HF logits computed with ref audio embeddings.")
        pt_logits_ref, _ = prefill_pt(input_ids, ref_ctx)
        pt_logits_ref = pt_logits_ref.float()
        print("[INFO] Torch Prefill wrapper logits (ref) computed.")
        full_logits_ref = run_full_model_prefill(full_model, embed, input_ids, ref_ctx)
        if full_logits_ref is not None:
            print("[INFO] Full HF model logits (ref) computed.")
    if exec_audio is not None:
        exec_ctx = prepare(exec_audio)
        hf_logits_exec = run_hf_prefill(lm_backbone, lm_head, embed, input_ids, exec_ctx)
        print("[INFO] HF logits computed with exec audio embeddings.")
        pt_logits_exec, _ = prefill_pt(input_ids, exec_ctx)
        pt_logits_exec = pt_logits_exec.float()
        print("[INFO] Torch Prefill wrapper logits (exec) computed.")
        full_logits_exec = run_full_model_prefill(full_model, embed, input_ids, exec_ctx)
        if full_logits_exec is not None:
            print("[INFO] Full HF model logits (exec) computed.")

    prefill_method = et_load_forward(args.prefill_pte)
    et_logits_ref = et_logits_exec = None
    if ref_audio is not None:
        et_logits_ref = et_call(prefill_method, [input_ids, ref_ctx])[0].float()
        print("[INFO] ExecuTorch logits computed with ref audio embeddings.")
    if exec_audio is not None:
        et_logits_exec = et_call(prefill_method, [input_ids, exec_ctx])[0].float()
        print("[INFO] ExecuTorch logits computed with exec audio embeddings.")

    print("\n=== Comparisons ===")
    if hf_logits_ref is not None and hf_logits_exec is not None:
        mae, mse, cos = compare_logits(hf_logits_exec, hf_logits_ref)
        print(f"HF(exec_audio) vs HF(ref_audio): MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if pt_logits_ref is not None and hf_logits_ref is not None:
        mae, mse, cos = compare_logits(pt_logits_ref, hf_logits_ref)
        print(f"TorchPrefill(ref) vs HF(ref)   : MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if pt_logits_exec is not None and hf_logits_exec is not None:
        mae, mse, cos = compare_logits(pt_logits_exec, hf_logits_exec)
        print(f"TorchPrefill(exec) vs HF(exec) : MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if pt_logits_exec is not None and pt_logits_ref is not None:
        mae, mse, cos = compare_logits(pt_logits_exec, pt_logits_ref)
        print(f"TorchPrefill(exec) vs TorchPrefill(ref): MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if et_logits_ref is not None and hf_logits_ref is not None:
        mae, mse, cos = compare_logits(et_logits_ref, hf_logits_ref)
        print(f"ET(ref_audio) vs HF(ref_audio): MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if et_logits_exec is not None and hf_logits_exec is not None:
        mae, mse, cos = compare_logits(et_logits_exec, hf_logits_exec)
        print(f"ET(exec_audio) vs HF(exec_audio): MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if et_logits_ref is not None and pt_logits_ref is not None:
        mae, mse, cos = compare_logits(et_logits_ref, pt_logits_ref)
        print(f"ET(ref_audio) vs TorchPrefill(ref): MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if et_logits_exec is not None and pt_logits_exec is not None:
        mae, mse, cos = compare_logits(et_logits_exec, pt_logits_exec)
        print(f"ET(exec_audio) vs TorchPrefill(exec): MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")

    if et_logits_exec is not None and hf_logits_ref is not None:
        mae, mse, cos = compare_logits(et_logits_exec, hf_logits_ref)
        print(f"ET(exec_audio) vs HF(ref_audio):  MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if full_logits_ref is not None and hf_logits_ref is not None:
        mae, mse, cos = compare_logits(full_logits_ref, hf_logits_ref)
        print(f"FullModel(ref) vs HF(ref)      : MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if full_logits_exec is not None and hf_logits_exec is not None:
        mae, mse, cos = compare_logits(full_logits_exec, hf_logits_exec)
        print(f"FullModel(exec) vs HF(exec)    : MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if full_logits_exec is not None and hf_logits_ref is not None:
        mae, mse, cos = compare_logits(full_logits_exec, hf_logits_ref)
        print(f"FullModel(exec) vs HF(ref)     : MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if full_logits_exec is not None and full_logits_ref is not None:
        mae, mse, cos = compare_logits(full_logits_exec, full_logits_ref)
        print(f"FullModel(exec) vs FullModel(ref): MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")

    print("\nInterpretation:")
    print("  - If HF(exec) vs HF(ref) already shows a large gap, the audio tower export is the source.")
    print("  - If TorchPrefill (and FullModel) align with HF but ET does not, the issue lies in the ET export/lowering.")


if __name__ == "__main__":
    main()
