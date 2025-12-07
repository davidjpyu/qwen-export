#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate ExecuTorch-exported Qwen2.5-Omni text LLM (prefill + decode) step consistency.
The key fix is rebuilding the HF cache from ET's KV tensors so both sides share the same starting state.

Defaults:
  --prefill_pte  llm_prefill_tokens_64p_750a_fp16.pte
  --decode_pte   llm_decode_cacheT1024_fp16.pte
  --audio_ctx    artifacts/golden_30s/audio_emb_exec.npy
  --l_text=64 --n_audio=750 --kv_cap=1024 --steps=5
"""

import os
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "sdpa")
torch.set_grad_enabled(False)

# ----------------- ExecuTorch helpers -----------------
def et_load_forward(pte_path: str):
    from executorch.runtime import Runtime
    rt = Runtime.get()
    prog = rt.load_program(pte_path)
    return prog.load_method("forward")

def et_call(method, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
    out_list = method.execute(inputs)
    res = []
    for x in out_list:
        if isinstance(x, torch.Tensor):
            res.append(x)
        else:
            try:
                res.append(x.to_torch())
            except Exception:
                res.append(torch.tensor(x))
    return res

# ----------------- HF backbone (text-only) -----------------
from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration
from transformers.cache_utils import DynamicCache, Cache

def load_text_only_fp16(model_id: str, revision: str):
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
                setattr(thinker, name, None); delattr(thinker, name)
            except Exception:
                pass
    if hasattr(omni, "perception"):
        try:
            omni.perception = None; delattr(omni, "perception")
        except Exception:
            pass

    lm_backbone = getattr(thinker, "model", thinker)
    lm_head = getattr(thinker, "lm_head", None) or getattr(thinker, "output", None) or getattr(omni, "lm_head", None)
    embed = getattr(lm_backbone, "embed_tokens", None) or getattr(thinker, "embed_tokens", None) \
         or getattr(lm_backbone, "tok_embeddings", None)
    if not isinstance(lm_head, nn.Module) or not isinstance(embed, nn.Embedding):
        raise RuntimeError("missing lm_head or embed_tokens")

    lm_backbone.half(); lm_head.half(); embed.half()
    return lm_backbone.eval(), lm_head.eval(), embed.eval()

# ----------------- KV flatten/unflatten -----------------
def flatten_pkv_from_cache(cache: Cache) -> List[torch.Tensor]:
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else list(cache)
    flat = []
    for k, v in legacy:
        flat += [k, v]
    return flat

def unflatten_to_legacy(flat: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    assert len(flat) % 2 == 0
    legacy = []
    for i in range(0, len(flat), 2):
        legacy.append((flat[i], flat[i+1]))
    return legacy

# ----------------- tokens -----------------
def build_tokens(text: str, l_text: int, model_id: str, revision: str) -> torch.Tensor:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, revision=revision, use_fast=True)
    ids = tok(text, add_special_tokens=True, return_tensors="pt")["input_ids"]  # [1, L0]
    if ids.shape[1] >= l_text:
        ids = ids[:, :l_text]
    else:
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        pad = torch.full((1, l_text - ids.shape[1]), pad_id, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    return ids.to(torch.int32)

# ----------------- KV helpers (pad/crop/masks) -----------------
def left_pad_kv_to_cap(kv_flat: List[torch.Tensor], kv_cap: int) -> Tuple[List[torch.Tensor], int]:
    """left-pad KV to kv_cap along time (dim=2); return (padded_kv, T_visible)."""
    out = []
    T0 = None
    for t in kv_flat:
        assert t.dim() == 4, f"KV rank must be 4, got {t.shape}"
        B, H, T, Dh = t.shape
        if T0 is None:
            T0 = T
        else:
            assert T0 == T, "all KV time dims must match"
        if T > kv_cap:
            t = t[:, :, -kv_cap:, :]
            T = kv_cap
        if T < kv_cap:
            pad = torch.zeros(B, H, kv_cap - T, Dh, dtype=t.dtype)
            t = torch.cat([pad, t], dim=2)
        out.append(t.contiguous())
    return out, int(min(T0, kv_cap))

def crop_kv_right(kv_flat: List[torch.Tensor], T_visible: int) -> List[torch.Tensor]:
    """keep rightmost T_visible tokens (remove left pads)."""
    out = []
    for t in kv_flat:
        B, H, T, Dh = t.shape
        out.append(t[:, :, -T_visible:, :].contiguous())
    return out

def slice_kv_to_cap(kv_flat: List[torch.Tensor], kv_cap: int) -> List[torch.Tensor]:
    """keep rightmost kv_cap tokens after each decode."""
    return [t[:, :, -kv_cap:, :].contiguous() for t in kv_flat]

def build_kv_vis_for_decode(T_visible: int, kv_cap: int) -> torch.Tensor:
    """
    [1, kv_cap+1] bool: left zeros (padding), right T_visible ones for history, last position True for current.
    """
    kv_vis = torch.zeros(1, kv_cap + 1, dtype=torch.bool)
    if T_visible > 0:
        kv_vis[0, (kv_cap - T_visible): kv_cap] = True
    kv_vis[0, -1] = True
    return kv_vis

# ----------------- metrics -----------------
def compare_logits(a: torch.Tensor, b: torch.Tensor, topk: int = 5):
    a = a.float().detach(); b = b.float().detach()
    diff = a - b
    mae = diff.abs().mean().item()
    mse = (diff * diff).mean().item()
    rel = (diff.norm() / (b.norm() + 1e-12)).item()
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
    ak = torch.topk(a, k=topk).indices
    bk = torch.topk(b, k=topk).indices
    top1_match = int(ak[0].item() == bk[0].item())
    inter = len(set(ak.tolist()) & set(bk.tolist()))
    return {"mae": mae, "mse": mse, "rel": rel, "cos": cos,
            "top1_match": top1_match, f"top{topk}_overlap": inter}

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill_pte", default="llm_prefill_tokens_64p_750a_fp16.pte")
    ap.add_argument("--decode_pte",  default="llm_decode_cacheT1024_fp16.pte")
    ap.add_argument("--audio_ctx",   default="artifacts/golden_30s/audio_emb_exec.npy")
    ap.add_argument("--model_id",    default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--revision",    default="main")
    ap.add_argument("--prompt",      default="User: transcribe the following audio. Assistant:")
    ap.add_argument("--l_text", type=int, default=64)
    ap.add_argument("--n_audio", type=int, default=750)
    ap.add_argument("--kv_cap",  type=int, default=1024)
    ap.add_argument("--steps",   type=int, default=5)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    assert os.path.exists(args.prefill_pte), f"missing {args.prefill_pte}"
    assert os.path.exists(args.decode_pte),  f"missing {args.decode_pte}"

    # audio_ctx
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    if args.audio_ctx and os.path.exists(args.audio_ctx):
        audio_ctx = torch.from_numpy(np.load(args.audio_ctx)).to(dtype).contiguous()
        print(f"[INFO] load audio_ctx from {args.audio_ctx}: {tuple(audio_ctx.shape)}")
    else:
        audio_ctx = torch.zeros(args.n_audio, 2048, dtype=dtype)
        print(f"[INFO] use zeros audio_ctx: {tuple(audio_ctx.shape)}")
    head_audio = audio_ctx[:5, :8].cpu().tolist()
    print(f"[PREFILL-TEST] audio_ctx_head: {head_audio}")

    # tokens
    input_ids = build_tokens(args.prompt, args.l_text, args.model_id, args.revision)  # int32 [1, L]
    print(f"[PREFILL-TEST] input_ids: {input_ids[0].tolist()}")

    # ---- ET prefill ----
    et_prefill = et_load_forward(args.prefill_pte)
    et_decode  = et_load_forward(args.decode_pte)

    out_list = et_call(et_prefill, [input_ids, audio_ctx])
    et_logits0 = out_list[0].float()
    et_kv_flat = [t.contiguous() for t in out_list[1:]]
    np.save("et_logits0.npy", et_logits0.cpu().numpy())

    print(f"[ET] prefill -> logits0: {tuple(et_logits0.shape)}, KV tensors: {len(et_kv_flat)}; example KV shape: {tuple(et_kv_flat[0].shape)}")

    # pad to static cap for ET decode, also prepare "HF-from-ET" cache
    T0 = int(et_kv_flat[0].shape[2])
    et_kv_flat, T_visible = left_pad_kv_to_cap(et_kv_flat, args.kv_cap)
    kv_vis = build_kv_vis_for_decode(T_visible, args.kv_cap)
    print(f"[ET] prefill KV T0={T0} -> padded to kv_cap={args.kv_cap}; kv_vis shape={tuple(kv_vis.shape)}, sum={int(kv_vis.sum())}")

    # ---- HF init from ET KV (eliminate starting-state mismatch) ----
    #   1) Remove left padding and keep the rightmost T_visible tokens
    #   2) Build an HF DynamicCache from those tensors
    hf_cache_from_et = DynamicCache.from_legacy_cache(
        unflatten_to_legacy(crop_kv_right(et_kv_flat, T_visible))
    )

    # HF modules
    lm_backbone, lm_head, embed = load_text_only_fp16(args.model_id, args.revision)
    if dtype == torch.float32:
        lm_backbone = lm_backbone.float()
        lm_head = lm_head.float()
        embed = embed.float()

    # Force both sides to consume the same next token at step 0
    cur_et = torch.argmax(et_logits0).view(1).to(torch.int32)
    cur_hf = cur_et.view(1, 1).to(torch.long)

    # Comparing HF prefill logits is optional but useful for sanity
    with torch.no_grad():
        x = embed(input_ids.to(torch.long))
        hidden0 = torch.cat([x, audio_ctx.unsqueeze(0)], dim=1)
        out0_ref = lm_backbone(inputs_embeds=hidden0, use_cache=True)
        hf_logits0_ref = lm_head(out0_ref.last_hidden_state[:, -1, :]).squeeze(0).float()
    np.save("hf_logits0.npy", hf_logits0_ref.cpu().numpy())

    m0 = compare_logits(et_logits0, hf_logits0_ref, topk=5)
    print(f"[COMPARE prefill] MAE={m0['mae']:.3e}  MSE={m0['mse']:.3e}  Rel={m0['rel']:.3e}  Cos={m0['cos']:.6f}  Top1Match={bool(m0['top1_match'])}  Top5Overlap={m0['top5_overlap']}")

    diff = (
         - hf_logits0_ref).abs()
    top_diff_val, top_diff_idx = torch.topk(diff, k=10)
    print(f"[COMPARE prefill] top diff idx: {top_diff_idx.tolist()}")
    print(f"[COMPARE prefill] top diff val: {top_diff_val.tolist()}")

    # ---- decode loop (shared KV start + token sequence) ----
    cur_T = T_visible
    for step in range(args.steps):
        # ET decode
        past_seen = torch.tensor([cur_T], dtype=torch.int64)
        et_out = et_call(et_decode, [cur_et, past_seen, kv_vis] + et_kv_flat)
        et_logit = et_out[0].float()
        np.save(f"et_logits_step{step+1}.npy", et_logit.cpu().numpy())
        et_kv_flat = [t.contiguous() for t in et_out[1:]] # Slice to capacity is now handled by the exported model

        # HF decode using the ET-derived cache
        with torch.no_grad():
            # CRITICAL FIX: Re-create the HF cache from the ET cache *at each step*.
            # This ensures both models see the exact same left-padded KV cache layout.
            hf_cache_this_step = DynamicCache()
            legacy_cache = unflatten_to_legacy(et_kv_flat)
            hf_cache_this_step.key_cache = [k for k, v in legacy_cache]
            hf_cache_this_step.value_cache = [v for k, v in legacy_cache]
            hf_cache_this_step.seen_tokens = cur_T

            out1 = lm_backbone(
                input_ids=cur_hf,
                past_key_values=hf_cache_this_step,
                attention_mask=kv_vis, # Use the same explicit mask as the ET model
                use_cache=True,
            )
            hf_logit = lm_head(out1.last_hidden_state[:, -1, :]).squeeze(0).float()
        np.save(f"hf_logits_step{step+1}.npy", hf_logit.cpu().numpy())

        m = compare_logits(et_logit, hf_logit, topk=5)
        print(f"[step {step}] MAE={m['mae']:.3e}  Cos={m['cos']:.6f}  Top1Match={bool(m['top1_match'])}  Top5Overlap={m['top5_overlap']}")

        # Always use the ExecuTorch argmax token to prevent divergence
        cur_et = torch.argmax(et_logit).view(1).to(torch.int32)
        cur_hf = cur_et.view(1,1).to(torch.long)

        # Maintain lengths and visibility bookkeeping
        if cur_T < args.kv_cap:
            cur_T += 1
        kv_vis = build_kv_vis_for_decode(cur_T, args.kv_cap)

    print("\n== Done ==")

if __name__ == "__main__":
    main()
