#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-Omni 3B -> ExecuTorch (text-only) prefill + decode export
(FP16, static KV capacity, explicit kv_vis + cache_position).
Outputs:
  - llm_prefill_tokens_64p_750a_fp16.pte
  - llm_decode_cacheT{cache_T}_fp16.pte
"""

import os, gc, argparse
from typing import List, Tuple, Optional
import torch, torch.nn as nn

os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "sdpa")
os.environ.setdefault("PYTORCH_DISABLE_MPS_FALLBACK", "1")
torch.set_grad_enabled(False)

def _simple_sdpa_mask_interface(
    *, batch_size:int, cache_position:torch.Tensor, kv_length:int, kv_offset:int,
    mask_function, attention_mask:Optional[torch.Tensor]=None,
    allow_is_causal_skip:bool=True, dtype:torch.dtype=torch.float32, config=None,
):
    device = cache_position.device
    B = int(batch_size); Q = int(cache_position.numel()); K = int(kv_length)
    kv_abs = kv_offset + torch.arange(K, device=device)
    q_abs  = cache_position.view(Q, 1).to(kv_abs.dtype)
    visible = (kv_abs.view(1, K) <= q_abs)            # [Q,K]
    mask = visible.unsqueeze(0).unsqueeze(1)          # [1,1,Q,K]
    if B != 1:
        mask = mask.expand(B, 1, Q, K).clone()
    if isinstance(attention_mask, torch.Tensor):
        if attention_mask.dim() == 2:
            kv_vis = attention_mask[:, None, None, :].to(torch.bool)
        elif attention_mask.dim() == 3:
            kv_vis = attention_mask[:, :, None, :].to(torch.bool)
        else:
            kv_vis = attention_mask.view(B, -1)[:, None, None, :].to(torch.bool)
        mask = mask & kv_vis
    return mask.to(torch.bool)

try:
    import transformers.masking_utils as _mu
    _mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]  = _simple_sdpa_mask_interface
    _mu.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = _simple_sdpa_mask_interface
    print("[patch] mask interface set to simple(sdpa/eager)")
except Exception as e:
    print("[WARN] patch mask failed:", e)

from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration
from transformers.cache_utils import DynamicCache, Cache

def flatten_pkv_from_cache(cache: Cache) -> List[torch.Tensor]:
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else list(cache)
    flat = []
    for k, v in legacy: flat += [k, v]
    return flat

def slice_kv_to_cap(kv_flat: List[torch.Tensor], kv_cap: int) -> List[torch.Tensor]:
    """keep rightmost kv_cap tokens after each decode."""
    # This function needs to be defined in export_prefill_decode.py for tracing.
    return [t[:, :, -kv_cap:, :].contiguous() for t in kv_flat]

def unflatten_to_legacy(flat: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    assert len(flat) % 2 == 0
    return [(flat[i], flat[i+1]) for i in range(0, len(flat), 2)]

def load_text_only_fp16(model_id: str, revision: str):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, config=cfg, trust_remote_code=True, revision=revision,
        device_map="cpu", low_cpu_mem_usage=True, torch_dtype=torch.float16,
    ).eval()
    thinker = getattr(omni, "thinker", getattr(omni, "model", omni))
    for name in ["audio_tower","vision_tower","audio_projector","visual_projector"]:
        if hasattr(thinker, name):
            try: setattr(thinker, name, None); delattr(thinker, name)
            except: pass
    if hasattr(omni, "perception"):
        try: omni.perception=None; delattr(omni,"perception")
        except: pass
    lm_backbone = getattr(thinker, "model", thinker)
    lm_head = getattr(thinker,"lm_head",None) or getattr(thinker,"output",None) or getattr(omni,"lm_head",None)
    embed = getattr(lm_backbone,"embed_tokens",None) or getattr(thinker,"embed_tokens",None) or getattr(lm_backbone,"tok_embeddings",None)
    if not isinstance(lm_head, nn.Module) or not isinstance(embed, nn.Embedding):
        raise RuntimeError("missing lm_head or embed_tokens")
    lm_backbone.half(); lm_head.half(); embed.half()
    return lm_backbone.eval(), lm_head.eval(), embed.eval()

class PrefillTokensWrapper(nn.Module):
    def __init__(self, lm_backbone, lm_head, embed):
        super().__init__(); self.backbone=lm_backbone; self.lm_head=lm_head; self.embed=embed
    def forward(self, input_ids:torch.Tensor, audio_ctx:torch.Tensor):
        x = self.embed(input_ids.to(torch.long))
        hidden = torch.cat([x, audio_ctx.to(torch.float16).unsqueeze(0)], dim=1)
        out = self.backbone(inputs_embeds=hidden, use_cache=True)
        logits0 = self.lm_head(out.last_hidden_state[:, -1, :]).squeeze(0)
        kv_flat = flatten_pkv_from_cache(out.past_key_values)
        return (logits0, *kv_flat)

class DecodeWrapper(nn.Module):
    """
    forward(token_id, past_seen, kv_vis, *kv_in)
      - token_id: [1] int32/int64
      - past_seen: [1] int64  (actual visible history length T_visible)
      - kv_vis: [1, cache_T+1] bool
      - *kv_in: KV tensors per layer [B, H, cache_T, Dh]
    """
    def __init__(self, lm_backbone, lm_head):
        super().__init__(); self.backbone=lm_backbone; self.lm_head=lm_head
    def forward(self, token_id:torch.Tensor, past_seen:torch.Tensor, kv_vis:torch.Tensor, *kv_in:torch.Tensor):
        # Manually construct the DynamicCache to be compatible with older transformers versions
        # that don't support the `seen_tokens` argument in `from_legacy_cache`.
        cache = DynamicCache()
        legacy_cache = unflatten_to_legacy(list(kv_in))
        cache.key_cache = [k for k, v in legacy_cache]
        cache.value_cache = [v for k, v in legacy_cache]
        cache.seen_tokens = past_seen.item()  # This is traceable by torch.export
        out = self.backbone(
            input_ids=token_id.view(1,1).to(torch.long),
            past_key_values=cache,
            attention_mask=kv_vis.to(torch.bool),
            cache_position=past_seen.to(torch.long),   # current position equals the true visible length
            use_cache=True,
        )
        logits = self.lm_head(out.last_hidden_state[:, -1, :]).squeeze(0)
        
        # CRITICAL FIX: Directly access the full underlying tensors from the DynamicCache.
        # `to_legacy_cache()` would slice them to `seen_tokens`, which is incorrect for a static cache model.
        # The model updates the cache in-place, so we just need to return the full tensors.
        kv_out_flat = []
        for k, v in zip(out.past_key_values.key_cache, out.past_key_values.value_cache):
            kv_out_flat.extend([k, v])
        return (logits, *kv_out_flat)

def build_fixed_capacity_kv_examples(prefill_mod: nn.Module, cache_T: int) -> List[torch.Tensor]:
    with torch.no_grad():
        tok_min = torch.ones(1,1,dtype=torch.int32)
        aud_min = torch.zeros(1,2048,dtype=torch.float16)
        flat_kv_min = list(prefill_mod(tok_min, aud_min)[1:])
    # print(f"DEBUG: flat_kv_min (from 1-token prefill) shape: {flat_kv_min[0].shape}") # For debugging
    fixed=[]
    for t in flat_kv_min:
        B,H,_,Dh = t.shape
        fixed.append(torch.zeros(B,H,cache_T,Dh,dtype=t.dtype))
    return fixed

def ep_to_pte(ep:"torch.export.ExportedProgram", out_path:str):
    from executorch.exir import to_edge_transform_and_lower
    exec_prog = to_edge_transform_and_lower(ep, partitioner=[]).to_executorch()
    with open(out_path,"wb") as f: f.write(exec_prog.buffer)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--l_text", type=int, default=64)
    ap.add_argument("--n_audio", type=int, default=750)
    ap.add_argument("--cache_T", type=int, default=1024)
    ap.add_argument("--prefill_out", default="llm_prefill_tokens_64p_750a_fp16.pte")
    ap.add_argument("--decode_out_tmpl", default="llm_decode_cacheT{cache_T}_fp16.pte")
    args = ap.parse_args()

    lm_backbone, lm_head, embed = load_text_only_fp16(args.model_id, args.revision); gc.collect()

    # prefill
    input_ids_ex = torch.ones(1,args.l_text,dtype=torch.int32)
    audio_ctx_ex = torch.zeros(args.n_audio,2048,dtype=torch.float16)
    from torch.export import export as texport
    prefill = PrefillTokensWrapper(lm_backbone,lm_head,embed).eval()
    ep = texport(prefill, (input_ids_ex, audio_ctx_ex))
    ep_to_pte(ep, args.prefill_out); print(f"[OK] prefill -> {args.prefill_out}")
    # print(f"DEBUG: Prefill output KV shape (first key): {list(ep.graph_signature.output_specs)[1].arg.shape}") # For debugging

    # decode
    kv_examples = build_fixed_capacity_kv_examples(prefill, cache_T=args.cache_T)
    decode = DecodeWrapper(lm_backbone,lm_head).eval()
    token_ex  = torch.tensor([1], dtype=torch.int32)
    past_ex   = torch.tensor([0], dtype=torch.int64)                 # placeholder
    kv_vis_ex = torch.ones(1, args.cache_T+1, dtype=torch.bool)
    # print(f"DEBUG: kv_examples (for decode export) shape: {kv_examples[0].shape}") # For debugging
    ep2 = texport(decode, (token_ex, past_ex, kv_vis_ex, *kv_examples))
    decode_out = args.decode_out_tmpl.format(cache_T=args.cache_T)
    ep_to_pte(ep2, decode_out); print(f"[OK] decode  -> {decode_out} (cache_T={args.cache_T})")

if __name__ == "__main__":
    main()
