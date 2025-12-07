#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a debug prefill graph that surfaces per-layer hidden states (last token)
and compare its ExecuTorch outputs against the PyTorch wrapper. This is the
suggested "method 1" to catch which layer first drifts after lowering.
"""

import argparse
import contextlib
import gc
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, Qwen2_5OmniForConditionalGeneration

os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "sdpa")
os.environ.setdefault("PYTORCH_DISABLE_MPS_FALLBACK", "1")
torch.set_grad_enabled(False)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ----------------- Patches -----------------
def _patch_transformers_mask():
    def _simple_sdpa_mask_interface(
        *,
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int,
        mask_function,
        attention_mask: Optional[torch.Tensor] = None,
        allow_is_causal_skip: bool = True,
        dtype: torch.dtype = torch.float32,
        config=None,
    ):
        device = cache_position.device
        B = int(batch_size)
        Q = int(cache_position.numel())
        K = int(kv_length)
        kv_abs = kv_offset + torch.arange(K, device=device)
        q_abs = cache_position.view(Q, 1).to(kv_abs.dtype)
        visible = kv_abs.view(1, K) <= q_abs
        mask = visible.unsqueeze(0).unsqueeze(1)
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

        _mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = _simple_sdpa_mask_interface
        _mu.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = _simple_sdpa_mask_interface
        print("[patch] mask interface set to simple(sdpa/eager)")
    except Exception as exc:
        print(f"[WARN] patch mask failed: {exc}")


_patch_transformers_mask()


# ----------------- ExecuTorch helpers -----------------
def et_load_forward(pte_path: str):
    try:
        from executorch.runtime import Runtime
    except Exception as exc:
        raise RuntimeError("ExecuTorch runtime is required to run the debug PTE") from exc

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


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float, float]:
    a = a.float().detach()
    b = b.float().detach()
    diff = a - b
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-12)
    mae = diff.abs().mean().item() / scale
    mse = (diff * diff).mean().item() / (scale * scale)
    cos = F.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=-1).item()
    max_abs = diff.abs().max().item() / scale
    return mae, mse, cos, max_abs


def maybe_load(path: Optional[str]) -> Optional[torch.Tensor]:
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.from_numpy(np.load(path)).to(torch.float16).contiguous()


# ----------------- HF modules -----------------
def load_text_only_fp16(model_id: str, revision: str):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        config=cfg,
        trust_remote_code=True,
        revision=revision,
        low_cpu_mem_usage=True,
        dtype=torch.float16,
    ).eval()
    omni.to(torch.device("cpu"))
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


# ----------------- Layer taps -----------------
class _TapModule(nn.Module):
    """Wraps a module and stores its primary output (first element if tuple)."""

    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod
        self.last_out: Optional[torch.Tensor] = None

    def forward(self, *args, **kwargs):
        out = self.mod(*args, **kwargs)
        if isinstance(out, tuple):
            self.last_out = out[0]
        else:
            self.last_out = out
        return out


def try_instrument_layer0(
    lm_backbone: nn.Module, need_attn: bool, need_residuals: bool, need_qkv: bool
) -> Tuple[Optional[_TapModule], Optional[_TapModule], Optional[_TapModule], Optional[_TapModule], Optional[_TapModule], Optional[_TapModule], Optional[_TapModule], Optional[int], Optional[int]]:
    """Wrap layer0 pieces so their outputs become visible to the wrapper."""

    def _get_layers(mod: nn.Module):
        if hasattr(mod, "layers"):
            return mod.layers
        if hasattr(mod, "model") and hasattr(mod.model, "layers"):
            return mod.model.layers
        if hasattr(mod, "decoder") and hasattr(mod.decoder, "layers"):
            return mod.decoder.layers
        return None

    layers = _get_layers(lm_backbone)
    if layers is None or len(layers) == 0:
        print("[WARN] Cannot locate backbone layers; skip tap.")
        return None, None

    layer0 = layers[0]
    attn = getattr(layer0, "self_attn", None) or getattr(layer0, "attention", None)
    mlp = getattr(layer0, "mlp", None) or getattr(layer0, "feed_forward", None)
    if attn is None or mlp is None:
        print("[WARN] layer0 is missing self_attn/mlp; skip tap.")
        return (None, None, None, None, None, None, None, None, None)

    attn_ln = getattr(layer0, "input_layernorm", None) or getattr(layer0, "ln1", None)
    post_ln = getattr(layer0, "post_attention_layernorm", None) or getattr(layer0, "ln2", None)

    # QKV taps must be installed on the real attention module, not the wrapper.
    attn_target = attn
    if isinstance(attn_target, _TapModule):
        attn_target = attn_target.mod
    q_proj = getattr(attn_target, "q_proj", None)
    k_proj = getattr(attn_target, "k_proj", None)
    v_proj = getattr(attn_target, "v_proj", None)
    num_heads = getattr(attn_target, "num_heads", None) or getattr(attn_target, "n_heads", None) or getattr(attn_target, "n_head", None)
    head_dim = getattr(attn_target, "head_dim", None) or getattr(attn_target, "hidden_size_head", None)

    if need_qkv:
        if q_proj is not None and not isinstance(q_proj, _TapModule):
            q_proj = _TapModule(q_proj)
            setattr(attn_target, "q_proj", q_proj)
        if k_proj is not None and not isinstance(k_proj, _TapModule):
            k_proj = _TapModule(k_proj)
            setattr(attn_target, "k_proj", k_proj)
        if v_proj is not None and not isinstance(v_proj, _TapModule):
            v_proj = _TapModule(v_proj)
            setattr(attn_target, "v_proj", v_proj)

    # Avoid double wrapping if reusing the same module.
    if need_attn and not isinstance(attn, _TapModule):
        attn = _TapModule(attn)
        setattr(layer0, "self_attn", attn)
    if need_attn and not isinstance(mlp, _TapModule):
        mlp = _TapModule(mlp)
        setattr(layer0, "mlp", mlp)
    if need_residuals:
        if attn_ln is not None and not isinstance(attn_ln, _TapModule):
            attn_ln = _TapModule(attn_ln)
            setattr(layer0, "input_layernorm", attn_ln)
        if post_ln is not None and not isinstance(post_ln, _TapModule):
            post_ln = _TapModule(post_ln)
            setattr(layer0, "post_attention_layernorm", post_ln)

    if not need_attn:
        attn = None
        mlp = None
    print("[INFO] layer0 wrapped for taps:",
          f"attn/mlp={need_attn}, residuals={need_residuals}, qkv={need_qkv}")
    return attn, mlp, attn_ln, post_ln, q_proj, k_proj, v_proj, num_heads, head_dim


# ----------------- Debug wrapper -----------------
class PrefillHiddenWrapper(nn.Module):
    """
    Returns:
      - logits for the last token
      - last hidden_state token (backbone output)
      - per-layer hidden_states last token (embedding + each block), optionally truncated
      - optional: layer0 attention output and MLP output (last token)
    """

    def __init__(self, lm_backbone, lm_head, embed, max_hidden_layers: Optional[int],
                 attn_tap: Optional[_TapModule] = None, mlp_tap: Optional[_TapModule] = None,
                 attn_ln_tap: Optional[_TapModule] = None, post_ln_tap: Optional[_TapModule] = None,
                 q_tap: Optional[_TapModule] = None, k_tap: Optional[_TapModule] = None, v_tap: Optional[_TapModule] = None,
                 tap_residuals: bool = False, tap_qkv: bool = False, tap_attn_scores: bool = False,
                 attn_num_heads: Optional[int] = None, attn_head_dim: Optional[int] = None):
        super().__init__()
        self.backbone = lm_backbone
        self.lm_head = lm_head
        self.embed = embed
        self.max_hidden_layers = max_hidden_layers
        self.attn_tap = attn_tap
        self.mlp_tap = mlp_tap
        self.attn_ln_tap = attn_ln_tap
        self.post_ln_tap = post_ln_tap
        self.q_tap = q_tap
        self.k_tap = k_tap
        self.v_tap = v_tap
        self.tap_residuals = tap_residuals
        self.tap_qkv = tap_qkv
        self.tap_attn_scores = tap_attn_scores
        self.attn_num_heads = attn_num_heads
        self.attn_head_dim = attn_head_dim

    def forward(self, input_ids: torch.Tensor, audio_ctx: torch.Tensor):
        tokens = self.embed(input_ids.to(torch.long))
        hidden = torch.cat([tokens, audio_ctx.to(torch.float16).unsqueeze(0)], dim=1)
        out = self.backbone(inputs_embeds=hidden, use_cache=True, output_hidden_states=True)
        logits = self.lm_head(out.last_hidden_state[:, -1, :]).squeeze(0)
        last_token = out.last_hidden_state[:, -1, :]
        hidden_states = out.hidden_states if out.hidden_states is not None else ()
        if self.max_hidden_layers is not None:
            hidden_states = tuple(hidden_states[: (self.max_hidden_layers + 1)])
        last_tokens = tuple(h[:, -1, :] for h in hidden_states)
        squeezed = tuple(t.squeeze(0) for t in last_tokens)

        def _last_token_general(t: torch.Tensor) -> torch.Tensor:
            if t.dim() < 2:
                return t
            # pick the largest non-batch dim as sequence and take last element
            dims = list(t.shape)
            seq_dim = 1 + max(range(len(dims) - 1), key=lambda i: dims[i + 1])
            return t.select(seq_dim, dims[seq_dim] - 1)

        extra = []
        if self.attn_tap is not None and self.mlp_tap is not None:
            if self.attn_tap.last_out is None or self.mlp_tap.last_out is None:
                raise RuntimeError("Taps enabled but no outputs captured; did layer0 run?")
            extra.extend([self.attn_tap.last_out.squeeze(0), self.mlp_tap.last_out.squeeze(0)])

        if self.tap_residuals:
            if self.attn_ln_tap is None or self.post_ln_tap is None:
                raise RuntimeError("Residual taps requested but ln modules missing.")
            if self.attn_ln_tap.last_out is None or self.post_ln_tap.last_out is None:
                raise RuntimeError("Residual taps enabled but ln outputs missing.")
            emb_full = hidden_states[0] if len(hidden_states) > 0 else hidden  # [1, T, H]
            attn_out_full = self.attn_tap.last_out if self.attn_tap is not None else None
            mlp_out_full = self.mlp_tap.last_out if self.mlp_tap is not None else None
            if attn_out_full is None or mlp_out_full is None:
                raise RuntimeError("Residual taps expect attn/mlp outputs; missing.")
            ln1_last = self.attn_ln_tap.last_out[:, -1, :].squeeze(0)
            ln2_last = self.post_ln_tap.last_out[:, -1, :].squeeze(0)
            attn_resid_last = (emb_full[:, -1, :] + attn_out_full[:, -1, :]).squeeze(0)
            mlp_resid_last = attn_resid_last + mlp_out_full[:, -1, :].squeeze(0)
            extra.extend([ln1_last])
            extra.append(attn_resid_last)
            extra.extend([ln2_last])
            extra.append(mlp_resid_last)

        if self.tap_qkv:
            for name, tap in [("q", self.q_tap), ("k", self.k_tap), ("v", self.v_tap)]:
                if tap is None or tap.last_out is None:
                    raise RuntimeError(f"qkv tap missing output for {name}")
                extra.append(_last_token_general(tap.last_out).squeeze(0))

        if self.tap_attn_scores:
            if self.q_tap is None or self.k_tap is None or self.v_tap is None:
                raise RuntimeError("attn_scores tap requires qkv taps.")
            if self.q_tap.last_out is None or self.k_tap.last_out is None or self.v_tap.last_out is None:
                raise RuntimeError("attn_scores tap missing qkv outputs.")
            q_full = self.q_tap.last_out.to(torch.float32)  # [B, T, Hq]
            k_full = self.k_tap.last_out.to(torch.float32)
            v_full = self.v_tap.last_out.to(torch.float32)
            B, T, H = q_full.shape
            ctx_merge = q_full[:, -1, :].squeeze(0)
            probs_last = torch.zeros(T, device=q_full.device, dtype=q_full.dtype)

            hd = int(self.attn_head_dim) if self.attn_head_dim else None
            if hd is None or hd <= 0:
                # fallback: try to infer from q_full (common head_dim=128)
                if H % 16 == 0:
                    hd = H // 16
                else:
                    hd = H
            q_heads = H // hd
            k_heads = max(1, int(k_full.shape[-1] // hd)) if k_full.shape[-1] % hd == 0 else 1
            v_heads = max(1, int(v_full.shape[-1] // hd)) if v_full.shape[-1] % hd == 0 else 1
            try:
                q = q_full.view(B, T, q_heads, hd).transpose(1, 2)  # [B, qh, T, hd]
                k = k_full.view(B, T, k_heads, hd).transpose(1, 2)  # [B, kh, T, hd]
                v = v_full.view(B, T, v_heads, hd).transpose(1, 2)  # [B, vh, T, hd]
                # Align k/v heads to q heads (multi-query: kh=vh=2, qh=16 -> repeat)
                if q_heads % k_heads == 0:
                    rep = q_heads // k_heads
                    k = k.repeat_interleave(rep, dim=1)
                if q_heads % v_heads == 0:
                    rep = q_heads // v_heads
                    v = v.repeat_interleave(rep, dim=1)
                # If still mismatched, slice to min
                min_h = min(q.shape[1], k.shape[1], v.shape[1])
                q = q[:, :min_h]
                k = k[:, :min_h]
                v = v[:, :min_h]
                q_last = q[:, :, -1:, :]                       # [B, Hh, 1, hd]
                attn_scores = torch.matmul(q_last, k.transpose(-2, -1)) / (hd ** 0.5)  # [B, Hh, 1, T]
                attn_probs = torch.softmax(attn_scores, dim=-1)
                ctx = torch.matmul(attn_probs, v)  # [B, Hh, 1, hd]
                ctx_merge = ctx.view(B, -1).squeeze(0)  # [Hh*hd]
                probs_last = attn_probs.view(B, min_h, T).squeeze(0)  # [Hh, T] or [T]
            except Exception as exc:
                print(f"[WARN] attn_scores tap failed ({exc}); using zero ctx/probs.")
                ctx_merge = torch.zeros_like(ctx_merge)
                probs_last = torch.zeros_like(probs_last)
            extra.extend([ctx_merge, probs_last])

        return (logits, last_token.squeeze(0), *squeezed, *extra)


# ----------------- Export -----------------
def export_debug_prefill(prefill: nn.Module, l_text: int, n_audio: int, out_path: str):
    print(f"[INFO] Exporting debug prefill to {out_path} ...")
    input_ids_ex = torch.ones(1, l_text, dtype=torch.int32)
    audio_ctx_ex = torch.zeros(n_audio, 2048, dtype=torch.float16)
    from torch.export import export as texport
    from executorch.exir import to_edge_transform_and_lower

    ep = texport(prefill, (input_ids_ex, audio_ctx_ex))
    exec_prog = to_edge_transform_and_lower(ep, partitioner=[]).to_executorch()
    with open(out_path, "wb") as f:
        f.write(exec_prog.buffer)
    print(f"[OK] debug prefill -> {out_path}")


# ----------------- Main flow -----------------
def _main_impl():
    ap = argparse.ArgumentParser(description="Layer-wise diff between PyTorch prefill and ExecuTorch export.")
    ap.add_argument("--audio", default="artifacts/golden_30s/audio_emb_exec.npy")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prompt", default="User: transcribe the following audio. Assistant:")
    ap.add_argument("--l_text", type=int, default=64)
    ap.add_argument("--n_audio", type=int, default=750)
    ap.add_argument("--pte_out", default="llm_prefill_tokens_debug.pte")
    ap.add_argument("--reuse_pte", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-3, help="Threshold to mark first drifting layer (by relative MAE).")
    ap.add_argument("--max_hidden_layers", type=int, default=None, help="Only keep this many hidden layers for export.")
    ap.add_argument("--tap_layer0", action="store_true", help="Export layer0 attention/MLP outputs.")
    ap.add_argument("--tap_layer0_residuals", action="store_true", help="Export layer0 ln1/attn_resid/ln2/mlp_resid (last token).")
    ap.add_argument("--tap_layer0_qkv", action="store_true", help="Export layer0 q/k/v projections (last token).")
    ap.add_argument("--tap_layer0_attn_scores", action="store_true", help="Export layer0 attention softmax (last token) and pre-o_proj context.")
    args = ap.parse_args()

    audio = maybe_load(args.audio)
    if audio is None:
        raise RuntimeError("Need --audio pointing to a .npy embedding.")
    print(f"[INFO] audio_emb: {tuple(audio.shape)}")

    lm_backbone, lm_head, embed = load_text_only_fp16(args.model_id, args.revision)
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, revision=args.revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    input_ids = build_prompt_tokens(tokenizer, args.prompt, args.l_text, args.use_chat_template)
    audio_ctx = align_audio_ctx(audio, args.n_audio)

    attn_tap = mlp_tap = attn_ln_tap = post_ln_tap = q_tap = k_tap = v_tap = None
    num_heads = head_dim = None
    if args.tap_layer0 or args.tap_layer0_residuals or args.tap_layer0_qkv or args.tap_layer0_attn_scores:
        attn_tap, mlp_tap, attn_ln_tap, post_ln_tap, q_tap, k_tap, v_tap, num_heads, head_dim = try_instrument_layer0(
            lm_backbone,
            need_attn=args.tap_layer0 or args.tap_layer0_residuals,
            need_residuals=args.tap_layer0_residuals,
            need_qkv=args.tap_layer0_qkv or args.tap_layer0_attn_scores,
        )
    prefill = PrefillHiddenWrapper(
        lm_backbone, lm_head, embed, args.max_hidden_layers,
        attn_tap, mlp_tap, attn_ln_tap, post_ln_tap, q_tap, k_tap, v_tap,
        tap_residuals=args.tap_layer0_residuals, tap_qkv=args.tap_layer0_qkv or args.tap_layer0_attn_scores,
        tap_attn_scores=args.tap_layer0_attn_scores, attn_num_heads=num_heads, attn_head_dim=head_dim,
    ).eval()

    needs_export = True
    if args.reuse_pte and os.path.exists(args.pte_out) and not (args.tap_layer0 or args.tap_layer0_residuals or args.tap_layer0_qkv or args.tap_layer0_attn_scores):
        needs_export = False
    if needs_export:
        if (args.tap_layer0 or args.tap_layer0_residuals or args.tap_layer0_qkv or args.tap_layer0_attn_scores) and args.reuse_pte:
            print("[INFO] Forcing re-export because tap options change the output signature.")
        export_debug_prefill(prefill, args.l_text, args.n_audio, args.pte_out)
    else:
        print(f"[INFO] Reusing existing debug PTE at {args.pte_out}")

    with torch.no_grad():
        pt_out = prefill(input_ids, audio_ctx)
    et_method = et_load_forward(args.pte_out)
    et_out = et_call(et_method, [input_ids, audio_ctx])

    if len(pt_out) != len(et_out):
        raise RuntimeError(f"Output count mismatch: torch={len(pt_out)} vs et={len(et_out)}")

    extra_names: List[str] = []
    if args.tap_layer0 and attn_tap is not None and mlp_tap is not None:
        extra_names += ["layer0_attn", "layer0_mlp"]
    if args.tap_layer0_residuals:
        extra_names += ["layer0_ln1", "layer0_attn_resid", "layer0_ln2", "layer0_mlp_resid"]
    if args.tap_layer0_qkv:
        extra_names += ["layer0_q", "layer0_k", "layer0_v"]
    if args.tap_layer0_attn_scores:
        extra_names += ["layer0_attn_ctx", "layer0_attn_probs"]

    extra_count = len(extra_names)
    n_hidden = len(pt_out) - 2 - extra_count
    if n_hidden <= 0:
        raise RuntimeError("Unexpected output layout; hidden_states missing?")

    torch_logits = pt_out[0].float()
    torch_final = pt_out[1].float()
    torch_layers = [pt_out[2 + i].float() for i in range(n_hidden)]
    torch_extras = [pt_out[2 + n_hidden + i].float() for i in range(extra_count)]

    et_logits = to_torch_tensor(et_out[0]).float()
    et_final = to_torch_tensor(et_out[1]).float()
    et_layers = [to_torch_tensor(et_out[2 + i]).float() for i in range(n_hidden)]
    et_extras = [to_torch_tensor(et_out[2 + n_hidden + i]).float() for i in range(extra_count)]

    print("\n=== Layer last-token diffs (PyTorch vs ExecuTorch, relative errors) ===")
    mae, mse, cos, mx = compare_tensors(torch_logits, et_logits)
    print(f"logits          : RelMAE={mae:.3e}  RelMSE={mse:.3e}  Cos={cos:.6f}  RelMaxAbs={mx:.3e}")
    mae, mse, cos, mx = compare_tensors(torch_final, et_final)
    print(f"backbone output : RelMAE={mae:.3e}  RelMSE={mse:.3e}  Cos={cos:.6f}  RelMaxAbs={mx:.3e}")

    first_bad = None
    for idx, (pt, et) in enumerate(zip(torch_layers, et_layers)):
        name = "emb" if idx == 0 else f"layer{idx:02d}"
        mae, mse, cos, mx = compare_tensors(pt, et)
        flag = "  <-- drift" if mae > args.tol and first_bad is None else ""
        if flag and first_bad is None:
            first_bad = name
        print(f"{name:12s}: RelMAE={mae:.3e}  RelMSE={mse:.3e}  Cos={cos:.6f}  RelMaxAbs={mx:.3e}{flag}")

    if extra_count:
        for name, t_pt, t_et in zip(extra_names, torch_extras, et_extras):
            mae, mse, cos, mx = compare_tensors(t_pt, t_et)
            print(f"{name:12s}: RelMAE={mae:.3e}  RelMSE={mse:.3e}  Cos={cos:.6f}  RelMaxAbs={mx:.3e}")

    if first_bad is None:
        print("\n[RESULT] No layer exceeded relative MAE threshold; logits mismatch likely elsewhere.")
    else:
        print(f"\n[RESULT] First layer over relative MAE threshold ({args.tol}): {first_bad}")


def main():
    log_path = "layer_diff_fp16.txt"
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = _Tee(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee):
            _main_impl()


if __name__ == "__main__":
    main()
