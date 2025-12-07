#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the Qwen2.5-Omni prefill through ExecuTorch on the Metal backend and
compare the logits against the existing CPU/XNNPACK export as well as the
pure PyTorch reference. This follows the "method 2" sanity check to see if
the large drift only appears with the CPU kernels.
"""

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from transformers import AutoConfig, AutoTokenizer, Qwen2_5OmniForConditionalGeneration
from transformers.cache_utils import Cache
from executorch.backends.transforms import get_shape
from executorch.exir.sym_util import eval_shape

os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "sdpa")
os.environ.setdefault("PYTORCH_DISABLE_MPS_FALLBACK", "1")
torch.set_grad_enabled(False)


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


def _patch_mps_split_dim():
    try:
        from executorch.backends.apple.mps.operators import shape_ops
        from typing import cast
    except Exception as exc:
        print(f"[WARN] MPS split patch unavailable: {exc}")
        return

    SplitVisitor = getattr(shape_ops, "SplitWithSizesVisitor", None)
    if SplitVisitor is None:
        print("[WARN] SplitWithSizesVisitor missing; skip patch.")
        return

    get_input_node = shape_ops.get_input_node

    def define_node(self, node: fx.Node, mps_graph):
        input_tensor = get_input_node(node, 0)
        input1_id = self.define_tensor(input_tensor, mps_graph)
        output_ids = self.define_tensor_list(node, mps_graph)
        split_sizes = eval_shape(cast(torch.SymInt, node.args[1]))
        dim = int(node.args[2])
        input_shape = get_shape(input_tensor)
        if dim < 0:
            dim += len(input_shape)
        if dim < 0 or dim >= len(input_shape):
            raise RuntimeError(
                f"[patched] split_copy: dim {dim} out of range for input tensor with {len(input_shape)} dimensions"
            )
        from executorch.backends.apple.mps.serialization.mps_graph_schema import (
            MPSNode,
            MPSSplitWithSizes,
        )

        mps_node = MPSNode(
            mpsnode_union=MPSSplitWithSizes(
                input1_id=input1_id,
                output_ids=output_ids,
                split_sizes=split_sizes,
                dim=dim,
            )
        )
        mps_graph.mps_nodes.append(mps_node)

    SplitVisitor.define_node = define_node
    print("[patch] SplitWithSizesVisitor now wraps negative dims.")


_patch_mps_split_dim()


# ----------------- ExecuTorch helpers -----------------
def et_load_forward(pte_path: str, runtime=None):
    from executorch.runtime import Runtime

    rt = runtime if runtime is not None else Runtime.get()
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


def maybe_load(path: Optional[str]) -> Optional[torch.Tensor]:
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.from_numpy(np.load(path)).to(torch.float16).contiguous()


# ----------------- HF modules -----------------
def load_hf_modules(model_id: str, revision: str):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        config=cfg,
        trust_remote_code=True,
        revision=revision,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    ).eval()
    thinker = getattr(omni, "thinker", getattr(omni, "model", omni))
    lm_backbone = getattr(thinker, "model", thinker)
    lm_head = getattr(thinker, "lm_head", None) or getattr(thinker, "output", None) or getattr(omni, "lm_head", None)
    embed = getattr(lm_backbone, "embed_tokens", None) or getattr(thinker, "embed_tokens", None) \
        or getattr(lm_backbone, "tok_embeddings", None)
    if not isinstance(lm_head, nn.Module) or not isinstance(embed, nn.Module):
        raise RuntimeError("Missing lm_head or embedding module in HF checkpoint")
    lm_backbone.half()
    lm_head.half()
    embed.half()
    return lm_backbone.eval(), lm_head.eval(), embed.eval()


def run_hf_prefill(
    lm_backbone: nn.Module,
    lm_head: nn.Module,
    embed: nn.Module,
    input_ids: torch.Tensor,
    audio_ctx: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        prompt_embeds = embed(input_ids.to(torch.long))
        hidden = torch.cat([prompt_embeds, audio_ctx.unsqueeze(0)], dim=1)
        out = lm_backbone(inputs_embeds=hidden, use_cache=False)
        logits = lm_head(out.last_hidden_state[:, -1, :]).squeeze(0).float()
    return logits


# ----------------- Prefill wrapper for export -----------------
def flatten_pkv_from_cache(cache: Cache) -> List[torch.Tensor]:
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else list(cache)
    flat = []
    for k, v in legacy:
        flat += [k, v]
    return flat


class PrefillTokensWrapper(nn.Module):
    """Same wrapper as used for the CPU/XNNPACK export."""

    def __init__(self, lm_backbone, lm_head, embed):
        super().__init__()
        self.backbone = lm_backbone
        self.lm_head = lm_head
        self.embed = embed

    def forward(self, input_ids: torch.Tensor, audio_ctx: torch.Tensor):
        x = self.embed(input_ids.to(torch.long))
        hidden = torch.cat([x, audio_ctx.to(torch.float16).unsqueeze(0)], dim=1)
        out = self.backbone(inputs_embeds=hidden, use_cache=True)
        logits0 = self.lm_head(out.last_hidden_state[:, -1, :]).squeeze(0)
        kv_flat = flatten_pkv_from_cache(out.past_key_values)
        return (logits0, *kv_flat)


# ----------------- Export helper -----------------
def export_metal_prefill(
    prefill_mod: nn.Module,
    l_text: int,
    n_audio: int,
    audio_dim: int,
    out_path: str,
    force: bool = False,
) -> str:
    if os.path.exists(out_path) and not force:
        print(f"[INFO] Reusing existing Metal prefill PTE: {out_path}")
        return out_path

    try:
        from torch.export import export as texport
        from executorch.backends.apple.mps.partition import MPSPartitioner
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        from executorch.exir.backend.backend_details import CompileSpec
    except Exception as exc:
        raise RuntimeError("Missing torch.export or ExecuTorch MPS dependencies") from exc

    example_tokens = torch.ones(1, l_text, dtype=torch.int32)
    example_audio = torch.zeros(n_audio, audio_dim, dtype=torch.float16)

    print("[INFO] Exporting Metal prefill PTE ...")
    with torch.no_grad():
        ep = texport(prefill_mod.eval(), (example_tokens, example_audio))

    def _normalize_split_dims(epg):
        gm = epg.graph_module
        changed = False
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            target_name = getattr(node.target, "__name__", "")
            if "split" not in target_name:
                continue
            args = list(node.args)
            if len(args) < 3:
                continue
            dim = int(args[2])
            if dim >= 0:
                continue
            rank = None
            arg0 = args[0]
            meta_val = getattr(arg0, "meta", {}).get("val", None) if isinstance(arg0, torch.fx.Node) else None
            if meta_val is not None and hasattr(meta_val, "ndim"):
                rank = meta_val.ndim
            elif hasattr(arg0, "meta") and "tensor_meta" in arg0.meta:
                rank = len(arg0.meta["tensor_meta"].shape)
            if rank is None or rank <= 0:
                continue
            args[2] = dim % rank
            node.args = tuple(args)
            changed = True
        if changed:
            gm.recompile()

    def _cast_int_matmul_inputs(epg):
        gm = epg.graph_module
        changed = False
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in (torch.ops.aten.matmul.default, torch.ops.aten.bmm.default):
                continue
            args = list(node.args)
            for idx in (0, 1):
                arg = args[idx]
                if not isinstance(arg, fx.Node):
                    continue
                meta = arg.meta.get("val") or arg.meta.get("example_value")
                dtype = None
                if isinstance(meta, torch.Tensor):
                    dtype = meta.dtype
                elif "tensor_meta" in arg.meta:
                    dtype = arg.meta["tensor_meta"].dtype
                if dtype is not None and dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                    continue
                with gm.graph.inserting_before(node):
                    cast_node = gm.graph.call_function(
                        torch.ops.aten._to_copy.default,
                        (arg, torch.float16, False, False),
                    )
                args[idx] = cast_node
                changed = True
            node.args = tuple(args)
        if changed:
            gm.recompile()

    _normalize_split_dims(ep)
    _cast_int_matmul_inputs(ep)

    compile_specs = [CompileSpec("use_fp16", bytes([1]))]
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[MPSPartitioner(compile_specs=compile_specs)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    exec_prog = edge.to_executorch()
    with open(out_path, "wb") as f:
        f.write(exec_prog.buffer)
    print(f"[OK] Metal prefill exported to {out_path}")
    return out_path


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Run ExecuTorch prefill on Metal and compare with CPU + HF baselines.")
    ap.add_argument("--exec_audio", default="artifacts/golden_30s/audio_emb_exec.npy")
    ap.add_argument("--ref_audio", default="artifacts/golden_30s/audio_emb_ref.npy")
    ap.add_argument("--cpu_pte", default="llm_prefill_tokens_64p_750a_fp16.pte")
    ap.add_argument("--metal_pte", default="llm_prefill_tokens_64p_750a_mps_fp16.pte")
    ap.add_argument("--force_export", action="store_true", help="Re-export the Metal PTE even if a cached file exists.")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prompt", default="User: transcribe the following audio. Assistant:")
    ap.add_argument("--l_text", type=int, default=64)
    ap.add_argument("--n_audio", type=int, default=750)
    ap.add_argument("--use_chat_template", action="store_true")
    args = ap.parse_args()

    exec_audio = maybe_load(args.exec_audio)
    ref_audio = maybe_load(args.ref_audio)
    if exec_audio is None and ref_audio is None:
        raise RuntimeError("Need at least one of --exec_audio or --ref_audio")
    print(f"[INFO] exec_audio: {None if exec_audio is None else tuple(exec_audio.shape)}")
    print(f"[INFO] ref_audio : {None if ref_audio is None else tuple(ref_audio.shape)}")

    audio_dim = None
    for cand in [exec_audio, ref_audio]:
        if cand is not None:
            audio_dim = cand.shape[1]
            break
    if audio_dim is None:
        audio_dim = 2048

    lm_backbone, lm_head, embed = load_hf_modules(args.model_id, args.revision)
    prefill_mod = PrefillTokensWrapper(lm_backbone, lm_head, embed).eval()

    metal_pte = export_metal_prefill(
        prefill_mod,
        args.l_text,
        args.n_audio,
        audio_dim,
        args.metal_pte,
        force=args.force_export,
    )
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    if not runtime.backend_registry.is_available("MPSBackend"):
        raise RuntimeError("This ExecuTorch build lacks the MPS backend; cannot run Metal prefill.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, revision=args.revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    input_ids = build_prompt_tokens(tokenizer, args.prompt, args.l_text, args.use_chat_template)

    def prepare(audio_tensor: torch.Tensor) -> torch.Tensor:
        return align_audio_ctx(audio_tensor.to(torch.float16).contiguous(), args.n_audio)

    ref_ctx = exec_ctx = None
    if ref_audio is not None:
        ref_ctx = prepare(ref_audio)
    if exec_audio is not None:
        exec_ctx = prepare(exec_audio)

    hf_logits_ref = hf_logits_exec = None
    if ref_ctx is not None:
        hf_logits_ref = run_hf_prefill(lm_backbone, lm_head, embed, input_ids, ref_ctx)
        print("[INFO] HF logits computed with ref audio embeddings.")
    if exec_ctx is not None:
        hf_logits_exec = run_hf_prefill(lm_backbone, lm_head, embed, input_ids, exec_ctx)
        print("[INFO] HF logits computed with exec audio embeddings.")

    cpu_logits_ref = cpu_logits_exec = None
    if args.cpu_pte and os.path.exists(args.cpu_pte):
        print(f"[INFO] Loading CPU/XNNPACK PTE: {args.cpu_pte}")
        cpu_method = et_load_forward(args.cpu_pte)
        if ref_ctx is not None:
            cpu_logits_ref = et_call(cpu_method, [input_ids, ref_ctx])[0].float()
            print("[INFO] ExecuTorch CPU logits (ref) computed.")
        if exec_ctx is not None:
            cpu_logits_exec = et_call(cpu_method, [input_ids, exec_ctx])[0].float()
            print("[INFO] ExecuTorch CPU logits (exec) computed.")
    else:
        print(f"[WARN] CPU PTE not found: {args.cpu_pte} (skipping CPU comparison)")

    metal_method = et_load_forward(metal_pte, runtime=runtime)
    metal_logits_ref = metal_logits_exec = None
    if ref_ctx is not None:
        metal_logits_ref = et_call(metal_method, [input_ids, ref_ctx])[0].float()
        print("[INFO] ExecuTorch Metal logits (ref) computed.")
    if exec_ctx is not None:
        metal_logits_exec = et_call(metal_method, [input_ids, exec_ctx])[0].float()
        print("[INFO] ExecuTorch Metal logits (exec) computed.")

    print("\n=== Comparisons ===")
    if hf_logits_ref is not None and hf_logits_exec is not None:
        mae, mse, cos = compare_logits(hf_logits_exec, hf_logits_ref)
        print(f"HF(exec_audio) vs HF(ref_audio)        : MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if cpu_logits_ref is not None and hf_logits_ref is not None:
        mae, mse, cos = compare_logits(cpu_logits_ref, hf_logits_ref)
        print(f"ET-CPU(ref_audio) vs HF(ref_audio)    : MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if cpu_logits_exec is not None and hf_logits_exec is not None:
        mae, mse, cos = compare_logits(cpu_logits_exec, hf_logits_exec)
        print(f"ET-CPU(exec_audio) vs HF(exec_audio)  : MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if metal_logits_ref is not None and hf_logits_ref is not None:
        mae, mse, cos = compare_logits(metal_logits_ref, hf_logits_ref)
        print(f"ET-Metal(ref_audio) vs HF(ref_audio)  : MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if metal_logits_exec is not None and hf_logits_exec is not None:
        mae, mse, cos = compare_logits(metal_logits_exec, hf_logits_exec)
        print(f"ET-Metal(exec_audio) vs HF(exec_audio): MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if metal_logits_ref is not None and cpu_logits_ref is not None:
        mae, mse, cos = compare_logits(metal_logits_ref, cpu_logits_ref)
        print(f"ET-Metal(ref) vs ET-CPU(ref)          : MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")
    if metal_logits_exec is not None and cpu_logits_exec is not None:
        mae, mse, cos = compare_logits(metal_logits_exec, cpu_logits_exec)
        print(f"ET-Metal(exec) vs ET-CPU(exec)        : MAE={mae:.3e}  MSE={mse:.3e}  Cos={cos:.6f}")

    print("\nInterpretation:")
    print("  • If ET-Metal aligns with HF while ET-CPU does not, the drift is coming from the CPU kernels.")
    print("  • If both ET versions disagree with HF, investigate the export graph itself (mask/cache logic).")


if __name__ == "__main__":
    main()
