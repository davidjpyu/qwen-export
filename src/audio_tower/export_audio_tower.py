import argparse
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration
from torch.export import export

# ExecuTorch (iOS / XNNPACK)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    to_edge_transform_and_lower,
)

logging.basicConfig(level=logging.INFO)

# --------------------------
# Static chunk plan for a fixed T
# --------------------------
def build_chunk_plan(T: int, n_window: int):
    """
    Build a static plan for the fixed time dimension T:
      - chunk_lengths: lengths of each chunk when splitting with base=2*n_window
      - aftercnn_lens: per-chunk length after the conv stack (~ceil(L/2))
    Returned as Python lists so torch.export sees only constants.
    """
    base = 2 * n_window
    n_chunks = math.ceil(T / base)
    chunk_lengths = [base] * n_chunks
    rem = T % base
    if rem != 0:
        chunk_lengths[-1] = rem
    aftercnn_lens = [(L - 1) // 2 + 1 for L in chunk_lengths]  # ceil(L/2)
    return chunk_lengths, aftercnn_lens


# --------------------------
# Static wrapper that avoids dynamic padding/masking
# --------------------------
class AudioTowerStatic(nn.Module):
    """
    Static (fixed-T) audio tower wrapper.
      Input:
        - input_features: [C=128, T=3000] 2-D tensor without a batch dim.
      Output:
        - token_audio: [N_chunks, D]
    Notes:
      - chunk_lengths / aftercnn_lens are baked in as Python constants;
      - no Tensor-driven slicing or `.tolist()` calls so torch.export stays happy.
    """
    def __init__(self, enc: nn.Module, T: int):
        super().__init__()
        self.enc = enc
        self.n_window = enc.n_window
        self.T = T

        # Precompute the chunk plan as Python constants
        self.chunk_lengths, self.aftercnn_lens_list = build_chunk_plan(T, self.n_window)

    @staticmethod
    def _pad_and_masks_static(chunk_list, chunk_lengths, device, dtype):
        """
        Constant-driven padding + mask construction.
        Args:
          - chunk_list[i]: [C, Ti]
          - chunk_lengths: Python list[int]
        Returns:
          - padded_feature: [N, C, Lmax]
          - batch_mask:     [N, 1, Lmax] (float 0/1)
          - mask_after:     [N, Lmax_after] (bool)
        """
        N = len(chunk_list)
        C = chunk_list[0].shape[0]
        Lmax = max(chunk_lengths)

        # 1) pad features using only Python int slices
        padded_feature = torch.zeros((N, C, Lmax), dtype=dtype, device=device)
        for i, L in enumerate(chunk_lengths):
            padded_feature[i, :, :L] = chunk_list[i]

        # 2) mask at the original resolution (broadcast compares only)
        lens_t = torch.tensor(chunk_lengths, device=device)      # [N]
        ar = torch.arange(Lmax, device=device)                   # [Lmax]
        batch_mask = (ar.unsqueeze(0) < lens_t.unsqueeze(1))     # [N, Lmax] bool
        batch_mask = batch_mask.to(padded_feature.dtype).unsqueeze(1)  # [N,1,Lmax] 0/1

        # 3) mask after the CNN downsample (again broadcast only)
        after = ((lens_t - 1) // 2 + 1)                          # [N]
        Lmax_after = int(after.max().item())
        ar2 = torch.arange(Lmax_after, device=device)
        mask_after = (ar2.unsqueeze(0) < after.unsqueeze(1))     # [N, Lmax_after] bool

        return padded_feature, batch_mask, mask_after

    def forward(self, input_features: torch.Tensor):
        """
        input_features: [C, T] (e.g. [128, 3000])
        """
        device = input_features.device
        dtype = input_features.dtype

        # 1) split along time using the static chunk plan
        chunk_list = input_features.split(self.chunk_lengths, dim=1)  # each [C, Ti]

        # 2) pad + masks via the static helper
        padded_feature, padded_mask, padded_mask_after_cnn = self._pad_and_masks_static(
            chunk_list, self.chunk_lengths, device, dtype
        )
        # padded_feature: [N, C, Lmax]
        # padded_mask:    [N, 1, Lmax]
        # padded_mask_after_cnn: [N, Lmax_after] (bool)

        # 3) Two conv + GELU stages plus masking -> [N, T_after, H]
        x = F.gelu(self.enc.conv1(padded_feature)) * padded_mask
        x = F.gelu(self.enc.conv2(x)).transpose(1, 2)  # [N, T_after, H]

        # 4) Position embedding
        pos = self.enc.positional_embedding.positional_embedding[: x.shape[1], :].unsqueeze(0)
        x = x + pos.to(x.dtype)

        # 5) Concatenate chunks using constant slicing to avoid shape-dependent ops
        pieces = []
        for i, L in enumerate(self.aftercnn_lens_list):
            pieces.append(x[i, :L, :])
        hidden_states = torch.cat(pieces, dim=0)          # [sum(Ti_after), H]

        # 6) Build cu_seqlens and attention_mask using only constants
        after_lens_i32 = torch.tensor(self.aftercnn_lens_list, device=hidden_states.device, dtype=torch.int32)
        cu_seqlens = torch.cat(
            (torch.zeros(1, device=hidden_states.device, dtype=torch.int32), after_lens_i32.cumsum(0)),
            dim=0,
        )
        # Keep FA2 behavior identical to the original implementation: pass None
        attn_impl = getattr(self.enc.config, "_attn_implementation", None)
        if attn_impl == "flash_attention_2":
            attention_mask = None
        else:
            # Use a 2D additive mask (same chunk = 0, different chunk = -inf) to avoid 4D masks
            L = int(after_lens_i32.sum().item())
            # Build segment ids via constant concatenation instead of repeat_interleave
            dev = hidden_states.device
            seg_chunks = [torch.full((L,), i, device=dev, dtype=torch.long)
                        for i, L in enumerate(self.aftercnn_lens_list)]
            seg_ids = torch.cat(seg_chunks, dim=0)  # [L], length is constant sum(aftercnn_lens_list)
            
            same_seg = seg_ids.unsqueeze(0) == seg_ids.unsqueeze(1)  # [L, L]
            finfo_min = torch.finfo(hidden_states.dtype).min
            attention_mask = torch.where(
                same_seg,
                torch.tensor(0.0, dtype=hidden_states.dtype, device=hidden_states.device),
                torch.tensor(finfo_min, dtype=hidden_states.dtype, device=hidden_states.device),
            )  # [L, L]

        # 7) Encoder stack
        for layer in self.enc.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            )[0]

        # 8) Split back into chunks and run pool/ln/proj
        outs = []
        for each in hidden_states.split(self.aftercnn_lens_list, dim=0):  # Python list[int]
            y = self.enc.avg_pooler(each.transpose(0, 1)).transpose_(0, 1)
            y = self.enc.ln_post(y)
            y = self.enc.proj(y)
            outs.append(y)

        token_audio = torch.cat(outs, dim=0)  # [N_chunks, D]
        return token_audio


def main():
    parser = argparse.ArgumentParser(description="Export Qwen2.5-Omni audio_tower (T=3000, static) to ExecuTorch (iOS/XNNPACK)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--T", type=int, default=3000, help="Whisper(30s, hop=160) -> T = 3000")
    parser.add_argument("--output_file", type=str, default="qwen_audio_tower_30s_static.pte")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    # 1) Load the model + audio tower module
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_id, config=cfg, trust_remote_code=True, dtype=dtype, device_map=None
    ).eval()
    enc = getattr(getattr(omni, "thinker", getattr(omni, "model", None)), "audio_tower").eval()

    # 2) Wrap it with the static adapter
    wrapped = AudioTowerStatic(enc, T=args.T).eval()

    # 3) Example input (fixed [C=128, T=3000], 2-D tensor)
    C, T = 128, args.T
    example_feats = torch.randn(C, T, dtype=dtype)

    # 4) Optional sanity check forward pass
    with torch.no_grad():
        y = wrapped(example_feats)
    logging.info(f"Sanity check output shape: {tuple(y.shape)}")

    # 5) Patch SDPA to a pure eager implementation while exporting
    import math as _m
    _orig_sdp = F.scaled_dot_product_attention

    def _sdp_eager(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        # q,k,v: [..., Lq, Dh], k^T: [..., Dh, Lk]
        if scale is None:
            scale = 1.0 / _m.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask  # handle 2D/broadcastable additive masks
        if is_causal:
            Lq, Lk = scores.size(-2), scores.size(-1)
            causal = torch.ones((Lq, Lk), device=scores.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)

    F.scaled_dot_product_attention = _sdp_eager

    # 6) Export a fully static graph (no Dynamo slicing / SDPA guards)
    with torch.no_grad():
        ep = export(wrapped, args=(example_feats,), strict=True)

    # Restore the original SDPA implementation
    F.scaled_dot_product_attention = _orig_sdp

    # 7) Lower to ExecuTorch (iOS/XNNPACK target)
    edge: EdgeProgramManager = to_edge_transform_and_lower(
        ep,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    exec_prog = edge.to_executorch()

    with open(args.output_file, "wb") as f:
        exec_prog.write_to_file(f)
    logging.info(f"Exported ExecuTorch program to {args.output_file}")


if __name__ == "__main__":
    main()
