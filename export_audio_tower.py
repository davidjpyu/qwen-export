# import torch, inspect
# from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration

# import torch.nn.functional as F

# def compute_aftercnn_lens(feature_lens: torch.LongTensor, n_window: int) -> torch.LongTensor:
#     # 1) 每条样本需要多少个 chunk（chunk 长度基准为 2*n_window）
#     chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()                # shape: (B,)

#     # 2) 先假设每个 chunk 都是满长度 2*n_window
#     total_chunks = int(chunk_num.sum().item())
#     chunk_lengths = torch.full(
#         (total_chunks,), n_window * 2,
#         dtype=torch.long, device=feature_lens.device
#     )

#     # 3) 把每条样本的最后一个 chunk 改成“余数长度”（如果余数=0，再改回满长度）
#     tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]         # 累计得到每条样本最后一个 chunk 的索引
#     remainder = feature_lens % (n_window * 2)
#     chunk_lengths[tail_chunk_index] = remainder
#     chunk_lengths = torch.where(chunk_lengths == 0, n_window * 2, chunk_lengths)

#     # 4) CNN 后的长度（实现里等价于 ceil(L/2)）
#     aftercnn_lens = (chunk_lengths - 1) // 2 + 1                                # shape: (total_chunks,)
#     return aftercnn_lens

# MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
# DTYPE = torch.float32

# # 1) 加载模型
# cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
# omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=DTYPE, device_map=None
# ).eval()

# enc = getattr(getattr(omni, "thinker", getattr(omni, "model", None)), "audio_tower")
# print("enc.forward:", inspect.signature(enc.forward))
# # 应该打印: (input_features, feature_lens=None, aftercnn_lens=None, **kwargs)

# # 2) 造一组“对齐形状”的假输入： [128, T]
# C, T = 128, 2048
# feats = torch.randn(C, T, dtype=DTYPE)              
# feature_lens = torch.tensor([T], dtype=torch.long)   # 每条样本的帧长度，long

# n_window = enc.n_window  # 从模块里拿
# aftercnn_lens = compute_aftercnn_lens(feature_lens, n_window)

# # 3) 跑前向，应该不再报 split/pad 的错
# with torch.no_grad():
#     out = enc(input_features=feats, feature_lens=feature_lens, aftercnn_lens=aftercnn_lens)

# # 打印出主结果的形状/类型，确认OK（不同实现可能返回张量/元组/字典）
# if isinstance(out, dict):
#     print("keys:", out.keys())
# elif isinstance(out, (tuple, list)):
#     print("tuple len:", len(out), "shape0:", tuple(out[0].shape) if torch.is_tensor(out[0]) else type(out[0]))
# else:
#     print("tensor out:", tuple(out.shape))


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
# 固定 T 的 chunk 计划（Python 常量）
# --------------------------
def build_chunk_plan(T: int, n_window: int):
    """
    基于固定 T，构造：
      - chunk_lengths: 以 base=2*n_window 切分后的每块长度（最后一块可能为余数）
      - aftercnn_lens: 每块经一次 ~ceil(L/2) 下采样后的长度
    都返回为 Python list[int]，以避免 torch.export 的动态图限制。
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
# 静态 wrapper：不调用库里的 padded_and_mask_function
# --------------------------
class AudioTowerStatic(nn.Module):
    """
    静态（固定 T）版 audio_tower：
      输入:
        - input_features: [C=128, T=3000]  （二维张量，无 batch 维）
      输出:
        - token_audio: [N_chunks, D]
    注意：
      - 将 chunk_lengths / aftercnn_lens 固定为构造期常量；
      - 不使用任何由 Tensor 决定的切片或 .tolist()
    """
    def __init__(self, enc: nn.Module, T: int):
        super().__init__()
        self.enc = enc
        self.n_window = enc.n_window
        self.T = T

        # 构造期用 Python 常量生成计划
        self.chunk_lengths, self.aftercnn_lens_list = build_chunk_plan(T, self.n_window)

    @staticmethod
    def _pad_and_masks_static(chunk_list, chunk_lengths, device, dtype):
        """
        纯常量驱动的 pad + 两种 mask（避免动态切片）。
        输入:
          - chunk_list[i]: [C, Ti]
          - chunk_lengths: Python list[int]
        返回:
          - padded_feature: [N, C, Lmax]
          - batch_mask:     [N, 1, Lmax] (float 0/1)
          - mask_after:     [N, Lmax_after] (bool)
        """
        N = len(chunk_list)
        C = chunk_list[0].shape[0]
        Lmax = max(chunk_lengths)

        # 1) pad features（Python int 切片 → 静态）
        padded_feature = torch.zeros((N, C, Lmax), dtype=dtype, device=device)
        for i, L in enumerate(chunk_lengths):
            padded_feature[i, :, :L] = chunk_list[i]

        # 2) 原分辨率 mask（广播比较 → 无动态切片）
        lens_t = torch.tensor(chunk_lengths, device=device)      # [N]
        ar = torch.arange(Lmax, device=device)                   # [Lmax]
        batch_mask = (ar.unsqueeze(0) < lens_t.unsqueeze(1))     # [N, Lmax] bool
        batch_mask = batch_mask.to(padded_feature.dtype).unsqueeze(1)  # [N,1,Lmax] 0/1

        # 3) 下采样后的 mask（广播比较）
        after = ((lens_t - 1) // 2 + 1)                          # [N]
        Lmax_after = int(after.max().item())
        ar2 = torch.arange(Lmax_after, device=device)
        mask_after = (ar2.unsqueeze(0) < after.unsqueeze(1))     # [N, Lmax_after] bool

        return padded_feature, batch_mask, mask_after

    def forward(self, input_features: torch.Tensor):
        """
        input_features: [C, T] （例如 [128, 3000]）
        """
        device = input_features.device
        dtype = input_features.dtype

        # 1) 固定计划切块（沿时间维 dim=1；Python list[int]）
        chunk_list = input_features.split(self.chunk_lengths, dim=1)  # list of [C, Ti]

        # 2) pad + 两种 mask（自实现，避免动态切片）
        padded_feature, padded_mask, padded_mask_after_cnn = self._pad_and_masks_static(
            chunk_list, self.chunk_lengths, device, dtype
        )
        # padded_feature: [N, C, Lmax]
        # padded_mask:    [N, 1, Lmax]
        # padded_mask_after_cnn: [N, Lmax_after] (bool)

        # 3) 两层 Conv + GELU + 掩码，转为 [N, T_after, H]
        x = F.gelu(self.enc.conv1(padded_feature)) * padded_mask
        x = F.gelu(self.enc.conv2(x)).transpose(1, 2)  # [N, T_after, H]

        # 4) 位置编码
        pos = self.enc.positional_embedding.positional_embedding[: x.shape[1], :].unsqueeze(0)
        x = x + pos.to(x.dtype)

        # 5) 用常量切片拼接，避免布尔索引带来的数据相关形状, = hidden_states = x[padded_mask_after_cnn]
        pieces = []
        for i, L in enumerate(self.aftercnn_lens_list):   # 这些 L 是 Python int（编译期常量）
            pieces.append(x[i, :L, :])                    # [:L] 是静态切片
        hidden_states = torch.cat(pieces, dim=0)          # [sum(Ti_after), H]

        # 6) 构造 cu_seqlens 与 attention_mask（基于构造期常量）
        after_lens_i32 = torch.tensor(self.aftercnn_lens_list, device=hidden_states.device, dtype=torch.int32)
        cu_seqlens = torch.cat(
            (torch.zeros(1, device=hidden_states.device, dtype=torch.int32), after_lens_i32.cumsum(0)),
            dim=0,
        )
        # 如果是 FA2，保持与原实现一致：直接传 None
        attn_impl = getattr(self.enc.config, "_attn_implementation", None)
        if attn_impl == "flash_attention_2":
            attention_mask = None
        else:
            # 用 2D 加性 mask（同块 0，跨块 -inf），避免 4D mask 在 sdpa 包装里被切片
            L = int(after_lens_i32.sum().item())
            # 常量拼接替代torch.repeat_interleave
            dev = hidden_states.device
            seg_chunks = [torch.full((L,), i, device=dev, dtype=torch.long)
                        for i, L in enumerate(self.aftercnn_lens_list)]
            seg_ids = torch.cat(seg_chunks, dim=0)  # [L]，长度是纯常量 sum(self.aftercnn_lens_list)
            
            same_seg = seg_ids.unsqueeze(0) == seg_ids.unsqueeze(1)  # [L, L]
            finfo_min = torch.finfo(hidden_states.dtype).min
            attention_mask = torch.where(
                same_seg,
                torch.tensor(0.0, dtype=hidden_states.dtype, device=hidden_states.device),
                torch.tensor(finfo_min, dtype=hidden_states.dtype, device=hidden_states.device),
            )  # [L, L]

        # 7) Encoder 堆叠
        for layer in self.enc.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            )[0]

        # 8) split 回 chunk，做 pool/ln/proj
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
    parser.add_argument("--T", type=int, default=3000, help="Whisper(30s, hop=160) → T = 3000")
    parser.add_argument("--output_file", type=str, default="qwen_audio_tower_30s_static.pte")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    # 1) 加载模型与子模块
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_id, config=cfg, trust_remote_code=True, dtype=dtype, device_map=None
    ).eval()
    enc = getattr(getattr(omni, "thinker", getattr(omni, "model", None)), "audio_tower").eval()

    # 2) 包装为静态版
    wrapped = AudioTowerStatic(enc, T=args.T).eval()

    # 3) 示例输入（固定 [C=128, T=3000]，二维！）
    C, T = 128, args.T
    example_feats = torch.randn(C, T, dtype=dtype)

    # 4) 先做一次 sanity check（可选）
    with torch.no_grad():
        y = wrapped(example_feats)
    logging.info(f"Sanity check output shape: {tuple(y.shape)}")

    # 5) --- 方案一：猴补 SDPA 为 eager 实现，仅在导出期间启用 ---
    import math as _m
    _orig_sdp = F.scaled_dot_product_attention

    def _sdp_eager(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        # q,k,v: [..., Lq, Dh], k^T: [..., Dh, Lk]
        if scale is None:
            scale = 1.0 / _m.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask  # 2D/可广播加性 mask
        if is_causal:
            Lq, Lk = scores.size(-2), scores.size(-1)
            causal = torch.ones((Lq, Lk), device=scores.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)

    F.scaled_dot_product_attention = _sdp_eager

    # 6) export（完全静态；避免 Dynamo 动态切片/SDPA guard）
    with torch.no_grad():
        ep = export(wrapped, args=(example_feats,), strict=True)

    # 恢复原 SDPA
    F.scaled_dot_product_attention = _orig_sdp

    # 7) Lower 到 ExecuTorch（iOS/XNNPACK）
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
