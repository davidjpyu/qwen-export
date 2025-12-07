#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_golden_audio_tower.py
生成 Audio Tower 的黄金样本，并做 ExecuTorch vs HF 数值对齐。

产物（默认写入 artifacts/golden_30s/）：
- mel.npy                : float32 [1,128,3000]
- audio_emb_exec.npy     : float32 [N_chunks, D]
- audio_emb_ref.npy      : float32 [N_chunks, D]
- feature_lens.npy       : int64   [1]
- aftercnn_lens.npy      : int64   [N_chunks]
- meta.json              : 配置、容差、误差统计等
"""
import os
import math
import json
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration

# -----------------------------
# 配置（按需修改）
# -----------------------------
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
DTYPE = torch.float32
SR = 16000
DURATION_S = 30                  # 30s -> T = 3000（与你的静态 audio_tower 对齐）
PROC_PTE = "whisper_preprocess.pte"
ENC_PTE  = "qwen_audio_tower_30s_static.pte"

# -----------------------------
# 工具：按 HF 实现计算 aftercnn_lens
# -----------------------------
def compute_aftercnn_lens_from_T(T: int, n_window: int) -> torch.LongTensor:
    """
    与 Qwen2.5-Omni audio_tower 内部的分块保持一致：
    - base = 2 * n_window
    - 把时间维 T 切成若干 base 的块（最后一块可短）
    - 每块长度 L -> after = (L - 1) // 2 + 1
    """
    base = 2 * n_window
    chunk_num = math.ceil(T / base)
    chunk_lengths = [base] * chunk_num
    rem = T % base
    if rem != 0:
        chunk_lengths[-1] = rem
    aftercnn_lens = [(L - 1) // 2 + 1 for L in chunk_lengths]
    return torch.tensor(aftercnn_lens, dtype=torch.long)

# -----------------------------
# ExecuTorch 运行工具
# -----------------------------
def run_executorch(proc_pte: str, enc_pte: str, wave_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入:
        wave_1d: torch.float32 [SR*DURATION_S]
    输出:
        mel:   torch.float32 [1,128,3000]
        y_exec:torch.float32 [N_chunks, D]
    """
    from executorch.runtime import Runtime
    rt = Runtime.get()

    # whisper preprocess
    proc_prog = rt.load_program(proc_pte)
    proc_fwd  = proc_prog.load_method("forward")
    mel_list = proc_fwd.execute([wave_1d])
    mel = mel_list[0]
    if not isinstance(mel, torch.Tensor):
        # 保险：某些版本可能返回 ET 自己的张量
        try:
            mel = mel.to_torch()
        except Exception:
            mel = torch.tensor(mel)

    # audio tower
    enc_prog = rt.load_program(enc_pte)
    enc_fwd  = enc_prog.load_method("forward")
    feats_CT = mel.squeeze(0)  # [128, T]
    enc_out_list = enc_fwd.execute([feats_CT])
    y_exec = enc_out_list[0]
    if not isinstance(y_exec, torch.Tensor):
        try:
            y_exec = y_exec.to_torch()
        except Exception:
            y_exec = torch.tensor(y_exec)

    # 统一类型与内存
    mel   = mel.to(torch.float32).contiguous()
    y_exec = y_exec.to(torch.float32).contiguous()
    return mel, y_exec

# -----------------------------
# HF 参考前向（音频塔）
# -----------------------------
def run_hf_audio_tower(model_id: str, revision: str, dtype: torch.dtype,
                       feats_CT: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """
    输入:
        feats_CT: torch.float32 [128, T]
        T: int
    返回:
        y_ref:           torch.float32 [N_chunks, D]
        feature_lens:    torch.int64   [1]
        aftercnn_lens:   torch.int64   [N_chunks]
    """
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, config=cfg, trust_remote_code=True, dtype=dtype, device_map=None, revision=revision
    ).eval()

    # 兼容不同权重结构：优先 thinker.audio_tower，否则 model.audio_tower
    enc = getattr(getattr(omni, "thinker", getattr(omni, "model", None)), "audio_tower").eval()

    feature_lens = torch.tensor([T], dtype=torch.long)
    aftercnn_lens = compute_aftercnn_lens_from_T(T, enc.n_window)

    with torch.no_grad():
        y_ref = enc(
            input_features=feats_CT,  # [128, T]
            feature_lens=feature_lens,
            aftercnn_lens=aftercnn_lens
        ).last_hidden_state

    y_ref = y_ref.to(torch.float32).contiguous()
    return y_ref, feature_lens, aftercnn_lens

# -----------------------------
# 音频准备
# -----------------------------
def make_wave(args) -> torch.Tensor:
    """
    生成/读取 16kHz 单声道 30s 波形，float32 [SR*DURATION_S]
    """
    need = SR * DURATION_S
    if args.wav is not None:
        try:
            import soundfile as sf
        except Exception as e:
            raise RuntimeError("需要 soundfile 读取 wav：pip install soundfile") from e
        wave_np, sr = sf.read(args.wav, dtype="float32", always_2d=False)
        assert sr == SR, f"期望 SR={SR}, 实际 {sr}"
        if wave_np.ndim == 2:
            # 简单下混
            wave_np = wave_np.mean(axis=1).astype(np.float32)
        if len(wave_np) >= need:
            wave_np = wave_np[:need]
        else:
            pad = np.zeros(need - len(wave_np), dtype=np.float32)
            wave_np = np.concatenate([wave_np, pad], 0)
        return torch.from_numpy(wave_np)
    elif args.silence:
        return torch.zeros(need, dtype=torch.float32)
    else:
        torch.manual_seed(0)
        return torch.randn(need, dtype=torch.float32)

# -----------------------------
# 误差评估
# -----------------------------
def compare_and_print(y_exec: torch.Tensor, y_ref: torch.Tensor):
    assert y_exec.shape == y_ref.shape, \
        f"形状不一致：ExecuTorch {tuple(y_exec.shape)} vs HF {tuple(y_ref.shape)}"

    diff = (y_exec - y_ref)
    mae = diff.abs().mean().item()
    mse = (diff * diff).mean().item()
    max_abs = diff.abs().max().item()
    rel_err = (diff.norm() / (y_ref.norm() + 1e-12)).item()

    cos = F.cosine_similarity(y_exec, y_ref, dim=-1)
    cos_mean = cos.mean().item()
    cos_min  = cos.min().item()
    cos_p01  = cos.kthvalue(max(1, int(0.01 * cos.numel()))).values.item() if cos.numel() > 10 else cos_min

    print("\n=== ExecuTorch vs HF : 数值对比 ===")
    print(f"MAE               : {mae:.6e}")
    print(f"MSE               : {mse:.6e}")
    print(f"Max |diff|        : {max_abs:.6e}")
    print(f"Relative ||diff|| : {rel_err:.6e}")
    print(f"Cosine mean       : {cos_mean:.6f}")
    print(f"Cosine min        : {cos_min:.6f}")
    print(f"Cosine p01        : {cos_p01:.6f}")

    return {
        "mae": mae, "mse": mse, "max_abs": max_abs, "rel_err": rel_err,
        "cos_mean": cos_mean, "cos_min": cos_min, "cos_p01": cos_p01
    }

# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default=None, help="可选：真实 wav（16kHz 单声道/双声道均可）")
    parser.add_argument("--silence", action="store_true", help="使用静音样本")
    parser.add_argument("--outdir", type=str, default="artifacts/golden_30s")
    parser.add_argument("--revision", type=str, default="main", help="HF 模型 revision，建议固定具体 commit")
    parser.add_argument("--expect_T", type=int, default=3000, help="静态 audio_tower 期望 T（默认 3000）")
    parser.add_argument("--proc_pte", type=str, default=PROC_PTE)
    parser.add_argument("--enc_pte",  type=str, default=ENC_PTE)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) 准备波形
    wave = make_wave(args)

    # 2) ExecuTorch 前向：mel 与 audio_tower
    mel, y_exec = run_executorch(args.proc_pte, args.enc_pte, wave)
    print("[ExecuTorch] mel.shape:", tuple(mel.shape))
    assert mel.dim() == 3 and mel.shape[1] == 128, "梅尔通道维应为 128"
    T = int(mel.shape[2])
    if args.expect_T is not None:
        assert T == args.expect_T, f"T={T} 与静态 audio_tower 的 {args.expect_T} 不一致"

    print("[ExecuTorch] audio_tower out:", tuple(y_exec.shape))

    # 3) HF 参考前向（音频塔）
    y_ref, feature_lens, aftercnn_lens = run_hf_audio_tower(
        MODEL_ID, args.revision, DTYPE, mel.squeeze(0), T
    )
    print("[HF] audio_tower out:", tuple(y_ref.shape))

    # 4) 数值对齐
    metrics = compare_and_print(y_exec, y_ref)

    tolerances = {
        "mae": 5e-4,
        "max_abs": 1e-2,
        "cos_mean": 0.9995
    }
    ok = (metrics["mae"] < tolerances["mae"]) and \
         (metrics["max_abs"] < tolerances["max_abs"]) and \
         (metrics["cos_mean"] > tolerances["cos_mean"])
    print("\nResult:", "✅ PASS" if ok else "⚠️  CHECK RESULTS")

    # 5) 落盘
    np.save(os.path.join(args.outdir, "mel.npy"), mel.cpu().numpy())
    np.save(os.path.join(args.outdir, "audio_emb_exec.npy"), y_exec.cpu().numpy())
    np.save(os.path.join(args.outdir, "audio_emb_ref.npy"), y_ref.cpu().numpy())
    np.save(os.path.join(args.outdir, "feature_lens.npy"), feature_lens.cpu().numpy())
    np.save(os.path.join(args.outdir, "aftercnn_lens.npy"), aftercnn_lens.cpu().numpy())

    meta = {
        "model_id": MODEL_ID,
        "revision": args.revision,
        "dtype": str(DTYPE),
        "sr": SR,
        "duration_s": DURATION_S,
        "mel_shape": list(mel.shape),
        "emb_shape": list(y_ref.shape),
        "tolerances": tolerances,
        "metrics": metrics,
        "source": "silence" if args.silence else ("wav" if args.wav else "random"),
        "proc_pte": args.proc_pte,
        "enc_pte": args.enc_pte
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
