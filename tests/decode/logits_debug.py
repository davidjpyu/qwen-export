#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to inspect ExecuTorch vs HF prefill logits pairs.
Given two .npy files (et_logits0.npy / hf_logits0.npy by default),
it reports basic error metrics and decodes the tokens with the largest
absolute differences so we know which IDs deviate the most.
"""

import argparse
import os
from typing import Sequence

import numpy as np
import torch
from transformers import AutoTokenizer


def load_logits(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing logits file: {path}")
    arr = np.load(path)
    return torch.from_numpy(arr).float().view(-1)


def decode_top_tokens(tokenizer: AutoTokenizer, indices: Sequence[int]) -> Sequence[str]:
    return [tokenizer.decode([idx], skip_special_tokens=False) for idx in indices]


def main():
    ap = argparse.ArgumentParser(description="Compare ET/HF prefill logits (npz dumps).")
    ap.add_argument("--et_logits", default="et_logits0.npy", help="Path to ExecuTorch logits npy file.")
    ap.add_argument("--hf_logits", default="hf_logits0.npy", help="Path to HF reference logits npy file.")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--topk", type=int, default=10, help="How many largest diffs to decode.")
    args = ap.parse_args()

    et = load_logits(args.et_logits)
    hf = load_logits(args.hf_logits)
    if et.shape != hf.shape:
        raise RuntimeError(f"Logits shape mismatch: ET {tuple(et.shape)} vs HF {tuple(hf.shape)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, revision=args.revision, use_fast=True
    )

    diff = (et - hf).abs()
    mae = diff.mean().item()
    mse = (diff * diff).mean().item()
    rel = diff.norm().item() / (hf.norm().item() + 1e-12)
    max_val, max_idx = diff.max(dim=0)
    print(f"MAE={mae:.3e}  MSE={mse:.3e}  Rel={rel:.3e}  MaxAbs={max_val.item():.3e} (idx={int(max_idx)})")

    topk = min(args.topk, diff.numel())
    top_val, top_idx = torch.topk(diff, k=topk)
    decoded = decode_top_tokens(tokenizer, top_idx.tolist())
    print("\nTop diff tokens:")
    for rank, (idx, val, text) in enumerate(zip(top_idx.tolist(), top_val.tolist(), decoded), 1):
        print(f"{rank:2d}. id={idx:<8} diff={val:.6f} token={text!r}")

    et_top_val, et_top_idx = torch.topk(et, k=topk)
    hf_top_val, hf_top_idx = torch.topk(hf, k=topk)
    et_decoded = decode_top_tokens(tokenizer, et_top_idx.tolist())
    hf_decoded = decode_top_tokens(tokenizer, hf_top_idx.tolist())

    print("\nExecuTorch logits top tokens:")
    for rank, (idx, val, text) in enumerate(zip(et_top_idx.tolist(), et_top_val.tolist(), et_decoded), 1):
        print(f"{rank:2d}. id={idx:<8} logit={val:.6f} token={text!r}")

    print("\nHF logits top tokens:")
    for rank, (idx, val, text) in enumerate(zip(hf_top_idx.tolist(), hf_top_val.tolist(), hf_decoded), 1):
        print(f"{rank:2d}. id={idx:<8} logit={val:.6f} token={text!r}")


if __name__ == "__main__":
    main()
