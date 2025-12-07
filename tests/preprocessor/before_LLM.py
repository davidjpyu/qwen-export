#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick prompt input-id checker for ExecuTorch vs HuggingFace."""

import argparse
import json
import os
from typing import Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer


def build_prompt_tokens(
    tokenizer: AutoTokenizer,
    prompt: str,
    l_text: int,
    use_chat_template: bool,
) -> Tuple[torch.Tensor, Sequence[int]]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )[0].tolist()
    else:
        ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0].tolist()

    if len(ids) > l_text:
        used = ids[-l_text:]
    elif len(ids) < l_text:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        used = ids + [pad_id] * (l_text - len(ids))
    else:
        used = ids
    return torch.tensor([used], dtype=torch.int32), used


def compare_tokens(et_ids: torch.Tensor, hf_ids: torch.Tensor) -> dict:
    print(f"[INFO] ExecuTorch input_ids shape {tuple(et_ids.shape)}")
    print(f"[INFO] HuggingFace input_ids shape {tuple(hf_ids.shape)}")
    if et_ids.shape != hf_ids.shape:
        print("[WARN] Shapes differ; treat as mismatch.")
        return {"exact_match": False, "num_mismatch": -1, "mismatch_indices": []}

    diff = et_ids != hf_ids
    mismatch_idx = diff.nonzero(as_tuple=False)
    count = int(mismatch_idx.shape[0])
    if count == 0:
        print("[INFO] input_ids match exactly.")
        return {"exact_match": True, "num_mismatch": 0, "mismatch_indices": []}

    preview = [(int(i), int(j)) for i, j in mismatch_idx.tolist()[:10]]
    print(f"[WARN] Found {count} mismatched positions; first few: {preview}")
    return {"exact_match": False, "num_mismatch": count, "mismatch_indices": preview}


def save_array(path: str, arr: torch.Tensor):
    np.save(path, arr.cpu().numpy())
    print(f"[INFO] saved {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Compare ExecuTorch vs HF prompt input_ids")
    parser.add_argument("--prompt", type=str, required=True, help="ExecuTorch prompt text")
    parser.add_argument("--hf_prompt", type=str, default=None, help="Optional alternate prompt for HF path")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--l_text", type=int, default=64, help="Static prompt length expected by prefill")
    parser.add_argument("--no_chat_template", action="store_true", help="Disable chat template for ExecuTorch tokens")
    parser.add_argument(
        "--hf_no_chat_template",
        action="store_true",
        help="Disable chat template for HF tokens (defaults to same setting as ExecuTorch)",
    )
    parser.add_argument("--outdir", type=str, default="artifacts/prompt_check")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, revision=args.revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    et_ids, et_used = build_prompt_tokens(
        tokenizer,
        args.prompt,
        args.l_text,
        use_chat_template=(not args.no_chat_template),
    )
    hf_prompt = args.hf_prompt if args.hf_prompt is not None else args.prompt
    hf_ids, hf_used = build_prompt_tokens(
        tokenizer,
        hf_prompt,
        args.l_text,
        use_chat_template=(not args.hf_no_chat_template),
    )

    preview_et = tokenizer.decode(et_used, skip_special_tokens=False)
    preview_hf = tokenizer.decode(hf_used, skip_special_tokens=False)
    print(f"[ET prompt]: {preview_et}")
    print(f"[HF prompt]: {preview_hf}")

    metrics = compare_tokens(et_ids, hf_ids)

    et_path = save_array(os.path.join(args.outdir, "input_ids_et.npy"), et_ids)
    hf_path = save_array(os.path.join(args.outdir, "input_ids_hf.npy"), hf_ids)
    with open(os.path.join(args.outdir, "prompt_et.txt"), "w", encoding="utf-8") as f:
        f.write(preview_et)
    with open(os.path.join(args.outdir, "prompt_hf.txt"), "w", encoding="utf-8") as f:
        f.write(preview_hf)

    summary = {
        "prompt_shape_et": list(et_ids.shape),
        "prompt_shape_hf": list(hf_ids.shape),
        "metrics": metrics,
        "input_ids_et": et_path,
        "input_ids_hf": hf_path,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] summary saved to {args.outdir}/summary.json")


+if __name__ == "__main__":
+    main()
