#!/usr/bin/env python3
"""Independent deterministic accuracy evaluator for the held-out integer set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from math_task import MODEL_ID, MODEL_REVISION, build_examples, extract_answer
from telemetry import GpuMonitor, query_gpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--eval-data", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing a CPU evaluation run.")
    if args.eval_data.exists():
        rows = [json.loads(line) for line in args.eval_data.read_text(encoding="utf-8").split("\n") if line.strip()]
    else:
        rows = build_examples(48, 20261022 + 1_000_003)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    gpu_start = query_gpu()
    monitor = GpuMonitor()
    monitor.start()
    predictions = []
    start = time.perf_counter()
    generated_tokens = 0
    try:
        for row in rows:
            prompt_messages = row["prompt"]
            inputs = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            generated_tokens += int(completion_ids.numel())
            predicted, source = extract_answer(completion)
            expected = int(row["answer"])
            predictions.append(
                {
                    "prompt": prompt_messages[0]["content"],
                    "expected": expected,
                    "completion": completion,
                    "predicted": predicted,
                    "parse_source": source,
                    "correct": predicted == expected,
                }
            )
        seconds = time.perf_counter() - start
        correct = sum(int(row["correct"]) for row in predictions)
        peak_alloc = torch.cuda.max_memory_allocated(0)
        peak_reserved = torch.cuda.max_memory_reserved(0)
        eval_bytes = args.eval_data.read_bytes() if args.eval_data.exists() else json.dumps(rows, sort_keys=True).encode()
        result = {
            "tag": args.tag,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adapter": str(args.adapter) if args.adapter else None,
            "seed": args.seed,
            "decode": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "thinking": False},
            "eval_count": len(rows),
            "correct_count": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
            "parse_rate": sum(row["predicted"] is not None for row in predictions) / len(rows) if rows else 0.0,
            "elapsed_seconds": seconds,
            "tokens_per_second": generated_tokens / seconds if seconds else None,
            "generated_tokens": generated_tokens,
            "eval_data_sha256": hashlib.sha256(eval_bytes).hexdigest(),
            "torch_peak_allocated_mib": peak_alloc / (1024**2),
            "torch_peak_reserved_mib": peak_reserved / (1024**2),
            "gpu_start": gpu_start,
            "gpu_monitor": monitor.summary(),
            "predictions": predictions,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2, ensure_ascii=False))
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
