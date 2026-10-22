#!/usr/bin/env python3
"""QLoRA GRPO experiment with explicit memory telemetry and OOM fallback."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as package_metadata
import json
import os
import random
import re
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import GRPOConfig, GRPOTrainer

from math_task import MODEL_ID, MODEL_REVISION, exact_math_reward
from telemetry import GpuMonitor, query_gpu


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


def git_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def make_fallbacks(initial: dict) -> list[dict]:
    configs = [deepcopy(initial)]
    current = deepcopy(initial)
    # Change one memory dimension at a time and preserve every attempted setting.
    for length in [224, 192, 160, 128, 96, 64, 48, 32, 24]:
        if current["max_completion_length"] > length:
            current["max_completion_length"] = length
            configs.append(deepcopy(current))
    for generations in [4, 2]:
        if current["num_generations"] > generations:
            current["num_generations"] = generations
            current["gradient_accumulation_steps"] = generations
            configs.append(deepcopy(current))
    for prompt_length in [160, 128, 96, 64, 48]:
        if current["prompt_token_limit"] > prompt_length:
            current["prompt_token_limit"] = prompt_length
            configs.append(deepcopy(current))
    # Remove duplicate configs if the selected profile starts below a fallback tier.
    unique: list[dict] = []
    for config in configs:
        if config not in unique:
            unique.append(config)
    return unique


def apply_prompt_token_limit(rows: list[dict], tokenizer, limit: int) -> list[dict]:
    """Trim unusually long prompt bodies while preserving both task and format tail."""
    prepared = deepcopy(rows)
    for row in prepared:
        messages = row["prompt"]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(rendered) <= limit or not messages:
            continue
        message = messages[-1]
        body_ids = tokenizer(str(message.get("content", "")), add_special_tokens=False)["input_ids"]
        overhead = max(0, len(rendered) - len(body_ids))
        budget = max(1, limit - overhead)
        if len(body_ids) > budget:
            prefix = max(1, budget // 2)
            suffix = max(0, budget - prefix)
            kept_ids = body_ids[:prefix] + (body_ids[-suffix:] if suffix else [])
            message["content"] = tokenizer.decode(kept_ids, skip_special_tokens=True)
    return prepared


def make_model(tokenizer, dtype: torch.dtype):
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
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    return model


def run_attempt(
    run_dir: Path,
    attempt_index: int,
    config: dict,
    train_rows: list[dict],
    data_path: Path,
    args: argparse.Namespace,
) -> dict:
    attempt_dir = run_dir / f"attempt-{attempt_index:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    (attempt_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    gpu_monitor = GpuMonitor()
    gpu_start = query_gpu()
    gpu_monitor.start()
    t0 = time.perf_counter()
    generation_stats = {"completion_count": 0, "completion_tokens": 0}
    model = trainer = tokenizer = None
    try:
        seed_everything(args.seed)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = make_model(tokenizer, dtype)
        trainable_before = {
            name: param.detach().float().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
        dataset = Dataset.from_list(apply_prompt_token_limit(train_rows, tokenizer, config["prompt_token_limit"]))
        training_args = GRPOConfig(
            output_dir=str(attempt_dir / "trainer"),
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_generations=config["num_generations"],
            max_completion_length=config["max_completion_length"],
            generation_batch_size=config["num_generations"],
            temperature=0.9,
            top_p=1.0,
            top_k=0,
            chat_template_kwargs={"enable_thinking": False},
            learning_rate=args.learning_rate,
            lr_scheduler_type="constant",
            warmup_steps=0,
            weight_decay=0.0,
            optim="adamw_torch",
            max_grad_norm=1.0,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            beta=0.0,
            loss_type="grpo",
            scale_rewards="group",
            use_vllm=False,
            report_to="none",
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=None if args.profile == "formal" else 2,
            seed=args.seed,
            data_seed=args.seed,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            disable_tqdm=True,
            log_completions=True,
            num_completions_to_print=2,
        )

        def tracked_math_reward(completions, answer, completion_ids=None, **kwargs):
            if completion_ids is not None:
                generation_stats["completion_count"] += len(completion_ids)
                generation_stats["completion_tokens"] += sum(len(ids) for ids in completion_ids)
            return exact_math_reward(completions, answer, **kwargs)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=tracked_math_reward,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        train_started = time.perf_counter()
        result = trainer.train()
        train_seconds = time.perf_counter() - train_started
        trainer.save_model(str(attempt_dir / "final_adapter"))
        trainer.state.save_to_json(str(attempt_dir / "trainer_state.json"))
        history = trainer.state.log_history
        (attempt_dir / "trainer_log_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

        changed_count = 0
        l1_update = 0.0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in trainable_before:
                    delta = (param.detach().float().cpu() - trainable_before[name]).abs()
                    l1_update += float(delta.sum().item())
                    changed_count += int(torch.count_nonzero(delta).item())
        if trainer.state.global_step < 1:
            raise RuntimeError("Trainer returned without an optimizer update")
        if changed_count == 0:
            raise RuntimeError("GRPO completed but no LoRA parameter changed; inspect group reward variance")

        train_metrics = dict(result.metrics)
        process_peak_allocated = torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else 0
        process_peak_reserved = torch.cuda.max_memory_reserved(0) if torch.cuda.is_available() else 0
        del trainable_before
        gpu_monitor.stop()
        summary = {
            "status": "success",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "attempt": attempt_index,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "software_versions": {name: package_version(name) for name in ["torch", "transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes"]},
            "base_precision": "4bit-nf4-double-quant",
            "compute_dtype": str(dtype),
            "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
            "use_vllm": False,
            "gradient_checkpointing": True,
            "seed": args.seed,
            "git": git_provenance(),
            "dataset": {"path": str(data_path), "count": len(train_rows), "sha256": sha256(data_path)},
            "code_sha256": {
                path.name: sha256(path)
                for path in [
                    Path(__file__),
                    Path(__file__).with_name("math_task.py"),
                    Path(__file__).with_name("telemetry.py"),
                    Path(__file__).with_name("eval_math.py"),
                    Path(__file__).with_name("prepare_data.py"),
                    Path(__file__).with_name("validate_reward.py"),
                    Path(__file__).with_name("audit_env.py"),
                    Path(__file__).with_name("check_runtime.py"),
                ]
            },
            "dependency_lock_sha256": sha256(Path("requirements-wsl2.lock.txt")) if Path("requirements-wsl2.lock.txt").exists() else None,
            "attempt_config": config,
            "optimizer_updates": trainer.state.global_step,
            "trainable_parameters": trainable_count,
            "changed_lora_parameter_count": changed_count,
            "lora_parameter_l1_update": l1_update,
            "train_seconds": train_seconds,
            "steps_per_second": trainer.state.global_step / train_seconds if train_seconds else None,
            "completions_per_second": trainer.state.global_step * config["num_generations"] / train_seconds if train_seconds else None,
            "observed_generation_completion_count": generation_stats["completion_count"],
            "observed_completion_tokens": generation_stats["completion_tokens"],
            "completion_tokens_per_second": generation_stats["completion_tokens"] / train_seconds if train_seconds else None,
            "trainer_metrics": train_metrics,
            "reward_log_entries": [{k: v for k, v in log.items() if "reward" in k.lower() or "complet" in k.lower()} for log in history if any("reward" in k.lower() or "complet" in k.lower() for k in log)],
            "torch_peak_allocated_mib": process_peak_allocated / (1024**2),
            "torch_peak_reserved_mib": process_peak_reserved / (1024**2),
            "gpu_start": gpu_start,
            "gpu_monitor": gpu_monitor.summary(),
            "checkpoints": sorted(str(path) for path in (attempt_dir / "trainer").glob("checkpoint-*")),
            "final_adapter": str(attempt_dir / "final_adapter"),
        }
        (attempt_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        return summary
    except BaseException as exc:
        # Preserve telemetry for failed/OOM attempts before releasing the model.
        gpu_monitor.stop()
        failure = {
            "status": "oom" if is_oom(exc) else "failed",
            "exception": repr(exc),
            "attempt_config": config,
            "elapsed_seconds": time.perf_counter() - t0,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else None,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(0) / (1024**2) if torch.cuda.is_available() else None,
            "gpu_start": gpu_start,
            "gpu_monitor": gpu_monitor.summary(),
        }
        (attempt_dir / "failure.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    finally:
        gpu_monitor.stop()
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "small", "formal"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--max-completion-length", type=int)
    parser.add_argument("--prompt-token-limit", type=int)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this WSL environment; refusing a CPU GRPO run.")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise SystemExit(f"Run directory is not empty: {args.run_dir}; choose a new run-dir to preserve prior artifacts.")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if not args.train_data.is_file():
        raise SystemExit(f"Training data missing: {args.train_data}; run scripts/prepare_data.py first.")
    rows = read_jsonl(args.train_data)
    if args.profile == "smoke":
        max_steps, generations, completion, prompt = args.max_steps or 1, 2, 32, 96
    elif args.profile == "small":
        max_steps, generations, completion, prompt = args.max_steps or 12, 4, 64, 128
    else:
        max_steps, generations, completion, prompt = args.max_steps or 500, 8, 128, 128
    args.max_steps = max_steps
    args.save_steps = args.save_steps or (50 if args.profile == "formal" else max(1, min(4, max_steps // 2)))
    initial = {
        "prompt_token_limit": prompt,
        "max_completion_length": completion,
        "num_generations": generations,
        "gradient_accumulation_steps": generations,
    }
    if args.num_generations is not None:
        if args.num_generations < 2:
            raise SystemExit("--num-generations must be at least 2 for GRPO group-relative advantages.")
        initial["num_generations"] = args.num_generations
        initial["gradient_accumulation_steps"] = args.num_generations
    if args.max_completion_length is not None:
        if args.max_completion_length < 1:
            raise SystemExit("--max-completion-length must be positive.")
        initial["max_completion_length"] = args.max_completion_length
    if args.prompt_token_limit is not None:
        if args.prompt_token_limit < 1:
            raise SystemExit("--prompt-token-limit must be positive.")
        initial["prompt_token_limit"] = args.prompt_token_limit
    attempts = []
    successful = None
    for index, config in enumerate(make_fallbacks(initial), start=1):
        try:
            successful = run_attempt(args.run_dir, index, config, rows, args.train_data, args)
            break
        except BaseException as exc:
            failed_dir = args.run_dir / f"attempt-{index:02d}"
            failure = {
                "attempt": index,
                "config": config,
                "status": "oom" if is_oom(exc) else "failed",
                "exception": repr(exc),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            attempts.append(failure)
            failed_dir.mkdir(parents=True, exist_ok=True)
            telemetry_path = failed_dir / "failure.json"
            if telemetry_path.exists():
                failure.update(json.loads(telemetry_path.read_text(encoding="utf-8")))
            (telemetry_path).write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if is_oom(exc):
                gc.collect()
                torch.cuda.empty_cache()
                if index < len(make_fallbacks(initial)):
                    continue
            raise
    (args.run_dir / "fallback_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n", encoding="utf-8")
    if successful is None:
        raise SystemExit("All memory fallback attempts failed.")
    root_summary = dict(successful)
    root_summary["profile"] = args.profile
    root_summary["oom_fallbacks"] = attempts
    (args.run_dir / "run_summary.json").write_text(json.dumps(root_summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(root_summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
