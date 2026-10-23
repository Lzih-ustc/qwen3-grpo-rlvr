#!/usr/bin/env python3
"""Local RTX smoke for a durable-token-budgeted correctness-gated GRPO update."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

import train_grpo as archived
from scripts.math_task import completion_text
from research.efficient_rlvr.rewards import correctness_gated_length_reward
from research.efficient_rlvr.group_filter import DynamicMixedGroupRollout
from research.efficient_rlvr.ledger import AttemptedTokenLedger, TokenBudgetExhausted


def cached_snapshot_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
    else:
        hf_cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
        path = hf_cache / "models--Qwen--Qwen3-0.6B" / "snapshots" / archived.MODEL_REVISION
    if not (path / "config.json").is_file() or not (path / "tokenizer.json").is_file():
        raise SystemExit(f"Pinned local model snapshot is incomplete or missing: {path}")
    return path


def load_base_for_adapter(dtype: torch.dtype, snapshot_path: Path):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(snapshot_path),
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    base.config.use_cache = False
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/efficient-rlvr-phase4b-gpu-smoke"))
    parser.add_argument("--report", type=Path, default=Path("reports/efficient_rlvr_phase4b_gpu_smoke.json"))
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--length-weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-candidate-groups", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--snapshot-path", type=Path, help="Optional path to the pinned local Qwen3 snapshot")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this script refuses to substitute CPU training for the requested GPU smoke")
    if args.generations < 2 or args.max_completion_length < 1 or args.token_budget < 1 or args.max_steps < 1:
        raise SystemExit("generations, max-completion-length, token-budget, and max-steps must be positive")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise SystemExit(f"Run directory is nonempty; choose a new path to preserve all smoke attempts: {args.run_dir}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = args.run_dir / "attempt-01"
    attempt_dir.mkdir()
    snapshot_path = cached_snapshot_path(args.snapshot_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    gpu_monitor = archived.GpuMonitor()
    gpu_start = archived.query_gpu()
    gpu_monitor.start()

    seed = args.seed
    archived.seed_everything(seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path), padding_side="left", local_files_only=True, trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot_path), quantization_config=quantization, torch_dtype=dtype,
        device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation="sdpa", local_files_only=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    before = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    train_data_path = ROOT / "data/train.jsonl"
    train_rows = archived.read_jsonl(train_data_path)
    if not train_rows:
        raise SystemExit("Synthetic smoke train split is empty")
    smoke_row = train_rows[0]
    dataset = Dataset.from_list([smoke_row])
    ledger = AttemptedTokenLedger(
        attempt_dir / "attempted_tokens.sqlite3",
        run_id=f"local-rtx-smoke-seed-{args.seed}",
        target_tokens=args.token_budget,
        group_size=args.generations,
        max_completion_tokens=args.max_completion_length,
        max_candidates_per_group=args.max_candidate_groups,
    )
    rollout = DynamicMixedGroupRollout(
        train_rows,
        group_size=args.generations,
        max_candidates_per_group=args.max_candidate_groups,
        seed=args.seed,
        ledger=ledger,
    )
    generation_stats: dict[str, int] = {"retained_completion_count": 0, "retained_completion_tokens": 0}
    reward_rows: list[dict[str, object]] = []

    def reward_func(completions, answer, completion_ids=None, effective_group=None, **kwargs):
        if completion_ids is not None:
            generation_stats["retained_completion_count"] += len(completion_ids)
            generation_stats["retained_completion_tokens"] += sum(len(ids) for ids in completion_ids)
        rewards = correctness_gated_length_reward(
            completions,
            answer,
            completion_ids,
            max_completion_tokens=args.max_completion_length,
            length_weight=args.length_weight,
            effective_group=effective_group,
        )
        reward_rows.extend({
            "completion": completion_text(completion),
            "target": str(target),
            "completion_tokens": len(ids),
            "reward": reward,
        } for completion, target, ids, reward in zip(completions, answer, completion_ids, rewards, strict=True))
        return rewards

    config = GRPOConfig(
        output_dir=str(attempt_dir / "trainer"),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.generations,
        num_generations=args.generations,
        generation_batch_size=args.generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=1.0,
        top_k=0,
        chat_template_kwargs={"enable_thinking": False},
        learning_rate=5e-5,
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
        save_steps=1,
        save_total_limit=None,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        disable_tqdm=True,
        log_completions=True,
        num_completions_to_print=args.generations,
    )
    (attempt_dir / "config.json").write_text(json.dumps({
        "model_id": archived.MODEL_ID,
        "model_revision": archived.MODEL_REVISION,
        "seed": seed,
        "group_size": args.generations,
        "max_completion_tokens": args.max_completion_length,
        "global_attempted_token_budget": args.token_budget,
        "length_weight": args.length_weight,
        "temperature": args.temperature,
        "max_candidate_groups_per_retained_group": args.max_candidate_groups,
        "reward_mode": "correctness_gated_length_reward",
        "loss_type": "grpo",
        "scale_rewards": "group",
        "use_vllm": False,
        "gradient_checkpointing": True,
        "smoke_prompt": smoke_row,
        "smoke_train_data_sha256": hashlib.sha256(train_data_path.read_bytes()).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_func=rollout,
    )
    start = time.perf_counter()
    token_budget_stopped = False
    train_error = None
    train_result = None
    try:
        train_result = trainer.train()
        ledger.mark_complete()
    except TokenBudgetExhausted as exc:
        token_budget_stopped = True
        train_error = str(exc)
    except Exception as exc:
        train_error = f"{type(exc).__name__}: {exc}"
        ledger.record_run_exception(stage="trainer.train", error=exc)
    train_seconds = time.perf_counter() - start
    global_step = trainer.state.global_step
    if global_step > 0 and not (attempt_dir / "trainer" / f"checkpoint-{global_step}").is_dir():
        trainer._save_checkpoint(trainer.model, trial=None)
    adapter_dir = attempt_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    trainer.state.save_to_json(str(attempt_dir / "trainer_state.json"))

    changed_parameters = 0
    changed_values = 0
    for name, parameter in trainer.model.named_parameters():
        if name in before:
            delta = (parameter.detach().float().cpu() - before[name]).abs()
            changed_parameters += int(torch.count_nonzero(delta).item() > 0)
            changed_values += int(torch.count_nonzero(delta).item())

    saved_adapter_state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(trainer.model).items()}
    peak_allocated = torch.cuda.max_memory_allocated(0) / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved(0) / (1024**2)
    history = list(trainer.state.log_history)
    checkpoint_dir = attempt_dir / "trainer" / f"checkpoint-{global_step}"
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    reload_base = load_base_for_adapter(dtype, snapshot_path)
    reloaded_adapter = PeftModel.from_pretrained(reload_base, str(adapter_dir), is_trainable=False)
    reloaded_state = get_peft_model_state_dict(reloaded_adapter)
    checkpoint_reload_match = (
        set(saved_adapter_state) == set(reloaded_state)
        and all(torch.equal(saved_adapter_state[key], reloaded_state[key].detach().cpu()) for key in saved_adapter_state)
    )
    del reloaded_adapter, reload_base
    gc.collect()
    torch.cuda.empty_cache()
    gpu_monitor.stop()

    grad_norms = [
        entry["grad_norm"] for entry in history
        if isinstance(entry.get("grad_norm"), (int, float))
    ]
    positive_rewards = [float(item["reward"]) for item in reward_rows if isinstance(item["reward"], (int, float))]
    correct_reward_count = sum(value > 0 for value in positive_rewards)
    ledger_summary = ledger.summary()
    summary = {
        "status": "passed" if (
            global_step >= 1
            and changed_values > 0
            and checkpoint_reload_match
            and not ledger_summary["actual_count_unknown"]
            and ledger_summary["budget_overrun_tokens"] == 0
            and ledger_summary["attempted_completion_tokens"] == rollout.stats.attempted_completion_tokens
        ) else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "local RTX 3060 one-update smoke; no AutoDL/cloud/paid service",
        "model_id": archived.MODEL_ID,
        "model_revision": archived.MODEL_REVISION,
        "local_snapshot_path_supplied": args.snapshot_path is not None,
        "seed": seed,
        "device": torch.cuda.get_device_name(0),
        "software_versions": {name: archived.package_version(name) for name in ["torch", "transformers", "trl", "peft", "accelerate", "bitsandbytes"]},
        "config": {"group_size": args.generations, "max_completion_tokens": args.max_completion_length, "length_weight": args.length_weight, "temperature": args.temperature, "dtype": str(dtype)},
        "optimizer_updates": global_step,
        "trainer_metrics": train_result.metrics if train_result is not None else None,
        "token_budget_stopped": token_budget_stopped,
        "train_error": train_error,
        "gradient_norms": grad_norms,
        "changed_lora_parameter_tensors": changed_parameters,
        "changed_lora_parameter_values": changed_values,
        "completion_count": rollout.stats.attempted_completions,
        "generated_tokens": rollout.stats.attempted_completion_tokens,
        "attempted_token_ledger": ledger_summary,
        "retained_completion_count": generation_stats["retained_completion_count"],
        "retained_completion_tokens": generation_stats["retained_completion_tokens"],
        "group_filter_stats": rollout.stats.as_dict(),
        "train_seconds": train_seconds,
        "generation_tokens_per_train_second": rollout.stats.attempted_completion_tokens / train_seconds if train_seconds else None,
        "reward_behavior": {
            "completion_rewards": reward_rows,
            "correct_nonzero_reward_count": correct_reward_count,
            "reward_min": min(positive_rewards) if positive_rewards else None,
            "reward_max": max(positive_rewards) if positive_rewards else None,
            "positive_reward_variance": len(set(positive_rewards)) > 1,
        },
        "checkpoint": {
            "path": str(checkpoint_dir),
            "exists": checkpoint_dir.is_dir(),
            "adapter_path": str(adapter_dir),
            "adapter_config_exists": (adapter_dir / "adapter_config.json").is_file(),
            "adapter_weights_exists": (adapter_dir / "adapter_model.safetensors").is_file(),
            "reload_matches_saved_adapter_tensors": checkpoint_reload_match,
        },
        "torch_peak_allocated_mib": peak_allocated,
        "torch_peak_reserved_mib": peak_reserved,
        "gpu_start": gpu_start,
        "gpu_monitor": gpu_monitor.summary(),
        "git_commit": archived.git_provenance().get("commit"),
        "run_summary_is_not_a_model_quality_result": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if summary["status"] != "passed":
        raise SystemExit("Smoke failed: inspect report without interpreting it as a quality evaluation")


if __name__ == "__main__":
    main()
