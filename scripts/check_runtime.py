#!/usr/bin/env python3
"""Fast compatibility preflight before downloading model weights or training."""
import json
from pathlib import Path

import torch
import transformers
import trl
import peft
import bitsandbytes
from trl import GRPOConfig
from train_grpo import make_fallbacks


config = GRPOConfig(
    output_dir="/tmp/qwen-grpo-api-preflight",
    max_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=2,
    max_completion_length=32,
    generation_batch_size=2,
    use_vllm=False,
    loss_type="grpo",
)
formal_fallbacks = make_fallbacks({
    "prompt_token_limit": 256,
    "max_completion_length": 256,
    "num_generations": 8,
    "gradient_accumulation_steps": 8,
})
assert formal_fallbacks[0]["max_completion_length"] == 256
assert all(item["num_generations"] >= 2 for item in formal_fallbacks)
assert all(item["gradient_accumulation_steps"] == item["num_generations"] for item in formal_fallbacks)
completion_lengths = [item["max_completion_length"] for item in formal_fallbacks]
assert all(left >= right for left, right in zip(completion_lengths, completion_lengths[1:]))
assert {item["num_generations"] for item in formal_fallbacks} == {2, 4, 8}
result = {
    "status": "passed",
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "trl": trl.__version__,
    "peft": peft.__version__,
    "bitsandbytes": bitsandbytes.__version__,
    "grpo_use_vllm": config.use_vllm,
    "grpo_loss_type": config.loss_type,
    "grpo_completion_limit": config.max_completion_length,
    "oom_fallback_ladder": {
        "completion_lengths": sorted({item["max_completion_length"] for item in formal_fallbacks}, reverse=True),
        "generation_groups": sorted({item["num_generations"] for item in formal_fallbacks}, reverse=True),
        "gradient_accumulation_tracks_group": True,
    },
}
print(json.dumps(result, indent=2))
Path("reports").mkdir(exist_ok=True)
Path("reports/runtime_preflight.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
