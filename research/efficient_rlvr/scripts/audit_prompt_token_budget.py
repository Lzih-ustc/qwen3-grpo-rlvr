#!/usr/bin/env python3
"""Audit locked prompt lengths with the pinned Qwen3 chat tokenizer offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def summarize(values: list[int], cap: int) -> dict[str, Any]:
    ordered = sorted(values)
    nearest = lambda percentile: ordered[max(0, int((percentile * len(ordered) + 0.999999)) - 1)]
    return {
        "count": len(values),
        "min": min(values),
        "mean": mean(values),
        "p50_nearest_rank": nearest(0.50),
        "p95_nearest_rank": nearest(0.95),
        "p99_nearest_rank": nearest(0.99),
        "max": max(values),
        "over_cap": sum(value > cap for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "efficient-rlvr-math-v1")
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--output-json", type=Path, default=ROOT / "reports" / "efficient_rlvr_phase3_prompt_length_audit.json")
    args = parser.parse_args()
    if args.max_prompt_length < 1:
        raise ValueError("max-prompt-length must be positive")
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    snapshot = cache_root / "models--Qwen--Qwen3-0.6B" / "snapshots" / MODEL_REVISION
    if not (snapshot / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Pinned local tokenizer snapshot is missing: {snapshot}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    results = {}
    for split in ("train", "validation", "test"):
        path = args.data_dir / manifest["splits"][split]["file"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["splits"][split]["sha256"]:
            raise AssertionError(f"Locked {split} data hash changed")
        rows = read_jsonl(path)
        lengths = [
            len(tokenizer.apply_chat_template(
                row["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            ))
            for row in rows
        ]
        results[split] = summarize(lengths, args.max_prompt_length)
    status = "passed" if all(item["over_cap"] == 0 for item in results.values()) else "failed"
    report = {
        "status": status,
        "audit_scope": "Offline tokenizer only; no model weights, GPU generation, training, or evaluation.",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "chat_template": {"enable_thinking": False, "add_generation_prompt": True},
        "max_prompt_length": args.max_prompt_length,
        "split_results": results,
        "data_sha256": {name: manifest["splits"][name]["sha256"] for name in results},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if status != "passed":
        raise SystemExit("Prompt token budget audit failed")


if __name__ == "__main__":
    main()
