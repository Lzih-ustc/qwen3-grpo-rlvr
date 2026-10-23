# Phase 4 — proposed AutoDL RTX 4090 experiment plan

Status: **plan only; no AutoDL instance has been created or started.** Paid execution requires the user's explicit approval after reviewing this plan.

## Questions and arms

Measure whether correctness-gated length shaping and correctness-mixed group filtering can reduce generated tokens and serving latency while retaining verified accuracy, including the extra rollout cost of filtering.

| Arm | Group size | Reward | Group filter |
|---|---:|---|---|
| A — standard GRPO | 4 | Exact integer correctness, 0/1 | No |
| B — CGLR | 4 | Correctness-gated length reward, `lambda=0.2` | No |
| C — CGLR + GFRL | 4 | Same CGLR | Retain mixed-correctness groups; at most 12 candidates/group |

The reward weight and candidate cap are fixed before looking at the independent test. The three arms use the same base revision, prompt template, chat-template settings, batch/group size, completion cap, LoRA/NF4 setup, optimizer, verifier, and decoding settings. The new data source and split hashes are in `data/efficient-rlvr-math-v1/manifest.json` after the reproducibility check closes.

## Seeds and budgets

Use three paired training seeds (`260201`, `260202`, `260203`) for each arm.

1. **Primary, fixed generated-token budget:** 16,000 attempted completion tokens per run, counting every GFRL candidate and every discarded completion. Stop only at a full completion/group boundary; report the overshoot, retained tokens, unique prompts, effective groups, updates, and elapsed time.
2. **Secondary, fixed GPU-time budget:** 25 minutes of active training per run after model load and warm-up, plus a 5-minute save/teardown allowance. Report each run's actual attempted/retained tokens and updates at the cutoff.

The matrix is 3 arms × 3 seeds × 2 budgets = 18 runs. A 30-minute billed ceiling per run reserves up to 9 GPU-hours for training. If the local 3060 smoke's group filter has a very low retained-group yield, keep that as a measured cost rather than silently increasing its token budget.

## Data and evaluation protocol

- Train only on the locked 1,500-row `train` split. Use the 300-row `validation` split to choose checkpoint/stopping rules and report selection uncertainty.
- Never use consumed GSM8K/SVAMP tests for selection. The 500-row `test` split is reserved for one final evaluation after all checkpoints, reward coefficients, group-filter settings, and report tables are frozen.
- The new test is generated from the upstream extrapolation regime, not from historical GSM8K/SVAMP items. A text-level duplicate audit and data hashes are stored alongside the split. This does not prove absence from Qwen pretraining.
- For each selected model, Base, and each fixed-time/token-budget arm, generate once per prompt up to 256 completion tokens with the same Qwen3 chat template and `enable_thinking=False`. Save original token IDs, full text, exact-verifier output, prompt ID, seed/arm/checkpoint, and per-question elapsed time.
- Score prefixes of the same saved completion at 32/64/96/128/192/256 tokens. This is the LCE curve; it avoids repeated sampling differences. Also report full-cap accuracy, parse failures, truncation, actual tokens, p50/p95 latency, and throughput.
- Use fixed prompt order and batch size 1 for per-question latency; warm up on 20 non-scored prompts, synchronize CUDA around timing, and state whether model load is excluded. Report active training GPU-hours and AutoDL billed hours separately.
- Report accuracy with paired bootstrap intervals and McNemar comparisons. Show per-seed outcomes and between-seed variation separately from question-level intervals. Compute correct test solutions per active training GPU-hour and per billed GPU-hour; include the complete train/eval cost.

## Hardware and memory

One AutoDL NVIDIA RTX 4090, 24 GB, one GPU only. Use QLoRA NF4, double quantization, LoRA r=8, gradient checkpointing, `use_vllm=False`, BF16 when supported, SDPA attention, group size 4, prompt cap 256 and completion cap 256. The local smoke peak was about 4.75 GiB whole-card and 1.11 GiB PyTorch allocated, leaving headroom, but the cloud preflight must log its actual baseline and peak. No other GPU size or provider is included in this budget.

Before the matrix: run environment/model/tokenizer/data-hash checks, then a short preflight that confirms one backward pass, nonzero gradient, checkpoint save/reload, and no OOM. If it fails, stop and preserve the failure. Do not repeatedly tune on validation. Any necessary sequence-cap change must be applied uniformly to all arms and seeds before the matrix begins.

## Stop rules

- Hard stop at 30 billed minutes per run, 14 total billed GPU-hours, or the approved spend cap, whichever occurs first.
- If the first three updates of a run have non-finite loss/gradient or no adapter parameter changes, stop that run and save the logs/checkpoint state.
- For GFRL, if fewer than 10% of candidate groups are retained across two consecutive 1,000-attempted-token windows, stop the arm as low-yield; do not relax the candidate cap or reward after seeing validation/test.
- On OOM, preserve logs, lower the completion cap once only before the full matrix, then rerun the preflight. If it still OOMs, stop the paid work and redesign the whole matrix rather than changing one arm.
- Stop when results are mixed or negative; do not expand seeds/steps automatically.

## Current cost estimate

AutoDL's [public price page](https://www.autodl.com/home) listed RTX 4090 24 GB at **¥1.88/GPU-hour** when checked on 2026-10-23. The [billing documentation](https://www.autodl.com/docs/price/) says to use the live site quote and explains that an on-demand instance is billed from power-on to power-off to the second, not only for GPU kernel time; optional paid data-disk capacity is separate. Reserve an estimated 1 hour for setup/preflight and up to 4 hours for validation/test inference and result export.

| Item | Estimated maximum |
|---|---:|
| 18 × 30-minute run slots | 9 GPU-hours |
| Setup/preflight | 1 GPU-hour |
| Evaluation/export | 4 GPU-hours |
| Total planning ceiling | 14 GPU-hours |
| At ¥1.88/hour | ¥26.32 |
| With 30% contingency | ¥34.22 |
| Proposed hard spend cap | **¥40** |

The plan assumes no paid data disk beyond the free storage allowance, no premium image, no long-duration storage, no membership discount, and no availability surcharge. Recheck the exact live quote before launch; if it exceeds ¥2.20/hour or requires separately billed storage, pause and obtain a revised approval. Keep the instance shut down outside active work because billing follows instance power state.

## Approval gate

The user must explicitly approve paid AutoDL execution and the ¥40 cap before the first instance is created or started. This document is an estimate and protocol, not that approval.
