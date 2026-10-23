# Phase 2 baselines and local smoke

## Baseline definitions

These are controlled engineering baselines. CGLR overlaps prior length-aware reward work, and correctness-mixed group filtering follows DAPO-style dynamic sampling; neither is presented as a new algorithm.

### Length-Controlled Evaluation (LCE)

Generate one completion up to a common maximum, retain its original token IDs, and score prefixes of that same completion at predeclared token budgets. Report exact match, parse rate, truncation, actual generated tokens, and latency. This evaluates answer availability under a token cap; it does not claim equal actual generation cost when evaluating different policies. Historical Phase C predictions lack token IDs, so the new helper is for future saved trajectories.

Implemented in [`evaluation.py`](evaluation.py). The helper records per-prefix token count, parser result, correctness, and truncation, and can aggregate paired length curves. The tokenizer/template, decode settings, prompt ordering, and cap list must be frozen before a formal run.

### Correctness-Gated Length Reward (CGLR)

```text
correct_i = exact_integer_verifier(completion_i, answer_i)
length_i = min(completion_token_ids_i / max_completion_tokens, 1)
reward_i = correct_i * (1 - lambda * length_i)
```

The implementation requires completion token IDs, `0 <= lambda < 1`, and equal input lengths. Incorrect and unparsed completions receive zero. Correct completions stay positive and shorter correct completions receive more reward. The smoke used `lambda=0.2`; it is not a selected optimum.

Implemented in [`rewards.py`](rewards.py).

### Group-Filtered RL (GFRL)

The rollout baseline samples groups of four, retains a group only when the existing exact verifier finds both correct and incorrect/unparsed completions, and retries up to a fixed candidate limit. Every attempted completion token, including discarded groups, is counted. Candidate prompts and their verifier answers move together; exhausted groups return `None` rewards so TRL marks them unscorable and forces their advantages to zero.

This is a group-level, DAPO-style dynamic-sampling baseline, distinct from GFPO's response-level ranking. Its yield and wasted rollout cost must be included in every result.

Implemented in [`group_filter.py`](group_filter.py). It uses TRL 1.16's experimental `rollout_func` and its private tokenization/generation methods. The dependency version is pinned. A TRL upgrade must rerun the adapter and rollout smoke before any experiment.

## RTX 3060 smoke

The final run was a single optimizer update on the local RTX 3060 using the archived local training file only to exercise the code path. It is not a quality estimate and does not use the new Phase 3 data.

| Measure | Result |
|---|---:|
| Configuration | Qwen3-0.6B, NF4 QLoRA, LoRA r=8, gradient checkpointing, non-vLLM, BF16 |
| Group size / completion cap / temperature | 4 / 64 tokens / 0.9 |
| Optimizer updates | 1 |
| Attempted completions / tokens | 8 / 371 |
| Retained completions / tokens | 4 / 237 |
| Candidate groups | 2: one all-correct discarded, one mixed retained |
| Correct positive rewards / incorrect zero rewards | 3 / 1 |
| Reward variation | nonzero |
| Gradient norm | 1.859375 |
| Changed LoRA tensors / scalar values | 196 / 2,750,317 |
| Checkpoint reload | checkpoint-1 and final adapter files present; reloaded adapter tensors match |
| Peak allocated / reserved | 1,108 / 1,554 MiB (PyTorch) |
| Whole-card sampled peak | 4,745 MiB |
| Attempted generated tokens / training second | 14.27 |

Source report: [`reports/efficient_rlvr_phase2_smoke_group_filter.json`](../../reports/efficient_rlvr_phase2_smoke_group_filter.json). This single update confirms gradient flow, an optimizer update, reward variation, token accounting, and checkpoint reload. It does not validate learning quality, generalization, or the stability of a full training run.

## Failed attempts retained

Earlier attempts are intentionally kept as negative engineering results:

1. Offline tokenizer loading attempted an HF metadata request before a local snapshot path was supplied. The loader now resolves the pinned cache snapshot directly.
2. A reward callback expected a `completion_text` field that TRL did not pass. The callback now decodes the completion through the pinned processing class.
3. Group4 produced unmarked answers under the historical strict parser; all rewards and gradients were zero.
4. The simple Group8 prompt produced identical correct completions and constant rewards; reward variance and gradients were zero.
5. The reasoning Group8 prompt failed the required answer marker on all samples; rewards and gradients were zero.

The three machine-readable GPU failure summaries are [`smoke.json`](../../reports/efficient_rlvr_phase2_smoke.json), [`smoke_group8.json`](../../reports/efficient_rlvr_phase2_smoke_group8.json), and [`smoke_group8_reasoning.json`](../../reports/efficient_rlvr_phase2_smoke_group8_reasoning.json). The first two software/setup failures did not produce completed run reports.

## Reproduction and tests

Run unit tests without model loading or CUDA:

```bash
HF_HUB_OFFLINE=1 python -m unittest discover -s research/efficient_rlvr/tests -v
```

The completed suite has 15 tests covering answer-boundary audit invariants, reward behavior, group selection/retries/exhaustion, prefix scoring, data-rebuild hashes, and saved-prompt extraction from Parquet-like rows. The final RTX smoke command (which performs one local update) is recorded in the run configuration and can be invoked with:

```bash
HF_HUB_OFFLINE=1 python research/efficient_rlvr/smoke.py \
  --run-dir runs/efficient-rlvr-phase2-smoke-group-filter \
  --report reports/efficient_rlvr_phase2_smoke_group_filter.json \
  --generations 4 --max-completion-length 64 --temperature 0.9 \
  --max-candidate-groups 12
```

The smoke refuses a nonempty run directory and requires the pinned model snapshot to exist locally. No AutoDL, cloud, or paid service was used.
