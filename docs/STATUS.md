# Project Status

Last updated: 2026-10-23 (Asia/Shanghai)

## Current gate

### Phase A — GRPO execution correctness: PASS for the local pilot

- Environment audit and pinned dependency lock are recorded in `reports/environment.json` and `requirements-wsl2.lock.txt`.
- Reward parser tests pass on correct, incorrect, unmarked, ambiguous, and boxed-integer outputs.
- The 1-step smoke completed generation, backward, a nonzero gradient, one optimizer update, and a measurable LoRA parameter update. Its two samples had nonzero reward variance.
- The 12-step pilot completed with finite loss/reward/gradient logs; 7 steps had nonzero gradient norms and 5 had zero reward variance and zero gradient norm.
- The smoke checkpoint contains adapter, optimizer, scheduler, RNG, and Trainer state. Small-run step 8 and step 12 checkpoints were separately reloaded for validation inference.
- `reports/quality_gate.json` passes. Peak VRAM and generation throughput are in the run summaries and experiment report.
- No OOM occurred; no fallback was required. No cloud GPU was rented or used.

### Phase B — RLVR baseline: GSM8K local pilot complete; longer baseline open

- The arithmetic smoke split remains reproducible; its 256 rows contain 4 duplicate prompts. Its 48-item exact match (24/48 base, 32/48 step 12) is only a parser/reward sanity result.
- A pinned MIT GSM8K subset now has 256 train, 96 validation, and two disjoint 128-row test slices. The pinned source revision, answer parser, license notice, split hashes, and overlap checks are stored under `data/gsm8k/`.
- With group 2 and 8 updates, the GSM8K pilot scored 4/96 on validation; group 4 and 4 updates scored 29/96. Both runs generated 16 completions, but token counts (304 vs 1,225) and wall time (24.1 vs 40.6 sec) differed, so this is not a compute-equivalent efficiency result.
- Validation-only selection chose group-4 step 4. On the untouched second test slice, exact match was 33/128 (25.78%) vs 12/128 (9.38%) for base. It generated about 3× as many output tokens and took about 5× as long; the sample and single seed are too small to establish a general gain.
- The group-4 final training step hit the 160-token completion limit for all four samples. Raise the completion budget in the next local profile and measure truncation before scaling the run.
- `reports/gsm8k_quality_gate.json` verifies finite metrics, nonzero gradients, actual adapter updates, checkpoint files/reloads, split hashes, validation-only selection, and the held-out comparison.
- A separate formal-stage dataset is now prepared at `data/gsm8k-formal/` (1,500 train, 500 validation, 1,063 test). The test excludes all 256 official-test rows seen in the two pilot slices; `reports/gsm8k_formal_data_validation.json` passes the source-index exclusion check. No formal training or formal-test evaluation has run.
- Formal-length single-step profiles at group 4 and 8, with prompt/completion limits 256, completed without OOM or clipping; sampled whole-card peaks were 3,774 and 3,769 MiB. Each profile generated fewer than 100 tokens, so neither estimates worst-case throughput. A 100-update group-8 run on the formal train split is the current local extension.

## Remaining phases

- **Phase B (mainline):** repeat the GSM8K run with a longer completion limit and more updates; use the prepared formal split, whose 1,063 test examples exclude both opened pilot slices. Repeat at least one seed if the local compute budget permits.
- **Phase C:** run a single controlled group-size/efficiency comparison under matched generated-token or GPU-time budgets; the current completion-count-matched pilot does not meet this gate.
- **Phase D:** not started; tool-use Agent RL is explicitly deferred until Phases A–C have reliable results and local compute budget is assessed.

## Next action

Use `data/gsm8k-formal/` for the next longer run after profiling a larger completion budget to address the observed truncation. Preserve both pilot test slices as already-used pilot data. Keep the arithmetic and first GSM8K pilot artifacts immutable. Continue on the local RTX 3060; any paid AutoDL run still requires explicit user approval.
