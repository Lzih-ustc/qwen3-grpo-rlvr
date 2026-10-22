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
- A separate formal-stage dataset is prepared at `data/gsm8k-formal/` (1,500 train, 500 validation, 1,063 test). The test excludes all 256 official-test rows seen in the two pilot slices; `reports/gsm8k_formal_data_validation.json` passes the source-index exclusion check. The full test was evaluated once after validation-only selection and is now consumed for this experiment.
- Formal-length single-step profiles at group 4 and 8, with prompt/completion limits 256, completed without OOM or clipping; sampled whole-card peaks were 3,774 and 3,769 MiB. Each profile generated fewer than 100 tokens, so neither estimates worst-case throughput.
- A 100-update group-8 extension completed on the formal train split with 8,603 generated tokens, 31.72 token/s, 4,136 MiB whole-card peak, and no OOM/clipping. Only 4/100 updates had zero reward variance/gradient. Validation selected step 100 (24/500) over step 50 (9/500), but base scored 46/500; formatting improved while accuracy fell. Treat this as a negative result and do not scale the same recipe unchanged.
- The step-100 checkpoint was selected by validation before the formal test was opened. On the 1,063-row test, base scored 95/1,063 (8.94%) and step100 scored 77/1,063 (7.24%); the adapter parsed more outputs and generated fewer tokens, but exact-match fell by 1.69 points. The test is consumed and must not be reused to tune future candidates.
- A fresh next-round dataset now lives in `data/gsm8k-next-fresh/`: 1,500 train, 500 new validation, 1,000 new test from unused official-train source rows. It excludes the previous formal train/validation's 2,000 source indices and passes `reports/gsm8k_next_fresh_data_validation.json`; the new test has not been model-evaluated.
- A deeper audit of the 100-step rollouts found only 18/800 exact answers and only 11/100 groups with any correct completion. 89 all-wrong groups still had nonzero reward variance in 85 cases because the inverse-distance shaping differentiates wrong integers; after `scale_rewards="group"`, this weak signal can still drive relative updates. Mean within-group reward std was 0.00764 in all-wrong groups versus 0.416 in groups containing a correct completion.
- Added a verifier-only `exact_format` reward mode (exact answer + optional 0.05 XML-format bonus, no distance shaping) and a generated-token-budget stop callback for fair compute comparisons. Reward unit cases and the fresh data/hash/source-index checks pass; training with the new mode has not run yet.
- The generated-token callback passed a minimum GPU update: 8 rollouts, a nonzero gradient, 2,748,622 changed LoRA parameters, and a step-1 checkpoint; target 1 token stopped at 274 tokens after the full group. Base evaluation on the untouched fresh validation is 54/500 (10.8%), parse rate 76%, 21,635 tokens at 54.57 token/s; its test remains sealed.
- A first 16,000-token `distance_shaped` pilot ended after 12 updates/96 completions (17,502 tokens; 9.4% group-boundary overshoot), with only 12 unique prompts. Rollouts had 16/96 exact answers, 47.9% parse rate, 31.25% average completion clipping, 3 zero-variance groups, 112.4 token/s, and a 4,430 MiB whole-card peak, no OOM. This is a train-sample diagnostic, not a quality estimate; no periodic full checkpoint was written because the run ended before step 25. A post-run rollout auditor records these metrics. The new terminal-checkpoint path passed a one-update GPU smoke: it saved optimizer/scheduler/RNG/Trainer state at checkpoint-1 after a 1-token target, and that adapter reloaded for independent inference.

## Remaining phases

- **Phase B (mainline):** run the reward ablation on the fresh split with equal generated-token budgets, select only on the fresh validation, then evaluate the selected candidate and base once on the still-sealed test. Repeat a seed if local compute permits.
- **Phase C:** after diagnosing reward signal, run a controlled group-size/efficiency comparison under matched generated-token or GPU-time budgets; the prior completion-count-matched pilot does not meet this gate.
- **Phase D:** not started; tool-use Agent RL is explicitly deferred until Phases A–C have reliable results and local compute budget is assessed.

## Next action

Finish a paired `distance_shaped` versus `exact_format` reward pilot against the fresh training split under the same generated-token target; compare clipping and zero-gradient incidence before deciding whether to raise the completion cap. Evaluate candidates only on the fresh validation and keep the new test sealed until checkpoint selection. Keep both pilot test slices, the consumed formal test, and the arithmetic artifacts immutable. Continue on the local RTX 3060; any paid AutoDL run still requires explicit user approval.
