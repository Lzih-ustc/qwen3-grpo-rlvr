# Project Status

Last updated: 2026-10-23 (Asia/Shanghai)

## New research branch — Efficient RLVR

The independent `efficient-rlvr` branch begins from the Phase C archive commit `b98a22d` and leaves Phase A–C results and selection records unchanged.

### Phase 1 — offline audit and literature review: complete

- Audited the saved Phase C GSM8K seed-1 Base/selected-Adapter test pair and SVAMP seed-2 Base/selected-Adapter pair offline. Source-file SHA-256 hashes and evaluation/decode metadata are recorded in `research/efficient_rlvr/phase1_answer_boundary_audit.json`.
- Added a separate post-hoc analysis of reconstructed output-token length, tokens before/inside/after the answer marker accepted by the historical verifier, parser failures, paired correctness, length-ranked accuracy/parse rate, exact repeated-sentence proxy, and recorded aggregate generation cost.
- Confirmed the audit does not revise original accuracy, parser status, candidate selection, splits, or test standards. Token region counts are re-tokenized from decoded text because original per-question token IDs were not saved. Tail text is not automatically treated as semantic redundancy.
- Reviewed GFPO, GRPO-LEAD, the two distinct papers titled Step-GRPO, DAPO, and Dr. GRPO in `research/efficient_rlvr/phase1_research_review.md`. The planned length reward and group filter are explicitly baselines with prior-work overlap; no novelty or gain is assumed.
- No model generation, GPU training, AutoDL, cloud GPU, or paid service was used in this phase.

### Phase 2 — controlled baseline implementation and local smoke: complete

- Implemented Length-Controlled Evaluation (LCE), Correctness-Gated Length Reward (CGLR), and a DAPO-style correctness-mixed Group-Filtered RL rollout baseline. The methods are explicitly treated as baselines with prior-work overlap, not as algorithmic novelty.
- Fifteen offline unit tests cover answer-boundary audit invariants, reward properties, group filtering/retry/exhaustion, prefix scoring, data rebuild verification, and Parquet prompt extraction.
- The final local RTX 3060 one-update smoke had nonzero gradient (1.859), nonzero reward variation, changed 2,750,317 LoRA parameter values, and reloaded checkpoint-1 with adapter tensor equality. It attempted 371 generated tokens and retained 237; sampled whole-card peak was 4,745 MiB. This verifies implementation plumbing only, not model quality.
- Five failed software/reward attempts remain documented; no result was silently overwritten. Full details are in `research/efficient_rlvr/phase2_design.md` and `phase2_smoke_history.md`.
- All runs were local; no AutoDL, cloud GPU, or paid service was used.

### Phase 3 — fresh data lock and protocol: complete

- Locked `data/efficient-rlvr-math-v1/` from the pinned DeepMind Mathematics Dataset generator commit `427f45075f84b8b9774950196ad63867ca20ffb3`: 1,500 train-regime rows, 300 interpolate-validation rows, and 500 extrapolate-test rows. The test is reserved for one final evaluation after all model and protocol choices are frozen.
- Split the pinned CPU dependencies into separate generator and Parquet-audit lock files: generator NumPy 1.26.4 preserves the recorded RNG stream, while PyArrow 26 requires NumPy 2 or newer. The separation keeps regeneration instructions installable without changing any locked rows.
- Fixed `PYTHONHASHSEED=0` and excluded upstream `_composed` modules after a two-process reproducibility probe found identity-hash-dependent prompt ordering in that code path. Two independent complete builds then matched all three locked split hashes. No upstream source was edited.
- Audited 730 archived JSON/JSONL/Parquet files under `data/`, `reports/`, and saved `runs/` traces (60,533 historical prompt-text entries): zero exact normalized or punctuation-insensitive question matches. All answers are signed integers; cross-split duplicates are zero. GSM8K/SVAMP test results were not used to construct or select these splits.
- Offline Qwen3 chat-template tokenization (`enable_thinking=False`) found all prompts below the 256-token cap: train max 130, validation max 105, test max 95. No model generation or evaluation was run on these new splits.
- The upstream repository's top-level Apache-2.0 license applies to its generator code; it does not separately state a license for generated output data. The license copy and provenance are included. JSONL splits are kept local and ignored by Git pending clearer output-data licensing; they can be deterministically regenerated and hash-checked.
- Audit artifacts: `research/efficient_rlvr/phase3_data_audit.md`, `reports/efficient_rlvr_phase3_data_audit.json`, `reports/efficient_rlvr_phase3_reproducibility.json`, and `reports/efficient_rlvr_phase3_prompt_length_audit.json`.

### Phase 4 — AutoDL plan: prepared, execution not approved

- Proposed one RTX 4090 24 GB, 3 arms × 3 paired seeds × fixed-token/fixed-time budgets, explicit stop rules, and an estimated 14-GPU-hour ceiling. Current public price estimate is ¥1.88/hour, with a proposed ¥40 hard spend cap including contingency.
- The plan is in `research/efficient_rlvr/phase4_autodl_plan.md`. No instance has been created or started. Paid execution remains gated on the user's explicit approval.

### Phase 5 — model comparison: not started

No formal training or new-data model evaluation has run. The RTX 3060 smoke used only archived local training data to test the implementation. Full comparisons must use the locked new splits, charge every discarded GFRL rollout to attempted-token cost, and report mixed/negative outcomes without consulting consumed GSM8K/SVAMP tests.

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
- A fresh next-round dataset now lives in `data/gsm8k-next-fresh/`: 1,500 train, 500 validation, 1,000 test from official-train source rows. After locking candidate selection on validation, the test was used once for a base/selected-adapter pair; it is now consumed. A separate audit found no overlap with historical training splits or 4,908 saved model predictions. 184 test rows appeared in a superseded intermediate data split that produced no model predictions and was never used for training; this is recorded in `reports/gsm8k_next_fresh_holdout_audit.json`.
- A deeper audit of the 100-step rollouts found only 18/800 exact answers and only 11/100 groups with any correct completion. 89 all-wrong groups still had nonzero reward variance in 85 cases because the inverse-distance shaping differentiates wrong integers; after `scale_rewards="group"`, this weak signal can still drive relative updates. Mean within-group reward std was 0.00764 in all-wrong groups versus 0.416 in groups containing a correct completion.
- Added a verifier-only `exact_format` reward mode (exact answer + optional 0.05 XML-format bonus, no distance shaping) and a generated-token-budget stop callback for fair compute comparisons. Reward unit cases and the fresh data/hash/source-index checks pass.
- The generated-token callback passed a minimum GPU update: 8 rollouts, a nonzero gradient, 2,748,622 changed LoRA parameters, and a step-1 checkpoint; target 1 token stopped at 274 tokens after the full group. Base evaluation on the fresh validation is 54/500 (10.8%), parse rate 76%, 21,635 tokens at 54.57 token/s.
- A first 16,000-token `distance_shaped` pilot ended after 12 updates/96 completions (17,502 tokens; 9.4% group-boundary overshoot), with only 12 unique prompts. Rollouts had 16/96 exact answers, 47.9% parse rate, 31.25% average completion clipping, 3 zero-variance groups, 112.4 token/s, and a 4,430 MiB whole-card peak, no OOM. This is a train-sample diagnostic, not a quality estimate; no periodic full checkpoint was written because the run ended before step 25. A post-run rollout auditor records these metrics. The new terminal-checkpoint path passed a one-update GPU smoke: it saved optimizer/scheduler/RNG/Trainer state at checkpoint-1 after a 1-token target, and that adapter reloaded for independent inference.
- The equal-token `exact_format` arm completed 13 updates/104 completions (17,272 tokens), with 21/104 exact answers, 58.7% parse rate, 20.19% average clipping, 3 zero-variance groups, 108.28 token/s, and a 4,384 MiB whole-card peak, no OOM. Validation scored 190/500 (38.0%) vs distance-shaping's 183/500 (36.6%); the paired difference was not significant (exact McNemar p=0.547), but the pre-registered highest-accuracy rule selected exact-format checkpoint-13. On the one-time new 1,000-row test, selected scored 336/1,000 (33.6%) vs base 97/1,000 (9.7%), paired p=2.37e-46. This is a positive single-seed, 13-update pilot result; validation does not establish that exact-format is better than distance-shaping, and selected generated 2.68× tokens and took 4.86× the inference time. Full selection and test audits are recorded under `reports/gsm8k_next_fresh_*`.

## Phase B — completed local baseline; reward-mode evidence limited

The 100-update formal run is a negative result: its selected adapter scored 77/1,063 on the consumed test versus 95/1,063 for Base. In a 16k-token reward comparison, exact-format validation scored 190/500 and distance-shaped scored 183/500 (paired McNemar p=0.547); the selected exact-format model scored 336/1,000 versus Base 97/1,000 on that one-time test. This is one seed and does not establish a reward-mode advantage. Full records remain under `reports/gsm8k_next_fresh_*` and in `reports/experiment_report.md`.

## Phase C — completed; group-size verdict: MIXED

The locked Phase C protocol used two training seeds, matched approximately 16k generated completion-token budgets, validation-only candidate selection, and one-time independent Base/selected-adapter tests. Both seeds used Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, `exact_format` reward, NF4 QLoRA, gradient checkpointing, `use_vllm=False`, prompt/completion limits of 256, and a 5e-5 learning rate.

| Seed | Group4 validation | Group8 validation | Selected | G4/G8 generated tokens | Updates | Peak whole-GPU training memory |
|---|---:|---:|---|---:|---:|---:|
| 1 | 139/500 (27.8%) | 176/500 (35.2%) | Group8 | 16,097 / 16,354 | 26 / 13 | 3,968 / 4,399 MiB |
| 2 | 183/500 (36.6%) | 163/500 (32.6%) | Group4 | 16,160 / 16,415 | 25 / 13 | 4,070 / 4,506 MiB |

The validation winner reverses across seeds, so the frozen group-size conclusion is **MIXED**. Token budgets differ by less than 1.6%, but optimizer updates, prompts covered, and backward-pass counts are not matched. Both runs completed without OOM. Validation paired tests/bootstrap intervals describe question-level uncertainty, not variation across training seeds.

Seed 1 selected Group8 scored 354/1,000 on its one-time GSM8K-train-source test versus Base 99/1,000 (paired p=2.12e-52). Seed 2 selected Group4 scored 675/1,000 on the one-time SVAMP test versus Base 267/1,000 (paired p=2.07e-85). Neither test compares Group4 with Group8. The selected adapters used 2.55×/2.94× the Base generation tokens and 4.58×/5.06× the Base wall time, respectively; these are quality/cost trade-offs, not inference-efficiency gains.

Saved-rollout diagnostics show correctness-mixed groups at 15/26 vs 11/13 in seed 1 and 14/25 vs 10/13 in seed 2 (Group4 vs Group8). Correctness-mixed groups per 1,000 generated tokens were 0.932/0.673 and 0.866/0.609. This is a verifier-signal-density proxy, not gradient efficacy. All 13 Group8 prompts in each seed also appeared in Group4; shared-prompt rollout accuracy differences were +1.0 and −1.0 percentage points. Source records are `reports/gsm8k_phasec_*`, `reports/phasec_seed2_diagnostic_report.md`, and `reports/gsm8k_phasec_prompt_overlap.json`.

An additional **post-hoc** audit of the saved test predictions found 273/1,000 Base and 260/1,000 adapter unparsed GSM8K outputs, and 167/1,000 Base and 114/1,000 adapter unparsed SVAMP outputs. The selected adapters produced longer decoded completions and consumed more aggregate inference tokens/time; per-question token IDs and latency were not saved, so re-tokenized string lengths are labelled separately from measured generation totals. This audit does not alter historical exact-match scores or the locked test protocol. See `reports/posthoc_prediction_length_audit.md` and `reports/posthoc_prediction_length_audit.json`.

All Phase C exit criteria are complete: both seeds, fixed-token-budget comparisons, locked validation choices, test-isolation audits, paired statistics, saved-output correctness/parse diagnostics, reward-group analysis, throughput and peak-memory records, and archived checkpoint-selection metadata. No additional Phase C training is planned in this archive.

## Phase D — not started

Only the CPU-only deterministic tool environment and policy-independent runner have been implemented. There is no Agent RL training. The installed TRL 1.16.0 / Transformers 4.57.3 combination blocks native `tools` and `environment_factory` at version checks; experimental `rollout_func` integration remains unimplemented. See `docs/agent_env_prototype.md`.

## Compute and data status

No training was started for this final archive, and no cloud GPU or paid service was used. The old AutoDL plan remains paused: only 23 unused official GSM8K-train rows remain, and SVAMP has been consumed as the Phase C seed 2 test. Any formal follow-on requires a newly selected and locked independent validation/test source. The Phase C checkpoint weights remain local-only; their paths, sizes, and SHA-256 digests are recorded in the artifact manifest.
