# Phase 4 — audited AutoDL RTX 4090 experiment plan

Status: **offline plan only; NO-GO for paid execution until the listed preflight blockers are closed and the user explicitly approves each paid stage.** No AutoDL instance has been created or started. No paid compute was used for this audit.

This plan preserves the locked dataset and all Phase A–C records. The independent test is opened once, after the protocol and all terminal checkpoints are frozen.

## Research question and fixed arms

Under fixed compute budgets, can the existing correctness-gated length reward (CGLR) and correctness-mixed group filtering (GFRL) reduce output-token use and serving latency while retaining exact-verifier accuracy?

| Arm | Group size | Reward | Filtering |
|---|---:|---|---|
| A — standard GRPO | 4 | Exact integer correctness, 0/1 | None |
| B — CGLR | 4 | Correctness-gated length reward, lambda=0.2 | None |
| C — CGLR + GFRL | 4 | Same CGLR | Retain mixed-correctness groups; maximum 12 candidate groups per requested group |

Use the same Qwen3-0.6B pinned revision, chat template with thinking disabled, prompt cap 256, completion cap 256, verifier, NF4 QLoRA, LoRA rank 8, gradient checkpointing, non-vLLM generation, optimizer, batch/group configuration, and decoding settings. Training uses only the locked 1,500-row train split. The three paired seeds are 260201, 260202, and 260203. Do not change arms, reward weight, data, token caps, stopping, or analysis after seeing validation or test outcomes.

## Phase sequence and evaluation inventory

All generated-token estimates below are completion tokens. The 128-token mean is a planning scenario, not an observed property of these new prompts. The 256-token column is the hard per-completion generation cap. Evaluation produces one full completion per model/prompt pair; LCE prefixes are calculated offline from that saved completion and add no generation.

| Stage | Model states | Scored model-prompt sequences | Planning generated tokens | Maximum generated tokens for the stage |
|---|---:|---:|---:|---:|
| Preflight | 2 adapter states from 1 base revision | 8 train-only diagnostic completions, plus up to 48 rollout completions | about 7,168 at 128 tokens/completion | 14,336 |
| Fixed-token main training | 9 terminal adapters | Training prompt count is data-dependent; about 1,125 completions across 9 runs at 128 tokens/completion | 144,000 attempted completion tokens total | 144,000 attempted tokens |
| Main validation | Base + 9 fixed-token adapters = 10 models | 10 × 300 = 3,000 | 409,600 including warmups | 819,200 |
| Fixed-GPU-time supplement training | 9 terminal adapters | Training prompt count is data-dependent | 675,000–1,350,000 scenario range, using 50–100 emitted tokens/s during 25 active minutes/run | No independent token ceiling; the hard limit is billed/active time |
| Supplement validation | 9 fixed-time adapters | 9 × 300 = 2,700 | 368,640 including warmups | 737,280 |
| Final independent test | Base + all 18 terminal adapters = 19 models | 19 × 500 = 9,500 | 1,264,640 including warmups | 2,529,280 |

The fixed-token estimate uses 16,000 tokens/run × 9 runs = 144,000. If mean completion length were 128, that is about 31 candidate groups or 125 completions/run at group size 4. GFRL retries can reduce distinct requested-prompt coverage; exact prompt IDs, candidate-group counts, completion counts, and effective updates must be logged. The fixed-time token range is a scenario only: the new 4090 training throughput is unknown until preflight.

The evaluation panel totals 19 models on validation (split into the two validation stages above) and 19 models on test. That is 15,200 scored model-prompt completions and 760 non-scored warmups. The combined completion-token envelope is 4,085,760, including warmups. At the 128-token scenario it is about 2,042,880. Repeated model loading, synchronization, and artifact export are included in the time reserve below.

The 20 warmups per model are drawn from a fixed, non-scored subset of train; no test prompt is used for warmup. At validation, score Base and all nine primary checkpoints before deciding whether the supplemental cohort may run. At the final test, score every pre-registered model once; do not choose a winner or change the analysis from test results.

## GFRL accounting and fixed-token enforcement

The current GroupFilterStats code counts every returned candidate completion and its completion-ID length before deciding whether the group is retained. Rejected all-correct and all-incorrect groups, retries on additional train rows, and candidates from groups that exhaust the 12-candidate limit all contribute to attempted completions and attempted completion tokens. Retained tokens include only completions from accepted mixed groups; discarded tokens are attempted minus retained. Exhausted requested groups are separately counted and their final returned outputs are marked ineffective for the optimizer.

This establishes the accounting semantics for a successful callback, but it is **not yet a formal budget guarantee**. Current code has no hard fixed-token stop, no append-only event ledger, and only aggregate in-memory telemetry. A process interruption before the final report could lose that aggregate. The formal 18-run controller is not implemented.

Before the main matrix, add and validate a budget-aware generation guard and durable per-candidate ledger:

1. Count completion IDs immediately after each generation call, before parsing/filtering; append candidate-group ID, source prompt ID, seed, arm, completion lengths, accepted/rejected/exhausted outcome, and cumulative attempted tokens.
2. Reserve the worst-case 4 × 256 = 1,024 tokens before starting each complete group. Start another candidate group only if at least 1,024 tokens remain. Never start a partial group. Apply the same accounting to retries.
3. Do not exceed 16,000 attempted completion tokens. The final unused slack will be less than 1,024 tokens; report actual tokens, requested groups, all candidate groups, unique prompts, completions, retained/discarded tokens, exhausted groups, effective groups, and optimizer updates.
4. Reconcile the ledger token sum against saved completion IDs and training-run summaries before accepting any run. On a process restart, resume only if the budget state and ledger are recovered exactly; otherwise stop and mark the run incomplete.

The code currently counts completion-ID list lengths, which includes a returned EOS token if present and excludes prompt tokens. Log the generation finish reason where available and identify cap-truncated sequences separately. Do not substitute retained-token totals for attempted-token budgets.

## LCE, actual generation cost, and latency

- **Offline LCE prefix accuracy:** For each saved completion, decode prefixes at 32, 64, 96, 128, 192, and 256 tokens; score each prefix with the same parser and verifier. Use actual prefix length min(saved completion length, requested prefix). This is a post-generation analysis of one fixed completion. It does not create a new rollout, stop a live decoder, or establish saved GPU time.
- **Actual generated tokens:** Count returned completion token IDs for the one full-cap generation per prompt, excluding prompt IDs and padding and including EOS when it is returned. Report tokens/question, total tokens, cap-hit rate, parse failures, and tokens per correct solution. All GFRL candidates, including discarded candidates, enter training-cost totals.
- **GPU generation latency:** Keep the model loaded, use batch size 1, synchronize CUDA immediately before and after generation, and record the model.generate interval per question. Also record prompt tokenization/transfer plus generation as end-to-end per-question latency, and report model load separately. Report p50 and p95 and the throughput distribution. This is the only latency result that can support a real serving-cost statement.

Do not label the offline LCE curve as latency or as actual early-stopping savings. A generation cap hit is an output that reaches 256 completion tokens without EOS; the rate remains unknown until the new validation outputs exist.

## Locked-data audit interpretation

The Phase 4B offline audit verified all three JSONL hashes against the manifest and did not rewrite the data. The pinned Qwen3 tokenizer/template audit found no prompt above 256 tokens: maximum lengths are 130 train, 105 validation, and 95 test. Completion truncation and answer-at-cap rates remain unknown until generation.

The data are not module-family-disjoint. Train and validation share 27 exact generator leaves. Train and test share one exact leaf and 13 expansion-normalized generator families; all 13 test leaves map to families represented in train after stripping the upstream suffixes. The test is an upstream extrapolation-regime split, not a novel-concept test. Its composition is also shifted: arithmetic is 70.4% of test versus 31.3% of train, mean visible numeric literals are 7.10 versus 3.80, and mean answer digits are 3.87 versus 2.93. Treat test accuracy as performance on this locked generated extrapolation mixture, not a broad math benchmark estimate. The full split-level summaries and domain counts are in phase4b_data_audit.md.

Prompt-cap integrity passes, but completion truncation is not measurable offline. After the primary validation only, if any arm has more than 5% of completions cap-truncated, do not start the supplement or open test. Preserve that outcome and seek approval for a new protocol; do not silently raise the cap or edit the data.

The upstream project licenses its generator code under Apache-2.0 but does not separately specify a generated-output license. Keep split JSONL local and out of Git. Before transferring the data to a paid remote instance, review the provider's handling terms and keep a copy of the locked hashes; do not redistribute generated rows as licensed source data.

## Pre-registered metrics and candidate rule

The primary accuracy is exact-verifier accuracy on the complete generated response at the fixed 256-token cap. Parse failures count as incorrect. LCE prefix accuracy is secondary and never substitutes for full-response accuracy.

For each budget cohort, report all three seed-level results and the mean; paired question-level uncertainty does not estimate training-seed uncertainty. There are four pre-registered accuracy contrasts across the two cohorts (CGLR vs GRPO and CGLR+GFRL vs GRPO in each cohort). For each contrast, run 20,000 paired bootstrap replicates with NumPy PCG64 seed 260423: resample the 500 prompt IDs with replacement and keep all three matched seed outputs for each sampled ID together. Use the empirical 1.25th percentile as the lower one-sided 98.75% bound (Bonferroni family-wise alpha 0.05 across four contrasts). Report McNemar tests per seed and apply Holm correction across the 12 seed-level tests (four contrasts × three seeds). State explicitly that three seeds are too few for a precise population-level seed variance estimate.

An arm is eligible to be described as a cost-efficient candidate only if all of the following hold against standard GRPO in the same budget cohort:

1. The lower one-sided 98.75% paired-bootstrap bound on exact-accuracy difference is above -2.0 percentage points. This is Bonferroni-adjusted across the four pre-registered accuracy contrasts spanning both budget cohorts.
2. Total actual generated test tokens per correct solution are at most 0.90× GRPO.
3. Paired median end-to-end per-question latency is at most 0.95× GRPO. Report p95 even though it is not a promotion gate.

These are fixed engineering thresholds, not predicted outcomes. Report every metric for every arm, including failure to qualify. If one or more arms qualify, report all qualifying arms; do not select one using a more favorable secondary metric. Do not choose checkpoints by validation: compare the terminal checkpoint fixed by each run budget. No extra seeds, replacement runs, reward tuning, or alternate metric gates are automatic.

Also report completion-token count per question, cap-hit and parse-failure rates, active and billed training hours, correct test solutions per active training GPU-hour, correct test solutions per billed training GPU-hour, peak VRAM, updates, unique prompts, and effective GFRL groups. Compute correct-solutions/GPU-hour as the sum of the three seed-level correct counts divided by the corresponding sum of run hours; report evaluation GPU-hours and total study spend separately. If an arm has zero correct answers, tokens per correct solution is infinite and it cannot qualify. Keep training efficiency separate from the inference-token/latency decision.

## Execution order, time, cost, and stop conditions

Prices below use the current public RTX 4090 24 GB quote of ¥1.88/hour checked 2026-10-23. AutoDL bills from instance power-on to power-off, not just GPU kernel time; paid data-disk storage is separate. The live quote must be checked again before any approved run. The 75-token/s evaluation planning rate is a reservation assumption, not a measured 4090 rate. Recalculate the full time/cost table from preflight; if the measured rate is lower or the live price/storage changes the cap, stop and seek approval again.

| Stage | Maximum billed time | Estimated cost at ¥1.88/hour | Stage stop |
|---|---:|---:|---|
| Preflight | 0.5 h | ¥0.94 | One setup only. Stop on failed budget-guard test, non-finite/no-gradient update, no adapter change, checkpoint mismatch, OOM, or telemetry reconciliation failure. |
| Fixed-token main: 9 runs × 40 min | 6.0 h | ¥11.28 | Each run reserves up to 10 min load/warmup, 25 min active training, and 5 min save. Stop at either 16,000 attempted tokens or 25 active minutes. Incomplete target means failed/incomplete run; no extension/replacement. |
| Main validation: 10 models × 300 | 3.5 h | ¥6.58 | One saved full-cap completion per item and 20 train warmups/model. Stop on hash/ID mismatch, missing output, parser/verifier mismatch, or >5% cap-truncated completions in any arm. |
| Fixed-time supplement: 9 runs × 40 min | 6.0 h | ¥11.28 | Each run allows 10 min load/warmup, exactly 25 active minutes, and up to 5 min save. Stop at the cutoff; no extension or replacement. |
| Supplement validation: 9 models × 300 | 3.25 h | ¥6.11 | One saved full-cap completion per item and 20 train warmups/model. Stop on artifact/metric failure or >5% cap-truncated completions in any supplemental arm. |
| Final independent test: 19 models × 500 | 10.0 h | ¥18.80 | One evaluation pass after hashes, terminal checkpoints, analysis code, and report tables are frozen. No test-based selection or reruns. |
| **Total planning ceiling** | **29.25 h** | **¥54.99** | **30% reserve yields ¥71.49; proposed rounded hard cap ¥75.** |

Evaluation reservations assume batch size 1, up to 256 generated tokens, 75 completion tokens/s, plus model reload, synchronization, and export overhead: 3.5 hours for main validation, 3.25 hours for supplement validation, and 10 hours for final test. If preflight cannot justify these reservations, revise the plan and obtain new approval before proceeding. The projected ¥75 cap includes no paid data disk, storage retention, premium image, or additional retries.

Paid work must be manually gated, one stage at a time. Finishing a stage never starts the next stage automatically. Do not create an instance, enable an automation that launches runs, extend a slot, buy storage, or retry a failed run without a new explicit approval. Any OOM, lost accounting, or failed stop condition ends the current stage; do not change just one arm's sequence cap or budget.

Price references: [AutoDL public GPU prices](https://www.autodl.com/) and [AutoDL billing rules](https://api.autodl.com/docs/price/).

## Current GO / NO-GO

**NO-GO for paid execution at this audit.** Offline data hashes, prompt caps, and current GFRL counter semantics pass; module-family generalization is limited and must be described accordingly. The formal run is blocked until the budget-aware stop, durable telemetry, and 4090 preflight are implemented and verified. This plan is not an approval to incur the estimated ¥75.
