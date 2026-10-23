# Phase 4B — formal-experiment final audit

Audit date: 2026-10-23

> Historical snapshot before implementation of the final engineering gate. Its GFRL-blocker decision is superseded by phase4b_engineering_gate.md, which records the CPU fault-injection results, local RTX 3060 smoke, and current paid-execution decision. Phase A–C experiment records remain unchanged.

## Decision

**NO-GO for paid execution.** This turn performed offline code/data/protocol review only. No GPU training, AutoDL instance, paid storage, or other paid service was started; spend for this audit is ¥0.

The data integrity and current prompt-cap checks pass. The formal design now has explicit evaluation counts, pre-registered metrics, stage budgets, and manual spend gates. The experiment cannot start until GFRL has a tested hard token-budget guard and durable accounting, the 4090 preflight is approved and passes, and the user explicitly approves the next paid stage. Completing one stage cannot auto-start another.

## Audit findings

| Area | Result | Evidence and consequence |
|---|---|---|
| Locked data integrity | PASS | Train, validation, and test file hashes match the manifest. No split files were edited. |
| Prompt length | PASS | Pinned tokenizer audit reports zero prompts over 256 tokens; maxima are 130/105/95 for train/validation/test. |
| Completion truncation | UNKNOWN | No new-split generation exists. At-cap and answer-truncation rates cannot be inferred offline. A >5% cap-hit rate in any arm on validation blocks the supplement and test. |
| Generator-module separation | MIXED | Train/validation share 27 exact leaf modules. Train/test share one exact leaf and 13 normalized module families. The test remains an upstream extrapolation-regime split, not a family-disjoint or novel-concept set. |
| Distribution comparability | LIMITED | Arithmetic is 31.3% of train and 70.4% of test; numeric literals/question average 3.80 vs 7.10; answer digits average 2.93 vs 3.87. Report test as this locked generated extrapolation mixture, not broad mathematical ability. |
| GFRL token telemetry | PARTIAL PASS | The callback counts every returned candidate's token IDs before deciding retention. Rejected all-correct/all-wrong groups, retry samples, and exhausted groups enter attempted totals; only mixed groups enter retained totals. |
| GFRL budget control | FAIL / BLOCKER | No fixed-token stop controller or append-only per-candidate ledger is integrated. Current aggregate in-memory stats may be lost on interruption. A 12-candidate cap bounds retries per requested group but does not cap run-level tokens. |
| LCE vs cost measures | PASS after protocol clarification | Offline prefixes are rescored from one saved full completion; they do not measure actual early stopping, generated tokens, or GPU latency. Actual token IDs and CUDA-synchronized timings are separate required fields. |
| Accuracy-cost rule | PRE-REGISTERED | Full-response exact accuracy is primary; accuracy non-inferiority and fixed token/latency reduction gates are set before test. Every arm is reported; no test-selected winner. |
| Cost estimate | CONDITIONAL | Planning ceiling is 29.25 billed GPU-hours and ¥54.99 at the current ¥1.88/hour public quote; with 30% reserve, proposed rounded cap is ¥75. Re-price after preflight and before each stage. |

The module and difficulty details, including per-domain row counts and all split hashes, are in phase4b_data_audit.md and reports/efficient_rlvr_phase4b_data_audit.json.

## Evaluation inventory

The terminal evaluation panel includes all 18 terminal adapters plus Base. The fixed-token and fixed-time cohorts each contain three arms × three paired seeds.

| Evaluation stage | Models | Scored prompts/model | Scored completions | 128-token planning scenario | 256-token maximum, with 20 warmups/model |
|---|---:|---:|---:|---:|---:|
| Main validation after fixed-token cohort | Base + 9 adapters = 10 | 300 | 3,000 | 409,600 | 819,200 |
| Supplement validation | 9 adapters | 300 | 2,700 | 368,640 | 737,280 |
| Final independent test | Base + 18 adapters = 19 | 500 | 9,500 | 1,264,640 | 2,529,280 |
| **Total** | **19 unique model states** | — | **15,200** | **2,042,880** | **4,085,760** |

Warmups add 200, 180, and 380 completions to the three rows, respectively. They use fixed non-scored train prompts, never test prompts. All counted evaluation completions are one full-cap generation. The LCE budgets (32/64/96/128/192/256) reuse those same saved completion IDs.

Primary fixed-token training has nine runs × 16,000 attempted tokens = 144,000 maximum attempted completion tokens. At a 128-token mean this is roughly 31 group candidates/run or 125 completion sequences/run; these are planning approximations, not unique-prompt counts. The fixed-time supplement has nine 25-active-minute runs. Before a 4090 preflight its token estimate is only a scenario: 50–100 generated tokens/s implies 75,000–150,000 tokens/run, or 675,000–1,350,000 total. Its run-level stop is time, not a token target.

## GFRL accounting audit

The existing GroupFilterStats behavior correctly distinguishes:

- attempted groups/completions/tokens: every candidate generation that returns;
- retained groups/completions/tokens: only correctness-mixed candidates;
- discarded tokens: attempted minus retained, including non-mixed and exhausted candidates;
- exhausted groups: requested groups that fail to find a mixed candidate within 12 candidate groups;
- effective groups: the accepted mixed groups returned with effective_group=true.

Thus the data structure's normal callback accounting includes all rejected and additional rollouts. It does **not** enforce 16,000 tokens for a whole training run. The formal controller must reserve 4 × 256 = 1,024 tokens before each full group attempt, refuse another attempt when less remains, count IDs before parsing/filtering, write each outcome durably, and reconcile ledger sums against saved IDs. Any lost or mismatched accounting invalidates that run; no token overshoot or replacement run is allowed.

## Metric and selection contract

For each cohort, report all seed-level rows, the three-seed mean/range, paired question-level intervals, and parse/cap rates. The independent test is one evaluation pass for all 19 pre-registered model states after code, terminal checkpoints, and reporting tables are frozen.

An intervention is a cost-efficient candidate only when, against standard GRPO in the same cohort:

1. Its lower one-sided 98.75% paired-bootstrap bound for full-response exact-accuracy difference is above -2.0 percentage points.
2. Its actual generated test tokens per correct solution are at most 90% of GRPO.
3. Its paired median end-to-end latency is at most 95% of GRPO.

There are four pre-registered accuracy contrasts across the two cohorts. For each contrast, run 20,000 paired bootstrap replicates with NumPy PCG64 seed 260423: resample 500 prompt IDs with replacement, keeping each prompt's three matched seed outputs together; use the empirical 1.25th percentile as the lower one-sided 98.75% bound (Bonferroni family-wise alpha 0.05 across four contrasts). Report McNemar tests per seed with Holm correction across the 12 seed-level tests (four contrasts × three seeds). The cost thresholds are fixed engineering criteria, not expected gains. Tokens per correct solution is infinite if an arm has zero correct answers. Compute correct-solutions/GPU-hour as the sum of three seed-level correct counts divided by the corresponding sum of run hours; report evaluation GPU-hours and all-in study spend separately. If multiple methods qualify, report them all; no single winner is chosen by a favorable secondary metric. No validation-best checkpoint selection is used: each comparison uses the terminal state determined by its registered budget.

Report p50/p95 end-to-end latency, CUDA-synchronized generation latency, model-load time, actual completion token counts, tokens/correct answer, full-cap accuracy, LCE offline prefix accuracy, cap-hit rate, parse-failure rate, peak VRAM, active/billed GPU hours, updates, unique prompts, attempted/retained/discarded GFRL tokens, effective groups, and correct solutions per training GPU-hour. Keep question-level uncertainty separate from uncertainty across only three training seeds.

## Revised paid-stage estimate

| Stage | Billed time ceiling | Estimate at ¥1.88/hour | Gate |
|---|---:|---:|---|
| Preflight | 0.50 h | ¥0.94 | One run only; stop on failed guard, gradient/update/checkpoint/telemetry check, or OOM. |
| Fixed-token main, nine runs | 6.00 h | ¥11.28 | Up to 40 billed min/run: 10 setup, 25 active, 5 save; stop at 16,000 tokens or active-time cutoff. |
| Main validation | 3.50 h | ¥6.58 | Base plus nine primary adapters; abort on artifact failure or >5% cap hits in an arm. |
| Fixed-time supplement, nine runs | 6.00 h | ¥11.28 | Up to 40 billed min/run, including 25 active minutes; no extension. |
| Supplement validation | 3.25 h | ¥6.11 | Nine supplemental adapters; stop on artifact failure or >5% cap hits in an arm. |
| Final independent test | 10.00 h | ¥18.80 | All 19 model states once; no result-based rerun or selection. |
| **Total** | **29.25 h** | **¥54.99** | With 30% reserve: **¥71.49; proposed rounded cap ¥75**. |

The validation/test time reserves use up-to-256-token generations, a planning decode rate of 75 tokens/s, plus model loading, synchronization, and export. That rate is not yet measured on the target instance. Recompute all stage ceilings from the preflight measurement and live quote; a lower rate, added storage charge, or higher quote requires a revised plan and explicit approval.

AutoDL states that RTX 4090 24 GB is currently ¥1.88/hour on its [public price page](https://www.autodl.com/). Its [billing documentation](https://api.autodl.com/docs/price/) says usage is charged from instance power-on through power-off, not only while kernels use the GPU, and separately billed data-disk storage may apply. Price was checked on 2026-10-23.

## Conditions before a GO decision

1. Implement and test the whole-run attempted-token guard and append-only GFRL ledger; verify that rejected, ineffective, and extra samples count toward the cap.
2. In a paid, explicitly approved preflight, verify one nonzero optimizer update, adapter parameter changes, checkpoint reload equality, exact token reconciliation, no OOM, GPU peak memory, and measured generation/training throughput.
3. Freeze all hashes, train-only warmup IDs, validation/test evaluation code, metric definitions, bootstrap seed, and terminal checkpoint rules before any test generation.
4. Confirm live AutoDL quote and storage charges. Obtain separate user approval for each paid stage, with the updated stage ceiling and no auto-start or retry.
5. Keep the generated split JSONL local and do not redistribute it; the upstream repository does not separately state an output-data license.

Phase A–C historical results, selection records, data, and checkpoints were not changed by this audit. No GPU run or paid stage is authorized by this report.
