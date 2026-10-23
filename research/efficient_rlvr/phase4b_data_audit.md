# Phase 4B locked-data audit

Status: **PASSED for integrity and prompt-input limits; completion-length risk remains unmeasured**. This is an offline audit and does not change the locked JSONL files.

## Split-level difficulty proxies

Question character/word/number counts and answer digit/magnitude distributions are descriptive proxies only; they are not validated difficulty labels. The held-out test comes from the source extrapolation regime.

| Split | Rows | Unique leaves | Question chars mean / p95 / max | Words/numbers mean / p95 / max | Numeric literals mean / p95 / max | Answer digits mean / p95 / max |
|---|---:|---:|---:|---:|---:|---:|
| train | 1500 | 30 | 47.088 / 83 / 156 | 10.087 / 17 / 45 | 3.796 / 8 / 21 | 2.925 / 8 / 21 |
| validation | 300 | 27 | 49.24 / 67 / 123 | 9.977 / 15 / 35 | 3.723 / 7 / 18 | 3.647 / 8 / 25 |
| test | 500 | 13 | 57.486 / 87 / 111 | 10.508 / 18 / 25 | 7.1 / 12 / 15 | 3.874 / 12 / 19 |

## Module overlap

- **train_vs_validation:** 27 exact leaf names; 27 expansion-normalized families; shared top-level domains: algebra, arithmetic, comparison, measurement, numbers, polynomials, probability.
- **train_vs_test:** 1 exact leaf names; 13 expansion-normalized families; shared top-level domains: algebra, arithmetic, comparison, measurement, numbers, probability.
- **validation_vs_test:** 1 exact leaf names; 12 expansion-normalized families; shared top-level domains: algebra, arithmetic, comparison, measurement, numbers, probability.

Exact generator-leaf disjointness does not imply concept-level independence. The expansion-normalized family view strips only the upstream naming suffixes _big, _longer, _more, and _more_samples; it is a structural overlap diagnostic, not a calibrated math-difficulty score.

## Generator-domain mix

| Split | Domain | Rows | Share |
|---|---|---:|---:|
| train | algebra | 253 | 16.9% |
| train | arithmetic | 469 | 31.3% |
| train | calculus | 10 | 0.7% |
| train | comparison | 67 | 4.5% |
| train | measurement | 96 | 6.4% |
| train | numbers | 432 | 28.8% |
| train | polynomials | 167 | 11.1% |
| train | probability | 6 | 0.4% |
| validation | algebra | 54 | 18.0% |
| validation | arithmetic | 89 | 29.7% |
| validation | comparison | 11 | 3.7% |
| validation | measurement | 21 | 7.0% |
| validation | numbers | 90 | 30.0% |
| validation | polynomials | 34 | 11.3% |
| validation | probability | 1 | 0.3% |
| test | algebra | 3 | 0.6% |
| test | arithmetic | 352 | 70.4% |
| test | comparison | 28 | 5.6% |
| test | measurement | 21 | 4.2% |
| test | numbers | 84 | 16.8% |
| test | probability | 12 | 2.4% |

## Prompt and completion truncation

The pinned Qwen3 tokenizer audit at a 256-token prompt cap passed with no over-cap prompts. Max chat-template prompt lengths were train 130, validation 105, and test 95 tokens. Thus the observed prompt-side cap margin is at least 126 tokens.

No outputs have been generated from these locked splits, so 256-token completion truncation rate, answer-at-cap rate, and CGLR reward effects at the cap are **unknown**. Do not infer them from target answer length or from the earlier smoke on different prompts. Measure them in validation; if the preregistered cap-hit threshold is exceeded, stop before opening the independent test.

## Integrity

All three source JSONL SHA-256 hashes match the immutable manifest. This report reads the files and does not rewrite them.
