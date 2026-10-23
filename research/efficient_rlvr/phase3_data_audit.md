# Phase 3 data lock audit

Status: **PASSED**. This is an offline data-integrity and exact-text overlap check. It did not generate model outputs, evaluate a checkpoint, or train a model.

## Locked source and splits

Source: [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset), pinned revision `427f45075f84b8b9774950196ad63867ca20ffb3`. The upstream repository's Apache-2.0 license covers its generator code; the repository does not separately state a generated-output license. A license copy is included, and the generated JSONL rows remain local/ignored by Git. Each row is newly generated from the upstream train, interpolate, or extrapolate regime and filtered to signed integer answers for the project's exact verifier.

| Split | Rows | SHA-256 | Generator modules |
|---|---:|---|---:|
| train | 1500 | `0b0e38b8b8fc608edeb930a519cc2c759886605d83f92c508c7e5ae5349e88a5` | 30 |
| validation | 300 | `0632c74da64263b398d4ba9e66663a11ed6918cc05d7229cb4f1ff65b93c82e3` | 27 |
| test | 500 | `28a1e8d26070c3c21057a3e28c0febbad9b158964bf7f2953c5e1fb17fddca6f` | 13 |

The `train` split is training-only. `validation` uses the upstream interpolation regime and may be used for candidate selection. `test` uses the separate extrapolation regime and is reserved for one final evaluation after all choices are frozen. All 2,300 normalized source questions are unique across splits.

Two separate full builds reproduced the three locked hashes; see [`efficient_rlvr_phase3_reproducibility.json`](../../reports/efficient_rlvr_phase3_reproducibility.json).

## Historical-use and duplicate audit

- Scanned 730 JSON/JSONL/Parquet files under the archived `data/`, `reports/`, and `runs/` trees (60,533 prompt-text entries); exact normalized source-question overlap count: **0**.
- Punctuation-insensitive text match count: **0**. This stricter diagnostic can include normalization collisions; it is reported separately and does not silently rewrite any locked rows.
- Cross-split exact duplicate count: **0**.
- GSM8K/SVAMP consumed test sets were not used to generate, select, rank, or tune the split. No new model quality result has been viewed.
- This is a text-level audit only. It does not detect paraphrase or prove the Qwen3 pretraining corpus never contained related public generator examples.

The detailed file list and SHA-256 hashes are in [`efficient_rlvr_phase3_data_audit.json`](../../reports/efficient_rlvr_phase3_data_audit.json); split content hashes and generation seeds are in [`manifest.json`](../../data/efficient-rlvr-math-v1/manifest.json).
