# Efficient RLVR math data v1

This directory records a locked, CPU-generated dataset for a future experiment. The original prompt rows are generated from the pinned [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset) code, using its train, interpolate, and extrapolate generator regimes. The project retains only signed integer answers so its exact integer verifier can score every item.

The source repository's software is Apache-2.0; its top-level license is copied to `UPSTREAM-LICENSE.txt`. The upstream repository does not separately state a license for generated output data. To avoid implying that the code license automatically covers output data, the three JSONL split files are kept local and ignored by Git. They can be rebuilt and checked against the SHA-256 values in `manifest.json`.

## Split policy

- `train.jsonl`: 1,500 rows from the upstream train regime; training only.
- `validation.jsonl`: 300 rows from the upstream interpolate regime; candidate selection may inspect it.
- `test.jsonl`: 500 rows from the separate extrapolate regime; do not inspect model results on it until the experiment, selection rule, reward coefficient, and report tables are frozen.

Seeds, source commit, dependency versions, Python hash seed, rejection counts, generator-module counts, and row hashes are recorded in `manifest.json`. Two independent full builds reproduced all three locked hashes. See `research/efficient_rlvr/phase3_data_audit.md` and `reports/efficient_rlvr_phase3_reproducibility.json`.

The text-level overlap check covered JSON/JSONL/Parquet files under archived `data/`, `reports/`, and saved `runs/` traces. It found no normalized exact or punctuation-insensitive match. This does not detect paraphrases or establish that Qwen3 pretraining never contained examples from the public generator.

## Rebuild

The generator and Parquet audit use separate CPU-only environments because the locked generator uses NumPy 1.26.4 while PyArrow 26 requires NumPy 2 or newer. Use CPython 3.12.3, as recorded in `manifest.json`, and run the commands from the repository root in WSL/Linux. Generate the splits in `.venv-data`:

```bash
python -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r research/efficient_rlvr/requirements-data.lock.txt
mkdir -p .cache/efficient_rlvr
git clone https://github.com/google-deepmind/mathematics_dataset.git .cache/efficient_rlvr/math_dataset_src
git -C .cache/efficient_rlvr/math_dataset_src checkout 427f45075f84b8b9774950196ad63867ca20ffb3
PYTHONHASHSEED=0 python research/efficient_rlvr/scripts/prepare_locked_math_data.py
```

Run the historical JSON/JSONL/Parquet overlap audit in `.venv-audit`:

```bash
python -m venv .venv-audit
source .venv-audit/bin/activate
python -m pip install -r research/efficient_rlvr/requirements-audit.lock.txt
python research/efficient_rlvr/scripts/audit_locked_math_data.py
```

The generator excludes upstream `_composed` leaves because their entity-description order depends on identity-hashed Python objects, and it records/skips the old generator's `random.randint(float, float)` calls rejected by CPython 3.12. It does not patch the upstream source.
