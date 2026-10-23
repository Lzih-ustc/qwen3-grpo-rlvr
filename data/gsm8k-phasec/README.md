# Phase C GSM8K split registry

Source: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), configuration `main`, pinned revision `740312add88f781978c0658806c59bc2815b9866`. The source dataset is MIT-licensed; the required notice is included in `LICENSE`.

This directory preserves the locked Phase C seed 1 validation and GSM8K-train-source test inputs, along with the source-index exclusion registry and SHA-256 records in `manifest.json`. `validation.jsonl` has 500 rows and `test.jsonl` has 1,000 rows. The train and test2 files are intentionally empty: the candidates used the shared `data/gsm8k-next-fresh/train.jsonl`, and the independent seed 1 comparison used the saved test file here. The test is consumed and must not be used to tune new candidates.

The manifest includes the dataset revision, source split indices, per-file hashes, parser rule, and overlap counts. It does not claim that a public benchmark was absent from Qwen pretraining.
