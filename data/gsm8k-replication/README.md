# Phase C seed 2 validation registry

Source: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), configuration `main`, pinned revision `740312add88f781978c0658806c59bc2815b9866`. The source dataset is MIT-licensed; the required notice is included in `LICENSE`.

`validation.jsonl` preserves the 500 locked GSM8K validation rows for the second training seed. The training input is shared at `../gsm8k-next-fresh/train.jsonl`; this directory's `train.jsonl`, `test.jsonl`, and `test2.jsonl` are intentionally empty. The independent seed 2 evaluation used the separately pinned full SVAMP set at `../svamp/test.jsonl`; that test is consumed and must not be reused for tuning.

`manifest.json` records source-index exclusions, hashes, parser settings, and overlap checks. The manifest is split provenance, not a statement about Qwen pretraining exposure.
