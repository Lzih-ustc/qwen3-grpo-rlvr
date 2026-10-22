# GSM8K fresh-split run

Source: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), configuration `main`, pinned revision `740312add88f781978c0658806c59bc2815b9866`.

This is the next candidate dataset: 1,500 train, 500 validation and 1,000 test examples sampled from unused rows of the official train split. All source indices previously used by `../gsm8k-formal/train.jsonl` and `../gsm8k-formal/validation.jsonl` were excluded before sampling. Train, validation and test within this directory are also source-index and prompt disjoint. The seed, split source, excluded indices, counts and SHA-256 hashes are in `manifest.json`; `reports/gsm8k_next_fresh_data_validation.json` verifies the split and its separation from the previous run.

No model has been evaluated on this split. Use validation for checkpoint selection, then evaluate the base model and selected checkpoint once on test. Labels are integer answers parsed from the pinned source's terminal `####` marker. See `LICENSE` for the dataset license notice.
