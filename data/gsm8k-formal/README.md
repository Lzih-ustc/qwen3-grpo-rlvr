# GSM8K formal-stage split

Source: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), configuration `main`, pinned revision `740312add88f781978c0658806c59bc2815b9866`.

This split contains 1,500 training and 500 validation examples sampled without overlap from the official train split. The formal test contains the 1,063 official test rows left after excluding the source indices used in `../gsm8k/test.jsonl` and `../gsm8k/test2.jsonl`. The preparation seed is `20261023`; exact counts, source indices, and SHA-256 hashes are in `manifest.json`. `reports/gsm8k_formal_data_validation.json` verifies the split hashes and exclusion of all 256 previously opened pilot test rows.

No model has been evaluated on this split's test rows. Use validation for checkpoint selection and report the formal test only once for the selected checkpoint. Labels are integer answers parsed from the pinned source's terminal `####` marker. See `LICENSE` for the dataset license notice.
