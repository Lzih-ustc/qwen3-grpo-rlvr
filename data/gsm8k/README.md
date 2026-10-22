# GSM8K pilot subset

Source: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), configuration `main`, pinned revision `740312add88f781978c0658806c59bc2815b9866`.

The source card describes GSM8K as multi-step grade-school math QA and lists an MIT license. This folder contains fixed local pilot subsets: 256 training rows and 96 validation rows sampled from the official train split, plus two disjoint 128-row test slices sampled from the official test split. `test.jsonl` was opened for early development diagnostics; `test2.jsonl` was opened after pilot checkpoint selection for the reported 33/128 versus 12/128 comparison. Both pilot slices have now been used and are not untouched final test data. Selection seed: `20261023`. Exact question text is deduplicated across all four subsets. `manifest.json` stores counts and SHA-256 hashes.

The separate `../gsm8k-formal/` directory is prepared for the next training stage: 1,500 train rows, 500 validation rows, and the remaining 1,063 official test rows. It excludes the source indices used by both pilot test slices, and its validation record verifies this separation. Its test split has not been evaluated. The separate directory keeps the pilot artifacts intact.

The source's `#### final answer` marker is parsed as a terminal integer. All rows included in the pilot must pass that parser; malformed or non-integer labels stop data preparation. Training prompts request a concise derivation and a single marked answer. The verifier awards reward only from the marked final integer.

See `LICENSE` for the source dataset license notice.
