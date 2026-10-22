# GSM8K next-run split

Source: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), configuration `main`, pinned revision `740312add88f781978c0658806c59bc2815b9866`.

This intermediate split reused the fixed 1,500-row train and 500-row validation subsets from `../gsm8k-formal/`; only its 1,000-row test was fresh. No model was evaluated on it. Before any next-round training, it was superseded by `../gsm8k-next-fresh/`, which also uses a fresh validation split. Keep this directory as a data-preparation record and use `gsm8k-next-fresh` for the next experiment.

No model has been evaluated on this test. Select checkpoints using validation, then evaluate base and the selected adapter once on this test. Labels are integer answers parsed from the pinned source's terminal `####` marker. See `LICENSE` for the source dataset license notice.
