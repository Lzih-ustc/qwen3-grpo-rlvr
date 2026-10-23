# Dataset and model artifact notices

This project redistributes small, fixed GSM8K and SVAMP dataset splits. The upstream license notices are retained alongside each relevant dataset.

- **GSM8K:** `openai/gsm8k`, configuration `main`, revision `740312add88f781978c0658806c59bc2815b9866`, MIT License, copyright OpenAI. Copies of the notice are under `data/gsm8k*/LICENSE`. The pinned dataset card identifies the MIT license: <https://huggingface.co/datasets/openai/gsm8k/blob/740312add88f781978c0658806c59bc2815b9866/README.md>.
- **SVAMP:** `arkilpatel/SVAMP`, commit `78e727689e1c1bebfc4be39c446898e8e10b0518`, MIT License, copyright Arkil Patel. The copied source notice is `data/svamp-source/LICENSE`. The pinned upstream license is <https://github.com/arkilpatel/SVAMP/blob/78e727689e1c1bebfc4be39c446898e8e10b0518/LICENSE>.
- **Qwen3-0.6B:** model revision `c1899de289a04d12100db370d81485cdf75e47ca`; upstream model card declares Apache-2.0: <https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca>. The model weights and adapter/checkpoint binaries are not included in this repository; scripts download the pinned model when a user runs training or inference.

The notices above describe the named upstream artifacts only. This repository does not add a blanket license grant for all project code or derived reports.
