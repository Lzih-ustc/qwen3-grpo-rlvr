# Qwen3-0.6B 本地 GRPO 实验

这是一个面向单张 RTX 3060 12GB 的可复现小型实验：在 WSL2 Ubuntu 中使用 TRL `GRPOTrainer`、4-bit NF4 QLoRA、梯度检查点和 Transformers 原生 `generate()`（`use_vllm=False`），以可验证的整数奖励完成 GRPO 更新，并由独立评测进程评估固定留出集。基础模型固定为 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`。包含算术正确性 smoke 和固定 revision 的 GSM8K 多步数学 pilot。

## 目录

- `scripts/audit_env.py`：记录 WSL、GPU、Python、PyTorch、CUDA、Transformers、TRL 等环境信息。
- `scripts/prepare_data.py`：按固定种子生成 JSONL 训练、验证和最终评测集。
- `scripts/prepare_gsm8k.py` / `scripts/validate_gsm8k.py`：下载固定 GSM8K revision，生成不重叠子集并校验来源、答案格式与文件 hash。
- `scripts/build_manifest.py`：刷新源码、数据、报告和运行产物的 SHA-256 与文件大小清单。
- `scripts/validate_reward.py`：验证答案提取、奖励函数和数据集隔离规则。
- `scripts/train_grpo.py`：先运行 1 步 smoke，再运行小规模训练；仅在 OOM 时按顺序缩短 completion、降低 generation group，并记录每次配置。
- `scripts/eval_math.py`：独立进程逐题贪心解码，评估基座模型或已保存的 LoRA adapter。
- `scripts/audit_run.py`：检查训练日志有限性、真实梯度/参数更新、generation 组配置、checkpoint 内容和 validation 选模结果。
- `scripts/audit_gsm8k.py`：审计 GSM8K pilot、validation checkpoint 选择和未参与选择的 test2 对照。
- `requirements-wsl2.txt`：直接依赖版本；运行成功后生成的 `requirements-wsl2.lock.txt` 冻结完整依赖树。
- `reports/`：环境审计、最小测试和实验结果。
- `reports/experiment_report.md`：本地 smoke、小规模训练、独立评测和 AutoDL 迁移计划汇总。
- `reports/artifact_manifest.json`：数据、代码、环境锁、checkpoint 和 adapter 的 SHA-256 与文件大小。
- `reports/data_split_validation.json`：三份数据与确定性生成结果、跨 split 题目隔离验证。
- `reports/gsm8k_quality_gate.json`：GSM8K 数据、训练、选模与留出对照质量门结果。
- `runs/`：checkpoint、最终 adapter、Trainer 日志、显存和吞吐量指标。
- `data/`：固定的本地 JSONL 数据。

## WSL2 安装与运行

在 WSL Ubuntu 终端进入 Windows 工作区挂载目录（如 `/mnt/e/强化学习项目`）：

```bash
python3 -m venv ~/.venvs/qwen3-grpo
~/.venvs/qwen3-grpo/bin/python -m pip install --upgrade pip
# requirements-wsl2.txt 中的 PyTorch wheel 使用 CUDA 12.1 runtime，和已审计的 WSL NVIDIA 驱动兼容。
~/.venvs/qwen3-grpo/bin/python -m pip install -r requirements-wsl2.txt --extra-index-url https://download.pytorch.org/whl/cu121
# 后续复现可直接使用完整冻结依赖树：
# ~/.venvs/qwen3-grpo/bin/python -m pip install -r requirements-wsl2.lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
~/.venvs/qwen3-grpo/bin/python scripts/audit_env.py --output reports/environment.json
~/.venvs/qwen3-grpo/bin/python scripts/check_runtime.py
~/.venvs/qwen3-grpo/bin/python scripts/prepare_data.py
~/.venvs/qwen3-grpo/bin/python scripts/validate_reward.py
~/.venvs/qwen3-grpo/bin/python scripts/train_grpo.py --profile smoke --run-dir runs/smoke-repro
~/.venvs/qwen3-grpo/bin/python scripts/train_grpo.py --profile small --run-dir runs/local-small-repro
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --tag base --output reports/eval_base.json
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --adapter runs/local-small-repro/attempt-01/trainer/checkpoint-8 --eval-data data/validation.jsonl --tag step8-validation --output reports/validation_step8_repro.json
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --adapter runs/local-small-repro/attempt-01/trainer/checkpoint-12 --eval-data data/validation.jsonl --tag step12-validation --output reports/validation_step12_repro.json
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --adapter runs/local-small-repro/attempt-01/final_adapter --tag local-small --output reports/eval_local_small.json
~/.venvs/qwen3-grpo/bin/python scripts/audit_run.py --smoke-run runs/smoke-repro --small-run runs/local-small-repro --step8-validation reports/validation_step8_repro.json --step12-validation reports/validation_step12_repro.json --output-dir reports/repro
```

GSM8K 子集由 `scripts/prepare_gsm8k.py` 固定 revision 和 seed 生成；复跑训练使用新的空目录。来源、MIT 许可、哈希和本轮实测结果见 [data/gsm8k/README.md](data/gsm8k/README.md) 与 [实验报告](reports/experiment_report.md)。

本地 GSM8K 采样组 pilot 可用以下命令复现。test2 只在 validation 选好 checkpoint 后用于最终对照：

```bash
~/.venvs/qwen3-grpo/bin/python scripts/prepare_gsm8k.py
~/.venvs/qwen3-grpo/bin/python scripts/validate_gsm8k.py
~/.venvs/qwen3-grpo/bin/python scripts/train_grpo.py --profile small --max-steps 8 --save-steps 4 --num-generations 2 --max-completion-length 160 --prompt-token-limit 192 --seed 20261023 --train-data data/gsm8k/train.jsonl --run-dir runs/gsm8k-pilot-repro
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --adapter runs/gsm8k-pilot-repro/attempt-01/trainer/checkpoint-4 --eval-data data/gsm8k/validation.jsonl --tag gsm8k-group2-step4-val --output reports/gsm8k_group2_step4_val.json --max-new-tokens 192
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --adapter runs/gsm8k-pilot-repro/attempt-01/trainer/checkpoint-8 --eval-data data/gsm8k/validation.jsonl --tag gsm8k-group2-step8-val --output reports/gsm8k_group2_step8_val.json --max-new-tokens 192
~/.venvs/qwen3-grpo/bin/python scripts/train_grpo.py --profile small --max-steps 4 --save-steps 2 --num-generations 4 --max-completion-length 160 --prompt-token-limit 192 --seed 20261023 --train-data data/gsm8k/train.jsonl --run-dir runs/gsm8k-group4-pilot-repro
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --adapter runs/gsm8k-group4-pilot-repro/attempt-01/trainer/checkpoint-2 --eval-data data/gsm8k/validation.jsonl --tag gsm8k-group4-step2-val --output reports/gsm8k_group4_step2_val.json --max-new-tokens 192
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --adapter runs/gsm8k-group4-pilot-repro/attempt-01/trainer/checkpoint-4 --eval-data data/gsm8k/validation.jsonl --tag gsm8k-group4-step4-val --output reports/gsm8k_group4_step4_val.json --max-new-tokens 192
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --tag gsm8k-base-test2 --eval-data data/gsm8k/test2.jsonl --output reports/gsm8k_base_test2.json --max-new-tokens 192
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --adapter runs/gsm8k-group4-pilot-repro/attempt-01/trainer/checkpoint-4 --eval-data data/gsm8k/test2.jsonl --tag gsm8k-selected-test2 --output reports/gsm8k_selected_test2.json --max-new-tokens 192
~/.venvs/qwen3-grpo/bin/python scripts/audit_gsm8k.py --group2-run runs/gsm8k-pilot-repro --group4-run runs/gsm8k-group4-pilot-repro --group2-step4-validation reports/gsm8k_group2_step4_val.json --group2-step8-validation reports/gsm8k_group2_step8_val.json --group4-step2-validation reports/gsm8k_group4_step2_val.json --group4-step4-validation reports/gsm8k_group4_step4_val.json --base-test reports/gsm8k_base_test2.json --selected-test reports/gsm8k_selected_test2.json --output-dir reports/gsm8k-repro
```

模型首次运行时从 Hugging Face 下载到默认 HF cache。实验本身不调用云端 GPU。为了便于复现，可在运行前设置 `HF_HOME` 指向持久磁盘目录。

## 奖励与指标

答案校验只接受唯一 `<answer>整数</answer>` 或唯一 `\boxed{整数}` 中的整数。`distance_shaped`历史奖励在此基础上加格式和数值距离分；新增`exact_format`对照只给精确答案1分、指定XML格式额外0.05分，不按错误答案距离排序。两种模式都有单测。`--max-generated-tokens`可令训练在完成当前组后停止，使不同采样组大小的比较按真实生成token对齐，并在summary记录目标与超额。新的奖励消融结果见`reports/experiment_report.md`；运行完成前不将任何候选当作已验证改进。

每次成功运行保存实际优化步数、LoRA 可训练参数和权重更新量、TRL 原始 log history/reward、训练耗时、step/s、生成 completion/s、PyTorch 峰值 allocated/reserved 显存，以及 `nvidia-smi` 采样到的整卡峰值显存。报告需同时保留桌面进程的显存基线，不能将 WSL 进程显存与整卡显存混为一谈。

## AutoDL 单张 24GB 迁移

先用新鲜validation确定奖励与训练配置，再在实例中记录 GPU 型号/驱动，按该驱动安装 PyTorch CUDA wheel，创建锁定环境并跑 reward 测试与 1 步 smoke。24GB 卡先用 NF4 QLoRA、gradient checkpointing、no-vLLM、group 8、prompt 256、completion 256 做单步 profile；确认余量后运行500步正式训练。旧官方test已在本地实验使用，后续正式运行不能再用它选模。`data/gsm8k-next-fresh/` 是新准备的独立轮次：从固定revision的GSM8K train split剩余行中排除既有1,500 train与500 validation，再采样新train/validation/test（1,500/500/1,000），source index校验通过。每50步保存checkpoint、每100步在validation评估。OOM时按代码回退顺序逐项缩短completion、降低group、降低prompt；每次记录配置与显存。

资源计划以单张 RTX 4090 24GB 为例：AutoDL 官方页面于2026-10-23列价为¥1.88/GPU小时，估计总用卡不超过2小时，硬上限4小时（GPU租用费最多¥7.52）；创建实例前重新核对当日价格、计费粒度和其他费用。该计划不是租卡授权，任何付费实例仍须先得到用户明确批准。

## 上游接口依据

- [TRL GRPOTrainer 文档](https://huggingface.co/docs/trl/grpo_trainer)
- [Qwen3-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen3 推理与 thinking 开关说明](https://huggingface.co/Qwen/Qwen3-0.6B)
