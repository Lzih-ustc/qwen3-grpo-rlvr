# AutoDL 单张 24GB GPU 正式训练迁移计划

## 迁移边界

本地已验证的主路径是单卡 Transformers 生成、TRL GRPO、NF4 QLoRA 和 gradient checkpointing；继续保持 `use_vllm=False`。先在新鲜split上对比历史`distance_shaped`奖励与`exact_format`奖励，再将validation选出的奖励和训练配置冻结到正式计划中；测试集不能用于选奖励或选checkpoint。正式训练固定使用 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca` 和一份新的完整环境审计。GPU 型号、驱动、CUDA runtime 和可用显存必须以 AutoDL 实例实际输出为准，不能把本机 RTX 3060 的驱动或显存数据带过去。

## 启动配置

1. 选择单张 24GB GPU 实例，记录产品型号、驱动、`nvidia-smi` 全量输出、Python、PyTorch CUDA、TRL 和 bitsandbytes 版本；不启用多卡、vLLM 或云端生成服务。
2. 用 `requirements-wsl2.lock.txt` 重建已验证的用户态依赖，并按实例驱动确认 CUDA wheel 可用；运行 `pip check`、`scripts/check_runtime.py`、`scripts/validate_gsm8k.py`。依赖版本或 CUDA wheel 有变动时，重新生成该实例的完整 lock 并保存审计结果。先跑小配置 smoke，再以正式配置做 1 步显存 profile，确认反向传播、非零 LoRA 更新、奖励日志和 checkpoint 保存均成功。
3. 正式配置起点：NF4 + double quant，LoRA `r=8, alpha=16`，梯度检查点，单样本 microbatch、generation group 8、prompt 上限 256、completion 上限 256、学习率 `5e-5`、单卡 500 updates；每 50 步保存 checkpoint、保留全部正式 checkpoint。24GB 卡先运行 `--profile formal --max-steps 1 --max-completion-length 256 --prompt-token-limit 256` 单独测峰值；显存压力过大时按下方回退阶梯降低 completion/group。GSM8K pilot 在 RTX 3060 上曾有 completion 撞到 160 token 上限，因此正式 profile 必须检查截断率。不要按本地 12GB GPU 的实测吞吐量推断 AutoDL 速度。
4. 使用固定 revision 的 `openai/gsm8k` `main` 配置。数据卡将其描述为需要多步推理的问题并列出 MIT 许可（[数据卡](https://huggingface.co/datasets/openai/gsm8k/blob/740312add88f781978c0658806c59bc2815b9866/README.md)，[上游许可](https://github.com/openai/grade-school-math/blob/master/LICENSE)）。之前的官方test已用于pilot和100-step实验，不能再用于后续选模。下一轮继续使用固定的1,500条train与500条validation，但从官方train剩余、未用过的source rows中排除前两split后另取1,000条test；固定样本ID与数据hash，并按(source_split, source_index)验证无重叠。奖励只接受GSM8K `####` 后的确定整数答案。

## 正式运行与监控

```bash
~/.venvs/qwen3-grpo/bin/python scripts/audit_env.py --output reports/autodl_environment.json
~/.venvs/qwen3-grpo/bin/python scripts/prepare_gsm8k.py --train-count 1500 --validation-count 500 --test-count 1000 --test2-count 0 --test-source-split train --output-dir data/gsm8k-next-fresh --exclude-train-data data/gsm8k-formal/train.jsonl --exclude-train-data data/gsm8k-formal/validation.jsonl
~/.venvs/qwen3-grpo/bin/python scripts/validate_gsm8k.py --data-dir data/gsm8k-next-fresh --exclude-source-data data/gsm8k-formal/train.jsonl --exclude-source-data data/gsm8k-formal/validation.jsonl --output reports/gsm8k_next_fresh_data_validation.json
~/.venvs/qwen3-grpo/bin/python scripts/train_grpo.py --profile formal --max-steps 1 --max-completion-length 256 --prompt-token-limit 256 --train-data data/gsm8k-next-fresh/train.jsonl --run-dir runs/autodl-profile24
~/.venvs/qwen3-grpo/bin/python scripts/train_grpo.py --profile formal --max-steps 500 --save-steps 50 --max-completion-length 256 --prompt-token-limit 256 --train-data data/gsm8k-next-fresh/train.jsonl --run-dir runs/autodl-formal
```

每 100 步用单独进程在 validation 上从保存的 checkpoint 执行 `scripts/eval_math.py`，保存完整逐题输出、exact-match、解析率、token/s和显存峰值。按validation选择checkpoint；新test只对选定模型与基座各评一次，不能参与选模或奖励调节。训练reward只用于诊断，不能代替评测。报告记录训练开始前桌面/系统占用的整卡显存、训练全程`nvidia-smi`峰值、Torch allocator峰值、生成token/s、completion/s、step/s、平均reward、准确率、所有checkpoint路径和config/data/code/environment hash。

## 本地奖励消融与 token 预算

100步历史运行的800条completion里只有18条正确，且89/100个组没有任何正确输出；距离奖励让其中85个全错组仍出现组内方差。下一轮先在`data/gsm8k-next-fresh/`上固定group 8、LoRA、学习率、seed、prompt/completion上限和约8,603个生成completion token，对照`distance_shaped`与`exact_format`。训练在实际token计数越过预算时停止，报告真实token数、超额、整卡峰值显存、训练时间、吞吐、组内正确率和zero-variance比例。基座与各候选只在新validation上选择；test继续封存直到奖励和checkpoint均确定。只有通过本地消融后，才将被选配置迁移到24GB正式训练。

## 显存回退顺序

遇到 OOM 时保留失败日志与峰值记录，一次只调整一个维度：completion 256 → 224 → 192 → 160 → 128 → 96 → 64 → 48 → 32 → 24；然后 group 8 → 4 → 2，并将 gradient accumulation 同步为 group 大小；最后将 prompt token 上限逐级降低。NF4、LoRA、梯度检查点、no-vLLM 和单样本 microbatch 始终保留。正式训练报告必须标明最终实际成功的配置和每一次回退。
