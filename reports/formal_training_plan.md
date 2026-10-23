# AutoDL 单张 24GB GPU 正式训练迁移计划

> **当前状态（2026-10-23）**：本计划的数据步骤已被Phase C seed2结果取代，不能照旧命令启动。Phase C seed1选group8、seed2选group4，组大小结论为MIXED；seed2已将完整SVAMP 1,000题用于一次base/selected配对test。GSM8K官方train来源只余23条未用行，不足以再划500题validation。AutoDL训练保持暂停，直到找到并审计新validation与独立test；没有租用云端GPU。

## 迁移边界

本地已验证的主路径是单卡 Transformers生成、TRL GRPO、NF4 QLoRA和gradient checkpointing；继续保持`use_vllm=False`。奖励对照按validation规则选了`exact_format`，但对`distance_shaped`的差异不显著（paired p=0.547）。Phase C匹配生成token预算的两seed结果相反：seed1 validation选group8，seed2选group4；group8的prompt覆盖、更新数与group4不同，当前没有跨seed最优group-size结论。seed1独立GSM8K-train-source test上被选中的group8为354/1,000，base为99/1,000；它未比较group4的test表现。seed2选中group4在SVAMP外部test为675/1,000，base为267/1,000，但该test同样未比较group8，且adapter耗时约5.06倍。SVAMP完整1,000题已用于这次测试，不可再作为新选模或独立test。未来正式run暂不指定group4或group8；先完成Phase C审计、明确预算估计目标并锁定新数据。固定`Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`。GPU型号、驱动、CUDA runtime和可用显存必须以目标实例实际输出为准，不能把本机RTX3060数据带过去。

## 启动配置

1. 选择单张 24GB GPU 实例，记录产品型号、驱动、`nvidia-smi` 全量输出、Python、PyTorch CUDA、TRL 和 bitsandbytes 版本；不启用多卡、vLLM 或云端生成服务。
2. 用 `requirements-wsl2.lock.txt` 重建已验证的用户态依赖，并按实例驱动确认 CUDA wheel 可用；运行 `pip check`、`scripts/check_runtime.py`、`scripts/validate_gsm8k.py`。依赖版本或 CUDA wheel 有变动时，重新生成该实例的完整 lock 并保存审计结果。先跑小配置 smoke，再以正式配置做 1 步显存 profile，确认反向传播、非零 LoRA 更新、奖励日志和 checkpoint 保存均成功。
3. 正式配置框架候选：NF4 + double quant，LoRA `r=8, alpha=16`，梯度检查点，单样本 microbatch、prompt/completion上限256、学习率`5e-5`；正式训练步数与每50步checkpoint频率仍需由新数据和预算确定。**当前不预设group4或group8**；若后续实验选择其中一个，须在新validation上锁定，并在目标24GB卡单独profile峰值。显存压力过大时按下方回退阶梯降低completion/group。GSM8K pilot曾有completion撞到160 token上限，且当前评测未保存逐题finish reason；后续profile/evaluator需记录逐题生成token与停止原因。不要按本地12GB GPU的实测吞吐量推断AutoDL速度。
4. 使用固定 revision 的 `openai/gsm8k` `main` 配置。数据卡将其描述为需要多步推理的问题并列出 MIT 许可（[数据卡](https://huggingface.co/datasets/openai/gsm8k/blob/740312add88f781978c0658806c59bc2815b9866/README.md)，[上游许可](https://github.com/openai/grade-school-math/blob/master/LICENSE)）。官方test、Phase B fresh test、Phase C GSM8K test均已用于评测，不能复用。GSM8K官方train共有7,473行；在seed2新划500题validation后，仅剩23条未使用source rows，无法构成新validation。原计划中的SVAMP完整1,000题现已用于Phase C seed2 base/selected test，不能再作为后续正式训练的验证或测试集。新数据源尚未锁定，需在Phase C结论后另行确定一份未使用validation和独立test，固定revision/hash/license并复跑source/text/prediction overlap审计；在完成前不得启动正式训练。奖励仍只接受verifier可核验的整数答案。

SVAMP source pin used for the completed Phase C seed2 test (2026-10-23): repository commit `78e727689e1c1bebfc4be39c446898e8e10b0518`, `SVAMP.json` SHA-256 `5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e`, and `LICENSE` SHA-256 `07ce3a34de7a865345634f9278bc26ddcacc2d0defd348c28e3f4603448b9ba4`. All 1,000 rows have been allocated to the one-time paired test; this source pin is recorded for provenance, not future evaluation reuse.

## 正式运行与监控

```bash
~/.venvs/qwen3-grpo/bin/python scripts/audit_env.py --output reports/autodl_environment.json
# No data-generation or training command is executable until a fresh validation/test
# pair has been selected, audited, and locked in a new pre-registration receipt.
# After that gate, recreate the environment and profile the chosen group size with:
# ~/.venvs/qwen3-grpo/bin/python scripts/audit_env.py --output reports/autodl_environment.json
# ~/.venvs/qwen3-grpo/bin/python scripts/train_grpo.py --profile formal --max-steps 1 --max-completion-length 256 --prompt-token-limit 256 --reward-mode exact_format --train-data <locked-train.jsonl> --run-dir runs/autodl-profile24
# A paid instance still requires explicit user authorization.
```

新validation/test pair锁定后，先按100题profile测验证长度、completion token终止原因和吞吐，再锁定checkpoint评估计划。用单独进程保存逐题prompt、completion token数、停止原因、答案提取结果与准确率；validation只用于checkpoint选择，test只评估一次base/预选模型配对。训练reward只用于诊断，不能代替评测。报告记录训练前整卡显存基线、`nvidia-smi`峰值、Torch峰值、生成token/s、completion/s、step/s、平均reward、准确率、checkpoint路径和config/data/code/environment hash。

## 资源估算与费用上限

AutoDL 官方 GPU 价格页在 2026-10-23 查询列出 RTX 4090 24GB 按量价 ¥1.88/GPU小时、RTX 3090 24GB ¥1.32/GPU小时（[AutoDL 官方价格页](https://www.autodl.com/home)）。此前“完整轮次不超过2 GPU小时”的粗估已撤回：本机 RTX 3060 上，500题Phase C candidate validation耗时36.1–44.7分钟；1,000题group8 test耗时74.1分钟、base test耗时16.2分钟。AutoDL 4090 的推理、训练速度尚未实测，不能直接按型号线性外推。

正式计划仍将总计费时长硬封顶为4 GPU小时（按该报价GPU租用费最多¥7.52），并留出至少0.5小时停机缓冲；这只是费用上限，不是对全流程能在4小时内完成的承诺。创建前先跑100题端到端profile，按实测生成token长度与吞吐重新估算500步训练、计划中的validation checkpoint评测和最终配对test；只有保守估算不超过3.5 GPU小时才启动完整轮次。若超出预算，应先减少validation checkpoint数量并重新锁定选择计划，或取得用户对更高上限的明确批准。真正启动前再次核对当前价格、计费粒度和其他费用；当前没有创建或使用AutoDL实例。模型下载、数据盘和公网流量等非GPU费用另行核对，不得产生未授权费用。

## 本地奖励消融与 token 预算

100步历史运行的800条completion里只有18条正确，且89/100个组没有任何正确输出；距离奖励让其中85个全错组仍出现组内方差。后续`exact_format` validation相对distance-shaping差异不显著（paired p=0.547）。匹配约16k生成token预算的Phase C复现中，seed1选group8，seed2选group4；Seed 2 group8出现正确性混合组的比例更高，但prompt覆盖、更新数不同，尚不能确定其准确率或效率优势。当前AutoDL计划不默认group8或group4，需先收束Phase C、找到未使用的数据并重新锁定评测方案。

## 显存回退顺序

遇到 OOM 时保留失败日志与峰值记录，一次只调整一个维度：completion 256 → 224 → 192 → 160 → 128 → 96 → 64 → 48 → 32 → 24；然后 group 8 → 4 → 2，并将 gradient accumulation 同步为 group 大小；最后将 prompt token 上限逐级降低。NF4、LoRA、梯度检查点、no-vLLM 和单样本 microbatch 始终保留。正式训练报告必须标明最终实际成功的配置和每一次回退。
