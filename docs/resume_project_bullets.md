# 简历项目描述

## 中文版

**资源受限环境下的 Qwen3-0.6B GRPO / RLVR 后训练**｜Python、PyTorch、TRL、PEFT、bitsandbytes、WSL2

- 在 WSL2 + RTX 3060 12GB 上搭建可复现的 Qwen3-0.6B GRPO 训练与独立评测流程：固定模型 revision 和数据 hash，使用 NF4 QLoRA、gradient checkpointing、Transformers 原生 no-vLLM rollout、可验证数学奖励，并记录梯度/LoRA 更新、checkpoint、训练吞吐、推理吞吐和峰值显存。
- 完成两 seed、约 16k 生成 token 预算的 Group4/Group8 比较：候选 validation 赢家在两个 seed 间反转（seed1 Group8 176/500，seed2 Group4 183/500），因此将结论冻结为 **MIXED**；进一步从 saved rollout 分析奖励方差、正确性混合组、prompt 覆盖和每千 token 信号密度，避免把更高混合组比例误读为优化收益。
- 按 validation 选出的 adapter 在一次性独立测试上分别达到 GSM8K-train-source 354/1,000（Base 99/1,000）和 SVAMP 675/1,000（Base 267/1,000）；同时报告 adapter 推理 token 约为 Base 的 2.55×/2.94×、wall time 4.58×/5.06×，不宣称 group-size 胜负稳定或推理效率提升。
- 追查 100-update 负结果：正式测试 exact-match 7.24%，低于 Base 8.94%；发现 distance shaping 可让全错组仍出现奖励方差，随后实现 verifier-only reward 对照、token-budget stopping、逐题 paired bootstrap/McNemar 审计和结果 manifest。
- 实现 CPU-only 工具环境原型（限权 AST 计算器、只读 SQLite、固定任务、轨迹和调用成本计量），补充单元测试；明确记录 TRL/Transformers 原生 tool API 版本门槛，未声称完成 Agent RL 训练。

## English version

**Low-memory Qwen3-0.6B GRPO/RLVR post-training** | Python, PyTorch, TRL, PEFT, bitsandbytes, WSL2

- Built a reproducible single-GPU GRPO pipeline on a 12 GB RTX 3060 using NF4 LoRA, gradient checkpointing, native Transformers generation (no vLLM), verifier-based math rewards, independent evaluation, and explicit checkpoint/throughput/peak-memory accounting.
- Completed a two-seed, approximately 16k generated-token Group4/Group8 study. Validation winners reversed across seeds, so the archived result is **MIXED**; analyzed reward variance, correctness-mixed groups, prompt overlap, and verifier-signal density without claiming optimizer efficacy.
- The validation-selected adapters scored 354/1,000 vs Base 99/1,000 on one GSM8K-train-source test and 675/1,000 vs Base 267/1,000 on one SVAMP test, while using 2.55×/2.94× generation tokens and 4.58×/5.06× wall time. The tests do not compare Group4 vs Group8 directly.
- Diagnosed a negative 100-update run (7.24% vs 8.94% Base), implemented a verifier-only reward comparison and token-budget stopping, and added paired statistical, saved-output, and artifact-integrity audits.
