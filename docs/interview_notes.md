# GRPO 项目面试技术问答

## 30 秒项目介绍

我在 WSL2 的 RTX 3060 12GB 上实现了 Qwen3-0.6B 数学 GRPO 后训练与独立评测流程，用 NF4 QLoRA、梯度检查点和 Transformers 原生 no-vLLM rollout。项目固定模型/数据版本，使用可验证答案奖励，记录真实梯度更新、checkpoint、吞吐和峰值显存。Phase C 完成了两 seed 的 Group4/Group8 固定生成 token 预算比较；validation 胜者在两个 seed 间反转，因此结论是 MIXED。我还做了保存预测的 post-hoc 长度/解析/推理成本分析，但没有重算原始准确率。

## GRPO 是怎么更新策略的？

对同一 prompt 采样 `G` 个 completion，verifier 分别计算 `r_i`，组内优势为：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + epsilon)
```

训练使用 TRL 的 clipped GRPO objective，不额外训练 PPO 的 value/critic 网络。本配置将组内 reward 标准化；若组内奖励完全相同，组没有相对优势信号。但“reward 有方差”仍不表示它一定区分数学正确性：格式奖励也可能让两个错误答案获得不同 reward。

基座以 NF4 double quant 加载，计算使用 BF16，LoRA `r=8, alpha=16`，开启 gradient checkpointing；生成走 Transformers 原生路径，`use_vllm=False`，训练和评测都关闭 thinking mode。这样可以在 12GB 显存完成真实更新，同时保持后端简单；no-vLLM 会牺牲一部分生成吞吐。

## 奖励为什么要从 distance shaping 改成 verifier-only？

历史 `distance_shaped` 奖励对错误整数也按数值距离打分。100-update 运行里，800 条 completion 只有18条精确正确，100个组里89个没有正确 completion，但其中85个仍有非零奖励方差。这说明弱的错误答案差异可能被组内标准化变成相对信号，reward 数值可计算不等于它对目标有帮助。

新增 `exact_format`：正确整数得1，正确且唯一 XML 格式可以额外得0.05；错误但格式合规仅得0.05；无法解析得0。Fresh validation 上 exact-format 190/500，distance-shaped 183/500，paired McNemar p=0.547；这没有证明 exact-format 更优。实验按预先锁定的最高准确率规则选 exact-format，但不作奖励算法优越的结论。

## Group4 与 Group8 怎样比较？

每个候选都以约16k实际生成 completion token 为停止目标，固定模型、奖励和 prompt/completion 上限。增加组大小可能提高单个组包含正确与错误样本的机会，但固定生成预算下会减少 prompt 数和 optimizer update。实验事实如下：

| Seed | Group4 validation | Group8 validation | Winner |
|---|---:|---:|---|
| 1 | 139/500 (27.8%) | 176/500 (35.2%) | Group8 |
| 2 | 183/500 (36.6%) | 163/500 (32.6%) | Group4 |

实际生成量只差约1.6%，但 Group4 更新25–26次、覆盖25–26个prompt，Group8更新13次、覆盖13个prompt。Group8 的 correctness-mixed group 比例更高，两个 seed 都如此；但是绝对混合组数更少，每千 token 的混合组密度也更低，validation 胜者仍反转。因此结论是 **MIXED**，不能说 Group8 稳定更好，也不能把 token-matched 当作总计算量匹配。

## 独立测试能支持什么结论？

Seed1 validation 选出的 Group8 在一次性 GSM8K-train-source test 上是354/1,000，Base 是99/1,000；seed2选出的 Group4 在一次性 SVAMP test 上是675/1,000，Base 是267/1,000。这些数据说明各自被选中的 adapter 在相应 test 与固定 verifier 下高于 Base。它们不比较 Group4 与 Group8 的测试成绩，也不能证明跨 seed 稳定，更不能把效果归因到组大小或 reward 模式。SVAMP是公开多年数据，重叠审计也不能证明基础模型预训练未见过它。

逐题配对的 McNemar 和 bootstrap 区间衡量固定题集上的问题级差异；两次训练 seed 不足以估计训练随机性。该区分很重要：大量题目不能代替多个独立训练 seed。

## 为什么准确率提高但推理反而更慢？

模型生成了更多内容。Seed1 adapter 比 Base 多生成2.55倍 token、耗时4.58倍；seed2多生成2.94倍 token、耗时5.06倍，token/s也更低。因此是本轮质量与推理成本的折衷，不是效率提升。post-hoc 分析显示错误 adapter 输出仍很长：重分词后平均长度在 seed1 为142.4 tokens、seed2为123.9 tokens，均高于对应正确 adapter 输出116.4和93.7。相关性不能说明更长推理导致错误或正确。

测试的 parse failure 也有变化：GSM8K上 Base/adapter 分别273/260题无法解析，SVAMP分别167/114题。主准确率仍按保存的固定 verifier 计分；即使人工能从表达式或文字推断答案，也不能事后修正历史 exact-match。Post-hoc 字符、词数和 tokenizer 统计只做诊断，且重分词不是原始生成 token ID。

## 负结果怎么处理？

100-step formal selected adapter 的 GSM8K test exact-match 为77/1,063（7.24%），低于 Base 的95/1,063（8.94%）；我保留这一结论，而没有用格式解析率或推理成本改善替代数学准确率。后续新增 verifier-only 奖励对照，并定位了历史距离 shaping 的全错组方差问题。这些证据支持继续审计奖励与采样预算，不支持直接扩大同一个训练配方。

## 下一步做什么？

Phase C 已封存，当前没有新的 GPU 训练计划。CPU-only Agent 环境原型已实现固定任务、受限 AST 计算器、只读 SQLite、逐步轨迹和工具/token/延迟计量；但没有连接模型或训练策略。TRL 1.16.0 与 Transformers 4.57.3 的 pinned 组合会拒绝原生 `tools`/`environment_factory` 版本要求，experimental `rollout_func` 还没有实现或通过真实更新验证。

AutoDL 正式训练也暂停：GSM8K官方 train 仅剩23条未用 source rows，不够构造 validation；SVAMP已用于 Phase C seed2 test。后续需先确定新的数据源、协议和预算，再讨论训练。没有租用云端 GPU 或启动付费服务。

## 常见追问

**为什么不用 Group8 就结束？** 两个 seed 的 validation 赢家相反。更大的组提升了 correctness-mixed group 比例，但固定 token 预算下覆盖更少 prompt、更新更少，每千 token 的混合组密度反而低。现在没有稳定胜者。

**Test 上 p 值很小，是否说明算法有效？** p 值针对已锁定题目集中的配对模型正确性差异，不估算跨 seed 波动，也不比较两个 group size 的 test 表现。模型选自 validation；test结果属于本轮所选模型对 Base 的一次性比较，解释范围有限。

**为什么不人工修正答案解析？** 解析规则是训练和评测协议的一部分。事后挑选答案格式修正会改变 outcome 定义并引入观察结果后的标准变化。可以另做明确标注的人工语义错误分析，但不能覆盖主要准确率。

**项目的工程产出是什么？** 可审计的低显存 GRPO 更新、固定 split/hash、checkpoint 选择与重载、独立逐题评测、配对统计、rollout 奖励组诊断、GPU与推理成本计量、离线 post-hoc 报告以及 CPU-only tool environment runner。
