# Qwen3-0.6B 本地 GRPO 实验报告

执行时间：2026-10-22 UTC 至 2026-10-23 Asia/Shanghai。全部训练和评测在本地 WSL2 RTX 3060 上完成；没有租用或使用 AutoDL/其他云端 GPU。模型和公开数据文件下载到本地缓存。

## 结果摘要

- 最小正确性检查通过；1 步 smoke 完成真实反向传播与优化器更新。
- 12 步小训练完成，LoRA 权重发生可量化变化，保留了完整 Trainer checkpoint 和最终 adapter。
- 全程没有 OOM，因此未启用显存回退。
- 固定 48 题留出集上，基座 exact-match 为 24/48（50.0%），微调后为 32/48（66.7%），增加 16.7 个百分点。这只是小样本 sanity 结果，不足以证明稳定泛化提升。
- 另完成固定 GSM8K revision 的本地 RLVR pilot：validation 选中的 group-4 step-4 在独立 test2 子集为 33/128（25.78%），base 为 12/128（9.38%）。这是单 seed、小子集结果；输出 token 与推理耗时明显增加，尚不能视为稳定提升或效率优化。

## 环境审计

首次安装前，WSL2 能看到 RTX 3060，但 PyTorch、Transformers、TRL、PEFT、Accelerate 和 bitsandbytes 均未安装。最终训练环境如下：

| 项目 | 审计结果 |
|---|---|
| 系统 | WSL2 / Ubuntu 24.04.3 LTS，Linux kernel 6.6.87.2 |
| Python | 3.12.3；虚拟环境 `/home/zihao/.venvs/qwen3-grpo` |
| GPU | NVIDIA GeForce RTX 3060，12,288 MiB，compute capability 8.6 |
| NVIDIA driver | 537.58；`nvidia-smi` 报告 driver CUDA 12.2 |
| PyTorch | 2.5.1+cu121；PyTorch CUDA runtime 12.1；CUDA 可用，BF16 可用 |
| Transformers / TRL | 4.57.3 / 1.16.0 |
| PEFT / Accelerate | 0.21.2 / 1.15.0 |
| Datasets / bitsandbytes | 4.7.0 / 0.50.2 |
| NumPy | 2.5.4 |
| 最终 `pip check` | 无依赖冲突 |

开始训练前最后一次环境审计记录到 2,057 MiB 已用、10,066 MiB 可用；最初审计为 2,111 MiB 已用、10,012 MiB 可用。占用主要来自桌面显示进程。逐次训练的 GPU 峰值是 0.5 秒间隔 `nvidia-smi` 采样到的整卡数字，包含桌面/其他进程；另列 PyTorch allocator 指标，二者不能相互替代。

环境完整记录见 `environment_preinstall.json` 和 `environment.json`。

## 模型、数据与可验证奖励

- 模型：`Qwen/Qwen3-0.6B`，固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`。
- 训练数据：256 条固定种子的整数加减乘题；SHA-256：`9493e73bcd6eb59f281449308fd0626dea9e41a7d3f2fbdfca5f7163846df011`。
- 验证集：96 条题目；SHA-256：`016c7a94489a2827f3b22eebf82b64d242f9f9679391311e1993c0ba606f276d`。按固定种子生成，并在采样时排除 train 与 final eval 已出现的题目。
- 最终评测集：48 条独立种子题；SHA-256：`fef149b7c763de8edc4e75d470e7dba79424891547c7f910ae7d4888b8705107`。
- `validate_reward.py` 对已保存 JSONL 与确定性生成器逐行比对，train/validation/eval 三组 prompt 文本交集均为 0。训练集 256 行中有 252 个唯一 prompt（4 行重复），验证集和评测集内无重复；正式数据应去重训练样本。
- 奖励只从唯一 `<answer>整数</answer>` 或唯一 `\boxed{整数}` 解析最终答案；未标记答案或多个答案标记不给分。正确答案加 1.0，正确 XML 格式加 0.05，距离 shaping 最多加 0.1。
- 5 项奖励/解析单测全部通过；包含正确、错误、无格式、重复标签和 boxed 负整数。

## 训练配置与实测指标

两次训练都使用 NF4 4-bit 基座、double quant、BF16 计算、LoRA `r=8 / alpha=16 / dropout=0`，注入 q/k/v/o/gate/up/down 投影；启用非 reentrant gradient checkpointing，单卡 Transformers 原生生成，`use_vllm=False`。没有加载 reference model（`beta=0`）。

| 指标 | 正确性 smoke | 小规模训练 |
|---|---:|---:|
| Prompt / completion token 上限 | 96 / 32 | 128 / 64 |
| Generation group / 梯度累积 | 2 / 2 | 4 / 4 |
| 优化器更新 | 1 | 12 |
| 训练耗时 | 24.35 秒 | 43.95 秒 |
| 更新速度 | 0.041 step/s | 0.273 step/s |
| Completion 速度 | 0.082 completion/s | 1.092 completion/s |
| 实际生成 token / 速度 | 51 / 2.09 token/s | 1,111 / 25.28 token/s |
| 平均训练 reward | 0.575 | 0.766（每步均值的算术平均） |
| reward 范围 | 0.575 | 0.000–1.150 |
| 零组内 reward 方差 | 0/1 步 | 5/12 步 |
| PyTorch 峰值 allocated / reserved | 1,107 / 1,432 MiB | 1,107 / 1,432 MiB |
| 采样整卡 GPU 峰值 | 4,457 MiB | 4,491 MiB |
| OOM / 回退 | 无 / 无 | 无 / 无 |

小训练共有 5,046,272 个可训练 LoRA 参数，其中 4,170,816 个参数在最终模型中发生变化；LoRA 参数绝对更新量之和为 638.29。训练日志 12 步 loss、reward 和 grad norm 均为有限数；7 步 grad norm 大于零，5 步 grad norm 为零。后 5 步正好对应组内 reward 方差为零，因此这些步没有有效的组内相对偏好信号；报告该情况供解释训练 reward 和后续正式配置时参考。

## Validation 选 checkpoint

先固定规则：按 validation exact-match accuracy 选 checkpoint；并列时选更早的 checkpoint。使用独立进程从 checkpoint 目录重载 adapter，在同一个 96 题 validation 集上贪心评测：

| Checkpoint | 正确题数 | Accuracy | 解析率 | 耗时 / 吞吐 |
|---|---:|---:|---:|---:|
| step 8 | 63/96 | 65.63% | 68.75% | 87.15 秒 / 29.28 token/s |
| step 12 | 65/96 | 67.71% | 69.79% | 68.58 秒 / 26.51 token/s |

因此选择 `checkpoint-12`。该选择只依据 validation；`quality_gate.json` 记录了候选、规则、hash 和 selected path。最终评测集 SHA 与 validation 不同，也没有用于选择模型。最终 adapter 的权重与 step 12 checkpoint 权重 SHA-256 相同。

## 独立留出评测

评测由独立进程加载基座或已保存 adapter，使用 greedy decoding、`max_new_tokens=48`、关闭 thinking；基座和 12 步模型使用同一最终评测文件及 seed。step 8/12 checkpoint 选择另用独立 validation 集。

| 模型 | Exact match | 解析率 | 耗时 / 生成吞吐 | PyTorch 峰值 allocated / reserved | 整卡采样峰值 |
|---|---:|---:|---:|---:|---:|
| 基座 | 24/48（50.0%） | 25/48（52.1%） | 41.92 秒 / 32.92 token/s | 820 / 846 MiB | 4,116 MiB |
| 12 步 adapter | 32/48（66.7%） | 32/48（66.7%） | 40.31 秒 / 22.03 token/s | 820 / 862 MiB | 4,509 MiB |

小评测集存在较大统计波动，评测准确率不能替代正式数学基准，也不应用训练 reward 选取正式模型。逐题预测、解析结果和准确率都保存在评测 JSON 中。测试结果显示本模型小样本 exact-match 上升，但这并不能证明通用数学推理能力提升。

## GSM8K 多步数学 RLVR pilot

算术实验验证了训练接口，但任务过于简单。后续 pilot 改用固定 revision 的 [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k/blob/740312add88f781978c0658806c59bc2815b9866/README.md)：官方数据卡将其描述为需要多步推理的年级数学题并列出 MIT 许可；上游 MIT 文本保存在 `data/gsm8k/LICENSE`。本地划分由固定样本种子生成，最终答案严格解析自来源答案末尾 `#### integer`，不符合整数格式的数据会使准备脚本失败。

| Split | 数量 | SHA-256 | 用途 |
|---|---:|---|---|
| train | 256 | `61cf86d39c754b0867b91547d8445afbcfe48a78dfd085d364f7ba6143c640bd` | GRPO |
| validation | 96 | `198be3c7a638cbbf6f71df4469dd30df5653d974ebf072dcac691ea04434fe16` | 选择 checkpoint |
| test（slice 1） | 128 | `ee7c56ff8834669a354c5b4eda69fdca8741471e0800872d82974fa64ca30ca8` | 初轮开发诊断；之后用于比较采样组，不作最终留出结论 |
| test2（slice 2） | 128 | `d422c8ecffd4aefcbcc8ded6de47b4d05970470da56cfaa028b6b7d3faf86350` | 最终 pilot 对照，选择 checkpoint 前未查看 |

四个 split 的题目文本均不重叠，逐行 JSONL hash 与 pinned manifest 匹配。训练配置都使用同一 Qwen3 revision、NF4 QLoRA、BF16、LoRA r=8/alpha=16、gradient checkpointing、`use_vllm=False`、学习率 `5e-5`、prompt 上限192和 completion 上限160。唯一主要变量为 GRPO group 与更新数；两组各采样16条 completion，但不是相同 token 或时间预算，因此这只是采样组探索，不是公平的效率消融。

| 训练条件 | Updates / group | Completion 数 / token 数 | train 时间 | 平均 reward / 平均组内 std | 被截断更新 | PyTorch allocated/reserved 峰值 | 整卡采样峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| group 2 | 8 / 2 | 16 / 304 | 24.08 秒 | 0.05296 / 0.01094 | step 2 有部分截断 | 1,117/1,432 MiB | 4,193 MiB |
| group 4 | 4 / 4 | 16 / 1,225 | 40.58 秒 | 0.24082 / 0.29572 | step 4 的4条全到160-token上限 | 1,144/1,432 MiB | 4,132 MiB |

两组都完成真实 LoRA 更新、日志数值有限、保留 optimizer/RNG/Trainer checkpoint，且无 OOM。按 validation exact-match 最高者从四个候选 checkpoint 中选出 group-4 step-4：base 为8/96（8.33%），group-2 step-4/8 均为4/96（4.17%），group-4 step-2 为14/96（14.58%），group-4 step-4 为29/96（30.21%）。所有候选都在同一 validation hash 上由独立进程重载评测；选择规则和实际结果见 `gsm8k_checkpoint_selection.json`。

在选择完成后，对未参与选模的 test2 做 base 与所选 checkpoint 的同协议对照（greedy，max_new_tokens=192，thinking 关闭）：

| 模型 | Exact match | 解析率 | 输出 token / token/s | 耗时 | PyTorch allocated/reserved | 整卡采样峰值 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 12/128（9.38%，95% Wilson CI 5.44–15.67%） | 78.13% | 5,188 / 49.30 | 105.23 秒 | 820/878 MiB | 3,520 MiB |
| 选中 group-4 step-4 | 33/128（25.78%，95% Wilson CI 18.99–33.99%） | 69.53% | 15,455 / 29.25 | 528.31 秒 | 820/900 MiB | 3,669 MiB |

test2 上两个模型对同一批题逐题配对：7 题都答对、5 题 base-only 正确、26 题 selected-only 正确、90 题都错；精确 McNemar 双侧 p=0.00019。该结果支持“这次固定采样下准确率上升”的描述，但只有128题、单 seed，而且所选模型生成约3倍 token、推理时间约5倍，解析率也低8.6个百分点；不应据此声称已获得普遍、稳定的 GSM8K 提升或训练加速。

首个 test slice 的早期开发诊断为 base 13/128、group-2 step-4 4/128；看到该结果后才探索 group 4。它不用于最终结论。test2 在 group-4 checkpoint 选择前保持未查看。训练 step-4 的 completion 截断率为100%，下一轮须先扩大长度预算并测显存/截断，然后用更多 updates 和新 seed 重复；另需给效率比较匹配生成 token 或 GPU 时间预算。

## Formal split 长度与组大小 profile

为决定本机12GB显存上的下一轮配置，在正式 split 的训练数据上分别做了1步 group-4 与 group-8 profile，seed `20261023`，prompt/completion 上限均为256，并保存了完整 checkpoint。profile 没有用于模型选取或能力结论；两次只各观察一个随机 prompt batch，completion 实际都较短，不能代表最坏长度或稳定吞吐。

| Group | Completion 数 / 实际 token | 更新耗时 / token 吞吐 | reward mean / std | PyTorch allocated/reserved | 整卡采样峰值 | OOM / 截断 |
|---|---:|---:|---:|---:|---:|---|
| 4 | 4 / 45 | 9.38秒 / 4.80 token/s | 0.05447 / 0.00231 | 1,107/1,432 MiB | 3,774 MiB | 无 / 0% |
| 8 | 8 / 74 | 12.73秒 / 5.81 token/s | 0.05592 / 0.00492 | 1,107/1,432 MiB | 3,769 MiB | 无 / 0% |

group-8比group-4单步生成更多token且用时增加，但不同completion数、短样本和不同采样结果使这一比较不能说明质量或效率优劣。随后以group-8、256-token completion上限完成100步扩展pilot；训练与选模结果见下节。两个profile运行记录见 `runs/profile-g4-c256/` 和 `runs/profile-g8-c256/`。

## Formal split 的100步扩展pilot

在固定正式训练split上完成group-8、100次optimizer update（seed `20261023`，学习率 `5e-5`，prompt/completion上限256）。运行记录的Git commit为 `c0c36832c2aa95d89ddcf84aff5e13f14ea1d871`；数据hash、配置、完整 reward/gradient history、completion parquet及checkpoint均保存在 `runs/gsm8k-formal-g8-100/`。

| 指标 | 结果 |
|---|---:|
| Optimizer updates / rollout group | 100 / 8 |
| Completion 数 / 生成 token 数 | 800 / 8,603 |
| 训练耗时 / step/s / completion/s / token/s | 271.21秒 / 0.369 / 2.950 / 31.72 |
| 平均训练 reward | 0.07811 |
| 零组内 reward 方差 / 零 gradient step | 4/100 / 4/100 |
| completion 截断 | 0/100 步 |
| 发生变化的 LoRA 参数 | 4,183,041 / 5,046,272 |
| PyTorch 峰值 allocated / reserved | 1,435 / 1,512 MiB |
| 整卡采样峰值 | 4,136 MiB |
| OOM / fallback | 无 / 无 |

validation按预先固定的exact-match规则选择step50与step100中更好的step100；但它仍低于同一validation上的基座：基座46/500（9.2%），step50为9/500（1.8%），step100为24/500（4.8%）。adapter解析率提高至97.6%，基座为74.8%，但格式改善没有转化为答案正确率。该运行因此是负向的扩展诊断，不支持能力提升结论。对训练parquet离线审计发现800条completion中只有18条精确正确（2.25%），754条能解析（94.25%），平均错误答案绝对距离为342.9；100个group中只有4组reward完全相同，对应4个零gradient步骤。零方差只占4%，不能单独解释准确率下降；大量错误但格式正确、距离很远的答案仍获得约0.05格式分和很弱的距离分，提示策略主要学会格式而未学会求解。step50/100均由独立进程重载并在相同500题、max-new-tokens192评测。

进一步按组统计，只有11/100组出现过正确答案，另有89组完全没有正确答案；其中仅4组reward方差为零，85组虽然全错仍因数值距离不同而产生非零方差。无正确答案组的平均reward标准差为0.00764，而出现正确答案组为0.416。当前`scale_rewards="group"`会将每组reward按组内统计量缩放，因此微弱的错误答案距离差异也可能为这85组提供相对优势。这是无效更新的一个可能来源。

据此新增`exact_format`奖励消融：正确整数得1分，唯一`<answer>`格式另加0.05分，错误整数不再按距离分层；原有奖励保留为`distance_shaped`对照。训练脚本还新增按真实生成completion token预算停止的回调，用于将两个条件配到约16,000个生成token并记录组边界造成的实际超额。新奖励已通过独立单元案例；两种奖励在新鲜split上的训练对照仍待执行，正式test保持封存。

新鲜split上尚未训练前的基座验证已完成：54/500（10.8%）精确正确，解析率76.0%，生成21,635 token，耗时396.45秒（54.57 token/s），PyTorch峰值allocated/reserved为820/882 MiB，采样整卡峰值3,481 MiB。新测试集仍未运行模型。以`token-budget-smoke`验证停止回调：目标1 token时完成1次真实optimizer update、8条completion、274 token，LoRA参数变化2,748,622个，step-1 checkpoint已保存；训练耗时15.24秒，峰值整卡3,940 MiB。随后用`terminal-checkpoint-smoke`验证预算在保存间隔前触发时也保存完整终态checkpoint：目标1、实际222 token、1次更新、8条completion，LoRA变化2,751,773个，训练10.05秒，22.08 token/s，峰值allocated/reserved 1,217/1,432 MiB、整卡3,920 MiB。checkpoint-1含optimizer/scheduler/RNG/Trainer状态，已从独立进程重新加载并在1条临时validation样本上成功生成。以上只验证工程链路，不作为性能结果。

### Fresh split 16k-token 距离奖励 pilot

为观察新split的生成长度和训练预算回调，先在group-8、256 completion上限下运行`distance_shaped`小pilot。运行12次更新，抽到12个不同训练prompt；96条completion生成17,502 token，较16,000目标超出1,502（9.4%）。训练耗时155.71秒，112.40 token/s；PyTorch峰值allocated/reserved为1,638/1,834 MiB，整卡峰值4,430 MiB，无OOM/回退。

这12个训练prompt上有16/96条精确正确（16.7%），解析率47.9%，XML答案标签率18.8%；这是训练rollout的样本内诊断，不能和500题基座validation直接比较。8/12组至少有一条completion被截断，平均组内截断比例31.25%（估算约30/96条）；3/12组reward方差及梯度为零。错误且可解析答案的平均绝对距离为49,588。该pilot因此不能作为策略效果结论，也提示256-token上限和旧距离奖励都可能产生大量无效样本。因为token预算在第25步保存间隔之前触发，首个pilot只保存final adapter；训练代码随后补上预算停止时保存完整终态Trainer checkpoint。逐条rollout见`reports/gsm8k_next_fresh_distance16k_rollout_audit.json`。

在完全相同的train split、seed、group 8、学习率、256-token长度上限和16,000-token目标下，完成了`exact_format`奖励对照。距离奖励组为12次更新/96条completion/17,502 token，训练155.71秒、112.40 token/s、整卡峰值4,430 MiB；exact-format组为13次更新/104条completion/17,272 token，训练159.52秒、108.28 token/s、整卡峰值4,384 MiB。两组均无OOM，预算超额分别为9.4%和8.0%。训练rollout精确正确率为16.7%（16/96）与20.2%（21/104），解析率47.9%与58.7%，平均completion截断比例31.25%与20.19%；两组都各有3个零方差/零梯度更新。以上是13个prompt左右的样本内诊断，不能说明validation泛化或奖励改进。距离组只有final adapter，没有完整终态resume checkpoint；exact-format组的checkpoint-13含完整状态。validation候选路径与选模规则已在`reports/gsm8k_next_fresh_reward_selection_plan.json`预先锁定，validation结果尚未运行；新test仍封存。

step100按validation选定后，基座和adapter各自在完整1,063题正式test上贪心评测一次（test SHA-256=`306c5d08ad76236c03c24a7fb60160453fc6e99161c988721893662e5e4f838f`）：

| 模型 | Exact match | 解析率 | 输出token / token/s | 耗时 | PyTorch峰值allocated/reserved | 整卡采样峰值 |
|---|---:|---:|---:|---:|---:|---:|
| 基座 | 95/1,063（8.94%，Wilson 95% CI 7.37–10.80%） | 75.63% | 52,137 / 54.84 | 950.73秒 | 820/888 MiB | 3,490 MiB |
| step100 adapter | 77/1,063（7.24%，Wilson 95% CI 5.83–8.96%） | 98.12% | 11,371 / 29.25 | 388.79秒 | 820/884 MiB | 3,479 MiB |

配对题目中两者都正确14题、仅基座正确81题、仅adapter正确63题、两者都错905题；exact McNemar双侧p=0.156。adapter少生成78.2% token、耗时少59.1%，但准确率低1.69个百分点，且token/s更低；这不是维持任务成功率前提下的成本优化。该正式test已经使用，后续新候选不能再用它调参或选模；需要从尚未查看的GSM8K来源行重新留出测试集，或使用新的独立基准。

## Checkpoint 与复现材料

小训练保留的 Trainer checkpoint：

- `runs/local-small/attempt-01/trainer/checkpoint-8`
- `runs/local-small/attempt-01/trainer/checkpoint-12`
- 最终 adapter：`runs/local-small/attempt-01/final_adapter`

checkpoint 保存 adapter、optimizer、scheduler、RNG 状态和 Trainer 状态，可供续训；评测直接使用最终 adapter。第 12 步 adapter 权重 SHA-256 为 `9fd7d119a81c377bbe752b2b7745648508489c289c4431b51eae4bcd15f5e7d7`；第 8 步权重 SHA-256 为 `c92b2a5e09af8f7e5dfd6d0af261c5104ffaf6a4753a6be02a84b984f303e578`。

完整文件大小与哈希清单在 `artifact_manifest.json`。训练依赖锁 `requirements-wsl2.lock.txt` 的 SHA-256 为 `9b220140188d5ea20bd457a5fffb04b00a3c1f2ce0c528aeabbcd0306a8665d6`。成功运行 summary 另记录模型 revision、数据/代码 hash、优化器更新数、reward log、显存峰值和 checkpoint 路径。

可在 WSL Ubuntu 工作区中用新目录重跑：

```bash
cd /mnt/e/强化学习项目
~/.venvs/qwen3-grpo/bin/python scripts/audit_env.py --output reports/environment.json
~/.venvs/qwen3-grpo/bin/python scripts/check_runtime.py
~/.venvs/qwen3-grpo/bin/python scripts/validate_reward.py
~/.venvs/qwen3-grpo/bin/python scripts/train_grpo.py --profile smoke --run-dir runs/smoke-repro
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --tag base --output reports/eval_base.json
~/.venvs/qwen3-grpo/bin/python scripts/train_grpo.py --profile small --run-dir runs/local-small-repro
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --adapter runs/local-small-repro/attempt-01/trainer/checkpoint-8 --eval-data data/validation.jsonl --tag step8-validation --output reports/validation_step8_repro.json
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --adapter runs/local-small-repro/attempt-01/trainer/checkpoint-12 --eval-data data/validation.jsonl --tag step12-validation --output reports/validation_step12_repro.json
~/.venvs/qwen3-grpo/bin/python scripts/eval_math.py --adapter runs/local-small-repro/attempt-01/final_adapter --tag local-small-repro --output reports/eval_local_small.json
~/.venvs/qwen3-grpo/bin/python scripts/audit_run.py --smoke-run runs/smoke-repro --small-run runs/local-small-repro --step8-validation reports/validation_step8_repro.json --step12-validation reports/validation_step12_repro.json --output-dir reports/repro
```

首次运行需从 Hugging Face 下载模型文件；以上命令的模型推理、反向传播和评测都在本机 GPU 执行。输出目录必须为空，复跑时使用新的目录名。

## 初次兼容性问题

第一次 smoke 暴露已安装 TRL `GRPOConfig` 不接受 `use_cache` 参数。该项已从 Trainer 配置移除，模型 config 保持 `use_cache=False`；修正后的 smoke 完成了有效参数更新。对应失败记录留在 `runs/smoke/attempt-01/failure.json`，成功记录留在 `runs/smoke-v2/run_summary.json`。另一次较新的 Transformers 与当前 PyTorch 组合不兼容，最终固定使用本报告列出的版本并由 `pip check` 验证。

## AutoDL 单张 24GB GPU 正式训练计划

详细迁移步骤见 `formal_training_plan.md`。核心顺序：在 AutoDL 记录实际 GPU/driver/CUDA，按驱动重建锁定环境并通过 reward/runtime 检查；运行 1 步小 smoke，再以正式长度/group 做 1 步 profile；稳定后使用单卡 NF4 QLoRA、gradient checkpointing、no-vLLM、group 8、prompt/completion 上限 256、500 optimizer updates、每 50 步保存。使用 1,500 条训练、500 条验证和 1,063 条独立留出题，每 100 步单独评测；保留全部 checkpoint、环境和数据 hash、吞吐、reward、准确率及显存曲线。OOM 时按计划逐项缩短 completion、降低 generation group，再降低 prompt 上限。

本机 12GB 显存上小训练的整卡峰值为 4,491 MiB，但 AutoDL 型号、桌面占用、驱动和吞吐都需在目标实例重新实测；不能根据这次结果线性外推。

`data/gsm8k-formal/`中的官方test已在step100选模后用于一次基座/adapter对照并视为已消耗。后续AutoDL迁移计划改为从官方train split剩余行中排除已用的1,500 train与500 validation source indices，另生成1,000条新test；新代码支持按source split/source index校验无重叠。更长的500步AutoDL正式训练尚未启动，也没有使用AutoDL/GPU云服务。

下一轮新split已准备在 `data/gsm8k-next-fresh/`：从官方train剩余source rows中排除之前正式实验的2,000个source indices，再新采样1,500 train、500 validation和1,000 test。新validation/test与此前分配和彼此均无重叠，校验结果在 `reports/gsm8k_next_fresh_data_validation.json`；该新test尚未用于模型推理。

## 上游接口依据

- [TRL GRPOTrainer 文档](https://huggingface.co/docs/trl/grpo_trainer)
- [Qwen3-0.6B 固定 revision 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)
- [Qwen3 模型卡与 thinking 选项](https://huggingface.co/Qwen/Qwen3-0.6B)
