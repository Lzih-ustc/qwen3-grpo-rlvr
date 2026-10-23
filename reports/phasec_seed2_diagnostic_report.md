# Phase C：第二随机种子复现与机制诊断

## 协议与数据隔离

Seed 2 的候选协议在训练和候选验证前锁定于`reports/gsm8k_phasec_seed2_plan.json`，锁定回执为`reports/gsm8k_phasec_seed2_plan_receipt.json`。训练使用固定的Qwen3-0.6B revision、`exact_format`奖励、学习率`5e-5`、prompt/completion上限256、NF4 LoRA、梯度检查点、`use_vllm=False`和16,000生成token目标。validation固定为500条此前未用的GSM8K官方train source rows；按最高exact-match选候选，test不参与选择。

独立测试固定为完整1,000题SVAMP，source commit与数据hash见`reports/svamp_holdout_audit.json`及`data/svamp/manifest.json`。SVAMP与本项目所有历史训练问题、历史模型预测的文本重叠审计通过。validation选择group4后，base与selected adapter各完成一次测试。SVAMP已视为已使用，不能再作为AutoDL模型选择或后续调参测试。本地重叠审计不能证明Qwen预训练阶段从未见过该公开数据集。

## 训练预算、吞吐和显存

| Seed 2候选 | 更新 | prompts / completions | 生成token | 训练秒 | token/s | completions/s | steps/s | Torch peak allocated/reserved | 整卡峰值 | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| group4 | 25 | 25 / 100 | 16,160 | 232.28 | 69.57 | 0.431 | 0.108 | 1,306 / 1,474 MiB | 4,070 MiB | `runs/gsm8k-phasec-seed2-g4-16k/attempt-01/trainer/checkpoint-25` |
| group8 | 13 | 13 / 104 | 16,415 | 150.20 | 109.29 | 0.692 | 0.087 | 1,702 / 1,910 MiB | 4,506 MiB | `runs/gsm8k-phasec-seed2-g8-16k/attempt-01/trainer/checkpoint-13` |

两组实际生成token相差1.58%，训练无OOM、无显存回退；LoRA权重确实变化，梯度范数非零。group8的吞吐更高、更新数与prompt覆盖约减半，峰值整卡显存比group4高436 MiB。token预算相同不意味着反向更新数、prompt数或总计算量相同。

## Validation与配对统计

Seed 2固定500题validation上，base为49/500（9.8%），group4为183/500（36.6%），group8为163/500（32.6%）。按预注册规则选择group4。逐题group4−group8差为+4.0个百分点，paired bootstrap 95% CI为[+0.2,+7.8]个百分点，精确McNemar p=0.0470；group4相对base差为+26.8个百分点，paired bootstrap 95% CI为[+22.4,+31.4]个百分点，精确McNemar p=3.33e-27。

| Seed | validation group4 | validation group8 | group4−group8差 | 配对bootstrap 95% CI | McNemar p | 预注册赢家 |
|---|---:|---:|---:|---:|---:|---|
| 1 | 139/500 (27.8%) | 176/500 (35.2%) | −7.4 pp | [−11.4, −3.6] pp | 0.000341 | group8 |
| 2 | 183/500 (36.6%) | 163/500 (32.6%) | +4.0 pp | [+0.2, +7.8] pp | 0.0470 | group4 |

两seed的validation胜者相反。bootstrap区间和McNemar检验描述的是固定题目集上的题目层面不确定性；两个训练seed不足以估计训练随机性，也不能据此称group4或group8稳定更优。Seed 1已有一次GSM8K train-source test：选中的group8为354/1,000，base为99/1,000；它没有比较group4的test表现。

## 组内奖励方差与有效正确性信号

`audit_grpo_run.py`从保存的逐completion parquet及Trainer日志重建每组的正确数、奖励方差、梯度和completion token。两种计数口径分别列出：

- **奖励有方差组**：组内奖励标准差大于0，代表GRPO有相对奖励差异可用；这不保证梯度一定改善数学能力。
- **正确性混合组**：同一组内至少一个completion正确、至少一个错误，代表本组明确对比了正确与错误答案。

| Seed / 组大小 | prompts / completions | 全错 / 全对 / 正误混合组 | 非零奖励方差组 | 零方差 / 零梯度 | 正误混合组比例 | 正误混合组token / 全部token | 每个正误混合组token |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 / group4 | 26 / 104 | 7 / 4 / 15（共26） | 24/26（92.3%） | 2 / 2 | 15/26（57.7%） | 9,507 / 16,097（59.1%） | 634 |
| 1 / group8 | 13 / 104 | 1 / 1 / 11（共13） | 13/13（100%） | 0 / 0 | 11/13（84.6%） | 12,979 / 16,354（79.4%） | 1,180 |
| 2 / group4 | 25 / 100 | 9 / 2 / 14（共25） | 19/25（76.0%） | 6 / 6 | 14/25（56.0%） | 7,745 / 16,160（47.9%） | 553 |
| 2 / group8 | 13 / 104 | 3 / 0 / 10（共13） | 11/13（84.6%） | 2 / 2 | 10/13（76.9%） | 11,664 / 16,415（71.1%） | 1,166 |

两个seed中group8都比group4有更高的正误混合组比例和更多token落在正误混合组上，说明大组更常在同一prompt内观察到正确与错误completion；但每个这样的group平均消耗约1.86–2.14倍completion token，而且prompt数/更新数约减半。绝对有效组数也并未增加：seed1为11对15，seed2为10对14。该信号差异跨seed方向相同，但validation准确率赢家反转，因此它尚不能解释最终性能，也不是对组大小因果效果的证明。

两seed都使用`exact_format`奖励：正确integer答案得到验证奖励，正确答案还可因XML格式得到小额bonus；错误但格式合规的整数也能得到0.05格式分，而无法解析输出得0分。因此，全错组仍可能仅因格式差异出现少量奖励方差。Seed 2无正确答案组的平均组内奖励标准差为0.0128（group4）和0.0081（group8），远低于含正确答案组的0.4053和0.4151。不能将所有非零方差组都解释成“学到数学正确性”。

独立Bernoulli正确率模型给出：若单次采样正确率为`p`，组大小`G`出现正误混合的概率为`1 − p^G − (1−p)^G`。例如`p=10%`时group4约34.4%、group8约57.0%；这是理论示例。实测混合组比例不同，原因包括题目难度不同、completion相关性、训练策略随更新变化，且两候选不覆盖完全相同的prompt集合。

### 保存训练样本的 prompt 覆盖交集（探索性）

为区分组大小信号与题目覆盖差异，`scripts/audit_phasec_prompt_overlap.py`从四次训练的 completion parquet 中提取每个更新的 prompt，并以训练 prompt 哈希比较同 seed 的候选。两臂的训练集 SHA-256 相同；两个 seed 中，group8 使用的13个 prompt 全部也出现在 group4，group4 另覆盖13题（seed1）和12题（seed2）。这是训练采样结果的事后描述，不是预注册的配对实验。

| Seed | 共享 prompt | 共享 prompt 上 G4 / G8 平均 exact match | G8−G4 | 正误混合组 G4 / G8 | 共享 prompt 生成 token G4 / G8 | 每个混合组 token G4 / G8 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 13/13个G8 prompt（Jaccard 0.50） | 34.6% / 35.6% | +1.0 pp | 6/13 / 11/13 | 7,006 / 16,354 | 1,168 / 1,487 |
| 2 | 13/13个G8 prompt（Jaccard 0.52） | 38.5% / 37.5% | −1.0 pp | 9/13 / 10/13 | 7,556 / 16,415 | 840 / 1,642 |

在共享13题上两 seed 的平均正确率差都只有约1个百分点，而且方向反转；Group8 两次都产生更多正误混合组，但其共享题 token 数约为 Group4 的2.17–2.33倍，而混合组数只增加1–5组。按此描述性口径，Group8在共享 prompt 上更常采到正确/错误并存的组，但每个混合组更耗 token；这没有证明其带来更有效的参数更新。每臂只有13个共享 prompt，采样completion彼此独立，未做显著性检验。结果不能消除更新数、prompt覆盖及训练seed的影响，也不能替代预注册的共享prompt、多seed实验。完整哈希与逐题统计保存在`reports/gsm8k_phasec_prompt_overlap.json`。

把全预算内“正确性混合组数 / 实际生成token × 1,000”作为信号密度代理，Group4在两个seed都高于Group8：

| Seed | Group4 mixed groups / 1k tokens | Group8 mixed groups / 1k tokens | G4 / G8 |
|---|---:|---:|---:|
| 1 | 0.9319 | 0.6726 | 1.39× |
| 2 | 0.8663 | 0.6092 | 1.42× |

这表明在已观测的约16k-token预算下，Group4用更多prompt覆盖换到了更多“至少含一条正确且一条错误completion”的组/生成token。它是可验证奖励信号的样本密度，不是有效梯度更新率：混合组内的梯度可能相互抵消或方向不一致，也未测量每次更新的反向计算成本。该指标只用于提出后续研究问题，不能单独支持Group4在模型质量或总计算上更优。数值由`audit_grpo_run.py`读取的保存completion与Trainer token日志重算，并加入四份rollout审计JSON。

## Base与adapter输出/解析审计

训练模板与独立评测均显式关闭Qwen3 thinking：训练端`chat_template_kwargs={"enable_thinking": false}`，评测端`enable_thinking=False`。评测使用相同prompt文本、greedy decode、`max_new_tokens=192`及相同整数答案提取器；seed1和seed2各自base与adapter也使用相同的seed和decode设置。逐题输出复解析后，所有保存的`predicted`、`correct`与解析器重算结果一致，没有记录层面的解析不一致；样本中没有可见`<think>`标签。

但严格exact-match的含义是“在192新token内生成可被integer verifier接受的最终答案”，不等同于纯粹的数学推理准确率。Seed1 GSM8K held-out test的base/选中group8结果为99/1,000与354/1,000，parse rate为72.7%与74.0%；parse source分布分别为base `answer_tag=643, boxed=84, unparsed=273`，adapter `answer_tag=441, boxed=299, unparsed=260`。base的accuracy差不能简单归因于总体parse rate（只差1.3 pp），但base有20.5%的输出含`<answer...`标记却未提取出唯一整数，adapter为4.8%。因此，输出格式和verifier可读性对部分题目有实际影响。

离线审计保存了40题的配对样本（每种配对正确性结果10题），不重新生成、不改写历史分数。样例中既有明确算术错误，也有语义上算出正确值、但把算式放进答案标签（例如`<answer>336 - 124 + 22</answer>`）或重复输出同一boxed答案，因严格唯一整数规则未计分；还有长推理在completion末尾停在答案之前的情况。早期评测json只保存总生成token数，没有逐题token数或finish reason，因而不能精确统计被192-token上限截断的样本比例。样本本身不能替代全量数学复核；主结果保留预注册的严格解析口径，不做事后修分。

seed2 base validation与选中group4 validation也进行了全量重解析及40题配对抽样，解析和`correct`标记一致。parse rate为75.6%与65.0%，exact-match为9.8%与36.6%；模型并非靠提高可解析率取得该validation结果，但严格指标仍会把没有完成、格式不合规或答案有歧义的输出计为错误。

## 一次性SVAMP外部测试

validation选中group4后，按预注册规则对完整1,000题SVAMP分别评估base与adapter一次。两边使用相同输入、greedy decoding、`max_new_tokens=192`、关闭thinking、seed 20261026和同一整数答案提取器；逐题配对与source hash均通过审计。

| 模型 | Exact match | Wilson 95% CI | 解析率 | 生成token / token/s | 耗时 | PyTorch allocated/reserved | 整卡start/peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 267/1,000（26.7%） | 24.1–29.5% | 83.3% | 35,598 / 48.75 | 730.28秒（12.2分钟） | 820/872 MiB | 3,655/4,518 MiB |
| 选中group4 adapter | 675/1,000（67.5%） | 64.5–70.3% | 88.6% | 104,482 / 28.26 | 3,697.65秒（61.6分钟） | 820/894 MiB | 3,553/4,743 MiB |

adapter−base exact-match差为+40.8个百分点，paired bootstrap 95% CI[+37.2,+44.4]个百分点，精确McNemar p=2.07e-85。两边都答对221题、仅base答对46题、仅adapter答对454题、都答错279题。Adapter生成token约为base的2.94倍，耗时约5.06倍，token/s较低；这不是效率提升。全量重解析一致；parse rate为83.3%与88.6%，parse source分别为base `answer_tag=758, boxed=75, unparsed=167`、adapter `answer_tag=344, boxed=542, unparsed=114`。带`<answer...`但未解析的输出分别154和24条。可解析率增加5.3个百分点，说明输出可解析性也有贡献，但不能单独解释40.8个百分点的exact-match差异。该测试只比较validation选中的group4和base，不比较group4与group8；不能消除两个训练seed中validation赢家相反的事实，也不能证明预训练未见过公开SVAMP题目。逐题配对报告、bootstrap与40题样本见`reports/svamp_phasec_seed2_final_test_audit.json`、`reports/svamp_phasec_seed2_test_bootstrap.json`和`reports/svamp_phasec_seed2_prediction_audit.json`。

审计代码、样本与统计分别见：

- `scripts/audit_saved_predictions.py`
- `scripts/audit_grpo_run.py`
- `scripts/paired_bootstrap.py`
- `reports/gsm8k_phasec_seed1_prediction_audit.json`
- `reports/gsm8k_phasec_seed2_validation_prediction_audit.json`
- `reports/gsm8k_phasec_seed1_test_bootstrap.json`
- `reports/gsm8k_phasec_group_validation_bootstrap.json`
- `reports/gsm8k_phasec_seed2_group_validation_bootstrap.json`
- `reports/gsm8k_phasec_seed2_selected_validation_bootstrap.json`
- `reports/gsm8k_phasec_g4_rollout_audit.json` / `reports/gsm8k_phasec_g8_rollout_audit.json`
- `reports/gsm8k_phasec_seed2_g4_rollout_audit.json` / `reports/gsm8k_phasec_seed2_g8_rollout_audit.json`
- `reports/svamp_phasec_seed2_final_test_audit.json`
- `reports/svamp_phasec_seed2_test_bootstrap.json`
- `reports/svamp_phasec_seed2_prediction_audit.json`
- `reports/svamp_phasec_seed2_base_test.json` / `reports/svamp_phasec_seed2_selected_test.json`

## 当前判断

**Phase C组大小比较：MIXED。** 固定生成token预算下，两个seed的validation赢家相反；group8虽然在两seed中显示更高的正误混合组比例和训练吞吐，却训练更新较少、覆盖prompt较少，seed2 validation低于group4。Seed 1的一次独立GSM8K test支持当次被选中的group8优于base，但没有在test上比较group4。Seed 2选中的group4在一次SVAMP test上为675/1,000，base为267/1,000；这支持本轮所选adapter在该数据和固定输出协议下优于base，但没有比较group4与group8，也不能消除训练seed间赢家相反的事实。

没有进入Phase D，也没有租用云端GPU。完成SVAMP后，Phase C优先做结论归档与实验设计，不立即追加训练。若将来继续，需先说明估计目标：固定生成token预算会让group4/group8的更新数和prompt覆盖不同；固定更新数则会让group8消耗更多生成token。下一轮应预注册预算口径、共享prompt顺序和多seed方案，并记录每条completion的准确生成token数及停止原因。AutoDL计划需重新选择未使用的validation/test数据；GSM8K train官方来源目前只剩23条未使用行，SVAMP也将在本轮后耗尽为独立测试用途。任何付费实例仍需用户明确授权。

## 最终归档补充：保存预测的 post-hoc 长度与成本分析

此段为 Phase C 冻结后新增的离线 **post-hoc** 分析。脚本从已保存的逐题 Base/selected-adapter JSON 读取 completion、`predicted`、`parse_source` 和原始配对正确性；没有重新推理、修复答案、覆盖预测文件、改动历史准确率或改变 test 标准。Qwen tokenizer 仅从本地缓存加载，重新 tokenize 已解码字符串；这不是原始生成 token ID。总生成 token、wall-clock 和 token/s 均仍采用原评测记录。详细逐题分组与 p50/p90 汇总见`reports/posthoc_prediction_length_audit.md`和同名JSON。

| 一次性独立测试 | Base / Adapter准确率 | Base / Adapter无法解析 | 平均重分词长度（正确输出，Base / Adapter） | 平均重分词长度（错误输出，Base / Adapter） | Base / Adapter总生成token | Base / Adapter耗时 |
|---|---:|---:|---:|---:|---:|---:|
| Seed1 GSM8K | 99/1,000 / 354/1,000 | 273 / 260 | 99.8 / 116.4 | 46.4 / 142.4 | 52,660 / 134,062 | 16.2 / 74.1 min |
| Seed2 SVAMP | 267/1,000 / 675/1,000 | 167 / 114 | 63.4 / 93.7 | 24.1 / 123.9 | 35,598 / 104,482 | 12.2 / 61.6 min |

Seed1逐题配对结果为两者都正确67题、base-only 32题、adapter-only 287题、两者都错614题；seed2为221、46、454、279题。Adapter在两份测试上都生成更多文本、使用更多总tokens且吞吐较低。尤其SVAMP上adapter正确与错误 completion 的平均重分词长度分别约93.7和123.9；较长输出与正确率提升同时出现，但较长本身不能解释答案质量。Seed1两者都错的问题中，adapter平均重分词长度约142.3、base约55.4，显示增加推理文本并不保证正确。不能将这些事后相关性解释为因果。

无法解析定义为保存的`predicted`为空，包含缺少标记答案或答案表达式不符合整数解析器的情况；它不等同于数学推理一定错误，也不允许人工补分。聚合“tokens / correct”只是每份测试总生成tokens除以正确题数的描述性比值，不是逐题推理成本。测试配对统计、准确率和候选选择仍以冻结的原始记录为准。
