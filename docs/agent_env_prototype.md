# CPU-only 工具环境原型

这个原型只为未来的工具使用 Agent RL 准备确定性的任务接口和评测契约；它不加载语言模型、不训练策略、不连接网络，也不使用 GPU。Phase D 的 Agent RL 训练仍未启动。

## 任务与动作

`agent_rl.environment.ToolEnvironment` 提供两个固定任务：

- `arithmetic_01`：通过受限算术工具计算 `(18 * 7) - 25`。
- `orders_tax_01`：从固定 SQLite 销售表查询 North 区已付款金额，再用算术工具计算加税后的美元金额。

调用 `reset(task_id, seed=...)` 开始新回合。`seed` 会记录在回合指标中；当前任务数据是固定 fixture，不随 seed 改变。动作使用简单映射：

```python
{"type": "tool_call", "tool": "python_calculate", "input": {"expression": "(18 * 7) - 25"}}
{"type": "tool_call", "tool": "sqlite_query", "input": {"query": "SELECT ..."}}
{"type": "submit", "answer": "101"}
```

只有最终答案由 Decimal 数值校验器判定，正确得 1.0，其他提交得 0.0。工具过程奖励为 0；回合会限制最大动作数和工具调用数。`step` 返回 observation、奖励、terminated、truncated 和计数器信息。

`agent_rl.runner` 提供不依赖 TRL 的回合执行器，policy 每步返回 JSON action 文本及可选的真实生成 token 数和生成耗时。它汇总任务成功率、成功任务平均工具调用尝试数、无效工具调用率、工具错误、输出 token 数和端到端延迟，并保留逐步轨迹。工具调用尝试数包含无效调用；无效率以全部工具调用尝试为分母。缺少 tokenizer token 计数时，报告保留 `null` 和覆盖率，不从文本长度推算 token。

## 安全边界与可复现性

算术工具用 Python AST 解析，仅接受数字、括号、正负号和基本算术运算；它不调用 `eval`，并限制表达式长度、指数及结果范围。SQL 工具只运行单条只读查询，SQLite authorizer 只允许读取当前任务表和受限聚合/字符串函数；结果最多返回100行，虚拟机指令数也有上限。SQLite 数据库在每次 reset 时从固定 fixture 重建。

原型是一个最小接口示范，不是生产级沙箱。当前 JSON action runner 能验证工具交互和计量契约，但尚未定义自然语言任务生成、模型调用后端、GRPO 训练数据格式、策略优化、对抗性安全审计或跨任务泛化评测。后续扩展应继续使用隔离进程/容器运行不可信工具代码；当前计算器并不执行 Python 代码。

## 固定依赖的 TRL 接口检查

本地 WSL 锁定环境为 `trl==1.16.0`、`transformers==4.57.3`。安装的 `GRPOTrainer` 签名确实包含 `tools`、`rollout_func` 和 `environment_factory`，源码也实现了多轮工具调用循环；但构造函数明确要求原生 `tools` 至少 Transformers 5.0.0、`environment_factory` 至少 5.2.0，因此当前锁定组合不能直接使用这两项原生集成。源码中另有 `rollout_func` 扩展点，不触发上述工具参数版本门槛，但它是实验接口，调用方必须返回 `prompt_ids`、`completion_ids`、`logprobs`，并正确处理工具反馈 token 的 loss mask。它可能提供不升级依赖的路径，目前尚未实现或与真实 GRPO 更新联调。没有升级依赖；版本门槛与扩展点均由本地安装源码核对，见 `requirements-wsl2.txt` 和 `requirements-wsl2.lock.txt`。

## 本地运行

仅需项目现有 Python，无需安装新依赖：

```bash
python -m unittest discover -s tests -v
python -m agent_rl.demo
```

demo 使用两条固定脚本轨迹，并输出聚合指标、逐步工具反馈和本机耗时。它标记为 contract demo，不是模型性能或 Agent RL 结果。耗时是运行时测量值，不作为确定性输出；任务状态与奖励应在重复运行间保持一致。

## 验证记录

在 WSL2 Ubuntu 的 Python 3.12.3 环境中运行上述命令：11项 runner 和环境单元测试全部通过；demo 的两个任务都得到 reward 1.0，分别用2步/1次工具调用和3步/2次工具调用完成。测试覆盖无效工具、缺少生成 token 元数据、无效 JSON、错误终答及汇总分母。该验证只执行 CPU 工具环境，不加载 Qwen、不触发训练，也不申请 GPU。
