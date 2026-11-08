# Agent 实验运行

本文的 `gb1transformer/` 指项目根目录；当前本地目录名为 `solutionmine/`。

运行命令和输出路径以 [项目 README](../README.md) 为准。实验完成状态以当前结果文件为准。

在 `gb1transformer/` 根目录、当前 uv 虚拟环境中执行命令。

## 有限轮次与上下文

每轮先完成候选提交和独立真实标签评估。
非最后一轮由 DeepSeek 生成阶段总结；最后一轮由程序导出指标和候选表，不调用 LLM 撰写结果分析。
下一轮新建 LangGraph Agent，只传入上一轮总结和本组全部已揭示实验记录。
原始工具轨迹保存到文件，不跨轮拼接；单轮内部保留正常工具调用上下文。
数据工具会加入历史真实标签，并排除已经实验过的候选。
两条模型路线、普通/知识增强模式以及不同实验运行的历史独立保存，预测模型在各轮保持固定。

当前配置的 `deepseek-flash` 经服务端元数据确认具有 1,048,576 token 上下文。
代码没有另设较小的输入上下文上限，也不截断工具历史。
单次输出上限显式设置为 65,536 token，可通过环境变量调整：

```bash
export DEEPSEEK_MAX_OUTPUT_TOKENS=131072
```

输出上限不同于上下文窗口，输入与生成内容合计仍受服务端容量约束。
逐轮总结减少跨轮积累，不保证任意长的单轮对话都不会溢出。
本流程不依赖 Mem0 或 embedding 服务。

## 启动与继续

```bash
python -m agent.experiment --model scratch --rounds 3 --output-dir outputs/experiments/scratch/run1
python -m agent.experiment --model esm2 --rounds 3 --output-dir outputs/experiments/esm2/run1
```

两组顺序执行。重复相同命令可继续未完成阶段，不重新生成已经保存的推荐、反馈和总结。
已有本组单轮推荐时，可增加 `--first-report <recommendations.json路径>` 作为第一轮。
改变模型组、知识增强模式或轮数时需使用新实验目录。

每轮目录包含：

- `recommendations.json`：锁定的十个候选、实际预测、推荐理由和工具轨迹
- `feedback.json`：独立评估器揭示的真实 fitness，以及与原始推荐对应的摘要校验值
- `summary.json`：仅非最后一轮生成，记录发现、局限和下一轮计划

实验根目录的 `revealed_history.json` 保存全部已揭示记录，`metrics.csv` 汇总逐轮真实 fitness。
`results.md` 是客观结果表，`results.json` 保存对应指标、候选记录及证据校验值。
推荐理由与阶段总结仅作为 Agent 过程记录；方法比较、成功失败原因和科学解释由研究者分析。
旧版 `final_report.json` / `final_report.md` 不读取或覆盖，不作为新结果。
旧实验已生成的最后一轮 `summary.json` 保留为历史文件，新流程不会读取或重新生成它。
`cumulative_best_fitness` 是累计最佳值，不等同于当轮性能逐轮提升。

## 模块职责

- `scientist.py`：单轮 LangGraph Agent 与 MCP 连接
- `evaluate.py`：隐藏标签评估，不注册为 Agent 工具
- `summary.py`：反馈后的 DeepSeek 总结
- `final_report.py`：根据结构化反馈生成结果表，不调用 LLM
- `experiment.py`：固定轮次编排和断点续跑
- `data_tools.py`：已知数据更新和候选排除
- `amino_acids.py`：氨基酸性质数值和文献来源
- `knowledge.py`：NetworkX 图谱及证据查询
- `knowledge_tools.py`：知识增强专用 MCP 工具
- `llm.py`：DeepSeek 配置及协议兼容

## 知识图谱增强

在现有命令上添加 `--knowledge`，无需另起服务。仍使用同一个 LangGraph Agent；
仅增加知识工具与提示，基础数据和预测工具保持一致。
使用当前共享 uv 环境，已将 `networkx==3.6.1` 显式加入依赖；交付环境也需要此依赖。
不需要图数据库、embedding 模型或额外 API。

```bash
# 单次分析，不提交候选、不揭示隐藏标签
python -m agent.scientist --model scratch --knowledge --task "分析 A20C 的已知单点证据和组合背景，说明性质变化与证据局限"

# 正式三轮实验：两条路线顺序运行，各自使用新目录
python -m agent.experiment --model scratch --knowledge --rounds 3 --output-dir outputs/experiments/scratch/knowledge_run1
python -m agent.experiment --model esm2 --knowledge --rounds 3 --output-dir outputs/experiments/esm2/knowledge_run1
```

不加 `--knowledge` 就是普通 Agent，知识工具不会注册。
不要把普通组的 `--first-report` 或 `--feedback` 传给知识增强组，代码会拒绝模式不一致的记录。
新的实验会保存 `experiment_id`，逐轮检查推荐和反馈是否属于同一次运行。
旧记录缺少知识字段时仅按普通实验兼容，不修改已有推荐、反馈和最终报告。

图谱关系为：

```text
位点 ─允许替换→ 突变 ─原残基/新残基→ 氨基酸（性质和来源）
变体 ─包含突变→ 突变
变体属性：完整序列、实验 fitness、数据来源
```

图谱完整加载初始训练集，不按 fitness 抽样。每轮从训练集和本实验已揭示记录重建，
不会读取 validation、test 或 private/pool.csv；公开 candidates.csv 仅用于核对已揭示 ID。
同一变体的 fitness 对应完整序列，不能拆分到两个突变节点。
查询返回的是既有实验观测和基础性质，不会替未知单点补标签。
每个返回的组合附带各组成单点的真实可见证据；完整对照齐全时，工具计算
`additive_deviation = 双点分数 - 单点 A 分数 - 单点 B 分数`，缺少单点则返回 `null`。
这里 WT 的对数分数为 0，算术偏离不等于统计显著的上位性。
知识工具没有固定的“有益突变名单”，也不会代替 LLM 选择候选。

| MCP 工具 | 输入示例 | 用途 |
| --- | --- | --- |
| `query_position_knowledge` | `{"position": 20}` | 查看 19 种替换及当前已知的单点证据 |
| `query_mutation_knowledge` | `{"mutation": "A20C", "limit": 3}` | 分开查看单点与组合背景、证据 ID、来源及性质变化 |
| `compare_residue_properties` | `{"original": "A", "target": "C"}` | 比较疏水性、极性、大小和简化电荷编码 |

性质数值核对自 AAindex，各尺度保持原始数值：

- [KYTJ820101：Kyte–Doolittle 疏水性](https://www.genome.jp/dbget-bin/www_bget?aaindex:KYTJ820101)
- [GRAR740102：Grantham 极性](https://www.genome.jp/dbget-bin/www_bget?aaindex:GRAR740102)
- [GRAR740103：Grantham 体积](https://www.genome.jp/dbget-bin/www_bget?aaindex:GRAR740103)
- [KLEP840101：Klein 等的净电荷编码](https://www.genome.jp/dbget-bin/www_bget?aaindex:KLEP840101)

电荷是简化编码，特别是 H 的 0 不代表所有 pH 下均不带电。
性质相似或差异较大只支持提出保守/激进替换假设，不保证 fitness 改善。
fitness 使用本项目由 Olson 实验计数计算的 WT 归一化对数富集值，
并非上述 AAindex 性质值；具体处理见 `src/prepare_data.py`。

```bash
python -m pytest tests/test_knowledge.py tests/test_knowledge_experiment.py -q
```

上述测试覆盖未揭示标签排除、组合背景区分、跨实验反馈拒绝、知识开关和断点续跑。
编排测试替代了外部 LLM，不作为正式实验结果。
接口通过或图谱分析成功，也不能证明知识增强优于普通 Agent；需要完成三轮实验后再比较真实 fitness。

## 随机推荐与模型直接推荐

`baselines.py` 复用当前 checkpoint、`FitnessPredictor` 和独立标签评估器，运行三轮、每轮 Top-10，不调用 LLM。
随机方法先按种子 42 对全部公开候选均匀无放回抽样，再计算入选者的预测值供记录；预测值不影响抽样。
两条模型路线使用相同随机候选，真实结果因此相同，不视为两次独立重复。
模型直接推荐给公开候选池中的全部候选打分，按预测值排序，三轮依次取尚未实验的十个最高分候选。
预测模型保持固定，全池分数只计算一次；每组只保留自己的反馈。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m agent.baselines --model scratch --method random --seed 42 --output-dir outputs/experiments/scratch/baselines_run1/random --device cuda:0
CUDA_VISIBLE_DEVICES=0 uv run python -m agent.baselines --model scratch --method direct --seed 42 --output-dir outputs/experiments/scratch/baselines_run1/direct --device cuda:0
CUDA_VISIBLE_DEVICES=0 uv run python -m agent.baselines --model esm2 --method random --seed 42 --output-dir outputs/experiments/esm2/baselines_run1/random --device cuda:0
CUDA_VISIBLE_DEVICES=0 uv run python -m agent.baselines --model esm2 --method direct --seed 42 --output-dir outputs/experiments/esm2/baselines_run1/direct --device cuda:0
```

上述基线命令使用 GPU 0，四组顺序执行；可修改 `CUDA_VISIBLE_DEVICES` 选择其他单张 GPU。
相同命令可以复用已保存的推荐、反馈和完整预测缓存。模型文件或候选池发生变化时需要新目录。
`predictions.csv` 只保存候选 ID 和模型预测，不包含隐藏标签；缓存绑定实际模型和候选文件的摘要。
随机组在第一轮前一次预测三十个入选候选，因此 `new_predictions` 为 30、0、0；全池排序组为候选池大小 N、0、0。

各组完成后，明确指定本次运行名称进行汇总：

```bash
python -m agent.compare_methods --baseline-run baselines_run1 --agent-run run1 --knowledge-run knowledge_run1 --output-dir outputs/experiments/comparison_run1
```

汇总只输出指标、表格、曲线和配置记录，不自动撰写结果分析。

汇总会重新核对固定种子抽样、全池模型 Top-30、候选去重、模型与数据摘要、提交与反馈对应关系、真实 fitness 和逐轮指标。
全池排序的预测次数高于 Agent；每种方法仅运行一次，不能据此声称统计显著优势或等计算量比较。

## React 交互页面

页面仅接入自训练 Transformer 的 MCP 方案，支持流式回答、工具进度、知识增强开关和 Top-10 推荐下载。
前端构建后，在 `gb1transformer/` 运行 `python -m agent.web --port 8000`，打开 `http://127.0.0.1:8000`。
环境准备、完整启动命令和使用范围见 [UI 说明](../ui/README.md)。
