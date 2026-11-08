# gb1transformer

使用 GB1 数据训练 Transformer、微调 ESM2，并通过 LangGraph Agent 和 MCP 完成虚拟定向进化实验。

## 0. clone project

本项目使用 Git LFS 管理模型权重和训练 checkpoint。克隆前请先安装 Git 和 Git LFS，避免缺少实际模型文件。

Ubuntu / Debian 可通过以下命令安装：

```bash
sudo apt update
sudo apt install git git-lfs
```

其他系统请参考 [Git LFS 安装说明](https://git-lfs.com/)。安装完成后，初始化 Git LFS 并克隆项目：

```bash
git lfs install
git clone https://github.com/void-wizard/gb1transformer.git
cd gb1transformer
```

克隆时会自动下载 LFS 管理的大文件，请等待下载完成。如果下载中断，或此前未安装 Git LFS，可在安装并初始化 Git LFS 后，进入项目根目录补充下载：

```bash
git lfs pull
```

以下命令均在克隆后的项目根目录 `gb1transformer/` 执行。

## 1. 准备环境和数据

先安装 uv；运行 UI 还需要 Node.js 24 和 npm。

```bash
uv sync --locked
```

命令会自动创建 `.venv` 并安装 Python 依赖，不需要手动激活环境。

下载数据和预训练模型，然后预处理、划分数据。已有数据和划分时跳过：

```bash
uv run python scripts/download.py
uv run python -m src.prepare_data
uv run python -m src.split_data
```

数据保存在 `data/`，预训练 ESM2 保存在 `models/esm2/`。

Agent 和 UI 使用项目根目录的 `.env`。已有配置直接保留；首次使用填写以下内容中的接口地址、密钥和模型名：

```dotenv
api_key=你的API密钥
base_url=你的DeepSeek接口地址
model=你的DeepSeek模型名
temperature=0
reasoning_effort=high
thinking=enabled
```

## 2. 训练模型

选择一张 GPU，下面以 GPU 0 为例。两条命令顺序执行：

```bash
export CUDA_VISIBLE_DEVICES=0

# 从零训练 Transformer
uv run python train.py

# 全参数微调 ESM2
uv run python train_esm.py
```

| 模型 | 结果目录 | 最佳模型 |
| --- | --- | --- |
| Transformer | `outputs/GB1transformer/` | `best.pt` |
| ESM2 | `outputs/esm2/` | `best_model/` |

各目录还保存训练日志、指标曲线和独立测试指标。训练完成后直接运行实验。

## 3. 运行实验

每条模型路线运行四种方法，每组三轮、每轮推荐十个候选。按顺序执行；Agent 组需要 `.env` 中的 API 配置。

```bash
export CUDA_VISIBLE_DEVICES=0

# !!! 如需重跑实验，可以将文件夹名改为定义的文件夹，比如 _run2， _run1后缀的为在 GBU 服务器运行的版本

# Transformer：随机推荐、模型直接推荐、普通 Agent、知识增强 Agent
CUDA_VISIBLE_DEVICES=0 uv run python -m agent.baselines --model scratch --method random --seed 42 --output-dir outputs/experiments/scratch/baselines_run1/random --device cuda:0

CUDA_VISIBLE_DEVICES=0 uv run python -m agent.baselines --model scratch --method direct --seed 42 --output-dir outputs/experiments/scratch/baselines_run1/direct --device cuda:0

uv run python -m agent.experiment --model scratch --rounds 3 --output-dir outputs/experiments/scratch/agent_run1

uv run python -m agent.experiment --model scratch --knowledge --rounds 3 --output-dir outputs/experiments/scratch/knowledge_run1

# ESM2：相同四组实验
CUDA_VISIBLE_DEVICES=0 uv run python -m agent.baselines --model esm2 --method random --seed 42 --output-dir outputs/experiments/esm2/baselines_run1/random --device cuda:0

CUDA_VISIBLE_DEVICES=0 uv run python -m agent.baselines --model esm2 --method direct --seed 42 --output-dir outputs/experiments/esm2/baselines_run1/direct --device cuda:0

uv run python -m agent.experiment --model esm2 --rounds 3 --output-dir outputs/experiments/esm2/agent_run1

uv run python -m agent.experiment --model esm2 --knowledge --rounds 3 --output-dir outputs/experiments/esm2/knowledge_run1
```

上述基线命令使用 GPU 0，可修改 `CUDA_VISIBLE_DEVICES` 选择其他单张 GPU。

每组目录中的 `metrics.csv` 保存逐轮指标，`round_01/` 至 `round_03/` 保存推荐及真实 fitness 反馈。中断后重复同一命令可续跑；重新训练或重新划分数据后应使用新的运行目录。

八组完成后汇总：

```bash
uv run python -m agent.compare_methods --baseline-run baselines_run1 --agent-run agent_run1 --knowledge-run knowledge_run1 --output-dir outputs/experiments/comparison_run1
```

汇总目录保存 `round_comparison.csv`、`selected_candidates.csv` 和 `fitness_comparison.png` 等结果。程序输出指标和图表

## 4. 启动 UI

UI 只使用训练好的 Transformer

```bash
# 需确保安装了nodejs、npm
# 根目录执行：
npm --prefix ui ci
npm --prefix ui run build
uv run python -m agent.web --port 8000
```

浏览器打开 <http://127.0.0.1:8000>。在远程服务器运行时，先在自己的电脑执行端口转发：

```bash
ssh -L 8000:127.0.0.1:8000 用户名@服务器地址
```

## 5. UI 使用示例

1. 新建对话，输入框上半部分保留下面的 GB1 野生型序列。
2. 点击启用“知识增强”和“推荐 Top-10”。
3. 在下半部分输入问题并发送，查看流式回答、工具调用和候选推荐。

野生型序列：

```text
MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE
```

问题示例：

> 根据当前已知实验和模型预测，从候选池中推荐 10 个值得验证的 GB1 突变体。请给出突变描述、完整序列、预测 fitness 和推荐依据，并区分已有证据与待验证假设。

结果和聊天记录保存在 `outputs/ui/`。

## 6. 主要结果

### 适应度预测

两条模型路线使用相同的数据划分，输入完整的 56 位 GB1 序列。根据验证集选择最佳模型后，在独立测试集上评估。

| 模型 | MSE | Spearman | Pearson | Top-100 命中率 | Top-100 平均真实 fitness |
| --- | ---: | ---: | ---: | ---: | ---: |
| 从零训练 Transformer | 0.2098 | 0.9800 | 0.9813 | 67% | 1.5895 |
| 全参数微调 ESM2 | 0.1879 | 0.9804 | 0.9818 | 61% | 1.5759 |

Top-100 命中率表示：按预测分数选出的前 100 个变体中，有多少比例属于真实 fitness 的前 100 名。

ESM2 的整体预测误差更低，Transformer 在 Top-100 高适应度变体筛选上表现更好。

### 虚拟定向进化

每组运行三轮，每轮推荐 10 个未在本组之前轮次中推荐过的候选，共评估 30 个不同候选。下表使用数据集中的真实 fitness，而非模型预测分数。fitness 为相对野生型的对数富集分数，野生型为 0，数值越大越好。

| 模型 | 推荐方法 | 30 个候选平均 fitness | 最佳候选 fitness |
| --- | --- | ---: | ---: |
| Transformer | 随机推荐 | -1.9413 | 1.3397 |
| Transformer | 模型直接推荐 | 1.7012 | 2.5295 |
| Transformer | 普通 Agent | 1.5702 | 2.0525 |
| Transformer | 知识增强 Agent | 1.5476 | 1.8128 |
| ESM2 | 随机推荐 | -1.9413 | 1.3397 |
| ESM2 | 模型直接推荐 | 1.7048 | 2.0109 |
| ESM2 | 普通 Agent | 1.4322 | 2.5295 |
| ESM2 | 知识增强 Agent | 1.3495 | 2.0525 |

两条模型路线中，模型直接推荐的平均 fitness 均最高，普通 Agent 和知识增强 Agent 均优于随机推荐。ESM2 普通 Agent 找到的最佳候选 fitness 为 2.5295，高于该模型直接推荐组的最佳值。本次实验中，知识增强没有提高推荐集合的平均 fitness。

详细结果保存在：

- `outputs/GB1transformer/test_metrics.csv`：Transformer 独立测试指标
- `outputs/esm2/test_results.json`：ESM2 独立测试指标
- `outputs/experiments/comparison_run1/`：逐轮指标、候选明细、位点统计和对比图
