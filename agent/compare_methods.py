import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.baselines import selection_order, sha256
from agent.data_tools import PUBLIC_COLUMNS
from src.prepare_data import WT_SEQUENCE


ROOT = Path(__file__).resolve().parents[1]
METHOD_NAMES = {
    "random": "随机推荐", "direct": "模型直接推荐",
    "agent": "普通 Agent", "knowledge": "知识增强 Agent",
}


def position_statistics(selected, by_round=False):
    """统计各位点发生突变的候选数，占比以对应组的不同候选数为分母"""

    columns = ["model", "method"] + (["round"] if by_round else [])
    records = []
    for keys, group in selected.groupby(columns, sort=False):
        group = group.drop_duplicates("candidate_id")
        counts = Counter()
        for positions in group["mutations"].str.findall(r"[A-Z](\d+)[A-Z]"):
            # 同一候选在同一位点只计一次，双点候选分别计入两个位点
            counts.update({int(position) for position in positions})
        for position in range(1, len(WT_SEQUENCE) + 1):
            records.append({
                **dict(zip(columns, keys)),
                "position": position,
                "candidate_count": counts[position],
                "total_candidates": len(group),
                "fraction": counts[position] / len(group),
            })
    return pd.DataFrame(records)


def compare(baseline_run, agent_run, knowledge_run, output_dir):
    """核对实验记录并导出对照表和曲线"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(ROOT / "data/splits/candidates.csv", usecols=PUBLIC_COLUMNS)
    pool = pd.read_csv(ROOT / "data/splits/private/pool.csv").set_index("id")
    rows, selected, checks = [], [], {}
    configs = []
    random_ids = {}
    for model in ["scratch", "esm2"]:
        for method in METHOD_NAMES:
            relative = (
                Path(model) / baseline_run / method if method in ["random", "direct"]
                else Path(model) / (agent_run if method == "agent" else knowledge_run)
            )
            directory = ROOT / "outputs/experiments" / relative
            config = json.loads((directory / "experiment.json").read_text())
            assert config["model_type"] == model and config["rounds"] == 3
            assert config["predictor_update"] == "fixed"
            configs.append({"model": model, "method": method, **config})
            expected = None
            if method in ["random", "direct"]:
                assert config["method"] == method
                assert config["candidates_sha256"] == sha256(ROOT / "data/splits/candidates.csv")
                assert config["train_sha256"] == sha256(ROOT / "data/splits/train.csv")
                for filename, digest in config["model_sha256"].items():
                    assert sha256(ROOT / filename) == digest
                for filename, digest in json.loads((directory / "source_manifest.json").read_text()).items():
                    snapshot = directory / "source_snapshot" / filename
                    assert sha256(snapshot if snapshot.exists() else ROOT / filename) == digest
                scores = None
                if method == "direct":
                    metadata = json.loads((directory / "predictions.json").read_text())
                    assert sha256(directory / "predictions.csv") == metadata["sha256"]
                    scores = pd.read_csv(directory / "predictions.csv")
                    assert len(scores) == len(candidates)
                    assert set(scores["id"]) == set(candidates["id"])
                expected = selection_order(candidates, method, config["seed"], scores).head(30)
            values, identifiers = [], []
            for number in range(1, 4):
                stage = directory / f"round_{number:02d}"
                report_path = stage / "recommendations.json"
                report = json.loads(report_path.read_text())
                feedback = json.loads((stage / "feedback.json").read_text())
                assert feedback["recommendation_sha256"] == sha256(report_path)
                assert report["model_type"] == feedback["model_type"] == model
                assert len(report["recommendations"]) == len(feedback["results"]) == 10
                if method in ["random", "direct"]:
                    assert report["method"] == feedback["method"] == method
                    assert report["experiment_id"] == feedback["experiment_id"] == config["experiment_id"]
                    assert [r["candidate_id"] for r in report["recommendations"]] == expected.iloc[(number-1)*10:number*10]["id"].tolist()
                current = []
                for recommended, measured in zip(report["recommendations"], feedback["results"]):
                    identifier = recommended["candidate_id"]
                    assert identifier == measured["candidate_id"] and identifier not in identifiers
                    record = pool.loc[identifier]
                    assert recommended["sequence"] == measured["sequence"] == record.sequence
                    assert recommended["mutations"] == measured["mutations"] == record.mutations
                    assert np.isclose(measured["measured_fitness"], record.fitness, rtol=0, atol=1e-10)
                    assert recommended["predicted_fitness"] == measured["predicted_fitness"]
                    if method == "direct":
                        predicted = expected.set_index("id").loc[identifier, "predicted_fitness"]
                        assert np.isclose(recommended["predicted_fitness"], predicted, rtol=0, atol=1e-10)
                    current.append(measured["measured_fitness"])
                    identifiers.append(identifier)
                    selected.append({"model": model, "method": method, "round": number, **measured})
                values.extend(current)
                assert np.isclose(np.mean(current), feedback["summary"]["mean_measured_fitness"], rtol=0, atol=1e-10)
                rows.append({"model": model, "method": method, "round": number,
                             "mean_measured_fitness": float(np.mean(current)),
                             "best_measured_fitness": max(current),
                             "cumulative_best_fitness": max(values), "n_evaluated": 10})
            if method == "random":
                random_ids[model] = identifiers
            history = json.loads((directory / "revealed_history.json").read_text())
            assert [r["candidate_id"] for r in history["results"]] == identifiers
            saved_metrics = pd.read_csv(directory / "metrics.csv")
            assert len(saved_metrics) == 3
            assert np.allclose(saved_metrics.mean_measured_fitness, [r["mean_measured_fitness"] for r in rows[-3:]], rtol=0, atol=1e-10)
            checks[f"{model}_{method}"] = {
                "passed": True, "n_unique": len(identifiers), "mean_fitness": float(np.mean(values)),
                "best_fitness": max(values), "source": str(directory.relative_to(ROOT)),
            }
    assert random_ids["scratch"] == random_ids["esm2"]

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "round_comparison.csv", index=False)
    selected_frame = pd.DataFrame(selected)
    selected_frame.to_csv(output_dir / "selected_candidates.csv", index=False)
    positions = position_statistics(selected_frame)
    positions.to_csv(output_dir / "position_counts.csv", index=False)
    position_statistics(selected_frame, by_round=True).to_csv(
        output_dir / "position_counts_by_round.csv", index=False,
    )
    (output_dir / "audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    # 导出各组运行配置
    (output_dir / "run_configs.json").write_text(
        json.dumps(configs, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    lines = ["# 四种推荐方法的三轮对照", "",
             "以下数值来自本次读取并核对的实验记录，fitness 为真实对数富集标签。", "",
             "| 模型 | 方法 | 第一轮均值 | 第二轮均值 | 第三轮均值 | 30 个推荐均值 | 最佳值 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for model, label in [("scratch", "Transformer"), ("esm2", "ESM2")]:
        for method, title in METHOD_NAMES.items():
            subset = frame[(frame.model == model) & (frame.method == method)]
            record = checks[f"{model}_{method}"]
            means = " | ".join(f"{v:.4f}" for v in subset.mean_measured_fitness)
            lines.append(f"| {label} | {title} | {means} | {record['mean_fitness']:.4f} | {record['best_fitness']:.4f} |")
    lines += ["", "## 推荐位点统计", "",
              "次数表示有多少条不同候选在该位置发生突变，占比为次数除以本组候选数。",
              "双点候选分别计入两个位置，因此各位置占比相加可以超过 100%。",
              "下表列出三轮合计出现最多的五个位点，同次数按位点编号排序；完整结果包含全部 56 个位置。", "",
              "| 模型 | 方法 | 不同候选数 | 位点：候选数（占比） |",
              "| --- | --- | ---: | --- |"]
    for model, label in [("scratch", "Transformer"), ("esm2", "ESM2")]:
        for method, title in METHOD_NAMES.items():
            subset = positions[(positions.model == model) & (positions.method == method)]
            top = subset[subset.candidate_count > 0].sort_values(
                ["candidate_count", "position"], ascending=[False, True],
            ).head(5)
            counts = "；".join(
                f"{row.position}：{row.candidate_count}（{row.fraction:.1%}）"
                for row in top.itertuples()
            )
            lines.append(f"| {label} | {title} | {int(subset.total_candidates.iloc[0])} | {counts} |")
    lines += ["", "## 指标定义", "",
              "- 每轮均值：该轮十个推荐候选的真实 fitness 平均值",
              "- 当轮最佳：该轮推荐候选的最大真实 fitness",
              "- 累计最佳：截至该轮全部已推荐候选的最大真实 fitness",
              "- 30 个推荐均值、最佳值：本组全部三轮候选的平均值、最大值",
              f"- 公开候选池大小：{len(candidates):,}",
              "", "## 结果文件", "",
              "- round_comparison.csv：每组每轮均值、当轮最佳和累计最佳",
              "- selected_candidates.csv：全部提交、预测分数、真实分数和原始推荐依据",
              "- position_counts.csv：每组的三轮合计位点统计",
              "- position_counts_by_round.csv：每组每轮的位点统计",
              "- 位点统计字段：position 为位点编号，candidate_count 为该位点发生突变的候选数，total_candidates 为本组或本轮候选总数，fraction 为两者之比（0—1）",
              "- audit.json：候选去重、固定种子选择、全池 Top-30、标签和提交摘要核对",
              "- run_configs.json：从各组 experiment.json 读取的实际配置，未记录的字段不推测",
              "- fitness_comparison.png / fitness_comparison.pdf：逐轮真实均值曲线", ""]
    lines += ["", "## 原始实验目录", ""]
    for record in checks.values():
        relative = Path(os.path.relpath(ROOT / record["source"], output_dir))
        lines.append(f"- [{relative}]({relative})")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    labels = ["Random", "Model ranking (full pool)", "LLM agent", "Knowledge-enhanced agent"]
    for ax, (model, title) in zip(axes, [("scratch", "Transformer"), ("esm2", "ESM2")]):
        for method, label in zip(METHOD_NAMES, labels):
            subset = frame[(frame.model == model) & (frame.method == method)]
            ax.plot(subset["round"], subset.mean_measured_fitness, "o-", label=label)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set(title=title, xlabel="Round", xticks=[1, 2, 3])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean measured log enrichment (10 selections)")
    fig.tight_layout()
    fig.savefig(output_dir / "fitness_comparison.png", dpi=180)
    fig.savefig(output_dir / "fitness_comparison.pdf")
    plt.close(fig)
    print("\n".join(lines[:14]))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--agent-run", required=True)
    parser.add_argument("--knowledge-run", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline_run, args.agent_run, args.knowledge_run, args.output_dir)
