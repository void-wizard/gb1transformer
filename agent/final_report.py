def build_final_report(model_type, evidence):
    """将逐轮反馈整理为结构化结果和 Markdown 表格"""

    candidates = [
        {
            "round": stage["round"],
            **{key: row[key] for key in (
                "candidate_id", "sequence", "mutations",
                "predicted_fitness", "measured_fitness",
            )},
        }
        for stage in evidence["rounds"]
        for row in stage["feedback"]["results"]
    ]
    lines = [
        "# 虚拟实验结果", "", f"模型：{model_type}", "",
        "fitness 为 WT 归一化的对数富集值；预测值与实验测量值分列展示。", "",
        "| 轮次 | 候选数 | 真实均值 | 当轮最佳 | 累计最佳 | 高于 WT 的比例 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in evidence["metrics"]:
        lines.append(
            f"| {row['round']} | {row['n_evaluated']} | "
            f"{row['mean_measured_fitness']:.6f} | {row['best_measured_fitness']:.6f} | "
            f"{row['cumulative_best_fitness']:.6f} | {row['fraction_above_wt']:.6f} |"
        )
    lines += [
        "", "| 轮次 | 候选 ID | 突变 | 完整序列 | 预测 fitness | 真实 fitness |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in candidates:
        lines.append(
            f"| {row['round']} | {row['candidate_id']} | {row['mutations']} | "
            f"{row['sequence']} | {row['predicted_fitness']:.6f} | {row['measured_fitness']:.6f} |"
        )
    lines += [
        "", "指标定义：真实均值、当轮最佳分别为本轮测量值的平均值、最大值；"
        "累计最佳为截至本轮全部测量值的最大值；高于 WT 的比例为真实 fitness > 0 的候选占比。",
    ]
    return {
        "model_type": model_type,
        "score_type": "measured_log_enrichment",
        "metrics": evidence["metrics"],
        "candidates": candidates,
        "content": "\n".join(lines) + "\n",
    }
