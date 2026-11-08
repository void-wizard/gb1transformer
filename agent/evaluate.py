import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agent.results import RecommendationBatch


ROOT = Path(__file__).resolve().parents[1]


def evaluate(recommendation_path):
    """校验已提交候选并返回真实 fitness"""

    recommendation_path = Path(recommendation_path)
    raw = recommendation_path.read_bytes()
    report = json.loads(raw)

    batch = RecommendationBatch.model_validate(report)

    # 提交后由评估器读取隐藏标签
    pool = pd.read_csv(
        ROOT / "data" / "splits" / "private" / "pool.csv",
        usecols=["id", "sequence", "mutations", "fitness"],
    ).set_index("id")

    rows = []
    for candidate, submitted in zip(
        batch.recommendations, report["recommendations"]
    ):
        candidate_id = candidate.candidate_id
        if candidate_id not in pool.index:
            raise ValueError(f"候选不在实验池中：{candidate_id}")

        measured = pool.loc[candidate_id]

        # 核对候选 ID 与序列、突变描述
        if (
            submitted["sequence"] != measured["sequence"]
            or submitted["mutations"] != measured["mutations"]
        ):
            raise ValueError(f"候选身份不一致：{candidate_id}")

        rows.append({
            **submitted,
            "measured_fitness": float(measured["fitness"]),
        })

    fitness = pd.Series([row["measured_fitness"] for row in rows])

    return {
        "model_type": report["model_type"],
        "knowledge": report.get("knowledge", False),
        "experiment_id": report.get("experiment_id"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_sha256": hashlib.sha256(raw).hexdigest(),
        "score_type": "measured_log_enrichment",
        "results": rows,
        "summary": {
            "n_evaluated": len(rows),
            "mean_measured_fitness": float(fitness.mean()),
            "best_measured_fitness": float(fitness.max()),
            # WT 归一化分数为 0
            "fraction_above_wt": float((fitness > 0).mean()),
        },
    }


def main():
    """评估推荐并保存反馈"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "recommendations",
        type=Path,
        help="已保存的 recommendations.json 路径",
    )
    args = parser.parse_args()

    output_path = args.recommendations.parent / "feedback.json"
    if output_path.exists():
        parser.error(f"反馈已存在，请查看已有结果：{output_path}")

    feedback = evaluate(args.recommendations)

    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(
            feedback,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(json.dumps(feedback["summary"], ensure_ascii=False, indent=2))
    print(f"实验反馈已保存：{output_path}")


if __name__ == "__main__":
    main()

