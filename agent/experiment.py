import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from agent.data_tools import check_scope
from agent.evaluate import evaluate
from agent.final_report import build_final_report
from agent.prompts import ROUND_TASK_TEMPLATE
from agent.scientist import run_agent
from agent.summary import summarize_round


ROOT = Path(__file__).resolve().parents[1]


def write_record(path, data):
    """独占写入阶段记录"""

    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


async def run_experiment(
    model_type, rounds=3, output_dir=None, first_report=None, knowledge=False,
):
    """逐轮执行推荐、评估和总结，支持断点续跑"""

    if rounds not in (2, 3):
        raise ValueError("当前实验运行 2 或 3 轮")
    if output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output_dir = ROOT / "outputs" / "experiments" / model_type / run_id
        if knowledge:
            output_dir = output_dir.parent / "knowledge" / run_id
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "experiment.json"
    config = {"model_type": model_type, "rounds": rounds, "top_k": 10, "predictor_update": "fixed"}
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in config.items()):
            raise ValueError("已有实验的模型组、轮次或推荐数量与本次启动不一致")
        check_scope(saved, model_type, knowledge)
        config = saved
    else:
        config.update(knowledge=knowledge, experiment_id=uuid4().hex)
        write_record(config_path, config)

    experiment_id = config.get("experiment_id")
    history = {
        "model_type": model_type, "knowledge": knowledge,
        "experiment_id": experiment_id, "results": [],
    }
    history_path = output_dir / "revealed_history.json"
    previous_summary = None
    metrics = []
    round_records = []
    print(f"实验目录：{output_dir}", flush=True)

    for round_number in range(1, rounds + 1):
        round_dir = output_dir / f"round_{round_number:02d}"
        round_dir.mkdir(exist_ok=True)
        report_path = round_dir / "recommendations.json"

        # 从已保存轮次恢复反馈历史
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        if not report_path.exists():
            if round_number == 1 and first_report is not None:
                imported = json.loads(Path(first_report).read_text(encoding="utf-8"))
                check_scope(imported, model_type, knowledge)
                if imported.get("experiment_id") not in (None, experiment_id):
                    raise ValueError("不能复用另一实验运行的推荐")
                imported.update(knowledge=knowledge, experiment_id=experiment_id)
                write_record(report_path, imported)
            else:
                task = ROUND_TASK_TEMPLATE.format(
                    round_number=round_number,
                    total_rounds=rounds,
                    previous_summary=json.dumps(previous_summary, ensure_ascii=False),
                    revealed_results=json.dumps(history["results"], ensure_ascii=False),
                )
                print(f"第 {round_number} 轮：生成推荐", flush=True)
                await run_agent(
                    model_type, task, recommend=True,
                    feedback_path=history_path, output_dir=round_dir,
                    knowledge=knowledge, experiment_id=experiment_id,
                )

        raw = report_path.read_bytes()
        report = json.loads(raw)
        check_scope(report, model_type, knowledge, experiment_id)
        already_tested = {row["candidate_id"] for row in history["results"]}
        if already_tested.intersection(
            row["candidate_id"] for row in report["recommendations"]
        ):
            raise ValueError("本轮推荐包含以前已经实验过的候选")

        feedback_path = round_dir / "feedback.json"
        if not feedback_path.exists():
            print(f"第 {round_number} 轮：揭示已提交候选的真实标签", flush=True)
            write_record(feedback_path, evaluate(report_path))
        feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
        check_scope(feedback, model_type, knowledge, experiment_id)
        if (
            feedback["model_type"] != model_type
            or feedback["recommendation_sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("反馈与已提交推荐不匹配")

        # 阶段总结供下一轮使用
        summary = None
        if round_number < rounds:
            summary_path = round_dir / "summary.json"
            if not summary_path.exists():
                print(f"第 {round_number} 轮：DeepSeek 总结实验", flush=True)
                summary = await summarize_round(report, feedback, previous_summary)
                write_record(summary_path, summary)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary["model_type"] != model_type
                or summary["recommendation_sha256"] != feedback["recommendation_sha256"]
            ):
                raise ValueError("总结与本轮实验记录不匹配")
            previous_summary = summary["summary"]
        round_records.append({
            "round": round_number,
            "feedback": feedback,
            "summary": summary["summary"] if summary else None,
        })
        history["results"].extend(feedback["results"])
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        metrics.append({
            "round": round_number,
            **feedback["summary"],
            "cumulative_best_fitness": max(
                row["measured_fitness"] for row in history["results"]
            ),
        })
        pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
        print(f"第 {round_number} 轮完成：{feedback['summary']}", flush=True)

    # 汇总各轮指标和候选记录
    evidence = {"rounds": round_records, "metrics": metrics}
    if experiment_id is not None:
        evidence.update(knowledge=knowledge, experiment_id=experiment_id)
    evidence_sha256 = hashlib.sha256(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    final_path = output_dir / "results.json"
    if not final_path.exists():
        print("正在汇总实验指标和候选记录", flush=True)
        final = build_final_report(model_type, evidence)
        final.update(knowledge=knowledge, experiment_id=experiment_id)
        final["evidence_sha256"] = evidence_sha256
        write_record(final_path, final)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if final["model_type"] != model_type or final["evidence_sha256"] != evidence_sha256:
        raise ValueError("结果汇总与本组实验记录不匹配")
    markdown_path = output_dir / "results.md"
    if not markdown_path.exists():
        with markdown_path.open("x", encoding="utf-8") as handle:
            handle.write(final["content"])
    print(f"最终输出：{markdown_path}", flush=True)

    return output_dir


def main():
    """启动或继续实验"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["scratch", "esm2"], required=True)
    parser.add_argument("--rounds", type=int, choices=[2, 3], default=3)
    parser.add_argument("--output-dir", type=Path, help="固定输出目录，也用于断点续跑")
    parser.add_argument("--first-report", type=Path, help="复用本组已经保存的第一轮推荐")
    parser.add_argument("--knowledge", action="store_true", help="启用知识图谱增强")
    args = parser.parse_args()
    asyncio.run(run_experiment(
        args.model, args.rounds, args.output_dir, args.first_report,
        knowledge=args.knowledge,
    ))


if __name__ == "__main__":
    main()
