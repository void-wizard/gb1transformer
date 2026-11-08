import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from agent.data_tools import PUBLIC_COLUMNS
from agent.evaluate import evaluate
from agent.experiment import write_record
from agent.mcp_server import MODEL_PATHS
from agent.results import save_report
from src.predictor import FitnessPredictor


ROOT = Path(__file__).resolve().parents[1]
TOP_K = 10
ROUNDS = 3


def sha256(path):
    """计算文件 SHA-256"""

    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def selection_order(candidates, method, seed=42, scores=None):
    """按随机抽样或预测分数排列候选"""

    candidates = candidates[PUBLIC_COLUMNS].sort_values("id").reset_index(drop=True)
    if method == "random":
        # 随机排列后按轮次无放回取样
        return candidates.sample(frac=1, random_state=seed).reset_index(drop=True)
    if method != "direct":
        raise ValueError("method 必须为 random 或 direct")
    ranked = candidates.merge(scores[["id", "predicted_fitness"]], on="id", validate="one_to_one")
    if len(ranked) != len(candidates) or not np.isfinite(ranked["predicted_fitness"]).all():
        raise ValueError("模型直接推荐必须覆盖全部候选，且预测分数必须有限")
    # 同分按 ID 排序
    return ranked.sort_values(["predicted_fitness", "id"], ascending=[False, True]).reset_index(drop=True)


def score_candidates(candidates, predictor, output_dir):
    """分块预测候选并缓存分数"""

    path = output_dir / "predictions.csv"
    metadata_path = output_dir / "predictions.json"
    if path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["sha256"] != sha256(path):
            raise ValueError("预测缓存内容与保存的摘要不一致")
        scores = pd.read_csv(path)
        if scores["id"].tolist() != candidates["id"].tolist():
            raise ValueError("预测缓存的候选顺序或覆盖范围不一致")
        return scores

    temporary = output_dir / "predictions.partial.csv"
    # 全池预测完成后替换正式缓存
    for start in range(0, len(candidates), 1024):
        batch = candidates.iloc[start:start + 1024]
        scores = predictor.predict(batch["sequence"].tolist())
        if len(scores) != len(batch) or not np.isfinite(scores).all():
            raise ValueError("模型返回的分数数量或数值异常")
        pd.DataFrame({"id": batch["id"].tolist(), "predicted_fitness": scores}).to_csv(
            temporary, index=False, mode="w" if start == 0 else "a", header=start == 0,
        )
        print(f"已预测 {min(start + 1024, len(candidates))}/{len(candidates)}", flush=True)
    temporary.replace(path)
    write_record(metadata_path, {"count": len(candidates), "sha256": sha256(path)})
    return pd.read_csv(path)


def run_baseline(model_type, method, output_dir, seed=42, device="cpu"):
    """运行三轮基线实验并保存推荐、反馈和指标"""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_PATHS[model_type]
    model_files = sorted(p for p in model_path.rglob("*") if p.is_file()) if model_path.is_dir() else [model_path]
    candidate_path = ROOT / "data/splits/candidates.csv"
    candidates = pd.read_csv(candidate_path, usecols=PUBLIC_COLUMNS).sort_values("id").reset_index(drop=True)
    if device.startswith("cuda") and torch.cuda.device_count() != 1:
        raise ValueError("请先通过 CUDA_VISIBLE_DEVICES 将进程限制到一张 GPU")

    config = {
        "model_type": model_type, "method": method, "knowledge": False,
        "rounds": ROUNDS, "top_k": TOP_K, "seed": seed, "predictor_update": "fixed",
        "candidate_count": len(candidates), "candidates_sha256": sha256(candidate_path),
        "train_sha256": sha256(ROOT / "data/splits/train.csv"),
        "model_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in model_files},
        "device": device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
    }
    config_path = output_dir / "experiment.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if any(saved.get(key) != value for key, value in config.items()):
            raise ValueError("数据、模型或实验配置发生变化，请使用新的实验目录")
        config = saved
    else:
        config["experiment_id"] = uuid4().hex
        write_record(config_path, config)
        paths = [Path(__file__), ROOT / "src/predictor.py", ROOT / "src/model.py",
                 ROOT / "src/tokenizer.py", ROOT / "agent/evaluate.py"]
        write_record(output_dir / "source_manifest.json", {
            str(p.relative_to(ROOT)): sha256(p) for p in paths
        })

    print(f"实验目录：{output_dir}", flush=True)
    predictor = FitnessPredictor(model_path, model_type, device)
    if method == "direct":
        scores = score_candidates(candidates, predictor, output_dir)
        ordered = selection_order(candidates, method, seed, scores)
    else:
        # 随机组先抽样，后预测
        ordered = selection_order(candidates, method, seed).head(ROUNDS * TOP_K).copy()
        ordered["predicted_fitness"] = predictor.predict(ordered["sequence"].tolist())

    history = {
        "model_type": model_type, "method": method, "knowledge": False,
        "experiment_id": config["experiment_id"], "results": [],
    }
    metrics = []
    for round_number in range(1, ROUNDS + 1):
        stage = output_dir / f"round_{round_number:02d}"
        stage.mkdir(exist_ok=True)
        chosen = ordered.iloc[(round_number - 1) * TOP_K:round_number * TOP_K]
        report_path = stage / "recommendations.json"
        if not report_path.exists():
            reason = (
                "固定种子对公开候选均匀无放回抽样；模型预测不参与选择"
                if method == "random" else "从全部尚未实验候选中按模型预测降序选择，同分按 ID 排序"
            )
            save_report({
                "model_type": model_type, "method": method, "knowledge": False,
                "experiment_id": config["experiment_id"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "recommendations": [
                    {"candidate_id": int(row.id), "sequence": row.sequence,
                     "mutations": row.mutations, "predicted_fitness": float(row.predicted_fitness),
                     "reason": reason}
                    for row in chosen.itertuples(index=False)
                ],
                "selection_rule": reason,
            }, stage)
        report = json.loads(report_path.read_text())
        if (
            report["experiment_id"] != config["experiment_id"]
            or report["method"] != method
            or [r["candidate_id"] for r in report["recommendations"]] != chosen["id"].tolist()
        ):
            raise ValueError("已保存的推荐与当前实验或确定性选择顺序不一致")

        feedback_path = stage / "feedback.json"
        if not feedback_path.exists():
            # 推荐保存后揭示标签
            feedback = evaluate(report_path)
            feedback["method"] = method
            write_record(feedback_path, feedback)
        feedback = json.loads(feedback_path.read_text())
        if (
            feedback["recommendation_sha256"] != sha256(report_path)
            or feedback["experiment_id"] != config["experiment_id"]
            or feedback["method"] != method
        ):
            raise ValueError("反馈与本组已提交推荐不一致")
        history["results"].extend(feedback["results"])
        metrics.append({
            "round": round_number, **feedback["summary"],
            "cumulative_best_fitness": max(r["measured_fitness"] for r in history["results"]),
            # 预测集中在第一轮前完成
            "new_predictions": (len(candidates) if method == "direct" else ROUNDS * TOP_K) if round_number == 1 else 0,
        })
        (output_dir / "revealed_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))
        pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
        print(f"第 {round_number} 轮完成：{feedback['summary']}", flush=True)

    values = [row["measured_fitness"] for row in history["results"]]
    summary = {
        "model_type": model_type, "method": method, "rounds_completed": ROUNDS,
        "n_evaluated": len(values), "mean_measured_fitness": sum(values) / len(values),
        "best_measured_fitness": max(values), "experiment_id": config["experiment_id"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_dir


def main():
    """启动基线实验"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["scratch", "esm2"], required=True)
    parser.add_argument("--method", choices=["random", "direct"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_baseline(args.model, args.method, args.output_dir, args.seed, args.device)


if __name__ == "__main__":
    main()
