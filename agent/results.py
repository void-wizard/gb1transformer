import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field, model_validator

from agent.prompts import SCIENTIST_SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parents[1]


class Recommendation(BaseModel):
    """候选 ID 与推荐依据"""

    candidate_id: int = Field(description="候选池中的 id")
    reason: str = Field(
        min_length=1,
        description="推荐依据及证据局限",
    )


class RecommendationBatch(BaseModel):
    """十个不同候选的推荐集合"""

    recommendations: list[Recommendation] = Field(
        min_length=10,
        max_length=10,
    )

    @model_validator(mode="after")
    def check_unique(self):
        """校验候选 ID 唯一性"""

        ids = [
            item.candidate_id
            for item in self.recommendations
        ]

        if len(ids) != len(set(ids)):
            raise ValueError("十个候选的 ID 必须不同")

        return self


def build_report(
    result, model_type, task, system_prompt=SCIENTIST_SYSTEM_PROMPT,
    excluded_ids=(),
):
    """校验候选及预测记录，生成推荐结果"""

    batch = result.get("structured_response")
    if batch is None:
        raise ValueError("Agent 未提交 RecommendationBatch，本轮推荐未保存")

    candidates = pd.read_csv(
        ROOT / "data" / "splits" / "candidates.csv",
        usecols=["id", "sequence", "mutations"],
    ).set_index("id")

    predictions = {}
    trace = []

    for message in result["messages"]:
        if isinstance(message, AIMessage):
            # 记录回答、工具调用和用量
            trace.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
                "usage": message.usage_metadata,
            })

        elif isinstance(message, ToolMessage):
            trace.append({
                "role": "tool",
                "name": message.name,
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            })

            if (
                message.name == "predict_fitness"
                and message.status == "success"
            ):
                # 从 artifact 读取 MCP 结构化结果
                data = (message.artifact or {}).get(
                    "structured_content"
                )

                if data is None or data["model"] != model_type:
                    raise ValueError(
                        "预测工具结果缺失或模型组不一致"
                    )

                for item in data["predictions"]:
                    predictions[item["sequence"]] = (
                        item["predicted_fitness"]
                    )

    rows = []
    for item in batch.recommendations:
        if item.candidate_id in excluded_ids:
            raise ValueError(f"候选已经实验过：{item.candidate_id}")
        if item.candidate_id not in candidates.index:
            raise ValueError(
                f"候选不在候选池中：{item.candidate_id}"
            )

        candidate = candidates.loc[item.candidate_id]
        sequence = candidate["sequence"]

        if sequence not in predictions:
            raise ValueError(
                f"候选尚未实际预测：{item.candidate_id}"
            )

        rows.append({
            "candidate_id": item.candidate_id,
            "sequence": sequence,
            "mutations": candidate["mutations"],
            "predicted_fitness": float(predictions[sequence]),
            "reason": item.reason,
        })

    return {
        "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "system_prompt": system_prompt,
        "recommendations": rows,
        "trace": trace,
    }


def save_report(report, output_dir=None):
    """保存推荐结果，默认按运行时间创建目录"""

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_dir = Path(output_dir) if output_dir is not None else (
        ROOT
        / "outputs"
        / "agent"
        / report["model_type"]
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "recommendations.json"

    # 独占创建，保留已提交结果
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)

    return output_path
