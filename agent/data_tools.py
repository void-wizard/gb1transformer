import json
from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import Field


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "splits"

PUBLIC_COLUMNS = [
    "id",
    "sequence",
    "mutations",
    "n_mutations",
]


def filter_position(frame, position):
    """按突变位点筛选，未指定时返回全部记录"""

    if position is None:
        return frame

    # 按完整突变标记匹配位点
    pattern = rf"(?:^|:)[A-Z]{position}[A-Z](?:$|:)"

    return frame.loc[
        frame["mutations"].str.contains(pattern, regex=True)
    ]


def check_scope(record, model_type, knowledge=False, experiment_id=None):
    """校验记录所属模型、知识模式和实验"""

    if record["model_type"] != model_type:
        raise ValueError("记录与当前预测模型组不一致")
    # knowledge 缺省为普通模式
    if record.get("knowledge", False) != knowledge:
        raise ValueError("记录与当前知识增强模式不一致")
    if experiment_id is not None and record.get("experiment_id") != experiment_id:
        raise ValueError("记录属于另一实验运行")


def load_experiment_data(
    feedback_path=None, model_type=None, knowledge=False, experiment_id=None,
):
    """加载训练集、本组已揭示反馈和剩余候选"""

    # 训练集作为初始已知实验
    known = pd.read_csv(
        DATA_DIR / "train.csv",
        usecols=PUBLIC_COLUMNS + ["fitness"],
    )
    known["source"] = "data/splits/train.csv"

    # 候选数据不含标签
    candidates = pd.read_csv(
        DATA_DIR / "candidates.csv",
        usecols=PUBLIC_COLUMNS,
    ).sort_values("id")

    if feedback_path is not None:
        history = json.loads(Path(feedback_path).read_text(encoding="utf-8"))
        check_scope(history, model_type, knowledge, experiment_id)
        if history["results"]:
            revealed = pd.DataFrame(history["results"]).rename(columns={
                "candidate_id": "id", "measured_fitness": "fitness",
            })
            # 合并本组已揭示的实验标签
            additional = candidates.merge(
                revealed[["id", "fitness"]], on="id", validate="one_to_one",
            )
            if len(additional) != len(revealed):
                raise ValueError("历史反馈包含不在候选池中的 ID")
            additional["source"] = (
                f"revealed_feedback:{history.get('experiment_id') or 'legacy'}"
            )
            known = pd.concat([known, additional], ignore_index=True)
            candidates = candidates.loc[~candidates["id"].isin(revealed["id"])]

    return known, candidates


def register_data_tools(mcp, known, candidates):
    """注册实验数据查询工具"""

    @mcp.tool(structured_output=True)
    def inspect_known_data(
        position: Annotated[
            int | None, Field(ge=2, le=56)
        ] = None,
        top_k: Annotated[
            int, Field(ge=1, le=100)
        ] = 10,
    ) -> dict[str, object]:
        """查询训练集及本组已揭示反馈的统计信息和高分变体

        position 为可选突变位点，top_k 为返回的高分记录数
        fitness 为实验计数计算的对数富集值
        位点统计包含单点和双点变体，不代表位点的独立效应
        """

        subset = filter_position(known, position)

        # 同分时按 ID 排序
        top_variants = subset.sort_values(
            ["fitness", "id"],
            ascending=[False, True],
        ).head(top_k)

        return {
            "source": "training_data_and_revealed_feedback",
            "score_type": "measured_log_enrichment",
            "count": len(subset),
            "mean_fitness": (
                float(subset["fitness"].mean())
                if len(subset)
                else None
            ),
            "top_variants": top_variants.to_dict("records"),
        }

    @mcp.tool(structured_output=True)
    def find_candidates(
        position: Annotated[
            int | None, Field(ge=2, le=56)
        ] = None,
        offset: Annotated[
            int, Field(ge=0)
        ] = 0,
        limit: Annotated[
            int, Field(ge=1, le=100)
        ] = 20,
    ) -> dict[str, object]:
        """按 ID 分页查询尚未揭示标签的候选，ID 顺序与 fitness 无关

        position 为可选突变位点，offset 为起始偏移，limit 为返回数量上限
        返回候选、匹配总数及 next_offset，无下一页时为 None
        """

        subset = filter_position(candidates, position)
        page = subset.iloc[offset:offset + limit]

        next_offset = offset + len(page)

        return {
            "total_matches": len(subset),
            "candidates": page.to_dict("records"),
            "next_offset": (
                next_offset
                if next_offset < len(subset)
                else None
            ),
        }
