import argparse
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.predictor import FitnessPredictor
from agent.data_tools import load_experiment_data, register_data_tools


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "scratch": ROOT / "outputs" / "GB1transformer" / "best.pt",
    "esm2": ROOT / "outputs" / "esm2" / "best_model",
}


def main():
    """加载预测模型并启动 MCP 服务"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["scratch", "esm2"],
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda:0"],
        default="cpu",
    )
    parser.add_argument("--feedback", type=Path, help="本组已揭示的历史反馈文件")
    parser.add_argument("--knowledge", action="store_true", help="启用知识图谱查询")
    parser.add_argument("--experiment-id", help="校验反馈所属实验运行")
    args = parser.parse_args()

    # 进程内复用预测模型
    predictor = FitnessPredictor(
        model_path=MODEL_PATHS[args.model],
        model_type=args.model,
        device=args.device,
    )

    mcp = FastMCP("gb1-fitness")

    known, candidates = load_experiment_data(
        args.feedback, args.model, args.knowledge, args.experiment_id,
    )
    register_data_tools(mcp, known, candidates)
    if args.knowledge:
        from agent.knowledge_tools import register_knowledge_tools

        register_knowledge_tools(mcp, known)

    @mcp.tool(structured_output=True)
    def predict_fitness(sequences: list[str]) -> dict[str, object]:
        """预测完整 GB1 序列的 fitness

        输入为 56 位标准氨基酸序列，适用于 GB1 野生型、单点和双点变体
        输出为预测对数富集值，越大表示预测结合富集越强，与输入顺序一致
        预测值不属于实验测量值
        """
        scores = predictor.predict(sequences)

        return {
            "model": args.model,
            "score_type": "predicted_log_enrichment",
            "predictions": [
                {
                    "sequence": sequence,
                    "predicted_fitness": score,
                }
                for sequence, score in zip(sequences, scores)
            ],
        }

    # stdio 用于 MCP 协议通信
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
