import argparse
import asyncio
import json
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.data_tools import check_scope
from agent.llm import make_llm
from agent.prompts import (
    KNOWLEDGE_SYSTEM_PROMPT,
    RECOMMENDATION_SYSTEM_PROMPT,
    SCIENTIST_SYSTEM_PROMPT,
    TASK_TEMPLATE,
)
from agent.results import RecommendationBatch, build_report, save_report
from agent.streaming import stream_agent
from agent.submission import require_recommendations
from src.prepare_data import WT_SEQUENCE


ROOT = Path(__file__).resolve().parents[1]


async def run_agent(
    model_type, task, recommend=False, feedback_path=None, output_dir=None,
    knowledge=False, experiment_id=None, on_event=None, messages=None,
):
    """连接 MCP 并执行 Agent 任务"""

    server_args = ["-m", "agent.mcp_server", "--model", model_type]
    if knowledge:
        server_args.append("--knowledge")
    excluded_ids = set()
    if feedback_path is not None:
        feedback_path = Path(feedback_path).resolve()
        history = json.loads(feedback_path.read_text(encoding="utf-8"))
        check_scope(history, model_type, knowledge, experiment_id)
        experiment_id = history.get("experiment_id", experiment_id)
        excluded_ids = {row["candidate_id"] for row in history["results"]}
        server_args.extend(["--feedback", str(feedback_path)])
    if experiment_id is not None:
        server_args.extend(["--experiment-id", experiment_id])

    parameters = StdioServerParameters(
        command=sys.executable,
        args=server_args,
        cwd=str(ROOT),
    )

    # 任务内复用 MCP 连接和模型
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            system_prompt = SCIENTIST_SYSTEM_PROMPT
            if recommend:
                system_prompt += RECOMMENDATION_SYSTEM_PROMPT
            if knowledge:
                system_prompt += KNOWLEDGE_SYSTEM_PROMPT

            llm = make_llm()
            scientist = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
                middleware=[require_recommendations] if recommend else [],
                response_format=(
                    ToolStrategy(RecommendationBatch)
                    if recommend
                    else None
                ),
            )

            inputs = {
                # 沿用历史消息及工具调用记录
                "messages": list(messages or []) + [
                    {
                        "role": "user",
                        "content": TASK_TEMPLATE.format(
                            wild_type=WT_SEQUENCE,
                            task=task,
                        ),
                    }
                ]
            }
            if on_event is None:
                result = await scientist.ainvoke(inputs)
            else:
                await on_event({"type": "status", "text": "工具已连接，Agent 正在分析任务"})
                result = await stream_agent(scientist, inputs, on_event)

            # 命令行输出轨迹，UI 使用流式事件
            for message in result["messages"] if on_event is None else ():
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        print(f"\n调用工具：{call['name']}")
                        print(f"工具参数：{call['args']}")

                    if message.content:
                        print(f"\nAgent：\n{message.content}")

                elif isinstance(message, ToolMessage):
                    print(f"\n工具结果：\n{message.content}")

            if recommend:
                report = build_report(
                    result, model_type, task, system_prompt=system_prompt,
                    excluded_ids=excluded_ids,
                )
                report["llm"] = {
                    "model": llm.model_name,
                    "max_output_tokens": llm.max_tokens,
                }
                report["knowledge"] = knowledge
                report["experiment_id"] = experiment_id
                output_path = save_report(report, output_dir=output_dir)
                result["report"] = report
                result["report_path"] = str(output_path)
                print(f"\n推荐结果已保存：{output_path}")

            return result


def main():
    """运行指定模型的 Agent 任务"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["scratch", "esm2"],
        required=True,
    )
    parser.add_argument(
        "--task",
        required=True,
        help="交给 Agent 的任务",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="提交十个候选并保存结构化推荐结果",
    )
    parser.add_argument("--feedback", type=Path, help="本组已揭示的历史反馈文件")
    parser.add_argument("--knowledge", action="store_true", help="启用知识图谱增强")
    args = parser.parse_args()

    if not args.task.strip():
        parser.error("--task 不能为空")

    task = args.task

    asyncio.run(
        run_agent(
            args.model, task, recommend=args.recommend, feedback_path=args.feedback,
            knowledge=args.knowledge,
        )
    )


if __name__ == "__main__":
    main()
