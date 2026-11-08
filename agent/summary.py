import json

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agent.llm import make_llm
from agent.prompts import ROUND_SUMMARY_SYSTEM_PROMPT


class RoundSummary(BaseModel):
    """实验发现、局限及下一轮计划"""

    findings: list[str] = Field(description="有实验依据的发现，引用候选 ID")
    limitations: list[str] = Field(description="预测偏差、失败假设和证据局限")
    next_round_plan: list[str] = Field(description="下一轮可检验的假设与搜索方向")


async def summarize_round(report, feedback, previous_summary):
    """根据本轮轨迹、反馈和此前总结生成下一轮上下文"""

    llm = make_llm()
    summarizer = create_agent(
        model=llm,
        tools=[],
        system_prompt=ROUND_SUMMARY_SYSTEM_PROMPT,
        response_format=ToolStrategy(RoundSummary),
    )
    result = await summarizer.ainvoke({
        "messages": [{
            "role": "user",
            "content": json.dumps({
                "previous_summary": previous_summary,
                "round_report": report,
                "feedback": feedback,
            }, ensure_ascii=False),
        }],
    })
    return {
        "model_type": report["model_type"],
        "recommendation_sha256": feedback["recommendation_sha256"],
        "summary": result["structured_response"].model_dump(),
        "system_prompt": ROUND_SUMMARY_SYSTEM_PROMPT,
        "llm": {"model": llm.model_name, "max_output_tokens": llm.max_tokens},
        "usage": [
            message.usage_metadata for message in result["messages"]
            if isinstance(message, AIMessage) and message.usage_metadata
        ],
    }
