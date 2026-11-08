from typing import NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import after_model
from langchain_core.messages import AIMessage, HumanMessage

from agent.prompts import RECOMMENDATION_RETRY_PROMPT


class SubmissionState(AgentState):
    submission_retried: NotRequired[bool]


@after_model(state_schema=SubmissionState, can_jump_to=["model"])
def require_recommendations(state, runtime):
    """普通文本结束时保留工具历史，提醒补交一次结构化推荐"""

    if state.get("structured_response") is not None:
        return None
    message = state["messages"][-1]
    if not isinstance(message, AIMessage) or message.tool_calls:
        return None
    if state.get("submission_retried"):
        raise ValueError("Agent 经提醒后仍未提交 RecommendationBatch，本轮推荐未保存")
    return {
        "messages": [HumanMessage(content=RECOMMENDATION_RETRY_PROMPT)],
        "submission_retried": True,
        "jump_to": "model",
    }
