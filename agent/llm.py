import os
from pathlib import Path

from dotenv import dotenv_values
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek


ROOT = Path(__file__).resolve().parents[1]


class ToolHistoryDeepSeek(ChatDeepSeek):
    """兼容 DeepSeek 工具调用历史中的 reasoning_content"""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(
            input_,
            stop=stop,
            **kwargs,
        )

        # 思考模式使用自动工具选择
        thinking = (payload.get("extra_body") or {}).get("thinking", {})
        if thinking.get("type") == "enabled" and payload.get("tool_choice") == "required":
            payload["tool_choice"] = "auto"

        # 补齐工具调用历史中的 reasoning_content
        messages = self._convert_input(input_).to_messages()

        for message, serialized in zip(messages, payload["messages"]):
            if isinstance(message, AIMessage):
                serialized["reasoning_content"] = (
                    message.additional_kwargs.get("reasoning_content") or ""
                )

        return payload


def make_llm():
    """从项目配置创建 DeepSeek 客户端"""

    config = dotenv_values(ROOT / ".env")

    return ToolHistoryDeepSeek(
        api_key=config["api_key"],
        api_base=config["base_url"],
        model=config["model"],
        temperature=float(config["temperature"]),
        reasoning_effort=config["reasoning_effort"],
        # 单次生成的输出上限
        max_tokens=int(os.environ.get(
            "DEEPSEEK_MAX_OUTPUT_TOKENS",
            config.get("max_output_tokens") or "65536",
        )),
        extra_body={
            "thinking": {
                "type": config["thinking"],
            }
        },
        timeout=120,
    )
