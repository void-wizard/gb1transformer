from langchain_core.messages import AIMessage, ToolMessage


async def stream_agent(scientist, inputs, on_event):
    """转发回答和工具事件，返回最终状态"""

    result = None
    async for mode, payload in scientist.astream(
        inputs, stream_mode=["messages", "updates", "values"], version="v1",
    ):
        if mode == "messages":
            message, _ = payload
            # 仅转发回答文本
            if isinstance(message, AIMessage) and message.text:
                await on_event({"type": "token", "text": message.text})
        elif mode == "updates":
            for update in payload.values():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages", []):
                    if isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            await on_event({
                                "type": "tool_start", "id": call["id"],
                                "name": call["name"], "args": call["args"],
                            })
                    elif isinstance(message, ToolMessage):
                        await on_event({
                            "type": "tool_end", "id": message.tool_call_id,
                            "name": message.name, "status": message.status,
                            "content": message.content,
                        })
        elif mode == "values":
            result = payload
    if result is None:
        raise ValueError("Agent 未返回最终状态")
    return result
