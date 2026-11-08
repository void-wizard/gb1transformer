import argparse
import asyncio
import json
import logging
from contextlib import suppress
from pathlib import Path
from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, field_validator
from langchain_core.messages import messages_from_dict, messages_to_dict

from agent import conversations
from agent.scientist import run_agent
from src.prepare_data import WT_SEQUENCE


ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="GB1 Agent")
run_lock = asyncio.Lock()


class AgentRequest(BaseModel):
    """聊天请求参数"""

    model_config = ConfigDict(extra="forbid")
    task: str
    sequence: str = WT_SEQUENCE
    knowledge: bool = False
    recommend: bool = True
    conversation_id: UUID | None = None

    @field_validator("task")
    @classmethod
    def nonempty_task(cls, value):
        """校验任务非空"""
        if not value.strip():
            raise ValueError("请输入任务")
        return value.strip()

    @field_validator("sequence")
    @classmethod
    def supported_sequence(cls, value):
        """校验 GB1 野生型背景"""
        value = "".join(value.split()).upper()
        if value != WT_SEQUENCE:
            raise ValueError("当前仅支持页面预填的 GB1 野生型序列")
        return value


@app.get("/api/config")
async def config():
    """返回野生型序列、模型名和推荐数量"""
    return {"sequence": WT_SEQUENCE, "model": "自训练 Transformer", "top_k": 10}


@app.post("/api/agent")
async def agent(request: AgentRequest):
    """流式返回 Agent 事件，断开时释放任务和 MCP 连接"""
    if run_lock.locked():
        raise HTTPException(409, "已有任务在运行或清理中，请稍后重试")
    conversation_id = str(request.conversation_id or uuid4())
    conversation = conversations.load_conversation(conversation_id) or {
        "id": conversation_id, "title": request.task[:36], "turns": [], "messages": [],
    }
    await run_lock.acquire()
    queue = asyncio.Queue()
    stopping = asyncio.Event()
    output_dir = ROOT / "outputs/ui" / uuid4().hex
    turn = {"task": request.task, "knowledge": request.knowledge,
            "recommend": request.recommend, "events": [], "status": "running"}
    conversation["turns"].append(turn)

    async def publish(event):
        # 合并连续文本事件
        events = turn["events"]
        if event["type"] == "token" and events and events[-1]["type"] == "token":
            events[-1] = {"type": "token", "text": events[-1]["text"] + event["text"]}
        else:
            events.append(event)
        await queue.put(event)

    async def produce():
        try:
            conversations.save_conversation(conversation)
            await queue.put({"type": "status", "text": "正在连接 Agent 和模型工具…"})
            result = await run_agent(
                "scratch", request.task, recommend=request.recommend,
                knowledge=request.knowledge, output_dir=output_dir, on_event=publish,
                messages=messages_from_dict(conversation["messages"]),
            )
            # 兼容 MCP 上下文吞掉取消异常的情况
            if stopping.is_set():
                return
            report = result.get("report")
            conversation["messages"] = messages_to_dict(result.get("messages", []))
            turn["status"] = "complete"
            done = {
                "type": "done", "recommendations": report["recommendations"] if report else [],
                "saved_to": str(output_dir.relative_to(ROOT) / "recommendations.json") if report else None,
                "conversation_id": conversation_id,
            }
            turn["events"].append(done)
            conversations.save_conversation(conversation)
            await queue.put(done)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not stopping.is_set():
                turn["status"] = "error"
                logging.exception("UI Agent 执行失败")
                # 前端仅返回异常类型
                await publish({"type": "error", "text": f"执行失败（{type(error).__name__}），请查看后端日志"})
        finally:
            try:
                if turn["status"] == "running":
                    turn["status"] = "stopped"
                conversations.save_conversation(conversation)
            finally:
                run_lock.release()

    worker = asyncio.create_task(produce())

    async def events():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] in {"done", "error"}:
                    break
        finally:
            stopping.set()
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


@app.get("/api/conversations")
async def conversation_list():
    return conversations.list_conversations()


@app.get("/api/conversations/{conversation_id}")
async def conversation_detail(conversation_id: UUID):
    conversation = conversations.load_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(404, "会话不存在")
    return conversations.public_conversation(conversation)


@app.delete("/api/conversations/{conversation_id}")
async def conversation_delete(conversation_id: UUID):
    if run_lock.locked():
        raise HTTPException(409, "请先停止当前任务，再删除会话")
    (conversations.DIRECTORY / f"{conversation_id}.json").unlink(missing_ok=True)
    return {"deleted": True}


# 静态资源限定为前端构建目录
dist = ROOT / "ui/dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
