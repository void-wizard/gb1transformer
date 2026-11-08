import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID


DIRECTORY = Path(__file__).resolve().parents[1] / "outputs/ui/conversations"


def load_conversation(conversation_id):
    """按 UUID 读取会话"""
    path = DIRECTORY / f"{UUID(str(conversation_id))}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_conversation(conversation):
    """原子保存会话记录"""
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    conversation["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = DIRECTORY / f"{UUID(conversation['id'])}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(conversation, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def public_conversation(conversation):
    """返回不含协议消息的会话数据"""
    return {key: value for key, value in conversation.items() if key != "messages"}


def list_conversations():
    """按更新时间返回会话列表"""
    rows = []
    for path in DIRECTORY.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({key: data[key] for key in ("id", "title", "updated_at")})
    return sorted(rows, key=lambda row: row["updated_at"], reverse=True)
