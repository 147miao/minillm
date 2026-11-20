"""事件模型与进程内事件总线。

事件是 MiniLLM 的唯一事实来源：所有结果型事件先落盘（JSONL），再经 EventBus 广播给
SSE 订阅者；流式增量（assistant.delta 等）只在内存中走 SSE，不进入 JSONL。

字段、seq/ts 规则与只 append 语义冻结在 docs/EVENTS.md，改动需同步文档。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any


class EventType:
    """事件类型常量（字符串，snake_case）。"""

    SESSION_CREATED = "session.created"
    USER_MESSAGE = "user.message"
    ASSISTANT_MESSAGE = "assistant.message"
    LLM_USAGE = "llm.usage"
    RUN_CODE_STARTED = "run_code.started"
    RUN_CODE_FINISHED = "run_code.finished"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    CONTEXT_COMPACTED = "context.compacted"
    SKILLS_INJECTED = "skills.injected"
    TODO_UPDATED = "todo.updated"
    SUBAGENT_SPAWNED = "subagent.spawned"
    SUBAGENT_MESSAGE = "subagent.message"
    SUBAGENT_COMPLETED = "subagent.completed"
    TURN_STARTED = "turn.started"
    TURN_FINISHED = "turn.finished"
    TURN_CANCELLED = "turn.cancelled"
    AGENT_ERROR = "agent.error"


ALL_EVENT_TYPES: frozenset[str] = frozenset(
    value
    for name, value in vars(EventType).items()
    if not name.startswith("_") and isinstance(value, str)
)


@dataclass(slots=True)
class Event:
    """一条持久化事件。"""

    seq: int
    ts: str
    session_id: str
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "session_id": self.session_id,
            "type": self.type,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Event:
        if not isinstance(raw, dict):
            raise ValueError("事件必须是 JSON 对象")
        try:
            seq = int(raw["seq"])
            ts = str(raw["ts"])
            session_id = str(raw["session_id"])
            event_type = str(raw["type"])
        except KeyError as exc:
            raise ValueError(f"事件缺少字段：{exc.args[0]}") from exc
        data = raw.get("data", {})
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("事件 data 必须是 JSON 对象")
        return cls(seq=seq, ts=ts, session_id=session_id, type=event_type, data=data)

    @classmethod
    def from_json(cls, line: str) -> Event:
        return cls.from_dict(json.loads(line))


_SENTINEL = object()


class EventSubscription:
    """单个 SSE/测试订阅者；非阻塞入队，关闭后以哨兵结束迭代。"""

    def __init__(self, bus: EventBus, session_id: str) -> None:
        self.session_id = session_id
        self._bus = bus
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._closed = False

    def __aiter__(self) -> EventSubscription:
        return self

    async def __anext__(self) -> Event:
        if self._closed and self._queue.empty():
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is _SENTINEL:
            raise StopAsyncIteration
        return item

    def _offer(self, event: Event) -> None:
        if self._closed:
            return
        self._queue.put_nowait(event)

    def close(self) -> None:
        """取消订阅；已入队事件仍可读完，之后迭代结束。"""

        if self._closed:
            return
        self._closed = True
        self._bus._unsubscribe(self)
        self._queue.put_nowait(_SENTINEL)


class EventBus:
    """按 session 分组的进程内发布/订阅总线。"""

    def __init__(self) -> None:
        self._subs: dict[str, set[EventSubscription]] = {}

    def subscribe(self, session_id: str) -> EventSubscription:
        sub = EventSubscription(self, session_id)
        self._subs.setdefault(session_id, set()).add(sub)
        return sub

    def _unsubscribe(self, sub: EventSubscription) -> None:
        subs = self._subs.get(sub.session_id)
        if not subs:
            return
        subs.discard(sub)
        if not subs:
            self._subs.pop(sub.session_id, None)

    def publish(self, event: Event) -> None:
        """非阻塞广播；任何订阅者异常都不影响发布方。"""

        for sub in tuple(self._subs.get(event.session_id, ())):
            try:
                sub._offer(event)
            except Exception:  # pragma: no cover - 防御性兜底
                continue

    async def close(self) -> None:
        for subs in tuple(self._subs.values()):
            for sub in tuple(subs):
                sub.close()
        self._subs.clear()

    @property
    def subscriber_count(self) -> int:
        return sum(len(subs) for subs in self._subs.values())
