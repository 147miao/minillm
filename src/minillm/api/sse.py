"""SSE 帧编码与事件流生成。

持久事件形如：
    id: <seq>
    data: <完整事件 JSON>

流式增量形如（没有 id，客户端用 type 区分）：
    data: {"seq": null, "type": "assistant.delta", "data": {"kind": "text", "text": "..."}}

客户端断开时，StreamingResponse 会取消生成器，finally 负责关闭订阅。这里不主动
轮询 request.is_disconnected()，避免 ASGITransport/部分服务器把“暂时没有消息”
误判为断开。
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

HEARTBEAT_SECONDS = 15.0
DELTA_TYPES = frozenset({"assistant.delta", "reasoning.delta"})
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def event_to_dict(raw: Any) -> dict[str, Any] | None:
    """把 Event / pydantic / dict / (type, data) 统一成 dict。"""

    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        event_type, data = raw
        if not isinstance(data, dict):
            data = {"text": data}
        return {"type": str(event_type), "data": data}
    if hasattr(raw, "to_dict"):
        value = raw.to_dict()
    elif hasattr(raw, "model_dump"):
        value = raw.model_dump()
    elif hasattr(raw, "__dict__"):
        value = dict(vars(raw))
    else:
        raise TypeError(f"无法序列化事件对象：{type(raw).__name__}")
    if not isinstance(value, dict):
        raise TypeError("事件对象必须能转成 dict")
    return value


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _event_seq(event: dict[str, Any]) -> int | None:
    return _as_int(event.get("seq"))


def _normalize_delta(event: dict[str, Any]) -> dict[str, Any] | None:
    event_type = str(event.get("type") or "")
    data = event.get("data")
    if not isinstance(data, dict):
        data = {}
    kind = str(data.get("kind") or ("reasoning" if event_type == "reasoning.delta" else "text"))
    if kind not in {"text", "reasoning"}:
        kind = "text"
    text = data.get("text")
    if text is None:
        text = data.get("content")
    if text is None or str(text) == "":
        return None
    out_type = event_type
    if out_type not in DELTA_TYPES:
        out_type = "reasoning.delta" if kind == "reasoning" else "assistant.delta"
    return {
        "seq": None,
        "type": out_type,
        "data": {"kind": kind, "text": str(text)},
    }


def _persistent_frame(event: dict[str, Any], session_id: str) -> str:
    payload = dict(event)
    payload["seq"] = int(event["seq"])
    payload.setdefault("session_id", session_id)
    if not isinstance(payload.get("data"), dict):
        payload["data"] = {}
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"id: {payload['seq']}\ndata: {body}\n\n"


def _delta_frame(event: dict[str, Any]) -> str:
    body = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"data: {body}\n\n"


class _QueueIterator:
    """把 asyncio.Queue 适配成异步迭代器；None 作为结束哨兵。"""

    def __init__(self, queue: asyncio.Queue[Any]) -> None:
        self._queue = queue

    def __aiter__(self) -> _QueueIterator:
        return self

    async def __anext__(self) -> Any:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item


def _as_iterator(subscription: Any) -> Any:
    if hasattr(subscription, "__aiter__"):
        return subscription.__aiter__()
    if hasattr(subscription, "get"):
        return _QueueIterator(subscription)
    raise TypeError("订阅对象必须支持异步迭代或提供 get()")


async def close_subscription(subscription: Any) -> None:
    """尽力关闭订阅；同步/异步 close 都兼容。"""

    if subscription is None:
        return
    for name in ("aclose", "close"):
        fn = getattr(subscription, name, None)
        if callable(fn):
            try:
                result = fn()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - 关闭失败不能影响响应收尾
                pass
            return


async def event_stream(
    subscription: Any,
    replay: Iterable[dict[str, Any]],
    session_id: str,
    after: int,
) -> AsyncIterator[str]:
    """先回放持久事件，再消费实时事件；按 seq 去重。"""

    last_seq = max(0, int(after))
    try:
        for event in sorted(replay, key=lambda item: _event_seq(item) or 0):
            if not isinstance(event, dict):
                continue
            if str(event.get("type") or "") in DELTA_TYPES:
                continue
            seq = _event_seq(event)
            if seq is None or seq <= last_seq:
                continue
            last_seq = seq
            yield _persistent_frame(event, session_id)

        iterator = _as_iterator(subscription)
        while True:
            try:
                raw = await asyncio.wait_for(iterator.__anext__(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            except StopAsyncIteration:
                break
            event = event_to_dict(raw)
            if event is None:
                continue
            event_type = str(event.get("type") or "")
            if event_type in {"heartbeat", "ping"}:
                continue
            seq = _event_seq(event)
            if event_type in DELTA_TYPES or seq is None:
                delta = _normalize_delta(event)
                if delta is not None:
                    yield _delta_frame(delta)
                continue
            if seq <= last_seq:
                continue
            last_seq = seq
            yield _persistent_frame(event, session_id)
    finally:
        await close_subscription(subscription)
