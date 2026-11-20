"""事件序列化与 EventBus 行为。"""

from __future__ import annotations

import asyncio
import json

import pytest

from minillm.events import ALL_EVENT_TYPES, Event, EventBus, EventType
from minillm.ids import utc_now_iso


def test_utc_timestamp_shape() -> None:
    ts = utc_now_iso()
    assert ts.endswith("Z")
    assert "T" in ts
    assert len(ts.split(".")[-1]) == 4  # 毫秒 + Z


def test_event_roundtrip() -> None:
    event = Event(
        seq=3,
        ts="2026-11-20T07:30:00.123Z",
        session_id="sess_abc",
        type=EventType.USER_MESSAGE,
        data={"content": "你好"},
    )
    payload = event.to_dict()
    assert payload["data"] == {"content": "你好"}
    assert Event.from_dict(payload) == event
    assert Event.from_json(event.to_json()) == event
    assert "你好" in event.to_json()


def test_event_rejects_missing_fields() -> None:
    with pytest.raises(ValueError):
        Event.from_dict({"seq": 1, "type": "user.message"})


def test_event_type_catalog_is_frozen() -> None:
    assert len(ALL_EVENT_TYPES) == 18
    for expected in (
        "session.created",
        "user.message",
        "assistant.message",
        "llm.usage",
        "run_code.started",
        "run_code.finished",
        "tool.call",
        "tool.result",
        "context.compacted",
        "skills.injected",
        "todo.updated",
        "subagent.spawned",
        "subagent.message",
        "subagent.completed",
        "turn.started",
        "turn.finished",
        "turn.cancelled",
        "agent.error",
    ):
        assert expected in ALL_EVENT_TYPES


async def test_bus_publish_and_close() -> None:
    bus = EventBus()
    subscription = bus.subscribe("sess_1")

    event = Event(1, utc_now_iso(), "sess_1", EventType.USER_MESSAGE, {"content": "hi"})
    bus.publish(event)

    received = await asyncio.wait_for(subscription.__anext__(), timeout=1)
    assert received == event
    assert bus.subscriber_count == 1

    subscription.close()
    assert bus.subscriber_count == 0
    with pytest.raises(StopAsyncIteration):
        await subscription.__anext__()

    # 关闭后继续 publish 不抛异常
    bus.publish(event)
    await bus.close()


async def test_bus_session_isolation_and_multi_subscriber() -> None:
    bus = EventBus()
    first = bus.subscribe("sess_a")
    second = bus.subscribe("sess_a")
    other = bus.subscribe("sess_b")

    bus.publish(Event(1, utc_now_iso(), "sess_a", EventType.TODO_UPDATED, {"items": []}))
    assert (await asyncio.wait_for(first.__anext__(), 1)).session_id == "sess_a"
    assert (await asyncio.wait_for(second.__anext__(), 1)).session_id == "sess_a"

    bus.publish(Event(2, utc_now_iso(), "sess_b", EventType.TURN_FINISHED, {"turn_id": "t"}))
    assert (await asyncio.wait_for(other.__anext__(), 1)).session_id == "sess_b"

    await bus.close()


async def test_bus_publish_without_subscribers() -> None:
    bus = EventBus()
    bus.publish(Event(1, utc_now_iso(), "nobody", EventType.AGENT_ERROR, {"message": "x"}))
    assert bus.subscriber_count == 0
    await bus.close()


async def test_async_for_iteration() -> None:
    bus = EventBus()
    subscription = bus.subscribe("sess_loop")
    bus.publish(Event(1, utc_now_iso(), "sess_loop", EventType.USER_MESSAGE, {"content": "a"}))
    bus.publish(Event(2, utc_now_iso(), "sess_loop", EventType.USER_MESSAGE, {"content": "b"}))

    seen = []
    async for event in subscription:
        seen.append(event.seq)
        if len(seen) == 2:
            break
    subscription.close()
    assert seen == [1, 2]
    await bus.close()


def test_event_json_is_compact_and_utf8() -> None:
    event = Event(1, utc_now_iso(), "sess_x", EventType.USER_MESSAGE, {"content": "中文"})
    payload = json.loads(event.to_json())
    assert payload["data"]["content"] == "中文"
