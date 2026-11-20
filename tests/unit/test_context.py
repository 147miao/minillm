"""L2/L3 上下文压缩测试（离线、不调真实模型）。"""

from __future__ import annotations

import json
from typing import Any

from minillm.agent.context import ContextManager, estimate_messages_tokens, estimate_tokens
from minillm.config import Settings
from minillm.errors import LLMError
from minillm.events import Event, EventType
from minillm.llm.types import LLMResult, Usage
from minillm.persistence.replay import build_messages, reduce_events


class FakeSmall:
    def __init__(self, content: str = "早期摘要", fail: bool = False) -> None:
        self.content = content
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> LLMResult:
        self.calls.append(kwargs)
        if self.fail:
            raise LLMError("small model down")
        return LLMResult(
            content=self.content,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="small-test",
        )


class Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append((event_type, data))


def _settings(**updates: Any) -> Settings:
    base = Settings(
        _env_file=None,
        llm_api_key="test-key",
        context_budget=20_000,
        prune_threshold=0.5,
        compact_threshold=0.8,
        keep_recent_turns=2,
        wal_fsync=False,
    )
    return base.model_copy(update=updates)


def _state(turns: int = 10, reasoning_len: int = 800):
    events: list[Event] = []
    seq = 0
    for turn in range(turns):
        seq += 1
        events.append(
            Event(
                seq=seq,
                ts="2026-01-01T00:00:00Z",
                session_id="sess_test",
                type=EventType.USER_MESSAGE,
                data={"content": f"任务 {turn}"},
            )
        )
        call_id = f"call_{turn}"
        seq += 1
        events.append(
            Event(
                seq=seq,
                ts="2026-01-01T00:00:00Z",
                session_id="sess_test",
                type=EventType.ASSISTANT_MESSAGE,
                data={
                    "content": "处理中",
                    "reasoning": "想" * reasoning_len,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "run_code",
                                "arguments": json.dumps({"code": "print(1)"}, ensure_ascii=False),
                            },
                        }
                    ],
                    "finish_reason": "tool_calls",
                },
            )
        )
        seq += 1
        events.append(
            Event(
                seq=seq,
                ts="2026-01-01T00:00:00Z",
                session_id="sess_test",
                type=EventType.RUN_CODE_FINISHED,
                data={
                    "call_id": call_id,
                    "exit_code": 0,
                    "stdout_preview": "ok",
                    "stderr_preview": "",
                    "content": "ok",
                    "duration_ms": 1,
                    "tool_calls": [],
                    "synthetic": False,
                },
            )
        )
    return reduce_events({"session_id": "sess_test"}, events)


def _assert_pairs(messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call_id = call.get("id")
            assert any(
                item.get("role") == "tool" and item.get("tool_call_id") == call_id
                for item in messages[index + 1 :]
            ), f"tool call {call_id} 没有配对结果"


def test_estimate_tokens_cjk_and_ascii() -> None:
    assert estimate_tokens("你好") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("你" * 100) == 50
    assert estimate_messages_tokens([{"role": "user", "content": "abcd"}]) >= 1


async def test_l2_prunes_reasoning_and_reduces_tokens() -> None:
    state = _state()
    messages = build_messages(state)
    before = estimate_messages_tokens(messages)
    emitter = Emitter()
    manager = ContextManager(_settings(), FakeSmall())
    view = await manager.prepare(
        state,
        messages,
        {"prompt_tokens": 11_000},
        session_id="sess_test",
        emit_event=emitter,
    )
    assert view.level == 2
    assert view.after_tokens < view.before_tokens
    assert view.summary
    assert view.messages
    _assert_pairs(view.messages)
    assert any(
        message.get("role") == "system" and "历史上下文摘要" in str(message.get("content"))
        for message in view.messages
    )
    assert not any(
        message.get("reasoning")
        for message in view.messages[: len(view.messages) - 6]
        if message.get("role") == "assistant"
    )
    events = [data for kind, data in emitter.events if kind == EventType.CONTEXT_COMPACTED]
    assert len(events) == 1
    assert events[0]["level"] == 2
    assert events[0]["pruned_upto_seq"] > 0
    assert events[0]["kept_message_count"] > 0
    assert any(kind == EventType.LLM_USAGE for kind, _ in emitter.events)
    assert before > 0


async def test_l3_full_compaction_keeps_recent() -> None:
    state = _state()
    messages = build_messages(state)
    emitter = Emitter()
    manager = ContextManager(_settings(), FakeSmall())
    view = await manager.prepare(
        state,
        messages,
        {"prompt_tokens": 17_000},
        session_id="sess_test",
        emit_event=emitter,
    )
    assert view.level == 3
    assert view.after_tokens < view.before_tokens
    assert view.messages[0]["role"] == "system"
    assert view.messages[0]["content"].startswith("[历史上下文摘要]")
    _assert_pairs(view.messages)
    event = [data for kind, data in emitter.events if kind == EventType.CONTEXT_COMPACTED][0]
    assert event["level"] == 3
    assert event["kept_message_count"] == len(view.messages) - 1


async def test_small_model_failure_falls_back() -> None:
    state = _state()
    messages = build_messages(state)
    emitter = Emitter()
    manager = ContextManager(_settings(), FakeSmall(fail=True))
    view = await manager.prepare(
        state,
        messages,
        {"prompt_tokens": 11_000},
        session_id="sess_test",
        emit_event=emitter,
    )
    assert view.level == 2
    assert view.fallback == "small_model_error"
    event = [data for kind, data in emitter.events if kind == EventType.CONTEXT_COMPACTED][0]
    assert event["fallback"] is True
    assert event["fallback_reason"] == "small_model_error"


async def test_no_compaction_below_prune_threshold() -> None:
    state = _state(turns=3)
    messages = build_messages(state)
    emitter = Emitter()
    manager = ContextManager(_settings(context_budget=1_000_000), FakeSmall())
    view = await manager.prepare(
        state,
        messages,
        {"prompt_tokens": 100},
        session_id="sess_test",
        emit_event=emitter,
    )
    assert view.level == 0
    assert view.messages == messages
    assert not emitter.events
