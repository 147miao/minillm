"""LLM 流式累积、scripted 记录/状态与请求头契约（全部离线）。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from minillm.errors import ConfigError, LLMError
from minillm.llm.client import OpenAICompatClient
from minillm.llm.scripted import ScriptedLLMClient
from minillm.llm.types import (
    LLMDone,
    LLMReasoningDelta,
    LLMTextDelta,
    LLMToolCallDelta,
    build_request_headers,
)

RUN_CODE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_code",
        "description": "执行 Python 程序",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


def _write_script(tmp_path, document: dict[str, Any]):
    script = tmp_path / "script.json"
    script.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return script


async def _drain(events) -> list[Any]:
    return [event async for event in events]


def _stream_kwargs() -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [RUN_CODE_TOOL],
        "session_id": "sess_test",
    }


def test_request_headers_override(settings) -> None:
    headers = build_request_headers(
        {"x-opencode-session": "base", "X-Custom": "1"}, "sess_1", "sub"
    )
    assert headers["x-opencode-session"] == "minillm-sess_1"
    assert headers["x-minillm-agent-kind"] == "sub"
    assert headers["User-Agent"] == "minillm/0.1.0"
    assert headers["X-Custom"] == "1"


async def test_scripted_stream_accumulates(tmp_path, settings) -> None:
    content = "第一段内容" * 8
    script = _write_script(
        tmp_path,
        {
            "responses": [
                {
                    "content": content,
                    "reasoning": "先想一想",
                    "tool_calls": [{"name": "run_code", "arguments": {"code": "print(1)"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "reasoning_tokens": 2,
                        "total_tokens": 15,
                    },
                }
            ],
            "small_responses": [{"content": "摘要内容"}],
        },
    )
    client = ScriptedLLMClient(settings, path=script)

    events = await _drain(client.stream_chat(**_stream_kwargs()))

    text = "".join(event.text for event in events if isinstance(event, LLMTextDelta))
    reasoning = "".join(event.text for event in events if isinstance(event, LLMReasoningDelta))
    tool_deltas = [event for event in events if isinstance(event, LLMToolCallDelta)]
    assert text == content
    assert reasoning == "先想一想"
    assert len(tool_deltas) == 1
    assert tool_deltas[0].name == "run_code"
    assert json.loads(tool_deltas[0].args_fragment) == {"code": "print(1)"}

    done = events[-1]
    assert isinstance(done, LLMDone)
    assert done.result.tool_calls[0].name == "run_code"
    assert done.result.tool_calls[0].arguments == {"code": "print(1)"}
    assert done.result.usage.prompt_tokens == 10
    assert done.result.finish_reason == "tool_calls"


async def test_scripted_complete_and_state(tmp_path, settings) -> None:
    script = _write_script(
        tmp_path,
        {
            "responses": [],
            "small_responses": [{"content": "摘要内容", "model": "small-test"}],
        },
    )
    client = ScriptedLLMClient(settings, path=script)
    result = await client.complete(
        messages=[{"role": "user", "content": "summarize"}], session_id="sess_test"
    )
    assert result.content == "摘要内容"
    assert result.model == "small-test"

    state = json.loads(script.with_name(script.name + ".state.json").read_text(encoding="utf-8"))
    assert state["used"]["small_responses"] == [0]


async def test_scripted_records_calls_and_headers(tmp_path, settings) -> None:
    script = _write_script(
        tmp_path,
        {"responses": [{"content": "ok"}], "small_responses": []},
    )
    client = ScriptedLLMClient(settings, path=script)
    await _drain(client.stream_chat(**_stream_kwargs()))

    calls_path = script.with_name(script.name + ".calls.jsonl")
    records = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["headers"]["x-opencode-session"] == "minillm-sess_test"
    assert record["headers"]["x-minillm-agent-kind"] == "main"
    assert record["headers"]["User-Agent"] == "minillm/0.1.0"
    assert record["tools"] == [RUN_CODE_TOOL]
    assert record["purpose"] == "chat"


async def test_scripted_state_survives_new_instance(tmp_path, settings) -> None:
    script = _write_script(
        tmp_path,
        {"responses": [{"content": "only once"}], "small_responses": []},
    )
    first = ScriptedLLMClient(settings, path=script)
    events = await _drain(first.stream_chat(**_stream_kwargs()))
    assert events[-1].result.content == "only once"

    second = ScriptedLLMClient(settings, path=script)
    with pytest.raises(LLMError):
        await _drain(second.stream_chat(**_stream_kwargs()))


async def test_scripted_match_filters(tmp_path, settings) -> None:
    script = _write_script(
        tmp_path,
        {
            "responses": [
                {"match": {"session_id": "sess_b"}, "content": "B"},
                {"match": {"agent_kind": "sub"}, "content": "SUB"},
                {"content": "default"},
            ],
            "small_responses": [],
        },
    )
    client = ScriptedLLMClient(settings, path=script)

    by_session = await _drain(
        client.stream_chat(
            messages=[{"role": "user", "content": "x"}],
            tools=[],
            session_id="sess_b",
        )
    )
    assert by_session[-1].result.content == "B"

    by_agent = await _drain(
        client.stream_chat(
            messages=[{"role": "user", "content": "x"}],
            tools=[],
            session_id="sess_c",
            agent_kind="sub",
        )
    )
    assert by_agent[-1].result.content == "SUB"

    fallback = await _drain(
        client.stream_chat(
            messages=[{"role": "user", "content": "x"}],
            tools=[],
            session_id="sess_c",
        )
    )
    assert fallback[-1].result.content == "default"


async def test_scripted_small_bucket_required(tmp_path, settings) -> None:
    script = _write_script(tmp_path, {"responses": [{"content": "main"}], "small_responses": []})
    client = ScriptedLLMClient(settings, path=script)
    with pytest.raises(LLMError):
        await client.complete(messages=[{"role": "user", "content": "x"}])


def test_scripted_missing_script_raises(settings, tmp_path) -> None:
    with pytest.raises(ConfigError):
        ScriptedLLMClient(settings, path=tmp_path / "missing.json")


def test_async_http_client_falls_back_without_env(monkeypatch) -> None:
    import httpx

    from minillm.llm.client import _build_async_http_client

    class _Stub:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def _factory(**kwargs):
        if kwargs.get("trust_env", True):
            raise httpx.InvalidURL("Invalid port: ':1]'")
        return _Stub(**kwargs)

    monkeypatch.setattr("minillm.llm.client.httpx.AsyncClient", _factory)
    built = _build_async_http_client(3.0)
    assert built.kwargs["trust_env"] is False


# ==== 真实客户端的离线替身 ====
class _FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[Any] | None = None,
    finish: str | None = None,
    usage: Any = None,
) -> Any:
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
        model_extra={},
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)],
        usage=usage,
    )


class _FakeAsyncOpenAI:
    chunks: list[Any] = []
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs: Any) -> _FakeStream:
        type(self).last_kwargs = kwargs
        return _FakeStream(type(self).chunks)

    async def close(self) -> None:
        return None


async def test_openai_client_accumulates_stream(settings, monkeypatch) -> None:
    _FakeAsyncOpenAI.chunks = [
        _chunk(content="你", reasoning="先想"),
        _chunk(
            content="好",
            tool_calls=[
                SimpleNamespace(
                    index=0,
                    id="call_1",
                    function=SimpleNamespace(name="run_code", arguments='{"code": '),
                )
            ],
        ),
        _chunk(
            tool_calls=[
                SimpleNamespace(
                    index=0,
                    id=None,
                    function=SimpleNamespace(name=None, arguments='"print(1)"}'),
                )
            ],
            finish="tool_calls",
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=7,
                completion_tokens=3,
                total_tokens=10,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
            ),
        ),
    ]
    _FakeAsyncOpenAI.last_kwargs = None
    monkeypatch.setattr("minillm.llm.client.AsyncOpenAI", _FakeAsyncOpenAI)

    client = OpenAICompatClient(settings)
    events = await _drain(client.stream_chat(**_stream_kwargs()))

    text = "".join(event.text for event in events if isinstance(event, LLMTextDelta))
    reasoning = "".join(event.text for event in events if isinstance(event, LLMReasoningDelta))
    assert text == "你好"
    assert reasoning == "先想"

    done = events[-1]
    assert isinstance(done, LLMDone)
    assert done.result.tool_calls[0].id == "call_1"
    assert done.result.tool_calls[0].name == "run_code"
    assert done.result.tool_calls[0].arguments == {"code": "print(1)"}
    assert done.result.usage.prompt_tokens == 7
    assert done.result.usage.reasoning_tokens == 2
    assert done.result.finish_reason == "tool_calls"

    kwargs = _FakeAsyncOpenAI.last_kwargs
    assert kwargs is not None
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["max_tokens"] == settings.llm_max_tokens
    assert kwargs["extra_headers"]["x-opencode-session"] == "minillm-sess_test"
    assert kwargs["extra_headers"]["x-minillm-agent-kind"] == "main"
    assert kwargs["extra_headers"]["User-Agent"] == "minillm/0.1.0"


async def test_openai_client_error_is_sanitized(settings, monkeypatch) -> None:
    class _RaisingAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs: Any) -> Any:
            raise RuntimeError("boom test-key")

    monkeypatch.setattr("minillm.llm.client.AsyncOpenAI", _RaisingAsyncOpenAI)

    client = OpenAICompatClient(settings)
    with pytest.raises(LLMError) as info:
        await _drain(client.stream_chat(**_stream_kwargs()))
    message = str(info.value)
    assert "test-key" not in message
    assert "***" in message
