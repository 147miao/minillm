"""JSONL 只追加、重放、压缩视图与崩溃恢复。"""

from __future__ import annotations

import json

import pytest

from minillm.errors import SessionError
from minillm.events import EventType
from minillm.persistence.replay import build_messages, load_state, reduce_events
from minillm.persistence.store import SessionStore


def _tool_calls(call_id: str = "call_1") -> list[dict]:
    return [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "run_code",
                "arguments": json.dumps({"code": "print(1)"}, ensure_ascii=False),
            },
        }
    ]


def _assistant(
    content: str = "",
    reasoning: str = "",
    tool_calls: list[dict] | None = None,
) -> dict:
    return {
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls or [],
        "finish_reason": "tool_calls" if tool_calls else "stop",
    }


def _assert_pairing(messages: list[dict]) -> None:
    index = 0
    while index < len(messages):
        message = messages[index]
        if message["role"] == "assistant" and message.get("tool_calls"):
            expected = [call["id"] for call in message["tool_calls"]]
            following = messages[index + 1 : index + 1 + len(expected)]
            assert [item["role"] for item in following] == ["tool"] * len(expected)
            assert [item["tool_call_id"] for item in following] == expected
            index += 1 + len(expected)
        else:
            assert message["role"] != "tool", f"孤立 tool 消息：{message}"
            index += 1


def test_append_only_and_seq(store: SessionStore) -> None:
    meta = store.create_session({"title": "demo", "workspace": "/tmp/ws", "model": "m"})
    session_id = meta["session_id"]
    path = store.events_path(session_id)

    before = path.read_text(encoding="utf-8")
    first = store.append(session_id, EventType.USER_MESSAGE, {"content": "hi"})
    after_first = path.read_text(encoding="utf-8")
    assert after_first.startswith(before)

    second = store.append(session_id, EventType.ASSISTANT_MESSAGE, _assistant(content="hello"))
    after_second = path.read_text(encoding="utf-8")
    assert after_second.startswith(after_first)

    assert first.seq == 2
    assert second.seq == 3
    lines = after_second.strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[-1])["type"] == EventType.ASSISTANT_MESSAGE

    all_events = store.load_events(session_id)
    assert [event.seq for event in all_events] == [1, 2, 3]
    after = store.load_events(session_id, after=2)
    assert [event.seq for event in after] == [3]


def test_store_with_fsync_enabled(settings) -> None:
    store = SessionStore(settings.data_dir, fsync=True)
    session_id = store.create_session({"title": "fsync"})["session_id"]
    store.append(session_id, EventType.USER_MESSAGE, {"content": "x"})
    relative = store.write_artifact(session_id, "note.txt", "hello")
    assert relative == "artifacts/note.txt"
    assert len(store.load_events(session_id)) == 2


def test_meta_list_and_artifacts(store: SessionStore) -> None:
    first = store.create_session({"title": "A"})
    second = store.create_session({"title": "B", "parent_id": first["session_id"]})

    assert store.meta(first["session_id"])["title"] == "A"
    titles = {item["title"] for item in store.list_sessions()}
    assert titles == {"A", "B"}

    relative = store.write_artifact(first["session_id"], "../evil.txt", "内容")
    assert relative == "artifacts/evil.txt"
    assert store.read_artifact(first["session_id"], "evil.txt") == "内容"

    with pytest.raises(SessionError):
        store.meta("missing")
    with pytest.raises(SessionError):
        store.create_session({"session_id": "bad/id"})
    with pytest.raises(SessionError):
        store.create_session({"session_id": second["session_id"]})


def test_replay_openai_messages(store: SessionStore) -> None:
    session_id = store.create_session({"title": "replay"})["session_id"]
    store.append(session_id, EventType.USER_MESSAGE, {"content": "do it"})
    store.append(
        session_id,
        EventType.ASSISTANT_MESSAGE,
        _assistant(reasoning="先看代码", tool_calls=_tool_calls()),
    )
    store.append(session_id, EventType.RUN_CODE_STARTED, {"call_id": "call_1", "code": "print(1)"})
    store.append(
        session_id,
        EventType.RUN_CODE_FINISHED,
        {
            "call_id": "call_1",
            "exit_code": 0,
            "stdout_preview": "1",
            "stderr_preview": "",
            "content": "1",
            "duration_ms": 5,
            "tool_calls": [],
        },
    )
    store.append(session_id, EventType.ASSISTANT_MESSAGE, _assistant(content="done"))

    state = load_state(session_id, store=store)
    assert [message["role"] for message in state.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert state.pending_runs == {}
    assert state.last_seq == 6
    assert all(not key.startswith("_") for message in state.messages for key in message)

    messages = build_messages(state)
    assert messages[0] == {"role": "user", "content": "do it"}
    assert messages[1]["tool_calls"][0]["function"]["name"] == "run_code"
    assert messages[1]["reasoning_content"] == "先看代码"
    assert messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "1"}
    _assert_pairing(messages)

    stripped = build_messages(state, include_reasoning=False)
    assert "reasoning_content" not in stripped[1]


def test_level2_prunes_reasoning_and_adds_note(store: SessionStore) -> None:
    session_id = store.create_session({"title": "l2"})["session_id"]
    store.append(session_id, EventType.USER_MESSAGE, {"content": "u1"})
    assistant = store.append(
        session_id,
        EventType.ASSISTANT_MESSAGE,
        _assistant(content="a1", reasoning="旧思维链"),
    )
    store.append(session_id, EventType.USER_MESSAGE, {"content": "u2"})
    store.append(
        session_id, EventType.ASSISTANT_MESSAGE, _assistant(content="a2", reasoning="新思维链")
    )
    store.append(
        session_id,
        EventType.CONTEXT_COMPACTED,
        {
            "level": 2,
            "before_tokens": 100,
            "after_tokens": 40,
            "summary": "旧摘要",
            "pruned_upto_seq": assistant.seq,
        },
    )

    state = load_state(session_id, store=store)
    assert state.messages[1]["reasoning"] == ""
    assert state.messages[0]["role"] == "user"
    system_positions = [
        i for i, message in enumerate(state.messages) if message["role"] == "system"
    ]
    assert len(system_positions) == 1
    assert state.messages[system_positions[0]]["content"].startswith("[历史上下文摘要]")
    assert state.messages[-1]["reasoning"] == "新思维链"
    assert state.compactions[-1]["level"] == 2
    _assert_pairing(build_messages(state))


def test_l2_after_l3_keeps_full_summary(store: SessionStore) -> None:
    session_id = store.create_session({"title": "l2-after-l3"})["session_id"]
    store.append(session_id, EventType.USER_MESSAGE, {"content": "u1"})
    store.append(session_id, EventType.ASSISTANT_MESSAGE, _assistant(content="a1", reasoning="old"))
    store.append(session_id, EventType.ASSISTANT_MESSAGE, _assistant(content="a2"))
    store.append(
        session_id,
        EventType.CONTEXT_COMPACTED,
        {
            "level": 3,
            "before_tokens": 900,
            "after_tokens": 90,
            "summary": "全量摘要",
            "kept_message_count": 2,
            "pruned_upto_seq": 4,
        },
    )
    store.append(session_id, EventType.USER_MESSAGE, {"content": "u2"})
    store.append(session_id, EventType.ASSISTANT_MESSAGE, _assistant(content="a3"))
    store.append(
        session_id,
        EventType.CONTEXT_COMPACTED,
        {
            "level": 2,
            "before_tokens": 500,
            "after_tokens": 200,
            "summary": "剪枝摘要",
            "pruned_upto_seq": 6,
        },
    )
    store.append(
        session_id,
        EventType.CONTEXT_COMPACTED,
        {
            "level": 2,
            "before_tokens": 400,
            "after_tokens": 150,
            "summary": "更新后的剪枝摘要",
            "pruned_upto_seq": 6,
        },
    )

    state = load_state(session_id, store=store)
    contents = [message["content"] for message in state.messages if message["role"] == "system"]
    assert any("全量摘要" in content for content in contents)
    assert any("更新后的剪枝摘要" in content for content in contents)
    assert not any("剪枝摘要" == content.split("] ")[-1] for content in contents)
    assert len([message for message in state.messages if message["role"] == "system"]) == 2


def test_level3_rebuild_keeps_recent_messages(store: SessionStore) -> None:
    session_id = store.create_session({"title": "l3"})["session_id"]
    store.append(session_id, EventType.USER_MESSAGE, {"content": "u1"})
    store.append(
        session_id,
        EventType.ASSISTANT_MESSAGE,
        _assistant(content="a1", reasoning="r1", tool_calls=_tool_calls()),
    )
    store.append(
        session_id,
        EventType.RUN_CODE_FINISHED,
        {"call_id": "call_1", "exit_code": 0, "content": "out", "stdout_preview": "out"},
    )
    store.append(session_id, EventType.ASSISTANT_MESSAGE, _assistant(content="a2"))
    store.append(session_id, EventType.USER_MESSAGE, {"content": "u2"})
    store.append(session_id, EventType.ASSISTANT_MESSAGE, _assistant(content="a3"))
    store.append(
        session_id,
        EventType.CONTEXT_COMPACTED,
        {
            "level": 3,
            "before_tokens": 1000,
            "after_tokens": 100,
            "summary": "结构化摘要",
            "kept_message_count": 2,
            "pruned_upto_seq": 7,
        },
    )

    state = load_state(session_id, store=store)
    assert [message["role"] for message in state.messages] == ["system", "user", "assistant"]
    assert state.messages[0]["content"].startswith("[历史上下文摘要]")
    assert state.messages[1]["content"] == "u2"
    assert state.messages[2]["content"] == "a3"
    assert state.compactions[-1]["kept_message_count"] == 2
    _assert_pairing(build_messages(state))


def test_compaction_boundary_never_leaves_orphan_tool(store: SessionStore) -> None:
    session_id = store.create_session({"title": "boundary"})["session_id"]
    store.append(session_id, EventType.USER_MESSAGE, {"content": "u1"})
    store.append(
        session_id,
        EventType.ASSISTANT_MESSAGE,
        _assistant(tool_calls=_tool_calls()),
    )
    store.append(
        session_id,
        EventType.RUN_CODE_FINISHED,
        {"call_id": "call_1", "exit_code": 0, "content": "out", "stdout_preview": "out"},
    )
    store.append(
        session_id,
        EventType.CONTEXT_COMPACTED,
        {
            "level": 3,
            "before_tokens": 500,
            "after_tokens": 50,
            "summary": "S",
            "kept_message_count": 1,
            "pruned_upto_seq": 4,
        },
    )

    state = load_state(session_id, store=store)
    assert [message["role"] for message in state.messages] == ["system", "tool"]
    messages = build_messages(state)
    assert [message["role"] for message in messages] == ["system"]
    _assert_pairing(messages)


async def test_recover_fills_synthetic_results(store: SessionStore) -> None:
    session_id = store.create_session({"title": "recover"})["session_id"]
    store.append(session_id, EventType.USER_MESSAGE, {"content": "u"})
    store.append(
        session_id,
        EventType.ASSISTANT_MESSAGE,
        _assistant(tool_calls=_tool_calls()),
    )
    store.append(session_id, EventType.RUN_CODE_STARTED, {"call_id": "call_1", "code": "print(1)"})
    store.append(
        session_id,
        EventType.TOOL_CALL,
        {"call_id": "tool_1", "run_call_id": "call_1", "tool": "read_file", "args_preview": "{}"},
    )

    state = await store.recover(session_id)
    assert state.pending_runs == {}
    events = store.load_events(session_id)
    finished = [event for event in events if event.type == EventType.RUN_CODE_FINISHED]
    results = [event for event in events if event.type == EventType.TOOL_RESULT]
    assert len(finished) == 1 and finished[0].data["synthetic"] is True
    assert len(results) == 1 and results[0].data["synthetic"] is True
    assert results[0].data["error"]["kind"] == "interrupted"
    _assert_pairing(build_messages(state))

    # 幂等：再次恢复不重复补写
    await store.recover(session_id)
    events = store.load_events(session_id)
    assert len([event for event in events if event.type == EventType.RUN_CODE_FINISHED]) == 1


def test_pending_tool_calls_are_not_sent_without_results(store: SessionStore) -> None:
    session_id = store.create_session({"title": "pending"})["session_id"]
    store.append(session_id, EventType.USER_MESSAGE, {"content": "u"})
    store.append(
        session_id,
        EventType.ASSISTANT_MESSAGE,
        _assistant(tool_calls=_tool_calls()),
    )
    store.append(session_id, EventType.RUN_CODE_STARTED, {"call_id": "call_1", "code": "print(1)"})

    state = load_state(session_id, store=store)
    assert set(state.pending_runs) == {"call_1"}
    messages = build_messages(state)
    assert [message["role"] for message in messages] == ["user"]
    _assert_pairing(messages)


def test_partial_tail_is_ignored_and_repaired(settings) -> None:
    store = SessionStore(settings.data_dir, fsync=False)
    session_id = store.create_session({"title": "partial"})["session_id"]
    store.append(session_id, EventType.USER_MESSAGE, {"content": "one"})
    path = store.events_path(session_id)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 99, "ts"')

    restarted = SessionStore(settings.data_dir, fsync=False)
    events = restarted.load_events(session_id)
    assert [event.seq for event in events] == [1, 2]

    appended = restarted.append(session_id, EventType.USER_MESSAGE, {"content": "two"})
    assert appended.seq == 3
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[-1])["seq"] == 3


def test_reduce_accepts_empty_events() -> None:
    state = reduce_events({"title": "empty"}, [])
    assert state.messages == []
    assert state.last_seq == 0
    assert build_messages(state) == []
