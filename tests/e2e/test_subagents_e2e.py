"""子 agent e2e：真实 run_code 内 spawn + wait + 独立会话。"""

from __future__ import annotations

import asyncio
import json

import httpx

SUBAGENT_CODE = (
    "info = tools.spawn_subagent("
    "task='子任务：回复 child done', inherit_context=True); "
    "print(info); "
    "result = tools.wait_agent(agent_id=info['agent_id'], timeout=20); "
    "print(result)"
)


def _client(server) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=server.base_url,
        trust_env=False,
        timeout=httpx.Timeout(20.0, read=60.0),
    )


async def test_parent_spawns_child_and_inherits_context(
    server_factory,
    script_writer,
    sse_collector,
    calls_reader,
    events_reader,
    tmp_path,
):
    script = script_writer(
        tmp_path,
        [
            {
                "match": {"agent_kind": "main"},
                "content": "调用子 agent",
                "tool_calls": [
                    {"name": "run_code", "arguments": {"code": SUBAGENT_CODE}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
            {
                "match": {"agent_kind": "subagent"},
                "content": "child done",
                "usage": {"prompt_tokens": 80, "completion_tokens": 5},
            },
            {
                "match": {"agent_kind": "main"},
                "content": "parent done",
                "usage": {"prompt_tokens": 120, "completion_tokens": 5},
            },
        ],
        name="subagents",
    )

    async with server_factory(tmp_path, script) as server:
        async with _client(server) as client:
            created = await client.post("/api/sessions", json={"title": "父子会话"})
            assert created.status_code == 201
            parent_id = created.json()["id"]

            opened = asyncio.Event()
            stream_task = asyncio.create_task(
                sse_collector(
                    client,
                    f"/api/sessions/{parent_id}/stream?after=0",
                    opened=opened,
                )
            )
            await asyncio.wait_for(opened.wait(), timeout=8)
            sent = await client.post(
                f"/api/sessions/{parent_id}/messages",
                json={"content": "请创建子agent"},
            )
            assert sent.status_code == 202, sent.text
            frames = await asyncio.wait_for(stream_task, timeout=40)
            types = [
                frame["event"].get("type")
                for frame in frames
                if isinstance(frame["event"], dict)
            ]
            assert "turn.finished" in types

            events = await client.get(f"/api/sessions/{parent_id}/events?after=0")
            assert events.status_code == 200
            event_items = events.json()["events"]
            spawned = [
                item for item in event_items if item["type"] == "subagent.spawned"
            ]
            completed = [
                item for item in event_items if item["type"] == "subagent.completed"
            ]
            assert spawned and completed
            assert spawned[0]["data"]["agent_id"]
            assert completed[0]["data"]["status"] in {
                "completed",
                "idle",
            }
            assert "child done" in json.dumps(
                completed[0]["data"], ensure_ascii=False
            )

            view = await client.get(f"/api/sessions/{parent_id}")
            assert view.status_code == 200
            parent_data = view.json()
            assert parent_data["messages"][-1]["content"] == "parent done"
            assert "child done" in json.dumps(
                parent_data["messages"], ensure_ascii=False
            )
            assert parent_data["agents"], "父会话 agents 为空"
            agent = parent_data["agents"][0]
            assert agent["agent_id"] == spawned[0]["data"]["agent_id"]
            assert agent["parent_session_id"] == parent_id
            assert agent["task"]
            assert agent["status"] in {"completed", "idle"}

            sessions = await client.get("/api/sessions")
            children = [
                item
                for item in sessions.json()["sessions"]
                if item.get("parent_id") == parent_id
            ]
            assert len(children) == 1, children
            child_id = children[0]["id"]

            child_events = events_reader(server.data_dir, child_id)
            assert child_events, "子会话没有 events.jsonl"
            child_types = [event["type"] for event in child_events]
            assert "assistant.message" in child_types
            child_messages = [
                event
                for event in child_events
                if event["type"] == "assistant.message"
            ]
            assert any(
                "child done" in str(event["data"].get("content", ""))
                for event in child_messages
            )

            calls = calls_reader(script)
            child_calls = [
                call for call in calls if call.get("agent_kind") == "subagent"
            ]
            assert child_calls, "没有子 agent 的 LLM 调用记录"
            child_messages_json = json.dumps(
                child_calls[0]["messages"], ensure_ascii=False
            )
            assert "请创建子agent" in child_messages_json
