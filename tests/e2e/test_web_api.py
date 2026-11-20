"""真实 uvicorn + scripted LLM 的 Web API/SSE e2e。"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

SLOW_CODE = "import time" + chr(10) + "time.sleep(8)" + chr(10) + "print('slow')"
ONE_LINE_CODE = "print('hello-web')"


def _client(server) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=server.base_url,
        trust_env=False,
        timeout=httpx.Timeout(15.0, read=45.0),
    )


def _web_script(script_writer, root):
    return script_writer(
        root,
        [
            {
                "content": "先跑一下",
                "tool_calls": [
                    {"name": "run_code", "arguments": {"code": ONE_LINE_CODE}}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                },
            },
            {
                "content": "完成",
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 5,
                    "total_tokens": 125,
                },
            },
        ],
        name="web-api",
    )


async def test_sessions_sse_artifacts_and_config(
    server_factory, script_writer, sse_collector, tmp_path
):
    script = _web_script(script_writer, tmp_path)
    async with server_factory(tmp_path, script) as server:
        async with _client(server) as client:
            health = await client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            config = await client.get("/api/config")
            assert config.status_code == 200
            config_data = config.json()
            assert config_data["model"]
            assert config_data["small_model"]
            assert config_data["context_budget"] == 200000
            assert config_data["mcp_servers"] == []
            assert isinstance(config_data["skills"], list)

            created = await client.post(
                "/api/sessions", json={"title": "E2E 会话"}
            )
            assert created.status_code == 201
            created_data = created.json()
            session_id = created_data["id"]
            assert created_data["session_id"] == session_id
            assert created_data["status"] in {"idle", "running"}

            listed = await client.get("/api/sessions")
            assert listed.status_code == 200
            listed_ids = [item["id"] for item in listed.json()["sessions"]]
            assert session_id in listed_ids

            opened = asyncio.Event()
            stream_task = asyncio.create_task(
                sse_collector(
                    client,
                    f"/api/sessions/{session_id}/stream?after=0",
                    opened=opened,
                )
            )
            await asyncio.wait_for(opened.wait(), timeout=8)

            sent = await client.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "请执行 hello-web"},
            )
            assert sent.status_code == 202, sent.text
            assert sent.json()["turn_id"]

            frames = await asyncio.wait_for(stream_task, timeout=35)
            assert frames, "SSE 没有返回任何帧"

            persisted = [
                frame
                for frame in frames
                if frame["id"] is not None and isinstance(frame["event"], dict)
            ]
            types = [frame["event"].get("type") for frame in persisted]
            assert "user.message" in types
            assert "assistant.message" in types
            assert "run_code.started" in types
            assert "run_code.finished" in types
            assert "turn.finished" in types
            assert any(
                isinstance(frame["event"].get("seq"), int) for frame in persisted
            )

            deltas = [
                frame
                for frame in frames
                if frame["id"] is None
                and isinstance(frame["event"], dict)
                and frame["event"].get("type") == "assistant.delta"
            ]
            assert deltas, "缺少无 id 的 assistant.delta"
            assert any(
                frame["event"].get("data", {}).get("text") for frame in deltas
            )

            view = await client.get(f"/api/sessions/{session_id}")
            assert view.status_code == 200
            view_data = view.json()
            assert view_data["meta"]["id"] == session_id
            assert view_data["meta"]["session_id"] == session_id
            assert view_data["context"]["budget"] == 200000
            assert isinstance(view_data["agents"], list)
            assert view_data["last_seq"] >= 1
            roles = [message["role"] for message in view_data["messages"]]
            assert roles == ["user", "assistant", "tool", "assistant"]
            assert any("hello-web" in message["content"] for message in view_data["messages"])

            events = await client.get(f"/api/sessions/{session_id}/events?after=0")
            assert events.status_code == 200
            event_types = [event["type"] for event in events.json()["events"]]
            assert "run_code.finished" in event_types
            assert "turn.finished" in event_types

            artifacts_dir = (
                server.data_dir / "sessions" / session_id / "artifacts"
            )
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "note.txt").write_text("artifact-body", encoding="utf-8")
            artifact = await client.get(
                f"/api/sessions/{session_id}/artifacts/note.txt"
            )
            assert artifact.status_code == 200
            assert artifact.text == "artifact-body"
            missing = await client.get(
                f"/api/sessions/{session_id}/artifacts/missing.txt"
            )
            assert missing.status_code == 404
            traversal = await client.get(
                f"/api/sessions/{session_id}/artifacts/..%2Fmeta.json"
            )
            assert traversal.status_code in {400, 404}

            outside = await client.post(
                "/api/sessions", json={"workspace": "/tmp/minillm-outside"}
            )
            assert outside.status_code == 400


async def test_busy_409_and_stop(
    server_factory, script_writer, sse_collector, status_waiter, tmp_path
):
    script = script_writer(
        tmp_path,
        [
            {
                "content": "慢任务",
                "tool_calls": [
                    {"name": "run_code", "arguments": {"code": SLOW_CODE}}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                },
            },
            {"content": "不该到达"},
        ],
        name="slow",
    )
    async with server_factory(tmp_path, script) as server:
        async with _client(server) as client:
            created = await client.post("/api/sessions", json={})
            assert created.status_code == 201
            session_id = created.json()["id"]

            opened = asyncio.Event()
            stream_task = asyncio.create_task(
                sse_collector(
                    client,
                    f"/api/sessions/{session_id}/stream?after=0",
                    opened=opened,
                )
            )
            await asyncio.wait_for(opened.wait(), timeout=8)

            sent = await client.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "跑慢任务"},
            )
            assert sent.status_code == 202, sent.text

            running = await status_waiter(client, session_id, "running", timeout=8)
            assert running.get("status") == "running", json.dumps(
                running, ensure_ascii=False
            )[:500]

            busy = await client.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "再来一次"},
            )
            assert busy.status_code == 409, busy.text
            assert "运行" in busy.text or "busy" in busy.text.lower()

            stopped = await client.post(f"/api/sessions/{session_id}/stop", json={})
            assert stopped.status_code == 200
            assert stopped.json().get("ok") is True

            frames = await asyncio.wait_for(stream_task, timeout=25)
            types = [
                frame["event"].get("type")
                for frame in frames
                if isinstance(frame["event"], dict)
            ]
            assert "turn.cancelled" in types

            idle = await status_waiter(client, session_id, "idle", timeout=8)
            assert idle.get("status") == "idle", json.dumps(
                idle, ensure_ascii=False
            )[:500]
