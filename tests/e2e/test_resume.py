"""断点恢复 e2e：杀进程重启后继续会话；未完成 run 的 synthetic 结果。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from minillm.persistence.store import SessionStore

RESUME_CODE = (
    "tools.write_file(path='resume.txt', content='one'); print('written')"
)


def _client(server) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=server.base_url,
        trust_env=False,
        timeout=httpx.Timeout(15.0, read=45.0),
    )


def _script(script_writer, root):
    return script_writer(
        root,
        [
            {
                "content": "",
                "tool_calls": [
                    {"name": "run_code", "arguments": {"code": RESUME_CODE}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            },
            {"content": "第一轮完成"},
            {"content": "第二轮完成"},
        ],
        name="resume",
    )


async def test_restart_and_continue_same_session(
    server_factory, script_writer, sse_collector, calls_reader, tmp_path
):
    script = _script(script_writer, tmp_path)
    data_dir = tmp_path / "data"
    workspace_root = tmp_path / "workspace"

    async with server_factory(
        tmp_path, script, data_dir=data_dir, workspace_root=workspace_root
    ) as server:
        async with _client(server) as client:
            created = await client.post("/api/sessions", json={"title": "恢复测试"})
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
                json={"content": "第一轮：写文件"},
            )
            assert sent.status_code == 202
            await asyncio.wait_for(stream_task, timeout=35)

            first_view = await client.get(f"/api/sessions/{session_id}")
            assert first_view.status_code == 200
            first_messages = first_view.json()["messages"]
            assert [message["role"] for message in first_messages] == [
                "user",
                "assistant",
                "tool",
                "assistant",
            ]
            assert first_messages[-1]["content"] == "第一轮完成"
            assert (
                workspace_root / session_id / "resume.txt"
            ).read_text(encoding="utf-8") == "one"

    # 进程已 SIGTERM；重新起一个同 DATA_DIR/WORKSPACE 的服务继续同一会话
    async with server_factory(
        tmp_path, script, data_dir=data_dir, workspace_root=workspace_root
    ) as server:
        async with _client(server) as client:
            restored = await client.get(f"/api/sessions/{session_id}")
            assert restored.status_code == 200
            restored_messages = restored.json()["messages"]
            assert any(
                message.get("content") == "第一轮完成"
                for message in restored_messages
            )
            assert any(message.get("role") == "tool" for message in restored_messages)

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
                json={"content": "第二轮：继续"},
            )
            assert sent.status_code == 202

            frames = await asyncio.wait_for(stream_task, timeout=35)
            types = [
                frame["event"].get("type")
                for frame in frames
                if isinstance(frame["event"], dict)
            ]
            assert "turn.finished" in types

            final_view = await client.get(f"/api/sessions/{session_id}")
            assert final_view.status_code == 200
            messages = final_view.json()["messages"]
            contents = [message.get("content", "") for message in messages]
            assert "第一轮完成" in contents
            assert messages[-1]["content"] == "第二轮完成"

            calls = calls_reader(script)
            assert len(calls) >= 3, calls
            third_messages = json.dumps(
                calls[2]["messages"], ensure_ascii=False
            )
            assert "第一轮完成" in third_messages
            assert "第一轮：写文件" in third_messages


async def test_unfinished_run_gets_synthetic_result(
    server_factory, script_writer, tmp_path
):
    data_dir = tmp_path / "data"
    workspace_root = tmp_path / "workspace"
    session_id = "sess_interrupted"
    workspace = workspace_root / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    store = SessionStore(data_dir, fsync=False)
    store.create_session(
        {
            "session_id": session_id,
            "title": "中断会话",
            "workspace": str(workspace),
            "model": "scripted",
            "parent_id": None,
            "status": "idle",
        }
    )
    call_id = "call_interrupted"
    store.append(session_id, "user.message", {"content": "跑一个长任务"})
    store.append(
        session_id,
        "assistant.message",
        {
            "content": "",
            "reasoning": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "run_code",
                        "arguments": json.dumps(
                            {"code": "print(1)"}, ensure_ascii=False
                        ),
                    },
                }
            ],
            "finish_reason": "tool_calls",
        },
    )
    store.append(
        session_id,
        "run_code.started",
        {"call_id": call_id, "code": "print(1)"},
    )

    script = script_writer(tmp_path, [], name="empty")
    async with server_factory(
        tmp_path, script, data_dir=data_dir, workspace_root=workspace_root
    ) as server:
        async with _client(server) as client:
            view = await client.get(f"/api/sessions/{session_id}")
            assert view.status_code == 200
            data = view.json()
            roles = [message["role"] for message in data["messages"]]
            assert roles == ["user", "assistant", "tool"]
            tool_message = data["messages"][-1]
            assert "服务重启" in tool_message["content"]

            events = await client.get(
                f"/api/sessions/{session_id}/events?after=0"
            )
            assert events.status_code == 200
            finished = [
                event
                for event in events.json()["events"]
                if event["type"] == "run_code.finished"
            ]
            assert finished
            assert finished[-1]["data"]["synthetic"] is True
            assert finished[-1]["data"]["call_id"] == call_id
