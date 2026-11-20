"""MCP e2e：真实服务启动 stdio MCP server 并在 run_code 中调用。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_CODE = "print(tools.mcp__stub__echo(text='hi-mcp'))"


def _client(server) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=server.base_url,
        trust_env=False,
        timeout=httpx.Timeout(20.0, read=60.0),
    )


async def test_mcp_server_registered_and_called(
    server_factory, script_writer, sse_collector, tmp_path
):
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "stub",
                        "command": sys.executable,
                        "args": [str(REPO_ROOT / "scripts" / "mcp_stub_server.py")],
                        "cwd": str(REPO_ROOT),
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    script = script_writer(
        tmp_path,
        [
            {
                "content": "调用 MCP",
                "tool_calls": [
                    {"name": "run_code", "arguments": {"code": MCP_CODE}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
            {"content": "mcp done"},
        ],
        name="mcp-script",
    )

    async with server_factory(
        tmp_path, script, extra_env={"MCP_CONFIG": str(mcp_config)}
    ) as server:
        async with _client(server) as client:
            config = await client.get("/api/config")
            assert config.status_code == 200
            assert "stub" in config.json()["mcp_servers"], config.text

            created = await client.post("/api/sessions", json={"title": "MCP"})
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
                json={"content": "调用 MCP echo"},
            )
            assert sent.status_code == 202, sent.text
            frames = await asyncio.wait_for(stream_task, timeout=40)
            types = [
                frame["event"].get("type")
                for frame in frames
                if isinstance(frame["event"], dict)
            ]
            assert "turn.finished" in types

            view = await client.get(f"/api/sessions/{session_id}")
            assert view.status_code == 200
            data = view.json()
            assert data["messages"][-1]["content"] == "mcp done"
            assert "hi-mcp" in json.dumps(data["messages"], ensure_ascii=False)

            events = await client.get(f"/api/sessions/{session_id}/events?after=0")
            finished = [
                event
                for event in events.json()["events"]
                if event["type"] == "run_code.finished"
            ]
            assert finished
            assert "hi-mcp" in json.dumps(
                finished[-1]["data"], ensure_ascii=False
            )
