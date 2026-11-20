"""上下文压缩 e2e：真实服务 + 真实事件重放 + scripted 小模型。"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from minillm.persistence.store import SessionStore


def _client(server) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=server.base_url,
        trust_env=False,
        timeout=httpx.Timeout(15.0, read=45.0),
    )


def _seed_history(
    data_dir, session_id: str, workspace, prompt_tokens: int
) -> None:
    store = SessionStore(data_dir, fsync=False)
    store.create_session(
        {
            "session_id": session_id,
            "title": "压缩历史",
            "workspace": str(workspace),
            "model": "scripted",
            "parent_id": None,
            "status": "idle",
        }
    )
    for index in range(8):
        call_id = f"call_hist_{index}"
        store.append(
            session_id, "user.message", {"content": f"历史问题 {index}"}
        )
        store.append(
            session_id,
            "assistant.message",
            {
                "content": f"处理 {index}",
                "reasoning": "推理" * 40,
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
        store.append(
            session_id,
            "run_code.finished",
            {
                "call_id": call_id,
                "exit_code": 0,
                "stdout_preview": "1",
                "stderr_preview": "",
                "content": "1",
                "artifact": None,
                "duration_ms": 1,
                "tool_calls": [],
                "synthetic": False,
            },
        )
        store.append(
            session_id,
            "llm.usage",
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 10,
                "reasoning_tokens": 0,
                "total_tokens": prompt_tokens + 10,
                "model": "scripted",
                "purpose": "chat",
            },
        )


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_level"),
    [(5000, 2), (9000, 3)],
)
async def test_compaction_levels(
    server_factory,
    script_writer,
    sse_collector,
    tmp_path,
    prompt_tokens,
    expected_level,
):
    session_id = f"sess_compact_{expected_level}"
    workspace = tmp_path / "workspace" / session_id
    workspace.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    _seed_history(data_dir, session_id, workspace, prompt_tokens)

    script = script_writer(
        tmp_path,
        [{"content": "压缩后完成"}],
        [{"content": "历史摘要"}],
        name=f"compact-{expected_level}",
    )
    env = {
        "CONTEXT_BUDGET": "8000",
        "PRUNE_THRESHOLD": "0.5",
        "COMPACT_THRESHOLD": "0.8",
        "KEEP_RECENT_TURNS": "6",
    }
    async with server_factory(
        tmp_path,
        script,
        data_dir=data_dir,
        workspace_root=tmp_path / "workspace",
        extra_env=env,
    ) as server:
        async with _client(server) as client:
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
                json={"content": "继续完成任务"},
            )
            assert sent.status_code == 202, sent.text
            frames = await asyncio.wait_for(stream_task, timeout=35)

            compacted = [
                frame["event"]["data"]
                for frame in frames
                if isinstance(frame["event"], dict)
                and frame["event"].get("type") == "context.compacted"
            ]
            assert compacted, "没有 context.compacted 事件"
            event = compacted[0]
            assert event["level"] == expected_level
            assert event["after_tokens"] < event["before_tokens"]
            assert event["before_tokens"] >= prompt_tokens
            assert not event.get("fallback")

            view = await client.get(f"/api/sessions/{session_id}")
            assert view.status_code == 200
            compactions = view.json()["context"]["compactions"]
            assert compactions
            assert compactions[-1]["level"] == expected_level
            assert compactions[-1]["after_tokens"] < compactions[-1]["before_tokens"]
