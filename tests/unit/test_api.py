"""API 层离线测试：FakeService + ASGITransport，不启动真实模型。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from minillm.api.app import create_app
from minillm.config import Settings
from minillm.errors import SessionError


class FakeSubscription:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False

    def __aiter__(self) -> FakeSubscription:
        return self

    async def __anext__(self) -> Any:
        item = await self.queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    def close(self) -> None:
        self.closed = True


class FakeService:
    """按 AgentService 契约实现的最小替身。"""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.events_store: dict[str, list[dict[str, Any]]] = {}
        self.messages: dict[str, list[dict[str, Any]]] = {}
        self.agents: dict[str, Any] = {}
        self.view_fields: dict[str, dict[str, Any]] = {}
        self.live_events: dict[str, list[dict[str, Any]]] = {}
        self.subs: list[FakeSubscription] = []
        self.busy = False
        self.mcp = ["stub"]
        self.mcp_error = False
        self.closed = False

    def seed(self, session_id: str = "sess_0001", **meta: Any) -> dict[str, Any]:
        data: dict[str, Any] = {
            "session_id": session_id,
            "title": "测试会话",
            "workspace": "",
            "model": "test-model",
            "parent_id": None,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "status": "idle",
        }
        data.update(meta)
        self.sessions[session_id] = data
        self.events_store.setdefault(session_id, [])
        return data

    async def create_session(
        self, title: str | None = None, workspace: str | None = None
    ) -> dict[str, Any]:
        session_id = f"sess_{len(self.sessions) + 1:04d}"
        return self.seed(
            session_id,
            title=title or "未命名会话",
            workspace=workspace or "",
        )

    async def list_sessions(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.sessions.values()]

    async def get_view(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.sessions:
            raise SessionError(f"会话不存在：{session_id}")
        meta = dict(self.sessions[session_id])
        view: dict[str, Any] = {
            "meta": meta,
            "messages": list(self.messages.get(session_id, [])),
            "todos": [],
            "agents": self.agents.get(session_id, {}),
            "context": {"tokens": 0, "budget": 1000, "ratio": 0.0, "compactions": []},
            "skills": [],
            "last_seq": 0,
            "status": meta.get("status", "idle"),
        }
        view.update(self.view_fields.get(session_id, {}))
        return view

    async def submit_message(self, session_id: str, content: str) -> str:
        if session_id not in self.sessions:
            raise SessionError(f"会话不存在：{session_id}")
        if self.busy:
            raise SessionError("会话运行中", detail={"code": "session_busy"})
        return "turn_1"

    async def stop(self, session_id: str) -> None:
        if session_id not in self.sessions:
            raise SessionError(f"会话不存在：{session_id}")
        self.sessions[session_id]["status"] = "idle"

    async def events(self, session_id: str, after: int = 0) -> list[dict[str, Any]]:
        if session_id not in self.sessions:
            raise SessionError(f"会话不存在：{session_id}")
        return [
            event
            for event in self.events_store.get(session_id, [])
            if int(event.get("seq") or 0) > after
        ]

    async def subscribe(self, session_id: str) -> FakeSubscription:
        if session_id not in self.sessions:
            raise SessionError(f"会话不存在：{session_id}")
        sub = FakeSubscription()
        for event in self.live_events.get(session_id, []):
            sub.queue.put_nowait(event)
        sub.queue.put_nowait(None)
        self.subs.append(sub)
        return sub

    async def mcp_servers(self) -> list[Any]:
        if self.mcp_error:
            raise RuntimeError("mcp down")
        return list(self.mcp)

    async def close(self) -> None:
        self.closed = True


def make_settings(tmp_path: Any) -> Settings:
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        workspace_root=tmp_path / "workspace",
        skills_dir=tmp_path / "skills",
        mcp_config=tmp_path / "mcp.json",
        llm_model="test-model",
        small_llm_model="small-model",
        context_budget=1000,
        llm_api_key="test-key",
    )


def make_client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


@pytest.fixture
async def env(tmp_path: Any) -> Any:
    service = FakeService()
    settings = make_settings(tmp_path)
    app = create_app(settings, service=service, web_dist=tmp_path / "missing-dist")
    async with make_client(app) as client:
        yield client, service, settings, tmp_path


async def test_health_and_config(env: Any) -> None:
    client, service, settings, _ = env
    skill_dir = settings.skills_dir / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: 演示技能\nkeywords: [pytest]\n---\n# Demo\n",
        encoding="utf-8",
    )
    service.mcp = ["stub", {"name": "other"}]

    health = await client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "version": "0.1.0", "model": "test-model"}

    config = (await client.get("/api/config")).json()
    assert config["model"] == "test-model"
    assert config["small_model"] == "small-model"
    assert config["context_budget"] == 1000
    assert config["workspace_root"] == str(settings.workspace_root)
    assert config["skills"][0]["name"] == "demo"
    assert config["mcp_servers"] == ["stub", "other"]
    assert config["version"] == "0.1.0"

    service.mcp_error = True
    fallback = (await client.get("/api/config")).json()
    assert fallback["mcp_servers"] == []


async def test_create_list_and_workspace_validation(env: Any) -> None:
    client, _, _, tmp_path = env
    empty = await client.get("/api/sessions")
    assert empty.status_code == 200
    assert empty.json() == {"sessions": []}

    created = await client.post("/api/sessions", json={"title": "hello"})
    assert created.status_code == 201
    session = created.json()
    assert session["id"] == session["session_id"]
    assert session["title"] == "hello"

    listed = (await client.get("/api/sessions")).json()["sessions"]
    assert [item["id"] for item in listed] == [session["id"]]

    inside = tmp_path / "workspace" / "proj"
    inside.mkdir(parents=True)
    ok = await client.post("/api/sessions", json={"workspace": str(inside)})
    assert ok.status_code == 201
    assert ok.json()["workspace"] == str(inside.resolve())

    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    rejected = await client.post("/api/sessions", json={"workspace": str(outside)})
    assert rejected.status_code == 400
    assert "workspace" in rejected.json()["detail"]


async def test_view_normalization(env: Any) -> None:
    client, service, _, _ = env
    service.seed("sess_0001")
    service.view_fields["sess_0001"] = {
        "messages": [
            {
                "role": "assistant",
                "content": "hi",
                "_private": True,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_code", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "out", "_seq": 2},
        ],
        "agents": {
            "agent_1": {
                "agent_id": "agent_1",
                "parent_session_id": "sess_0001",
                "task": "子任务",
                "status": "running",
                "result_preview": "进行中",
                "updated_at": "2026-01-01T00:00:01Z",
            }
        },
        "context": {"tokens": 25, "budget": 1000, "ratio": 0.025, "compactions": []},
        "skills": [{"name": "demo", "description": "d", "path": "/tmp/demo"}],
        "last_seq": 7,
    }

    view = (await client.get("/api/sessions/sess_0001")).json()
    assert view["meta"]["id"] == "sess_0001"
    assert view["meta"]["session_id"] == "sess_0001"
    assert "_private" not in view["messages"][0]
    assert view["messages"][0]["tool_calls"][0]["function"]["name"] == "run_code"
    assert view["messages"][1]["tool_call_id"] == "call_1"
    assert set(view["agents"][0]) == {
        "agent_id",
        "parent_session_id",
        "task",
        "status",
        "result_preview",
        "updated_at",
    }
    assert view["context"] == {
        "tokens": 25,
        "budget": 1000,
        "ratio": 0.025,
        "compactions": [],
    }
    assert view["skills"][0]["name"] == "demo"
    assert view["last_seq"] == 7
    assert view["status"] == "idle"


async def test_events_after_and_404(env: Any) -> None:
    client, service, _, _ = env
    service.seed("sess_0001")
    service.events_store["sess_0001"] = [
        {"seq": 1, "type": "user.message", "data": {"content": "a"}},
        {"seq": 2, "type": "assistant.message", "data": {"content": "b"}},
    ]
    events = (await client.get("/api/sessions/sess_0001/events?after=1")).json()["events"]
    assert [item["seq"] for item in events] == [2]
    assert events[0]["session_id"] == "sess_0001"

    missing = await client.get("/api/sessions/unknown")
    assert missing.status_code == 404


async def test_message_validation_and_conflict(env: Any) -> None:
    client, service, _, _ = env
    service.seed("sess_0001")

    empty = await client.post("/api/sessions/sess_0001/messages", json={"content": "  "})
    assert empty.status_code == 400

    service.busy = True
    busy = await client.post("/api/sessions/sess_0001/messages", json={"content": "hi"})
    assert busy.status_code == 409
    assert busy.json()["code"] == "session_busy"

    service.busy = False
    ok = await client.post("/api/sessions/sess_0001/messages", json={"content": "hi"})
    assert ok.status_code == 202
    assert ok.json()["turn_id"] == "turn_1"

    stopped = await client.post("/api/sessions/sess_0001/stop")
    assert stopped.status_code == 200
    assert stopped.json() == {"ok": True}


async def test_sse_replay_live_dedupe_and_delta(env: Any) -> None:
    client, service, _, _ = env
    service.seed("sess_0001")
    service.events_store["sess_0001"] = [
        {"seq": 1, "ts": "t1", "type": "user.message", "data": {"content": "a"}}
    ]
    service.live_events["sess_0001"] = [
        {"seq": 2, "ts": "t2", "type": "turn.started", "data": {}},
        {"seq": 1, "ts": "t1", "type": "user.message", "data": {"content": "dup"}},
        {"seq": 3, "ts": "t3", "type": "turn.finished", "data": {}},
        {
            "seq": None,
            "type": "assistant.delta",
            "data": {"kind": "text", "text": "hello"},
        },
    ]

    response = await client.get("/api/sessions/sess_0001/stream")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "id: 1" in response.text
    data_lines = [
        line[6:] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    events = [json.loads(line) for line in data_lines]
    persistent = [item["seq"] for item in events if item.get("seq") is not None]
    assert persistent == [1, 2, 3]
    delta = [item for item in events if item.get("type") == "assistant.delta"]
    assert delta and delta[0]["data"]["text"] == "hello"
    assert service.subs and service.subs[0].closed


async def test_artifact_route_and_traversal(env: Any) -> None:
    client, service, settings, _ = env
    service.seed("sess_0001")
    artifacts = settings.data_dir / "sessions" / "sess_0001" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "out.txt").write_text("hello artifact", encoding="utf-8")

    ok = await client.get("/api/sessions/sess_0001/artifacts/out.txt")
    assert ok.status_code == 200
    assert ok.text == "hello artifact"

    missing = await client.get("/api/sessions/sess_0001/artifacts/nope.txt")
    assert missing.status_code == 404

    traversal = await client.get("/api/sessions/sess_0001/artifacts/%2e%2e")
    assert traversal.status_code in (400, 404)


async def test_static_hosting(tmp_path: Any) -> None:
    service = FakeService()
    settings = make_settings(tmp_path)
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<div id="root">app</div>', encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
    app = create_app(settings, service=service, web_dist=dist)
    async with make_client(app) as client:
        root = await client.get("/")
        assert root.status_code == 200
        assert '<div id="root">app</div>' in root.text

        deep = await client.get("/deep/route")
        assert deep.status_code == 200
        assert '<div id="root">app</div>' in deep.text

        asset = await client.get("/assets/app.js")
        assert asset.status_code == 200
        assert "console.log" in asset.text

        health = await client.get("/api/health")
        assert health.status_code == 200


async def test_static_missing_and_no_service(env: Any) -> None:
    client, _, _, tmp_path = env
    root = await client.get("/")
    assert root.status_code == 200
    assert "前端尚未构建" in root.text

    settings = make_settings(tmp_path)
    app = create_app(settings, service=None, web_dist=tmp_path / "missing-dist")
    async with make_client(app) as client:
        health = await client.get("/api/health")
        assert health.status_code == 200
        no_service = await client.get("/api/sessions")
        assert no_service.status_code == 503


async def test_lifespan_closes_service(tmp_path: Any) -> None:
    service = FakeService()
    settings = make_settings(tmp_path)
    app = create_app(settings, service=service, web_dist=tmp_path / "missing-dist")
    async with app.router.lifespan_context(app):
        assert service.closed is False
    assert service.closed is True
