"""E2E helpers: real uvicorn server with scripted LLM, SSE reader, script IO.

All servers use temporary data/workspace roots and a random free port. The
server is launched through the real CLI entry (python -m minillm serve) so the
HTTP/SSE tests exercise the actual app, not an in-process app object.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_START_TIMEOUT = 25.0
SERVER_STOP_TIMEOUT = 15.0
PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def free_port() -> int:
    """Bind an ephemeral localhost port, close it, return the number."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_script(
    root: Path,
    responses: list[dict[str, Any]],
    small_responses: list[dict[str, Any]] | None = None,
    *,
    name: str = "script",
) -> Path:
    """Write a ScriptedLLMClient JSON script; returns its persistent path."""

    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "responses": responses,
                "small_responses": small_responses or [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def read_calls(script: Path) -> list[dict[str, Any]]:
    """Read the scripted LLM call log (script.calls.jsonl)."""

    path = script.with_name(script.name + ".calls.jsonl")
    if not path.exists():
        return []
    calls: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            calls.append(item)
    return calls


def read_events(data_dir: Path, session_id: str) -> list[dict[str, Any]]:
    """Read a session's JSONL events directly (for child sessions)."""

    path = data_dir / "sessions" / session_id / "events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


@dataclass
class Server:
    proc: asyncio.subprocess.Process
    port: int
    root: Path
    data_dir: Path
    workspace_root: Path
    script: Path
    base_url: str
    stderr: str = ""


async def _wait_health(
    client: httpx.AsyncClient, server: Server, proc: asyncio.subprocess.Process
) -> None:
    deadline = asyncio.get_running_loop().time() + SERVER_START_TIMEOUT
    last_error = "未开始"
    while asyncio.get_running_loop().time() < deadline:
        if proc.returncode is not None:
            out = await proc.communicate()
            stderr = (out[1] or b"").decode("utf-8", errors="replace")
            server.stderr = stderr
            raise RuntimeError(
                f"minillm serve 提前退出（exit={proc.returncode}）：\n{stderr[-4000:]}"
            )
        try:
            response = await client.get("/api/health")
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except Exception as exc:  # noqa: BLE001 - startup probe
            last_error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(0.12)
    raise RuntimeError(f"minillm serve 未在 {SERVER_START_TIMEOUT}s 内就绪：{last_error}")


async def _stop(proc: asyncio.subprocess.Process, server: Server) -> None:
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=SERVER_STOP_TIMEOUT)
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
    with contextlib.suppress(Exception):
        out = await proc.communicate()
        server.stderr = (out[1] or b"").decode("utf-8", errors="replace")


@contextlib.asynccontextmanager
async def run_server(
    root: Path,
    script: Path,
    *,
    data_dir: Path | None = None,
    workspace_root: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> AsyncIterator[Server]:
    """Start the real CLI server and wait for /api/health."""

    data = Path(data_dir) if data_dir is not None else root / "data"
    workspace = Path(workspace_root) if workspace_root is not None else root / "workspace"
    data.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    port = free_port()
    env = os.environ.copy()
    for key in PROXY_KEYS:
        env.pop(key, None)
    env.update(
        {
            "NO_PROXY": "localhost,127.0.0.1,::1",
            "no_proxy": "localhost,127.0.0.1,::1",
            "MINILLM_LLM": "scripted",
            "MINILLM_SCRIPT": str(script),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    command = [
        sys.executable,
        "-m",
        "minillm",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--data-dir",
        str(data),
        "--workspace-root",
        str(workspace),
    ]
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    server = Server(
        proc=proc,
        port=port,
        root=root,
        data_dir=data,
        workspace_root=workspace,
        script=script,
        base_url=f"http://127.0.0.1:{port}",
    )
    try:
        async with httpx.AsyncClient(
            base_url=server.base_url, trust_env=False, timeout=httpx.Timeout(10.0)
        ) as probe:
            await _wait_health(probe, server, proc)
        yield server
    finally:
        await _stop(proc, server)


async def collect_sse(
    client: httpx.AsyncClient,
    url: str,
    *,
    opened: asyncio.Event | None = None,
    stop_types: tuple[str, ...] = ("turn.finished", "turn.cancelled"),
    timeout: float = 35.0,
    max_events: int = 500,
) -> list[dict[str, Any]]:
    """Read one SSE stream and return parsed frames.

    Each frame is {"id": <str|None>, "event": <dict>}. The default event name
    is used by the server; id is only present for persisted events.
    """

    events: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(timeout):
            async with client.stream("GET", url) as response:
                if opened is not None:
                    opened.set()
                response.raise_for_status()
                current_id: str | None = None
                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line.startswith(":"):
                        continue
                    if line.startswith("id:"):
                        current_id = line[3:].strip()
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                        continue
                    if line != "":
                        continue
                    if not data_lines:
                        current_id = None
                        continue
                    raw = "\n".join(data_lines)
                    data_lines = []
                    frame_id = current_id
                    current_id = None
                    try:
                        payload: Any = json.loads(raw)
                    except json.JSONDecodeError:
                        payload = raw
                    events.append({"id": frame_id, "event": payload})
                    if isinstance(payload, dict) and payload.get("type") in stop_types:
                        return events
                    if len(events) >= max_events:
                        return events
    except TimeoutError:
        return events
    return events


async def wait_status(
    client: httpx.AsyncClient,
    session_id: str,
    expected: str,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Poll GET /api/sessions/{id} until status matches; return last view."""

    deadline = asyncio.get_running_loop().time() + timeout
    view: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/sessions/{session_id}")
        if response.status_code == 200:
            view = response.json()
            if view.get("status") == expected:
                return view
        await asyncio.sleep(0.1)
    return view


@pytest.fixture
def server_factory():
    return run_server


@pytest.fixture
def script_writer():
    return write_script


@pytest.fixture
def sse_collector():
    return collect_sse


@pytest.fixture
def calls_reader():
    return read_calls


@pytest.fixture
def events_reader():
    return read_events


@pytest.fixture
def status_waiter():
    return wait_status
