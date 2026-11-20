"""run_bash / web_fetch / net 助手测试。"""

from __future__ import annotations

import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from minillm.artifacts import ArtifactStore
from minillm.config import Settings
from minillm.errors import ToolError
from minillm.net import make_async_http_client, sanitize_proxy_env
from minillm.tools.base import ToolContext


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/404":
            self.send_response(404)
            self.end_headers()
            return
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return
        body = (
            b"<html><head><title>T</title><script>bad()</script></head>"
            b"<body><h1>Hello</h1><p>world</p></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        return


@pytest.fixture
def http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _fetch_ctx(
    tool_ctx: ToolContext, settings: Settings, store, session_id: str
) -> ToolContext:
    private = settings.model_copy(update={"web_fetch_allow_private": True})
    return replace(
        tool_ctx,
        settings=private,
        artifacts=ArtifactStore(store, session_id, private),
    )


async def test_run_bash_basic(registry, tool_ctx) -> None:
    result = await registry.invoke("run_bash", {"command": "echo hi"}, tool_ctx)
    assert result.value["exit_code"] == 0
    assert result.value["stdout"] == "hi\n"
    assert result.value["stderr"] == ""
    assert result.value["truncated"] is False
    assert result.value["duration_ms"] >= 0


async def test_run_bash_exit_code(registry, tool_ctx) -> None:
    result = await registry.invoke("run_bash", {"command": "exit 3"}, tool_ctx)
    assert result.value["exit_code"] == 3


async def test_run_bash_cwd(registry, tool_ctx) -> None:
    sub = Path(tool_ctx.workspace) / "sub"
    sub.mkdir()
    result = await registry.invoke("run_bash", {"command": "pwd", "cwd": "sub"}, tool_ctx)
    assert result.value["stdout"].strip() == str(sub)
    with pytest.raises(ToolError) as escaped:
        await registry.invoke("run_bash", {"command": "pwd", "cwd": "../"}, tool_ctx)
    assert escaped.value.kind == "validation"


async def test_run_bash_timeout(registry, tool_ctx) -> None:
    result = await registry.invoke(
        "run_bash", {"command": "sleep 5", "timeout": 1}, tool_ctx
    )
    assert result.value["exit_code"] == -1
    assert result.meta["timed_out"] is True
    assert "timed out" in result.value["stderr"]
    assert result.value["duration_ms"] < 4000


async def test_run_bash_env_sanitized(registry, tool_ctx) -> None:
    command = (
        "python -c \"import os;"
        "print(repr(os.environ.get('LLM_API_KEY')), "
        "repr(os.environ.get('HTTP_PROXY')), "
        "repr(os.environ.get('NO_PROXY')))\""
    )
    result = await registry.invoke("run_bash", {"command": command}, tool_ctx)
    assert result.value["stdout"].strip() == "None None None"


async def test_run_bash_output_cap(registry, small_ctx) -> None:
    result = await registry.invoke(
        "run_bash", {"command": "python -c \"print('x' * 500)\""}, small_ctx
    )
    assert result.value["truncated"] is True
    assert result.artifact is not None
    assert "full output" in result.value["stdout"]
    assert "xxxx" in small_ctx.artifacts.read_text(result.artifact)


async def test_run_bash_stderr_cap(registry, small_ctx) -> None:
    command = '''python -c "import sys; print('e' * 500, file=sys.stderr)"'''
    result = await registry.invoke("run_bash", {"command": command}, small_ctx)
    assert result.value["truncated"] is True
    assert result.artifact is not None
    assert "full output" in result.value["stderr"]


async def test_web_fetch_html(
    registry, tool_ctx, settings, store, session_id, http_server
) -> None:
    ctx = _fetch_ctx(tool_ctx, settings, store, session_id)
    result = await registry.invoke("web_fetch", {"url": http_server}, ctx)
    assert "Hello" in result.value
    assert "world" in result.value
    assert "bad(" not in result.value
    assert result.meta["status"] == 200


async def test_web_fetch_redirect(
    registry, tool_ctx, settings, store, session_id, http_server
) -> None:
    ctx = _fetch_ctx(tool_ctx, settings, store, session_id)
    result = await registry.invoke("web_fetch", {"url": http_server + "/redirect"}, ctx)
    assert "Hello" in result.value
    assert result.meta["status"] == 200
    assert result.meta["url"].endswith("/")


async def test_web_fetch_private_rejected(registry, tool_ctx, http_server) -> None:
    with pytest.raises(ToolError) as rejected:
        await registry.invoke("web_fetch", {"url": http_server}, tool_ctx)
    assert rejected.value.kind == "validation"


async def test_web_fetch_http_error(
    registry, tool_ctx, settings, store, session_id, http_server
) -> None:
    ctx = _fetch_ctx(tool_ctx, settings, store, session_id)
    with pytest.raises(ToolError) as not_found:
        await registry.invoke("web_fetch", {"url": http_server + "/404"}, ctx)
    assert not_found.value.kind == "execution"


async def test_web_fetch_scheme_rejected(registry, tool_ctx) -> None:
    with pytest.raises(ToolError) as invalid:
        await registry.invoke("web_fetch", {"url": "file:///etc/passwd"}, tool_ctx)
    assert invalid.value.kind == "validation"


async def test_web_fetch_truncation(
    registry, tool_ctx, settings, store, session_id, http_server
) -> None:
    ctx = _fetch_ctx(tool_ctx, settings, store, session_id)
    result = await registry.invoke(
        "web_fetch", {"url": http_server, "max_chars": 10}, ctx
    )
    assert result.artifact is not None
    assert "full output" in result.value
    assert "Hello" in ctx.artifacts.read_text(result.artifact)


def test_sanitize_proxy_env() -> None:
    env = {
        "NO_PROXY": "localhost,127.0.0.1,[::1]",
        "no_proxy": "[::1],example.com",
    }
    cleaned = sanitize_proxy_env(env)
    assert "[" not in env["NO_PROXY"]
    assert "[" not in env["no_proxy"]
    assert cleaned == "localhost,127.0.0.1"
    assert env["no_proxy"] == "example.com"


async def test_http_client_constructs_with_bracket_no_proxy(
    settings, monkeypatch
) -> None:
    monkeypatch.setenv("NO_PROXY", "localhost,[::1]")
    monkeypatch.setenv("no_proxy", "localhost,[::1]")
    client = make_async_http_client(settings)
    try:
        assert client is not None
    finally:
        await client.aclose()
