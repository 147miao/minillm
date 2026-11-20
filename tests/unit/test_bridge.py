"""桥接帧与 worker 子进程的单元测试。"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path

import pytest

from minillm.errors import BridgeError
from minillm.runtime.bridge import Bridge, create_socketpair
from minillm.runtime.protocol import (
    MAX_FRAME_BYTES,
    decode_frame,
    encode_frame,
    read_frame,
)


def test_frame_roundtrip() -> None:
    payload = {"type": "tool_call", "call_id": "tc_1", "args": {"x": "中文"}}
    encoded = encode_frame(payload)
    assert encoded.endswith(b"\n")
    assert decode_frame(encoded) == payload


def test_frame_rejects_bad_json() -> None:
    with pytest.raises(BridgeError):
        decode_frame(b"{not json}\n")
    with pytest.raises(BridgeError):
        decode_frame(b'["not-object"]\n')
    with pytest.raises(BridgeError):
        decode_frame(b"no-newline")


def test_frame_rejects_oversize() -> None:
    huge = {"x": "a" * (MAX_FRAME_BYTES + 10)}
    with pytest.raises(BridgeError):
        encode_frame(huge)
    with pytest.raises(BridgeError):
        decode_frame(b"a" * (MAX_FRAME_BYTES + 1) + b"\n")


async def test_read_frame_eof() -> None:
    reader = asyncio.StreamReader()
    reader.feed_eof()
    assert await read_frame(reader) is None

    reader = asyncio.StreamReader()
    reader.feed_data(encode_frame({"type": "done"}))
    reader.feed_eof()
    assert await read_frame(reader) == {"type": "done"}


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


def _worker_env() -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": str(_repo_src()),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    return env


async def _spawn_worker(tmp_path: Path) -> tuple[asyncio.subprocess.Process, Bridge]:
    parent, child = create_socketpair()
    child_fd = child.fileno()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "minillm.runtime.worker",
        "--bridge-fd",
        str(child_fd),
        cwd=str(tmp_path),
        env=_worker_env(),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        pass_fds=(child_fd,),
        start_new_session=True,
    )
    child.close()
    bridge = await Bridge.connect(parent)
    return proc, bridge


async def _close_worker(proc: asyncio.subprocess.Process, bridge: Bridge) -> None:
    with contextlib.suppress(Exception):
        await bridge.close()
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), 9)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=3)


async def test_worker_positional_and_keyword_args(tmp_path: Path) -> None:
    proc, bridge = await _spawn_worker(tmp_path)
    try:
        code = (
            'print(tools.echo("hi"))\n'
            'print(tools.plus(2, 3))\n'
            'print(tools.echo(text="kw"))\n'
            'print("中文🙂")'
        )
        await bridge.send_exec(code, {"echo": ["text"], "plus": ["a", "b"]})
        first = await asyncio.wait_for(bridge.read(), timeout=10)
        assert first is not None
        assert first["type"] == "tool_call"
        assert first["name"] == "echo"
        assert first["args"] == {"text": "hi"}
        await bridge.reply({"call_id": first["call_id"], "ok": True, "result": "OK"})

        second = await asyncio.wait_for(bridge.read(), timeout=10)
        assert second is not None
        assert second["name"] == "plus"
        assert second["args"] == {"a": 2, "b": 3}
        await bridge.reply({"call_id": second["call_id"], "ok": True, "result": 5})

        third = await asyncio.wait_for(bridge.read(), timeout=10)
        assert third is not None
        assert third["args"] == {"text": "kw"}
        await bridge.reply({"call_id": third["call_id"], "ok": True, "result": "KW"})

        done = await asyncio.wait_for(bridge.read(), timeout=10)
        assert done == {"type": "done", "exit_code": 0}
        assert proc.stdout is not None and proc.stderr is not None
        stdout = (await proc.stdout.read()).decode()
        await proc.wait()
        assert "OK" in stdout and "5" in stdout and "KW" in stdout
        assert "中文🙂" in stdout
        assert proc.returncode == 0
    finally:
        await _close_worker(proc, bridge)


async def test_worker_tool_error_is_catchable(tmp_path: Path) -> None:
    proc, bridge = await _spawn_worker(tmp_path)
    try:
        code = (
            'try:\n'
            '    tools.echo("x")\n'
            'except ToolCallError as exc:\n'
            '    print(exc.kind, exc.tool)'
        )
        await bridge.send_exec(code, {"echo": ["text"]})
        frame = await asyncio.wait_for(bridge.read(), timeout=10)
        assert frame is not None and frame["type"] == "tool_call"
        await bridge.reply(
            {
                "call_id": frame["call_id"],
                "ok": False,
                "error": {
                    "kind": "validation",
                    "tool": "echo",
                    "message": "bad input",
                    "errors": [{"loc": ["text"], "msg": "bad"}],
                },
            }
        )
        done = await asyncio.wait_for(bridge.read(), timeout=10)
        assert done == {"type": "done", "exit_code": 0}
        assert proc.stdout is not None
        stdout = (await proc.stdout.read()).decode()
        await proc.wait()
        assert "validation echo" in stdout
    finally:
        await _close_worker(proc, bridge)


async def test_worker_unknown_tool_and_syntax_error(tmp_path: Path) -> None:
    proc, bridge = await _spawn_worker(tmp_path)
    try:
        await bridge.send_exec('tools.nope()', {})
        done = await asyncio.wait_for(bridge.read(), timeout=10)
        assert done == {"type": "done", "exit_code": 1}
        assert proc.stderr is not None
        stderr = (await proc.stderr.read()).decode()
        await proc.wait()
        assert "ToolCallError" in stderr or "未知工具" in stderr
    finally:
        await _close_worker(proc, bridge)

    proc, bridge = await _spawn_worker(tmp_path)
    try:
        await bridge.send_exec("if True print(1)", {})
        done = await asyncio.wait_for(bridge.read(), timeout=10)
        assert done == {"type": "done", "exit_code": 1}
        assert proc.stderr is not None
        stderr = (await proc.stderr.read()).decode()
        await proc.wait()
        assert "SyntaxError" in stderr
    finally:
        await _close_worker(proc, bridge)


async def test_worker_bridge_eof_exits(tmp_path: Path) -> None:
    proc, bridge = await _spawn_worker(tmp_path)
    await bridge.send_exec('print("before")', {})
    await bridge.close()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=5)
    if proc.returncode is None:
        await _close_worker(proc, bridge)
    assert proc.returncode is not None
