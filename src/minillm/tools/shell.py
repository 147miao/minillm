"""run_bash：在 workspace 内执行 shell 命令。

约束：
- 子进程环境使用白名单，绝不继承 LLM_API_KEY / 代理 / token；
- 独立进程组，超时 killpg；
- stdout/stderr 各 1 MiB 硬上限，再按 MAX_TOOL_OUTPUT_CHARS 落盘截断。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..errors import ToolError
from .base import ToolContext, ToolResult, ToolSpec, resolve_workspace_path

_OUTPUT_CAP = 1024 * 1024
_ALLOWED_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "PYTHONPATH",
    "TMPDIR",
    "USER",
    "SHELL",
    "TERM",
    "VIRTUAL_ENV",
    "PYTHONIOENCODING",
)


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunBashParams(_Params):
    command: str = Field(min_length=1, max_length=100_000)
    cwd: str | None = None
    timeout: int = Field(default=60, ge=1, le=300)


def _bash_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _ALLOWED_ENV if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.setdefault("HOME", str(Path.home()))
    env["PYTHONUNBUFFERED"] = "1"
    return env


async def _read_capped(stream: Any, cap: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    kept = 0
    total = 0
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if kept < cap:
            take = chunk[: cap - kept]
            chunks.append(take)
            kept += len(take)
    return b"".join(chunks), total > cap


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


async def run_bash(ctx: ToolContext, params: RunBashParams) -> ToolResult:
    cwd = resolve_workspace_path(
        ctx,
        params.cwd,
        tool="run_bash",
        must_exist=True,
        expect_dir=True,
    )
    bash = shutil.which("bash") or "/bin/bash"
    started = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            bash,
            "-c",
            params.command,
            cwd=str(cwd),
            env=_bash_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise ToolError(
            f"命令启动失败：{exc}", kind="execution", tool="run_bash"
        ) from exc
    if proc.stdout is None or proc.stderr is None:  # pragma: no cover - 防御性
        raise ToolError("子进程管道创建失败", kind="execution", tool="run_bash")

    out_task = asyncio.create_task(_read_capped(proc.stdout, _OUTPUT_CAP))
    err_task = asyncio.create_task(_read_capped(proc.stderr, _OUTPUT_CAP))
    timed_out = False
    try:
        await asyncio.wait_for(proc.wait(), timeout=params.timeout)
        exit_code = int(proc.returncode or 0)
    except TimeoutError:
        timed_out = True
        _kill_process_group(proc)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=5)
        exit_code = -1
    except asyncio.CancelledError:
        _kill_process_group(proc)
        with contextlib.suppress(Exception):
            await proc.wait()
        raise
    finally:
        out_bytes, out_capped = await out_task
        err_bytes, err_capped = await err_task

    duration_ms = int((time.perf_counter() - started) * 1000)
    stdout = out_bytes.decode("utf-8", errors="replace")
    stderr = err_bytes.decode("utf-8", errors="replace")
    if timed_out:
        stderr += f"\n[command timed out after {params.timeout}s; process killed]"
    stdout, stderr, artifact_truncated, artifact = ctx.artifacts.cap_streams(
        "run_bash", stdout, stderr
    )
    truncated = out_capped or err_capped or artifact_truncated
    meta: dict[str, Any] = {"timed_out": timed_out}
    if artifact is not None:
        meta["artifact"] = artifact
        meta["truncated"] = True
    value = {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": truncated,
        "duration_ms": duration_ms,
    }
    return ToolResult(ok=True, value=value, meta=meta, artifact=artifact)


SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="run_bash",
        description="在 workspace 内执行 bash 命令，带超时与输出上限",
        params=RunBashParams,
        signature=(
            "run_bash(command: str, cwd: str | None = None, timeout: int = 60) -> dict"
        ),
        handler=run_bash,
    ),
)
