"""run_code 执行器：启动 worker、桥接 tools.*、转发事件与结果。

边界：
- 只负责执行一次模型 Python 程序；工具实现与校验在 ToolRegistry。
- 每个 run_code 一个 worker 子进程；socketpair + pass_fds，无 TCP/token。
- 空闲超时在“工具调用进行中”暂停，避免 wait_agent 等长工具被误杀；硬上限始终生效。
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..errors import MiniLLMError, ToolError
from ..events import EventType
from ..tools.base import ToolContext
from ..tools.registry import ToolRegistry
from .bridge import Bridge, create_socketpair

_STREAM_CAP = 1024 * 1024
_SAFE_ENV_KEYS = {
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
}


@dataclass(slots=True)
class RunCodeResult:
    """一次 run_code 的完整结果；content 是喂回模型的 tool 消息文本。"""

    exit_code: int
    stdout: str
    stderr: str
    content: str
    artifact: str | None
    duration_ms: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    timed_out: bool = False
    cancelled: bool = False
    error: str | None = None


def _preview(value: Any, limit: int = 2000) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


class RunCodeExecutor:
    """模型 Python 程序的执行器。"""

    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry

    async def execute(
        self,
        *,
        code: str,
        call_id: str,
        ctx: ToolContext,
    ) -> RunCodeResult:
        """执行一段模型代码并返回结构化结果；取消时先发 finished 再 re-raise。"""

        started = time.monotonic()
        state: dict[str, Any] = {
            "timed_out": False,
            "cancelled": False,
            "error": None,
            "done_exit": None,
            "bridge_error": None,
        }
        activity: dict[str, Any] = {"last": started, "inflight": 0}

        code_artifact = None
        code_preview = code
        if len(code) > self.settings.max_tool_output_chars:
            code_artifact = ctx.artifacts.save_text("run_code_code", code, suffix=".py")
            code_preview = (
                code[: self.settings.max_tool_output_chars]
                + f"\n[truncated; full code: {code_artifact}]"
            )
        await ctx.emit_event(
            EventType.RUN_CODE_STARTED,
            {"call_id": call_id, "code": code_preview, "artifact": code_artifact},
        )

        tools_map = self._tools_map(ctx)
        tool_records: list[dict[str, Any]] = []
        parent_sock = None
        child_sock = None
        proc: asyncio.subprocess.Process | None = None
        bridge: Bridge | None = None
        stdout_task: asyncio.Task | None = None
        stderr_task: asyncio.Task | None = None
        bridge_task: asyncio.Task | None = None
        wd_task: asyncio.Task | None = None
        lock = asyncio.Lock()
        stdout_text = ""
        stderr_text = ""
        stdout_trunc = False
        stderr_trunc = False
        proc_rc: int | None = None

        try:
            try:
                Path(ctx.workspace).mkdir(parents=True, exist_ok=True)
                parent_sock, child_sock = create_socketpair()
                child_sock.set_inheritable(True)
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "minillm.runtime.worker",
                    "--bridge-fd",
                    str(child_sock.fileno()),
                    cwd=str(ctx.workspace),
                    env=self._child_env(),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    pass_fds=(child_sock.fileno(),),
                    start_new_session=True,
                )
                child_sock.close()
                child_sock = None

                bridge = await Bridge.connect(parent_sock)
                parent_sock = None
                await bridge.send_exec(code, tools_map)

                assert proc.stdout is not None and proc.stderr is not None
                stdout_task = asyncio.create_task(
                    self._read_stream(proc.stdout, activity)
                )
                stderr_task = asyncio.create_task(
                    self._read_stream(proc.stderr, activity)
                )
                bridge_task = asyncio.create_task(
                    self._bridge_loop(
                        bridge, ctx, call_id, tool_records, activity, state, lock
                    )
                )
                wd_task = asyncio.create_task(
                    self._watchdog(proc, activity, started, state)
                )
                await proc.wait()
            except asyncio.CancelledError:
                state["cancelled"] = True
                self._killpg(proc)
                raise
            except Exception as exc:  # noqa: BLE001 - 启动/桥接失败都要结构化返回
                state["error"] = f"{type(exc).__name__}: {exc}"
                self._killpg(proc)
            finally:
                if wd_task is not None:
                    wd_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await wd_task
                if bridge is not None:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await asyncio.wait_for(bridge.close(), timeout=2)
                if proc is not None:
                    try:
                        proc_rc = await asyncio.wait_for(proc.wait(), timeout=3)
                    except (TimeoutError, asyncio.CancelledError):
                        self._killpg(proc)
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            proc_rc = await asyncio.wait_for(proc.wait(), timeout=2)
                stream_tasks = [
                    task
                    for task in (stdout_task, stderr_task, bridge_task)
                    if task is not None
                ]
                if stream_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*stream_tasks, return_exceptions=True),
                            timeout=3,
                        )
                    except TimeoutError:
                        for task in stream_tasks:
                            task.cancel()
                        await asyncio.gather(*stream_tasks, return_exceptions=True)
                stdout_text, stdout_trunc = self._task_stream_value(stdout_task)
                stderr_text, stderr_trunc = self._task_stream_value(stderr_task)
                if bridge_task is not None and bridge_task.done():
                    try:
                        exc = bridge_task.exception()
                    except (asyncio.CancelledError, Exception):
                        exc = None
                    if exc is not None and not state.get("error"):
                        state["error"] = f"{type(exc).__name__}: {exc}"
                if parent_sock is not None:
                    with contextlib.suppress(OSError):
                        parent_sock.close()
                if child_sock is not None:
                    with contextlib.suppress(OSError):
                        child_sock.close()

            result, data = self._build_result(
                ctx=ctx,
                call_id=call_id,
                code_artifact=code_artifact,
                started=started,
                proc_rc=proc_rc,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                stdout_trunc=stdout_trunc,
                stderr_trunc=stderr_trunc,
                state=state,
                tool_records=tool_records,
            )
            await ctx.emit_event(EventType.RUN_CODE_FINISHED, data)
            return result
        except asyncio.CancelledError as cancel_exc:
            state["cancelled"] = True
            result, data = self._build_result(
                ctx=ctx,
                call_id=call_id,
                code_artifact=code_artifact,
                started=started,
                proc_rc=proc_rc,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                stdout_trunc=stdout_trunc,
                stderr_trunc=stderr_trunc,
                state=state,
                tool_records=tool_records,
            )
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(
                    ctx.emit_event(EventType.RUN_CODE_FINISHED, data)
                )
            raise cancel_exc

    # ==== 子进程 ====
    def _child_env(self) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV_KEYS}
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        src = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            [part for part in (src, existing) if part]
        )
        return env

    @staticmethod
    def _killpg(proc: asyncio.subprocess.Process | None) -> None:
        if proc is None or proc.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        except OSError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    async def _watchdog(
        self,
        proc: asyncio.subprocess.Process,
        activity: dict[str, Any],
        started: float,
        state: dict[str, Any],
    ) -> None:
        idle_limit = max(1, int(self.settings.run_code_idle_timeout))
        hard_limit = max(1, int(self.settings.run_code_max_timeout))
        while proc.returncode is None:
            await asyncio.sleep(0.05)
            if proc.returncode is not None:
                return
            now = time.monotonic()
            if now - started > hard_limit:
                state["timed_out"] = True
                self._killpg(proc)
                return
            # 关键：工具调用进行中暂停空闲计时（wait_agent 等长调用），只受硬上限约束
            if activity["inflight"]:
                continue
            if now - float(activity["last"]) > idle_limit:
                state["timed_out"] = True
                self._killpg(proc)
                return

    async def _read_stream(
        self,
        stream: asyncio.StreamReader,
        activity: dict[str, Any],
    ) -> tuple[str, bool]:
        chunks: list[bytes] = []
        total = 0
        truncated = False
        while True:
            data = await stream.read(65536)
            if not data:
                break
            activity["last"] = time.monotonic()
            if total < _STREAM_CAP:
                take = data[: _STREAM_CAP - total]
                chunks.append(take)
                total += len(take)
            if total >= _STREAM_CAP:
                truncated = True
        return b"".join(chunks).decode("utf-8", errors="replace"), truncated

    @staticmethod
    def _task_stream_value(task: asyncio.Task | None) -> tuple[str, bool]:
        if task is None or not task.done() or task.cancelled():
            return "", False
        try:
            value = task.result()
        except Exception:  # noqa: BLE001 - 读流出错按空处理
            return "", False
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], str)
        ):
            return value[0], bool(value[1])
        return "", False

    # ==== 工具映射与桥处理 ====
    def _tools_map(self, ctx: ToolContext) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for spec in self.registry.all_specs:
            if spec.kind == "subagent" and not ctx.allow_subagents:
                continue
            mapping[spec.name] = list(spec.params.model_fields.keys())
        return mapping

    async def _bridge_loop(
        self,
        bridge: Bridge,
        ctx: ToolContext,
        run_call_id: str,
        records: list[dict[str, Any]],
        activity: dict[str, Any],
        state: dict[str, Any],
        lock: asyncio.Lock,
    ) -> None:
        try:
            while True:
                frame = await bridge.read()
                if frame is None:
                    return
                activity["last"] = time.monotonic()
                kind = frame.get("type")
                if kind == "done":
                    try:
                        state["done_exit"] = int(frame.get("exit_code", 0))
                    except (TypeError, ValueError):
                        state["done_exit"] = None
                    return
                if kind != "tool_call":
                    continue
                await self._handle_tool_call(
                    bridge, ctx, run_call_id, frame, records, activity, lock
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 记录后交给清理路径
            state["bridge_error"] = f"{type(exc).__name__}: {exc}"

    async def _handle_tool_call(
        self,
        bridge: Bridge,
        ctx: ToolContext,
        run_call_id: str,
        frame: dict[str, Any],
        records: list[dict[str, Any]],
        activity: dict[str, Any],
        lock: asyncio.Lock,
    ) -> None:
        tool_call_id = str(frame.get("call_id") or "")
        name = str(frame.get("name") or "")
        raw_args = frame.get("args")
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        activity["last"] = time.monotonic()
        activity["inflight"] += 1
        try:
            args_text = json.dumps(args, ensure_ascii=False, default=str)
            args_artifact = None
            if len(args_text) > 2000:
                args_artifact = ctx.artifacts.save_json("tool_args", args)
                args_preview = args_text[:2000] + f"... [full: {args_artifact}]"
            else:
                args_preview = args_text
            await ctx.emit_event(
                EventType.TOOL_CALL,
                {
                    "call_id": tool_call_id,
                    "run_call_id": run_call_id,
                    "tool": name,
                    "args_preview": args_preview,
                    "args_artifact": args_artifact,
                },
            )

            started = time.perf_counter()
            ok = False
            value: Any = None
            meta: dict[str, Any] = {}
            artifact: str | None = None
            error: dict[str, Any] | None = None
            try:
                async with lock:
                    result = await self.registry.invoke(
                        name,
                        args,
                        dataclasses.replace(ctx, call_id=tool_call_id),
                    )
                ok = bool(result.ok)
                value = result.value
                meta = dict(result.meta or {})
                artifact = result.artifact
                if not ok:
                    error = {
                        "kind": "tool_error",
                        "tool": name,
                        "message": "工具返回 ok=false",
                        "errors": [],
                    }
            except ToolError as exc:
                error = exc.to_dict()
            except MiniLLMError as exc:
                error = {
                    "kind": "tool_error",
                    "tool": name,
                    "message": exc.message,
                    "errors": [],
                }
            except Exception as exc:  # noqa: BLE001 - 工具异常必须回给模型
                error = {
                    "kind": "execution",
                    "tool": name,
                    "message": f"{type(exc).__name__}: {exc}",
                    "errors": [],
                }
            duration_ms = int((time.perf_counter() - started) * 1000)
            records.append({"name": name, "ok": ok, "duration_ms": duration_ms})
            preview = _preview(value if ok else error)
            await ctx.emit_event(
                EventType.TOOL_RESULT,
                {
                    "call_id": tool_call_id,
                    "run_call_id": run_call_id,
                    "tool": name,
                    "ok": ok,
                    "preview": preview,
                    "artifact": artifact,
                    "error": error,
                    "synthetic": False,
                },
            )
            if ok:
                await bridge.reply(
                    {
                        "call_id": tool_call_id,
                        "ok": True,
                        "result": value,
                        "meta": meta,
                        "artifact": artifact,
                    }
                )
            else:
                await bridge.reply(
                    {
                        "call_id": tool_call_id,
                        "ok": False,
                        "error": error,
                    }
                )
        finally:
            activity["inflight"] = max(0, int(activity["inflight"]) - 1)
            activity["last"] = time.monotonic()

    # ==== 结果 ====
    def _build_result(
        self,
        *,
        ctx: ToolContext,
        call_id: str,
        code_artifact: str | None,
        started: float,
        proc_rc: int | None,
        stdout_text: str,
        stderr_text: str,
        stdout_trunc: bool,
        stderr_trunc: bool,
        state: dict[str, Any],
        tool_records: list[dict[str, Any]],
    ) -> tuple[RunCodeResult, dict[str, Any]]:
        timed_out = bool(state.get("timed_out"))
        cancelled = bool(state.get("cancelled"))
        error = state.get("error") or state.get("bridge_error")
        if cancelled or timed_out or error:
            exit_code = -1
        elif state.get("done_exit") is not None:
            exit_code = int(state["done_exit"])
        elif proc_rc is not None:
            exit_code = int(proc_rc)
        else:
            exit_code = -1

        lines: list[str] = [f"exit_code: {exit_code}"]
        if timed_out:
            lines.append("timed_out: true")
        if cancelled:
            lines.append("cancelled: true")
        if error:
            lines.append(f"error: {error}")
        lines.append("stdout:")
        lines.append(stdout_text.rstrip() or "(empty)")
        lines.append("stderr:")
        lines.append(stderr_text.rstrip() or "(empty)")
        if tool_records:
            lines.append("tool calls:")
            for record in tool_records:
                status = "ok" if record.get("ok") else "error"
                lines.append(
                    f"- {record.get('name')}: {status} ({record.get('duration_ms')}ms)"
                )
        full_content = "\n".join(lines)
        content, artifact = ctx.artifacts.cap_text(
            "run_code_output",
            full_content,
            max_chars=self.settings.run_code_output_chars,
            reason="run_code_output",
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        truncated = bool(
            code_artifact or stdout_trunc or stderr_trunc or artifact is not None
        )
        result = RunCodeResult(
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
            content=content,
            artifact=artifact,
            duration_ms=duration_ms,
            tool_calls=list(tool_records),
            truncated=truncated,
            timed_out=timed_out,
            cancelled=cancelled,
            error=error,
        )
        data: dict[str, Any] = {
            "call_id": call_id,
            "exit_code": exit_code,
            "stdout_preview": stdout_text[:2000],
            "stderr_preview": stderr_text[:2000],
            "content": content,
            "artifact": artifact,
            "duration_ms": duration_ms,
            "tool_calls": list(tool_records),
            "synthetic": False,
            "timed_out": timed_out,
            "cancelled": cancelled,
        }
        if error:
            data["error"] = error
        return result, data


__all__ = ["RunCodeExecutor", "RunCodeResult"]
