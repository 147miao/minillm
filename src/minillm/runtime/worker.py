"""run_code 子进程：执行模型输出的 Python，并通过 socket 代理 tools.*。

协议只走 --bridge-fd 传入的 socketpair；stdout/stderr 留给模型输出与 traceback，
绝不写协议帧。宿主关闭 fd/退出时 readline 返回 EOF，worker 立即终止，避免孤儿。
"""

from __future__ import annotations

import argparse
import builtins
import secrets
import socket
import sys
import traceback
from typing import Any, BinaryIO

from .protocol import BridgeError, read_frame_sync, write_frame_sync


class ToolCallError(Exception):
    """宿主工具调用失败；模型代码可以 except ToolCallError 自行处理。"""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "tool_error",
        tool: str | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.tool = tool
        self.errors = list(errors or [])

    def __str__(self) -> str:
        prefix = f"[{self.kind}]"
        if self.tool:
            prefix += f" {self.tool}:"
        return f"{prefix} {self.args[0] if self.args else ''}"


class _Tools:
    """按工具名暴露同步函数；位置参数按宿主给的参数顺序映射。"""

    def __init__(
        self,
        params: dict[str, list[str]],
        reader: BinaryIO,
        writer: BinaryIO,
    ) -> None:
        self._params = params
        self._reader = reader
        self._writer = writer

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._params:
            raise ToolCallError(f"未知工具：{name}", kind="unknown_tool", tool=name)
        order = self._params[name]

        def invoke(*args: Any, **kwargs: Any) -> Any:
            if len(args) > len(order):
                raise ToolCallError(
                    f"工具 {name} 最多接受 {len(order)} 个位置参数，实际 {len(args)} 个",
                    kind="validation",
                    tool=name,
                )
            payload: dict[str, Any] = dict(zip(order, args, strict=False))
            for key in payload:
                if key in kwargs:
                    raise ToolCallError(
                        f"工具 {name} 的参数 {key} 同时以位置和关键字传入",
                        kind="validation",
                        tool=name,
                    )
            payload.update(kwargs)
            return self._call(name, payload)

        invoke.__name__ = name
        return invoke

    def _call(self, name: str, args: dict[str, Any]) -> Any:
        call_id = f"tc_{secrets.token_hex(8)}"
        write_frame_sync(
            self._writer,
            {"type": "tool_call", "call_id": call_id, "name": name, "args": args},
        )
        frame = read_frame_sync(self._reader)
        if frame is None:
            raise ToolCallError("宿主桥接已关闭", kind="bridge", tool=name)
        if frame.get("call_id") != call_id:
            raise ToolCallError(
                f"桥接响应 id 不匹配：期望 {call_id}，实际 {frame.get('call_id')}",
                kind="bridge",
                tool=name,
            )
        if frame.get("ok"):
            return frame.get("result")
        error = frame.get("error") or {}
        raise ToolCallError(
            str(error.get("message") or "工具调用失败"),
            kind=str(error.get("kind") or "tool_error"),
            tool=str(error.get("tool") or name),
            errors=error.get("errors") if isinstance(error.get("errors"), list) else [],
        )


def _execute(code: str, tools: _Tools) -> int:
    """执行模型代码；返回进程 exit code。"""

    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__builtins__": builtins,
        "tools": tools,
        "ToolCallError": ToolCallError,
    }
    try:
        compiled = compile(code, "<run_code>", "exec")
        exec(compiled, namespace)
        return 0
    except SystemExit as exc:
        value = exc.code
        if value is None:
            return 0
        return value if isinstance(value, int) else 1
    except BaseException:  # noqa: BLE001 - 未捕获异常一律回给宿主
        traceback.print_exc()
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MiniLLM run_code worker")
    parser.add_argument("--bridge-fd", type=int, required=True)
    args = parser.parse_args(argv)

    try:
        sock = socket.socket(fileno=args.bridge_fd)
    except OSError as exc:
        print(f"worker 无法接管 bridge fd：{exc}", file=sys.stderr)
        return 2

    reader = sock.makefile("rb")
    writer = sock.makefile("wb")
    try:
        frame = read_frame_sync(reader)
        if frame is None:
            return 0
        if frame.get("type") != "exec":
            print("worker 期望 exec 帧", file=sys.stderr)
            return 2
        code = frame.get("code")
        if not isinstance(code, str):
            print("worker 的 code 必须是字符串", file=sys.stderr)
            return 2
        raw_tools = frame.get("tools") or {}
        tools_map: dict[str, list[str]] = {}
        if isinstance(raw_tools, dict):
            for name, order in raw_tools.items():
                if isinstance(order, list):
                    tools_map[str(name)] = [str(item) for item in order]
        exit_code = _execute(code, _Tools(tools_map, reader, writer))
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            write_frame_sync(writer, {"type": "done", "exit_code": exit_code})
        except (BrokenPipeError, OSError, BridgeError):
            pass
        return exit_code
    except BridgeError as exc:
        print(f"worker 桥接协议错误：{exc}", file=sys.stderr)
        return 1
    finally:
        try:
            reader.close()
        except OSError:
            pass
        try:
            writer.close()
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
