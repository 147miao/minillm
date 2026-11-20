#!/usr/bin/env python3
"""最小 MCP stdio server（测试与演示用）。

协议：newline-delimited JSON-RPC 2.0；stdout 只写协议消息，stderr 写日志。
工具：
- echo(text: str) -> text
- add(a: int, b: int) -> a + b
- slow() -> 延迟 1s（仅用于测试超时）
"""

from __future__ import annotations

import json
import signal
import sys
import time
from typing import Any

PROTOCOL_VERSION = "2025-06-18"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "回显传入的文本",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要回显的文本"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add",
        "description": "两个整数相加",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "第一个加数"},
                "b": {"type": "integer", "description": "第二个加数"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
    {
        "name": "slow",
        "description": "延迟 1s 后返回（仅测试超时用）",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


def _log(message: str) -> None:
    print(f"[mcp-stub] {message}", file=sys.stderr, flush=True)


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _result(request_id: Any, result: dict[str, Any]) -> None:
    _send({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id: Any, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def _text(tool: str, text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle(message: dict[str, Any]) -> None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") if isinstance(message.get("params"), dict) else {}

    if method == "initialize":
        _result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mcp-stub", "version": "0.1.0"},
            },
        )
        _log("initialize")
        return
    if method == "notifications/initialized":
        _log("initialized")
        return
    if method == "ping":
        _result(request_id, {})
        return
    if method == "tools/list":
        _result(request_id, {"tools": TOOLS})
        return
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        _log(f"call {name} {arguments}")
        if name == "echo":
            _result(request_id, _text("echo", str(arguments.get("text", ""))))
            return
        if name == "add":
            try:
                total = int(arguments.get("a", 0)) + int(arguments.get("b", 0))
            except (TypeError, ValueError):
                _result(
                    request_id,
                    {
                        "content": [{"type": "text", "text": "a/b 必须是整数"}],
                        "isError": True,
                    },
                )
                return
            _result(request_id, _text("add", str(total)))
            return
        if name == "slow":
            time.sleep(1.0)
            _result(request_id, _text("slow", "slow done"))
            return
        _result(
            request_id,
            {
                "content": [{"type": "text", "text": f"unknown tool: {name}"}],
                "isError": True,
            },
        )
        return
    if request_id is not None:
        _error(request_id, -32601, f"method not found: {method}")


def main() -> int:
    signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
    _log("started")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _log("invalid json line")
            continue
        if isinstance(message, dict):
            handle(message)
    _log("stdin closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
