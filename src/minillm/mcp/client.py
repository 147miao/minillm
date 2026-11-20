"""MCP stdio JSON-RPC 2.0 客户端与动态工具注册。

协议要点：
- 客户端启动子进程，stdin/stdout 按行传 JSON-RPC 2.0（UTF-8，一行一个消息）；
- 生命周期：initialize → notifications/initialized → tools/list → tools/call；
- stdout 只允许协议消息，stderr 单独 drain 到日志；
- 每个 server 一把 asyncio.Lock，请求串行，避免 stdio 并发交错。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..config import Settings
from ..errors import MCPError, ToolError
from ..tools.base import ToolContext, ToolSpec
from ..tools.registry import ToolRegistry
from .config import MCPServerConfig, load_mcp_config

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "minillm", "version": "0.1.0"}
DEFAULT_REQUEST_TIMEOUT = 30.0
_MAX_PAGES = 50
_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_]")
_FIELD_UNSAFE = re.compile(r"[^A-Za-z0-9_]")
# MCP server 默认不继承宿主的密钥类环境变量；需要时在 mcp.json 的 env 中显式提供。
_SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def sanitize_tool_name(server: str, tool: str) -> str:
    """把 server/tool 转成合法 Python 标识符：mcp__<server>__<tool>。"""

    safe_server = _NAME_UNSAFE.sub("_", str(server)) or "server"
    safe_tool = _NAME_UNSAFE.sub("_", str(tool)) or "tool"
    name = f"mcp__{safe_server}__{safe_tool}"
    if not re.fullmatch(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise MCPError(f"无法生成合法 MCP 工具名：{name!r}")
    return name


class MCPClient:
    """单个 MCP server 的 stdio 客户端。"""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.request_timeout = float(request_timeout)
        self._logger = logger or logging.getLogger(f"minillm.mcp.{config.name}")
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._next_id = 0
        self._stderr_task: asyncio.Task[None] | None = None
        self._tools: list[dict[str, Any]] | None = None
        self._broken = False
        self._closed = False

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    # ==== 生命周期 ====
    async def start(self) -> None:
        """启动子进程并完成 initialize 握手。"""

        if self._closed:
            raise MCPError(f"MCP 客户端已关闭：{self.config.name}")
        if self._proc is not None:
            return
        env = {
            key: value
            for key, value in os.environ.items()
            if not any(marker in key.upper() for marker in _SENSITIVE_ENV_MARKERS)
        }
        env.update({str(key): str(value) for key, value in self.config.env.items()})
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.cwd,
                env=env,
            )
        except (OSError, ValueError) as exc:
            self._closed = True
            raise MCPError(f"MCP server 启动失败：{self.config.name}: {exc}") from exc

        self._stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            result = await self._request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": dict(CLIENT_INFO),
                },
            )
            if not isinstance(result, dict):
                raise MCPError(f"MCP initialize 返回格式错误：{self.config.name}")
            version = result.get("protocolVersion")
            if version and version != PROTOCOL_VERSION:
                self._logger.warning(
                    "MCP server %s 协议版本不同：%s（客户端 %s）",
                    self.config.name,
                    version,
                    PROTOCOL_VERSION,
                )
            capabilities = result.get("capabilities")
            if not isinstance(capabilities, dict) or not isinstance(
                capabilities.get("tools"), dict
            ):
                raise MCPError(f"MCP server 未声明 tools capability：{self.config.name}")
            await self._notify("notifications/initialized", {})
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        """终止子进程并清理任务；可重复调用。"""

        if self._closed and self._proc is None:
            return
        self._closed = True
        proc = self._proc
        self._proc = None
        self._tools = None
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)
            self._stderr_task = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except (OSError, RuntimeError):
            pass
        if proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._logger.debug("[%s] %s", self.config.name, text[:2000])
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - stderr 日志失败不影响协议
            self._logger.debug("MCP stderr drain 结束", exc_info=True)

    # ==== 协议 I/O ====
    async def _write(self, message: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise MCPError(f"MCP 进程未启动：{self.config.name}")
        if proc.returncode is not None:
            self._broken = True
            raise MCPError(f"MCP 进程已退出：{self.config.name}（code={proc.returncode}）")
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            proc.stdin.write(payload.encode("utf-8"))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            self._broken = True
            raise MCPError(f"MCP stdin 写入失败：{self.config.name}") from exc

    async def _read_message(self) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise MCPError(f"MCP 进程未启动：{self.config.name}")
        line = await proc.stdout.readline()
        if not line:
            self._broken = True
            code = proc.returncode
            raise MCPError(
                f"MCP 进程已退出：{self.config.name}（code={code}）"
            )
        try:
            message = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._broken = True
            raise MCPError(f"MCP stdout 返回非 JSON 数据：{self.config.name}") from exc
        if not isinstance(message, dict):
            self._broken = True
            raise MCPError(f"MCP stdout 消息不是对象：{self.config.name}")
        return message

    async def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if self._broken:
            raise MCPError(f"MCP 连接已不可用：{self.config.name}")
        async with self._lock:
            if self._broken:
                raise MCPError(f"MCP 连接已不可用：{self.config.name}")
            self._next_id += 1
            request_id = self._next_id
            await self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            limit = self.request_timeout if timeout is None else float(timeout)
            try:
                while True:
                    message = await asyncio.wait_for(self._read_message(), timeout=limit)
                    if "id" not in message:
                        continue  # server 通知
                    if message.get("id") != request_id:
                        raise MCPError(
                            f"MCP 响应 id 不匹配：{self.config.name} "
                            f"（期望 {request_id}，收到 {message.get('id')}）"
                        )
                    error = message.get("error")
                    if error:
                        detail = error.get("data") if isinstance(error, dict) else None
                        code = error.get("code") if isinstance(error, dict) else None
                        text = error.get("message") if isinstance(error, dict) else str(error)
                        raise MCPError(
                            f"MCP 错误：{self.config.name}: {text}",
                            detail={"code": code, "data": detail},
                        )
                    return message.get("result")
            except TimeoutError as exc:
                self._broken = True
                raise MCPError(
                    f"MCP 请求超时：{self.config.name} {method} ({limit}s)"
                ) from exc

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._broken:
            raise MCPError(f"MCP 连接已不可用：{self.config.name}")
        async with self._lock:
            await self._write(
                {"jsonrpc": "2.0", "method": method, "params": params or {}}
            )

    # ==== MCP 方法 ====
    async def list_tools(self) -> list[dict[str, Any]]:
        """拉取工具列表；尽力支持 nextCursor 分页。"""

        if self._tools is not None:
            return list(self._tools)
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(_MAX_PAGES):
            params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
            result = await self._request("tools/list", params)
            if not isinstance(result, dict):
                raise MCPError(f"tools/list 返回格式错误：{self.config.name}")
            page = result.get("tools")
            if not isinstance(page, list):
                raise MCPError(f"tools/list.tools 不是数组：{self.config.name}")
            tools.extend(item for item in page if isinstance(item, dict))
            next_cursor = result.get("nextCursor")
            if not next_cursor:
                break
            cursor = str(next_cursor)
            if cursor in seen:
                raise MCPError(f"tools/list cursor 循环：{self.config.name}")
            seen.add(cursor)
        else:
            raise MCPError(f"tools/list 分页过多：{self.config.name}")
        self._tools = tools
        return list(tools)

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """调用 MCP 工具，返回原始 result（content/isError）。"""

        args = {} if arguments is None else arguments
        if not isinstance(args, dict):
            raise MCPError("MCP 工具参数必须是 JSON 对象")
        result = await self._request(
            "tools/call", {"name": name, "arguments": args}
        )
        if not isinstance(result, dict):
            raise MCPError(f"tools/call 返回格式错误：{self.config.name}")
        return result


# ==== inputSchema -> pydantic 模型 ====
def _annotation(schema: dict[str, Any]) -> Any:
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        non_null = [item for item in raw_type if item != "null"]
        raw_type = non_null[0] if non_null else "null"
    if raw_type == "string":
        return str
    if raw_type == "integer":
        return int
    if raw_type == "number":
        return float
    if raw_type == "boolean":
        return bool
    if raw_type == "object":
        return dict[str, Any]
    if raw_type == "array":
        items = schema.get("items")
        item_type = _annotation(items) if isinstance(items, dict) else Any
        return list[item_type]
    if raw_type == "null":
        return type(None)
    return Any


def _field_name(raw: str, used: set[str]) -> str:
    base = _FIELD_UNSAFE.sub("_", str(raw)).strip("_") or "field"
    if base[0].isdigit():
        base = f"f_{base}"
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def build_params_model(
    server: str, tool_name: str, input_schema: dict[str, Any] | None
) -> type[BaseModel]:
    """按 inputSchema 动态生成 pydantic 参数模型（支持 alias/额外字段策略）。"""

    schema = input_schema if isinstance(input_schema, dict) else {}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required_raw = schema.get("required")
    required = set(required_raw) if isinstance(required_raw, list) else set()
    used: set[str] = set()
    fields: dict[str, tuple[Any, Any]] = {}
    for raw_name, raw_prop in properties.items():
        prop = raw_prop if isinstance(raw_prop, dict) else {}
        field = _field_name(str(raw_name), used)
        annotation = _annotation(prop)
        description = str(prop.get("description") or "")
        if str(raw_name) in required:
            default: Any = ...
        else:
            annotation = annotation if annotation is Any else annotation | None
            default = None
        fields[field] = (
            annotation,
            Field(default, alias=str(raw_name), description=description),
        )
    extra = "forbid" if schema.get("additionalProperties") is False else "allow"
    model_name = "MCP_" + _FIELD_UNSAFE.sub("_", f"{server}_{tool_name}") + "_Params"
    return create_model(
        model_name,
        __config__=ConfigDict(extra=extra, populate_by_name=True),
        **fields,
    )


def _type_label(schema: dict[str, Any]) -> str:
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        raw_type = next((item for item in raw_type if item != "null"), "any")
    if raw_type == "array":
        items = schema.get("items")
        inner = _type_label(items) if isinstance(items, dict) else "any"
        return f"list[{inner}]"
    if isinstance(raw_type, str) and raw_type in {
        "string",
        "integer",
        "number",
        "boolean",
        "object",
    }:
        return raw_type
    return "any"


def build_signature(tool_name: str, input_schema: dict[str, Any] | None) -> str:
    schema = input_schema if isinstance(input_schema, dict) else {}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required_raw = schema.get("required")
    required = set(required_raw) if isinstance(required_raw, list) else set()
    parts: list[str] = []
    for raw_name, raw_prop in properties.items():
        prop = raw_prop if isinstance(raw_prop, dict) else {}
        suffix = "" if str(raw_name) in required else " = None"
        parts.append(f"{raw_name}: {_type_label(prop)}{suffix}")
    return f"{tool_name}({', '.join(parts)}) -> str"


def build_mcp_tool_spec(
    server: str, client: MCPClient, tool: dict[str, Any]
) -> ToolSpec:
    """把一个 MCP tool 适配成 ToolSpec。"""

    raw_name = str(tool.get("name") or "").strip()
    if not raw_name:
        raise MCPError(f"MCP server {server} 返回了没有 name 的工具")
    full_name = sanitize_tool_name(server, raw_name)
    schema = tool.get("inputSchema")
    model = build_params_model(server, raw_name, schema if isinstance(schema, dict) else {})
    description = str(tool.get("description") or f"MCP 工具 {raw_name}（server={server}）")
    signature = build_signature(full_name, schema if isinstance(schema, dict) else {})

    async def handler(_ctx: ToolContext, params: BaseModel) -> str:
        arguments = params.model_dump(by_alias=True, exclude_unset=True)
        result = await client.call_tool(raw_name, arguments)
        content = result.get("content")
        texts: list[str] = []
        if isinstance(content, list):
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    texts.append(item["text"])
        text = "\n".join(texts)
        if result.get("isError"):
            raise ToolError(
                text or f"MCP 工具 {raw_name} 返回错误", kind="tool_error", tool=full_name
            )
        if not text:
            text = json.dumps(result, ensure_ascii=False, default=str)
        return text

    return ToolSpec(
        name=full_name,
        description=description,
        params=model,
        signature=signature,
        handler=handler,
        kind="mcp",
    )


# ==== 管理器 ====
class MCPManager:
    """管理所有已配置的 MCP server，并把工具注册进 ToolRegistry。"""

    def __init__(
        self,
        settings: Settings,
        registry: ToolRegistry | None = None,
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self._registry = registry
        self.request_timeout = float(request_timeout)
        self._logger = logger or logging.getLogger("minillm.mcp")
        self._clients: dict[str, MCPClient] = {}
        self._tool_names: set[str] = set()
        self._started = False

    def servers(self) -> list[str]:
        """已连接 server 名称；/api/config 用。"""

        return list(self._clients)

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tool_names)

    async def start(self, registry: ToolRegistry | None = None) -> MCPManager:
        """连接配置中的全部 server；单个失败只告警，不影响其他。"""

        if registry is not None:
            self._registry = registry
        if self._started:
            return self
        self._started = True
        try:
            configs = load_mcp_config(self.settings.mcp_config)
        except MCPError as exc:
            self._logger.warning("MCP 配置不可用，跳过：%s", exc)
            return self

        for config in configs:
            client = MCPClient(
                config,
                request_timeout=self.request_timeout,
                logger=logging.getLogger(f"minillm.mcp.{config.name}"),
            )
            try:
                await client.start()
                tools = await client.list_tools()
            except Exception as exc:  # noqa: BLE001 - 单 server 失败不阻塞启动
                self._logger.warning("MCP server %s 启动/握手失败：%s", config.name, exc)
                await client.close()
                continue
            self._clients[config.name] = client
            if self._registry is not None:
                self._register_tools(config.name, client, tools)
        return self

    def _register_tools(
        self, server: str, client: MCPClient, tools: list[dict[str, Any]]
    ) -> None:
        assert self._registry is not None
        for tool in tools:
            try:
                spec = build_mcp_tool_spec(server, client, tool)
            except Exception as exc:  # noqa: BLE001 - 单个坏工具不影响其他
                self._logger.warning("MCP 工具适配失败（%s）：%s", server, exc)
                continue
            if self._registry.has(spec.name):
                self._logger.warning(
                    "MCP 工具名冲突，跳过：%s（server=%s）", spec.name, server
                )
                continue
            try:
                self._registry.register(spec)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("MCP 工具注册失败：%s：%s", spec.name, exc)
                continue
            self._tool_names.add(spec.name)

    async def close(self) -> None:
        for name, client in list(self._clients.items()):
            try:
                await client.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响其他
                self._logger.warning("MCP server %s 关闭失败", name, exc_info=True)
        self._clients.clear()

    async def __aenter__(self) -> MCPManager:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()


async def register_mcp_tools(
    registry: ToolRegistry,
    settings: Settings,
    *,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> MCPManager:
    """便利入口：连接 settings.mcp_config 中的 server 并注册工具。"""

    manager = MCPManager(settings, registry, request_timeout=request_timeout)
    await manager.start()
    return manager


__all__ = [
    "DEFAULT_REQUEST_TIMEOUT",
    "MCPClient",
    "MCPManager",
    "PROTOCOL_VERSION",
    "build_mcp_tool_spec",
    "build_params_model",
    "build_signature",
    "register_mcp_tools",
    "sanitize_tool_name",
]
