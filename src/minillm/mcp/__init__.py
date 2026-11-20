"""MCP（Model Context Protocol）stdio 客户端。

对外入口：
- MCPServerConfig / load_mcp_config：读取 mcp.json
- MCPClient：单 server 的 initialize/list/call/close
- MCPManager / register_mcp_tools：批量连接并注册进 ToolRegistry
- sanitize_tool_name：mcp__<server>__<tool> 命名规则
"""

from __future__ import annotations

from .client import (
    DEFAULT_REQUEST_TIMEOUT,
    PROTOCOL_VERSION,
    MCPClient,
    MCPManager,
    build_mcp_tool_spec,
    build_params_model,
    build_signature,
    register_mcp_tools,
    sanitize_tool_name,
)
from .config import MCPServerConfig, load_mcp_config

__all__ = [
    "DEFAULT_REQUEST_TIMEOUT",
    "MCPClient",
    "MCPManager",
    "MCPServerConfig",
    "PROTOCOL_VERSION",
    "build_mcp_tool_spec",
    "build_params_model",
    "build_signature",
    "load_mcp_config",
    "register_mcp_tools",
    "sanitize_tool_name",
]
