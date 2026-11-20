"""HTTP 客户端助手。

本机环境坑：NO_PROXY 含 [::1] 这类带方括号的 IPv6 条目时，httpx 0.28 的
URLPattern 解析会抛 InvalidURL。这里先清理代理环境，再构造 AsyncClient。
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import Any

import httpx

from .config import Settings

_BRACKET_CHARS = ("[", "]")


def _clean_no_proxy(value: str) -> str:
    parts: list[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if any(char in item for char in _BRACKET_CHARS):
            continue
        parts.append(item)
    return ",".join(parts)


def sanitize_proxy_env(
    environ: MutableMapping[str, str] | None = None,
) -> str:
    """清理 NO_PROXY/no_proxy 中的方括号条目，返回清理后的值。

    httpx 的 trust_env 在构造 Client 时读取 os.environ；这里直接原地修正，
    避免之后每次请求都重复处理。测试可传入 dict 做隔离验证。
    """

    env = environ if environ is not None else os.environ
    cleaned: dict[str, str] = {}
    for key in ("NO_PROXY", "no_proxy"):
        if key not in env:
            continue
        current = _clean_no_proxy(str(env[key]))
        env[key] = current
        cleaned[key] = current
    return cleaned.get("NO_PROXY") or cleaned.get("no_proxy") or ""


def make_async_http_client(settings: Settings, **kwargs: Any) -> httpx.AsyncClient:
    """构造带超时、UA 和代理环境清理的 httpx 异步客户端。"""

    sanitize_proxy_env()
    timeout = kwargs.pop("timeout", settings.web_fetch_timeout)
    follow_redirects = bool(kwargs.pop("follow_redirects", True))
    headers = {"User-Agent": "minillm/0.1.0"}
    extra_headers = kwargs.pop("headers", None) or {}
    headers.update({str(key): str(value) for key, value in dict(extra_headers).items()})
    return httpx.AsyncClient(
        trust_env=True,
        follow_redirects=follow_redirects,
        timeout=httpx.Timeout(float(timeout)),
        headers=headers,
        **kwargs,
    )
