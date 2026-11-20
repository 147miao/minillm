"""web_fetch：抓取 HTTP(S) 页面并转成文本。

安全：
- 默认拒绝私网/环回/链路本地地址（WEB_FETCH_ALLOW_PRIVATE=true 可关闭，用于测试/本机服务）；
- 手动处理重定向并逐跳校验；
- 响应 2 MiB 上限，输出再按字符上限落盘截断。
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from ..errors import ToolError
from ..net import make_async_http_client
from .base import ToolContext, ToolResult, ToolSpec

_MAX_BYTES = 2 * 1024 * 1024
_MAX_REDIRECTS = 5
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
_BLOCK_TAGS = {
    "p",
    "div",
    "br",
    "li",
    "tr",
    "section",
    "article",
    "header",
    "footer",
    "pre",
    "table",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebFetchParams(_Params):
    url: str = Field(min_length=1, max_length=8192)
    max_chars: int = Field(default=20000, ge=1, le=200000)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts).replace("\r", " ")
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        raw = re.sub(r" *\n *", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _validate_url(url: str) -> tuple[str, str, int]:
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise ToolError(f"URL 非法：{url}", kind="validation", tool="web_fetch") from exc
    if parts.scheme not in {"http", "https"}:
        raise ToolError(
            "只支持 http/https URL", kind="validation", tool="web_fetch"
        )
    if not parts.hostname:
        raise ToolError("URL 缺少主机名", kind="validation", tool="web_fetch")
    if parts.username or parts.password:
        raise ToolError(
            "URL 不允许携带用户名/密码", kind="validation", tool="web_fetch"
        )
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise ToolError(f"URL 端口非法：{url}", kind="validation", tool="web_fetch") from exc
    return parts.hostname, parts.scheme, port


async def _ensure_public(host: str, port: int, *, allow_private: bool) -> None:
    if allow_private:
        return
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    if literal is not None:
        addresses.append(literal)
    else:
        try:
            infos = await asyncio.to_thread(
                socket.getaddrinfo, host, port, type=socket.SOCK_STREAM
            )
        except OSError as exc:
            raise ToolError(
                f"DNS 解析失败：{host}", kind="execution", tool="web_fetch"
            ) from exc
        for info in infos:
            raw = info[4][0]
            try:
                addresses.append(ipaddress.ip_address(raw))
            except ValueError:
                continue
    if not addresses:
        raise ToolError(f"无法解析主机：{host}", kind="execution", tool="web_fetch")
    for address in addresses:
        if not address.is_global:
            raise ToolError(
                "拒绝访问私网/环回地址（可设置 WEB_FETCH_ALLOW_PRIVATE=true）",
                kind="validation",
                tool="web_fetch",
            )


def _extract_text(content_type: str, body: bytes, encoding: str) -> str:
    lowered = content_type.lower()
    if "html" in lowered:
        parser = _HTMLTextExtractor()
        parser.feed(body.decode(encoding, errors="replace"))
        parser.close()
        return parser.text()
    if (
        lowered.startswith("text/")
        or "json" in lowered
        or "xml" in lowered
        or "javascript" in lowered
        or not lowered
    ):
        return body.decode(encoding, errors="replace").strip()
    raise ToolError(
        f"不支持的 Content-Type：{content_type or 'unknown'}",
        kind="execution",
        tool="web_fetch",
    )


async def _fetch(
    ctx: ToolContext, params: WebFetchParams
) -> tuple[str, int, str, str, int]:
    client = make_async_http_client(ctx.settings, follow_redirects=False)
    try:
        current = params.url
        for _ in range(_MAX_REDIRECTS + 1):
            host, _scheme, port = _validate_url(current)
            await _ensure_public(
                host, port, allow_private=ctx.settings.web_fetch_allow_private
            )
            async with client.stream("GET", current) as response:
                status = response.status_code
                if 300 <= status < 400:
                    location = response.headers.get("location")
                    if not location:
                        raise ToolError(
                            "重定向缺少 Location", kind="execution", tool="web_fetch"
                        )
                    current = urljoin(current, location)
                    continue
                if status >= 400:
                    raise ToolError(
                        f"HTTP 状态码 {status}", kind="execution", tool="web_fetch"
                    )
                content_type = response.headers.get("content-type", "")
                encoding = response.charset_encoding or "utf-8"
                body = bytearray()
                total = 0
                capped = False
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if len(body) < _MAX_BYTES:
                        body.extend(chunk[: _MAX_BYTES - len(body)])
                    if total >= _MAX_BYTES:
                        capped = True
                        break
                text = _extract_text(content_type, bytes(body), encoding)
                if capped:
                    text += "\n[response truncated at 2 MiB]"
                return current, status, content_type, text, total
        raise ToolError("重定向次数过多", kind="execution", tool="web_fetch")
    finally:
        await client.aclose()


async def web_fetch(ctx: ToolContext, params: WebFetchParams) -> ToolResult:
    url, status, content_type, text, total = await _fetch(ctx, params)
    limit = min(params.max_chars, ctx.settings.max_tool_output_chars)
    text, artifact = ctx.artifacts.cap_text("web_fetch", text, max_chars=limit)
    meta: dict[str, Any] = {
        "url": url,
        "status": status,
        "content_type": content_type,
        "bytes": total,
        "chars": len(text),
    }
    if artifact is not None:
        meta["artifact"] = artifact
        meta["truncated"] = True
    return ToolResult(ok=True, value=text, meta=meta, artifact=artifact)


SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="web_fetch",
        description="抓取 http/https 页面并转为文本，默认拒绝私网地址",
        params=WebFetchParams,
        signature="web_fetch(url: str, max_chars: int = 20000) -> str",
        handler=web_fetch,
    ),
)
