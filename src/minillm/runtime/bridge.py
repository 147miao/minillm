"""宿主侧的 socketpair 桥接连接。

宿主创建 socketpair，把子端 fd 通过 pass_fds 交给 worker；宿主保留父端并用
asyncio 流包装。没有 TCP、没有 token/hello：fd 本身即鉴权，EOF 即对方退出。
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from typing import Any

from .protocol import MAX_FRAME_BYTES, read_frame, write_frame


def create_socketpair() -> tuple[socket.socket, socket.socket]:
    """创建一对本地双向 socket；父端给宿主，子端给 worker。"""

    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    return parent, child


class Bridge:
    """宿主侧桥接连接的薄封装。"""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self._closed = False

    @classmethod
    async def connect(cls, parent_sock: socket.socket) -> Bridge:
        parent_sock.setblocking(False)
        reader, writer = await asyncio.open_connection(
            sock=parent_sock,
            limit=MAX_FRAME_BYTES + 1024,
        )
        return cls(reader, writer)

    async def send_exec(self, code: str, tools: dict[str, list[str]]) -> None:
        await write_frame(
            self.writer,
            {"type": "exec", "code": code, "tools": tools},
        )

    async def read(self) -> dict[str, Any] | None:
        return await read_frame(self.reader)

    async def reply(self, payload: dict[str, Any]) -> None:
        await write_frame(self.writer, payload)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.writer.close()
        with contextlib.suppress(Exception):
            await self.writer.wait_closed()


__all__ = ["Bridge", "create_socketpair"]
