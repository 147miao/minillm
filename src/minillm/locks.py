"""按真实路径串行化的进程内异步文件锁。"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Entry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class FileLockManager:
    """同一路径的读改写必须持有同一把锁。

    说明：锁只在当前进程内有效；原子替换负责让跨进程读者看不到半截文件。
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    @staticmethod
    def key(path: str | Path) -> str:
        return os.path.realpath(os.fspath(path))

    @asynccontextmanager
    async def acquire(self, path: str | Path) -> AsyncIterator[None]:
        key = self.key(path)
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry()
            self._entries[key] = entry
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0 and self._entries.get(key) is entry:
                self._entries.pop(key, None)

    @property
    def active_locks(self) -> int:
        return len(self._entries)
