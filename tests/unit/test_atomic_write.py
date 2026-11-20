"""原子写与文件锁并发测试。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from minillm.locks import FileLockManager


async def test_write_uses_os_replace(registry, tool_ctx, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    original = os.replace

    def wrapper(src, dst):
        calls.append((os.fspath(src), os.fspath(dst)))
        return original(src, dst)

    monkeypatch.setattr(os, "replace", wrapper)
    await registry.invoke("write_file", {"path": "atomic.txt", "content": "ok\n"}, tool_ctx)
    assert calls
    assert calls[-1][1] == str(Path(tool_ctx.workspace) / "atomic.txt")


async def test_concurrent_writes_are_atomic(registry, tool_ctx) -> None:
    workspace = Path(tool_ctx.workspace)
    target = workspace / "shared.txt"
    initial = "initial\n"
    target.write_text(initial)
    contents = [f"writer-{index}-" + str(index % 10) * 100_000 + "\n" for index in range(20)]
    allowed = {initial, *contents}
    broken: list[int] = []
    stop = asyncio.Event()

    async def reader() -> None:
        while not stop.is_set():
            try:
                text = target.read_text()
            except FileNotFoundError:
                await asyncio.sleep(0)
                continue
            if text not in allowed:
                broken.append(len(text))
            await asyncio.sleep(0)

    reader_task = asyncio.create_task(reader())
    await asyncio.gather(
        *(
            registry.invoke("write_file", {"path": "shared.txt", "content": content}, tool_ctx)
            for content in contents
        )
    )
    stop.set()
    await reader_task

    assert broken == []
    assert target.read_text() in allowed
    leftovers = [path.name for path in workspace.iterdir() if path.name.endswith(".tmp")]
    assert leftovers == []


async def test_file_lock_serializes_tasks() -> None:
    manager = FileLockManager()
    active = 0
    peak = 0

    async def worker() -> None:
        nonlocal active, peak
        async with manager.acquire("/tmp/minillm-lock-target"):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.005)
            active -= 1

    await asyncio.gather(*(worker() for _ in range(10)))
    assert peak == 1
    assert manager.active_locks == 0
