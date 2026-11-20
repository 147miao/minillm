"""文件读写、编辑与列目录工具。

安全边界：所有路径都必须解析到 workspace 内；符号链接逃逸会被拒绝。
原子边界：写/改先写同目录临时文件，fsync 后 os.replace，避免读者看到半截文件。
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..errors import ToolError
from .base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    relative_to_workspace,
    resolve_workspace_path,
)

_IGNORED_DIRS = {".git", "node_modules", ".venv", "data", "dist", "__pycache__"}


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadFileParams(_Params):
    path: str = Field(min_length=1, max_length=4096)
    start_line: int = Field(default=1, ge=1)
    limit: int = Field(default=2000, ge=1, le=10000)


class WriteFileParams(_Params):
    path: str = Field(min_length=1, max_length=4096)
    content: str = Field(max_length=20_000_000)


class EditFileParams(_Params):
    path: str = Field(min_length=1, max_length=4096)
    old_string: str = Field(min_length=1)
    new_string: str = ""
    replace_all: bool = False


class ListFilesParams(_Params):
    path: str = "."
    pattern: str = Field(default="**/*", min_length=1, max_length=512)
    max_results: int = Field(default=200, ge=1, le=2000)


def _is_binary(path: Path, tool: str) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(8192)
    except OSError as exc:
        raise ToolError(f"文件不可读：{path.name}", kind="execution", tool=tool) from exc


def _read_exact(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return handle.read()
    except OSError as exc:
        raise ToolError(f"文件读取失败：{path.name}", kind="execution") from exc


def _count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:  # pragma: no cover - 部分文件系统不支持目录 fsync
        pass


def _atomic_write_text(target: Path, text: str) -> None:
    """同目录临时文件 + fsync + os.replace；保留已有文件 mode。"""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.is_dir():
        raise ToolError(f"目标路径是目录：{target}", kind="execution")
    mode: int | None = None
    if target.exists():
        try:
            mode = target.stat().st_mode & 0o777
        except OSError:  # pragma: no cover
            mode = None
    tmp = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            try:
                os.chmod(tmp, mode)
            except OSError:  # pragma: no cover
                pass
        os.replace(tmp, target)
        _fsync_dir(target.parent)
    except OSError as exc:
        raise ToolError(f"文件写入失败：{exc}", kind="execution") from exc
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover
                pass


async def read_file(ctx: ToolContext, params: ReadFileParams) -> ToolResult:
    target = resolve_workspace_path(
        ctx, params.path, tool="read_file", must_exist=True
    )
    if not target.is_file():
        raise ToolError(f"不是文件：{params.path}", kind="execution", tool="read_file")
    if _is_binary(target, "read_file"):
        raise ToolError(
            f"拒绝读取二进制文件：{params.path}", kind="execution", tool="read_file"
        )
    lines: list[str] = []
    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                if number < params.start_line:
                    continue
                if len(lines) >= params.limit:
                    break
                lines.append(line)
    except OSError as exc:
        raise ToolError(
            f"文件读取失败：{params.path}", kind="execution", tool="read_file"
        ) from exc

    text = "".join(lines)
    rel = relative_to_workspace(ctx, target, tool="read_file")
    meta: dict[str, Any] = {
        "path": rel,
        "start_line": params.start_line,
        "returned_lines": len(lines),
        "maybe_more": len(lines) == params.limit,
    }
    clipped, artifact = ctx.artifacts.cap_text("read_file", text)
    if artifact is not None:
        meta["artifact"] = artifact
        meta["truncated"] = True
    return ToolResult(ok=True, value=clipped, meta=meta, artifact=artifact)


async def write_file(ctx: ToolContext, params: WriteFileParams) -> dict[str, Any]:
    target = resolve_workspace_path(ctx, params.path, tool="write_file")
    if target.exists() and target.is_dir():
        raise ToolError(f"目标路径是目录：{params.path}", kind="execution", tool="write_file")
    created = not target.exists()
    async with ctx.locks.acquire(target):
        _atomic_write_text(target, params.content)
    rel = relative_to_workspace(ctx, target, tool="write_file")
    return {
        "path": rel,
        "bytes": len(params.content.encode("utf-8")),
        "lines": _count_lines(params.content),
        "created": created,
    }


async def edit_file(ctx: ToolContext, params: EditFileParams) -> dict[str, Any]:
    target = resolve_workspace_path(ctx, params.path, tool="edit_file", must_exist=True)
    if not target.is_file():
        raise ToolError(f"不是文件：{params.path}", kind="execution", tool="edit_file")
    if _is_binary(target, "edit_file"):
        raise ToolError(
            f"拒绝编辑二进制文件：{params.path}", kind="execution", tool="edit_file"
        )
    async with ctx.locks.acquire(target):
        text = _read_exact(target)
        count = text.count(params.old_string)
        if count == 0:
            raise ToolError(
                "old_string 未在文件中找到", kind="execution", tool="edit_file"
            )
        if count > 1 and not params.replace_all:
            raise ToolError(
                f"old_string 出现 {count} 次，请提供更精确的片段或 replace_all=True",
                kind="execution",
                tool="edit_file",
            )
        new_text = (
            text.replace(params.old_string, params.new_string)
            if params.replace_all
            else text.replace(params.old_string, params.new_string, 1)
        )
        _atomic_write_text(target, new_text)
    rel = relative_to_workspace(ctx, target, tool="edit_file")
    return {"path": rel, "replaced": count if params.replace_all else 1}


async def list_files(ctx: ToolContext, params: ListFilesParams) -> ToolResult:
    root = resolve_workspace_path(
        ctx, params.path, tool="list_files", must_exist=True, expect_dir=True
    )
    if Path(params.pattern).is_absolute() or ".." in Path(params.pattern).parts:
        raise ToolError(
            f"非法的 glob 模式：{params.pattern}", kind="validation", tool="list_files"
        )
    try:
        matches = list(root.glob(params.pattern))
    except (ValueError, NotImplementedError, OSError) as exc:
        raise ToolError(
            f"glob 模式非法：{params.pattern}", kind="validation", tool="list_files"
        ) from exc

    workspace = Path(ctx.workspace).resolve(strict=True)
    seen: set[str] = set()
    entries: list[str] = []
    for entry in matches:
        try:
            resolved = entry.resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        if resolved != workspace and workspace not in resolved.parents:
            continue
        try:
            rel = resolved.relative_to(workspace)
        except ValueError:
            continue
        if any(part in _IGNORED_DIRS for part in rel.parts):
            continue
        text = rel.as_posix()
        if entry.is_dir():
            text += "/"
        if text in seen:
            continue
        seen.add(text)
        entries.append(text)

    entries.sort()
    truncated = len(entries) >= params.max_results
    entries = entries[: params.max_results]
    meta: dict[str, Any] = {"count": len(entries), "truncated": truncated}
    artifact: str | None = None
    payload = json.dumps(entries, ensure_ascii=False)
    if len(payload) > ctx.settings.max_tool_output_chars:
        artifact = ctx.artifacts.save_json("list_files", entries)
        limit = ctx.settings.max_tool_output_chars
        while entries and len(json.dumps(entries, ensure_ascii=False)) > limit:
            entries.pop()
        meta["count"] = len(entries)
        meta["artifact"] = artifact
        meta["truncated"] = True
    return ToolResult(ok=True, value=entries, meta=meta, artifact=artifact)


SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="read_file",
        description="读取 workspace 内文本文件，可按行截取",
        params=ReadFileParams,
        signature="read_file(path: str, start_line: int = 1, limit: int = 2000) -> str",
        handler=read_file,
    ),
    ToolSpec(
        name="write_file",
        description="原子写入文本文件（临时文件 + os.replace）",
        params=WriteFileParams,
        signature="write_file(path: str, content: str) -> dict",
        handler=write_file,
    ),
    ToolSpec(
        name="edit_file",
        description="按 old_string 精确替换并原子写回，默认要求唯一匹配",
        params=EditFileParams,
        signature=(
            "edit_file(path: str, old_string: str, new_string: str, "
            "replace_all: bool = False) -> dict"
        ),
        handler=edit_file,
    ),
    ToolSpec(
        name="list_files",
        description="列出 workspace 内文件，忽略依赖与数据目录",
        params=ListFilesParams,
        signature=(
            "list_files(path: str = '.', pattern: str = '**/*', "
            "max_results: int = 200) -> list[str]"
        ),
        handler=list_files,
    ),
)
