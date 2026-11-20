"""search_files：优先 ripgrep，找不到或失败时回退 Python re。"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..errors import ToolError
from .base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    resolve_workspace_path,
)

_IGNORED_DIRS = {".git", "node_modules", ".venv", "data", "dist", "__pycache__"}


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchFilesParams(_Params):
    pattern: str = Field(min_length=1, max_length=2000)
    path: str = "."
    glob: str | None = Field(default=None, max_length=512)
    max_results: int = Field(default=100, ge=1, le=500)
    context: int = Field(default=0, ge=0, le=10)


def _ignored(rel: Path) -> bool:
    return any(part in _IGNORED_DIRS for part in rel.parts)


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(8192)
    except OSError:
        return True


def _rg_env() -> dict[str, str]:
    allowed = ("PATH", "HOME", "LANG", "LC_ALL")
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.setdefault("HOME", str(Path.home()))
    return env


def _relative_result(ctx: ToolContext, raw_path: str) -> str | None:
    workspace = Path(ctx.workspace).resolve(strict=True)
    text = raw_path
    if text.startswith("./"):
        text = text[2:]
    try:
        target = (workspace / text).resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    if target != workspace and workspace not in target.parents:
        return None
    try:
        return target.relative_to(workspace).as_posix()
    except ValueError:
        return None


def _parse_rg_json(
    ctx: ToolContext, stdout: bytes, max_results: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") not in {"match", "context"}:
            continue
        data = event.get("data") or {}
        path_doc = data.get("path") or {}
        raw_path = str(path_doc.get("text") or "")
        line_number = data.get("line_number")
        lines_doc = data.get("lines") or {}
        text = str(lines_doc.get("text") or "").rstrip("\n")
        if not raw_path or not isinstance(line_number, int):
            continue
        rel = _relative_result(ctx, raw_path)
        if rel is None:
            continue
        if _ignored(Path(rel)):
            continue
        key = (rel, line_number)
        if key in seen:
            continue
        seen.add(key)
        results.append({"path": rel, "line": line_number, "text": text})
        if len(results) >= max_results:
            break
    return results


def _rg_search(
    ctx: ToolContext, params: SearchFilesParams, search_arg: str
) -> list[dict[str, Any]] | None:
    rg = shutil.which("rg")
    if not rg:
        return None
    command = [rg, "--json", "--no-messages", "--color", "never"]
    if params.glob:
        command.extend(["--glob", params.glob])
    for ignored in sorted(_IGNORED_DIRS):
        command.extend(["--glob", f"!**/{ignored}/**"])
    if params.context:
        command.extend(["--context", str(params.context)])
    command.extend(["-e", params.pattern, "--", search_arg])
    try:
        proc = subprocess.run(
            command,
            cwd=str(Path(ctx.workspace).resolve(strict=True)),
            env=_rg_env(),
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode not in (0, 1):
        return None
    return _parse_rg_json(ctx, proc.stdout, params.max_results + 1)


def _python_search(
    ctx: ToolContext, params: SearchFilesParams, root: Path
) -> list[dict[str, Any]]:
    try:
        regex = re.compile(params.pattern)
    except re.error as exc:
        raise ToolError(
            f"非法正则表达式：{exc}", kind="validation", tool="search_files"
        ) from exc

    workspace = Path(ctx.workspace).resolve(strict=True)
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    budget = params.max_results * (2 * params.context + 1) + 10
    for current, dirnames, filenames in os.walk(root):
        if len(results) >= budget:
            break
        dirnames[:] = sorted(name for name in dirnames if name not in _IGNORED_DIRS)
        for filename in sorted(filenames):
            if len(results) >= budget:
                break
            path = Path(current) / filename
            try:
                resolved = path.resolve(strict=False)
            except (OSError, RuntimeError):
                continue
            if resolved != workspace and workspace not in resolved.parents:
                continue
            rel = resolved.relative_to(workspace)
            if _ignored(rel):
                continue
            rel_text = rel.as_posix()
            if params.glob and not fnmatch.fnmatch(rel_text, params.glob):
                continue
            if _is_binary(path):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            for index, line in enumerate(lines):
                if not regex.search(line):
                    continue
                for around in range(
                    max(0, index - params.context),
                    min(len(lines), index + params.context + 1),
                ):
                    key = (rel_text, around + 1)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(
                        {"path": rel_text, "line": around + 1, "text": lines[around]}
                    )
                    if len(results) >= budget:
                        break
                if len(results) >= budget:
                    break
    return results


async def search_files(ctx: ToolContext, params: SearchFilesParams) -> ToolResult:
    root = resolve_workspace_path(
        ctx, params.path, tool="search_files", must_exist=True, expect_dir=True
    )
    workspace = Path(ctx.workspace).resolve(strict=True)
    search_arg = "." if root == workspace else root.relative_to(workspace).as_posix()

    results = _rg_search(ctx, params, search_arg)
    if results is None:
        results = _python_search(ctx, params, root)

    results.sort(key=lambda item: (item["path"], item["line"], item["text"]))
    truncated = len(results) > params.max_results
    results = results[: params.max_results]
    meta: dict[str, Any] = {"count": len(results), "truncated": truncated}
    artifact: str | None = None
    payload = json.dumps(results, ensure_ascii=False)
    if len(payload) > ctx.settings.max_tool_output_chars:
        artifact = ctx.artifacts.save_json("search_files", results)
        limit = ctx.settings.max_tool_output_chars
        while results and len(json.dumps(results, ensure_ascii=False)) > limit:
            results.pop()
        meta["count"] = len(results)
        meta["artifact"] = artifact
        meta["truncated"] = True
    return ToolResult(ok=True, value=results, meta=meta, artifact=artifact)


SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search_files",
        description="按正则搜索 workspace 内文本，优先 rg，支持 glob 与上下文行",
        params=SearchFilesParams,
        signature=(
            "search_files(pattern: str, path: str = '.', glob: str | None = None, "
            "max_results: int = 100, context: int = 0) -> list[dict]"
        ),
        handler=search_files,
    ),
)
