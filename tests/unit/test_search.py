"""search_files 测试：rg 路径、Python 回退、glob、context、截断。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minillm.errors import ToolError
from minillm.tools import search as search_module


async def _write(registry, ctx, path: str, content: str) -> None:
    await registry.invoke("write_file", {"path": path, "content": content}, ctx)


async def test_search_with_rg(registry, tool_ctx, monkeypatch) -> None:
    if not search_module.shutil.which("rg"):
        pytest.skip("本机没有 rg")
    await _write(registry, tool_ctx, "a.py", "needle\nother\n")
    monkeypatch.setattr(
        search_module,
        "_python_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应回退")),
    )
    result = await registry.invoke("search_files", {"pattern": "needle"}, tool_ctx)
    assert result.value == [{"path": "a.py", "line": 1, "text": "needle"}]


async def test_search_with_rg_context_and_glob(registry, tool_ctx, monkeypatch) -> None:
    if not search_module.shutil.which("rg"):
        pytest.skip("本机没有 rg")
    await _write(registry, tool_ctx, "a.py", "zero\ntarget\none\n")
    await _write(registry, tool_ctx, "b.txt", "target\n")
    monkeypatch.setattr(
        search_module,
        "_python_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应回退")),
    )
    result = await registry.invoke(
        "search_files", {"pattern": "target", "glob": "*.py", "context": 1}, tool_ctx
    )
    assert [(item["path"], item["line"]) for item in result.value] == [
        ("a.py", 1),
        ("a.py", 2),
        ("a.py", 3),
    ]


async def test_search_with_rg_max_results(registry, tool_ctx, monkeypatch) -> None:
    if not search_module.shutil.which("rg"):
        pytest.skip("本机没有 rg")
    await _write(registry, tool_ctx, "many.txt", "\n".join(["needle"] * 20) + "\n")
    monkeypatch.setattr(
        search_module,
        "_python_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应回退")),
    )
    result = await registry.invoke(
        "search_files", {"pattern": "needle", "max_results": 3}, tool_ctx
    )
    assert len(result.value) == 3
    assert result.meta["truncated"] is True


async def test_search_with_rg_subdir(registry, tool_ctx, monkeypatch) -> None:
    if not search_module.shutil.which("rg"):
        pytest.skip("本机没有 rg")
    await _write(registry, tool_ctx, "src/a.py", "needle\n")
    await _write(registry, tool_ctx, "other.py", "needle\n")
    monkeypatch.setattr(
        search_module,
        "_python_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应回退")),
    )
    result = await registry.invoke(
        "search_files", {"pattern": "needle", "path": "src"}, tool_ctx
    )
    assert result.value == [{"path": "src/a.py", "line": 1, "text": "needle"}]


async def test_search_python_fallback(registry, tool_ctx, monkeypatch) -> None:
    monkeypatch.setattr(search_module.shutil, "which", lambda name: None)
    await _write(registry, tool_ctx, "src/a.py", "zero\nneedle\none\n")
    result = await registry.invoke("search_files", {"pattern": "needle"}, tool_ctx)
    assert result.value == [{"path": "src/a.py", "line": 2, "text": "needle"}]


async def test_glob_and_context(registry, tool_ctx, monkeypatch) -> None:
    monkeypatch.setattr(search_module.shutil, "which", lambda name: None)
    await _write(registry, tool_ctx, "a.py", "zero\ntarget\none\n")
    await _write(registry, tool_ctx, "b.txt", "target\n")
    only_py = await registry.invoke(
        "search_files", {"pattern": "target", "glob": "*.py"}, tool_ctx
    )
    assert [item["path"] for item in only_py.value] == ["a.py"]

    with_context = await registry.invoke(
        "search_files", {"pattern": "target", "context": 1}, tool_ctx
    )
    assert [(item["path"], item["line"]) for item in with_context.value] == [
        ("a.py", 1),
        ("a.py", 2),
        ("a.py", 3),
        ("b.txt", 1),
    ]


async def test_invalid_regex(registry, tool_ctx, monkeypatch) -> None:
    monkeypatch.setattr(search_module.shutil, "which", lambda name: None)
    with pytest.raises(ToolError) as invalid:
        await registry.invoke("search_files", {"pattern": "["}, tool_ctx)
    assert invalid.value.kind == "validation"


async def test_max_results(registry, tool_ctx, monkeypatch) -> None:
    monkeypatch.setattr(search_module.shutil, "which", lambda name: None)
    await _write(registry, tool_ctx, "many.txt", "\n".join(["needle"] * 20) + "\n")
    result = await registry.invoke(
        "search_files", {"pattern": "needle", "max_results": 3}, tool_ctx
    )
    assert len(result.value) == 3
    assert result.meta["truncated"] is True


async def test_artifact_when_results_too_large(registry, small_ctx, monkeypatch) -> None:
    monkeypatch.setattr(search_module.shutil, "which", lambda name: None)
    await _write(registry, small_ctx, "many.txt", "\n".join(["needle"] * 50) + "\n")
    result = await registry.invoke(
        "search_files", {"pattern": "needle", "max_results": 50}, small_ctx
    )
    assert len(result.value) < 50
    assert result.artifact is not None
    full = json.loads(small_ctx.artifacts.read_text(result.artifact))
    assert len(full) == 50


async def test_ignored_dirs(registry, tool_ctx, monkeypatch) -> None:
    monkeypatch.setattr(search_module.shutil, "which", lambda name: None)
    workspace = Path(tool_ctx.workspace)
    (workspace / ".git").mkdir()
    (workspace / ".git" / "secret.py").write_text("needle\n")
    (workspace / "src").mkdir()
    (workspace / "src" / "a.py").write_text("needle\n")
    result = await registry.invoke("search_files", {"pattern": "needle"}, tool_ctx)
    assert [item["path"] for item in result.value] == ["src/a.py"]


async def test_path_escape(registry, tool_ctx) -> None:
    with pytest.raises(ToolError) as escaped:
        await registry.invoke("search_files", {"pattern": "x", "path": "../"}, tool_ctx)
    assert escaped.value.kind == "validation"
