"""文件工具测试：读写、编辑、列目录、路径安全、截断。"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from minillm.errors import ToolError


async def _invoke(registry, ctx, tool: str, args: dict):
    return await registry.invoke(tool, args, ctx)


async def test_write_read_edit_flow(registry, tool_ctx) -> None:
    written = await registry.invoke(
        "write_file", {"path": "pkg/main.py", "content": "a\nb\nc\n"}, tool_ctx
    )
    assert written.value["path"] == "pkg/main.py"
    assert written.value["bytes"] == 6
    assert written.value["lines"] == 3
    assert written.value["created"] is True

    read = await registry.invoke("read_file", {"path": "pkg/main.py"}, tool_ctx)
    assert read.value == "a\nb\nc\n"

    edited = await registry.invoke(
        "edit_file",
        {"path": "pkg/main.py", "old_string": "b", "new_string": "B"},
        tool_ctx,
    )
    assert edited.value == {"path": "pkg/main.py", "replaced": 1}
    assert (Path(tool_ctx.workspace) / "pkg/main.py").read_text() == "a\nB\nc\n"


async def test_read_file_line_window(registry, tool_ctx) -> None:
    await registry.invoke(
        "write_file", {"path": "lines.txt", "content": "1\n2\n3\n4\n"}, tool_ctx
    )
    read = await registry.invoke(
        "read_file", {"path": "lines.txt", "start_line": 2, "limit": 2}, tool_ctx
    )
    assert read.value == "2\n3\n"
    assert read.meta["returned_lines"] == 2
    assert read.meta["maybe_more"] is True

    empty = await registry.invoke(
        "read_file", {"path": "lines.txt", "start_line": 99}, tool_ctx
    )
    assert empty.value == ""


async def test_write_preserves_mode(registry, tool_ctx) -> None:
    target = Path(tool_ctx.workspace) / "script.sh"
    target.write_text("echo old\n")
    target.chmod(0o640)
    await registry.invoke("write_file", {"path": "script.sh", "content": "echo new\n"}, tool_ctx)
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


async def test_edit_not_found_and_ambiguous(registry, tool_ctx) -> None:
    await registry.invoke("write_file", {"path": "x.txt", "content": "same same\n"}, tool_ctx)
    with pytest.raises(ToolError) as missing:
        await registry.invoke(
            "edit_file", {"path": "x.txt", "old_string": "nope", "new_string": "y"}, tool_ctx
        )
    assert missing.value.kind == "execution"

    with pytest.raises(ToolError) as ambiguous:
        await registry.invoke(
            "edit_file", {"path": "x.txt", "old_string": "same", "new_string": "y"}, tool_ctx
        )
    assert ambiguous.value.kind == "execution"
    assert "replace_all" in ambiguous.value.message

    replaced = await registry.invoke(
        "edit_file",
        {"path": "x.txt", "old_string": "same", "new_string": "y", "replace_all": True},
        tool_ctx,
    )
    assert replaced.value["replaced"] == 2


async def test_edit_preserves_crlf(registry, tool_ctx) -> None:
    target = Path(tool_ctx.workspace) / "crlf.txt"
    target.write_bytes(b"a\r\nb\r\n")
    await registry.invoke(
        "edit_file", {"path": "crlf.txt", "old_string": "a", "new_string": "x"}, tool_ctx
    )
    assert target.read_bytes() == b"x\r\nb\r\n"


async def test_path_escape_rejected(registry, tool_ctx, tmp_path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    for path in ("../outside.txt", str(outside)):
        with pytest.raises(ToolError) as escaped:
            await registry.invoke("read_file", {"path": path}, tool_ctx)
        assert escaped.value.kind == "validation"


async def test_symlink_escape_rejected(registry, tool_ctx, tmp_path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = Path(tool_ctx.workspace) / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(ToolError) as escaped:
        await registry.invoke("read_file", {"path": "link.txt"}, tool_ctx)
    assert escaped.value.kind == "validation"
    with pytest.raises(ToolError):
        await registry.invoke("write_file", {"path": "link.txt", "content": "x"}, tool_ctx)


async def test_read_binary_and_missing(registry, tool_ctx) -> None:
    binary = Path(tool_ctx.workspace) / "bin.dat"
    binary.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ToolError) as binary_error:
        await registry.invoke("read_file", {"path": "bin.dat"}, tool_ctx)
    assert binary_error.value.kind == "execution"

    with pytest.raises(ToolError) as missing:
        await registry.invoke("read_file", {"path": "missing.txt"}, tool_ctx)
    assert missing.value.kind == "execution"

    with pytest.raises(ToolError):
        await registry.invoke("read_file", {"path": "."}, tool_ctx)


async def test_read_truncation_artifact(registry, small_ctx) -> None:
    await registry.invoke("write_file", {"path": "big.txt", "content": "x" * 300}, small_ctx)
    result = await registry.invoke("read_file", {"path": "big.txt"}, small_ctx)
    assert result.artifact is not None
    assert result.artifact.startswith("artifacts/")
    assert result.value.endswith(f"full output: {result.artifact}]")
    assert small_ctx.artifacts.read_text(result.artifact) == "x" * 300


async def test_list_files_ignores_and_sorts(registry, tool_ctx) -> None:
    workspace = Path(tool_ctx.workspace)
    for ignored in (".git", "node_modules", ".venv", "data", "dist", "__pycache__"):
        folder = workspace / ignored
        folder.mkdir()
        (folder / "junk.txt").write_text("junk")
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("print(1)\n")
    (workspace / "README.md").write_text("hi\n")

    listed = await registry.invoke("list_files", {"pattern": "**/*"}, tool_ctx)
    assert listed.value == ["README.md", "src/", "src/app.py"]
    assert listed.meta["truncated"] is False

    only_py = await registry.invoke("list_files", {"pattern": "**/*.py"}, tool_ctx)
    assert only_py.value == ["src/app.py"]

    capped = await registry.invoke(
        "list_files", {"pattern": "**/*", "max_results": 1}, tool_ctx
    )
    assert len(capped.value) == 1
    assert capped.meta["truncated"] is True


async def test_list_files_rejects_bad_pattern(registry, tool_ctx) -> None:
    with pytest.raises(ToolError) as absolute:
        await registry.invoke("list_files", {"pattern": "/etc/*"}, tool_ctx)
    assert absolute.value.kind == "validation"
    with pytest.raises(ToolError):
        await registry.invoke("list_files", {"pattern": "../*"}, tool_ctx)
