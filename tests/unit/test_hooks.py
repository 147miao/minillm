"""Hooks 扩展点单元测试。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from minillm.config import Settings
from minillm.hooks import (
    AuditHook,
    HookContext,
    HookEvent,
    HookManager,
    SkillsHook,
    ToolDecision,
    load_hook_modules,
    register_builtin_hooks,
)
from minillm.skills import clear_skill_cache

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FakeResult:
    ok: bool
    value: str = ""


def make_ctx(settings: Settings, emitted: list[tuple[str, dict]] | None = None) -> HookContext:
    async def emit(event_type: str, data: dict) -> None:
        if emitted is not None:
            emitted.append((event_type, data))

    return HookContext(
        session_id="sess_test",
        settings=settings,
        agent_id="main",
        emit_event=emit,
    )


@pytest.fixture(autouse=True)
def _clear_skill_cache() -> None:
    clear_skill_cache()
    yield
    clear_skill_cache()


async def test_register_order(settings: Settings) -> None:
    manager = HookManager()
    calls: list[str] = []

    async def first(ctx: HookContext, payload: dict) -> None:
        calls.append("first")

    async def second(ctx: HookContext, payload: dict) -> None:
        calls.append("second")

    manager.register(HookEvent.BEFORE_LLM, first)
    manager.register(HookEvent.BEFORE_LLM, second)
    await manager.emit(HookEvent.BEFORE_LLM, make_ctx(settings), {"messages": []})
    assert calls == ["first", "second"]
    assert manager.registered(HookEvent.BEFORE_LLM) == (first, second)


async def test_register_rejects_invalid_and_duplicate(settings: Settings) -> None:
    manager = HookManager()

    async def hook(ctx: HookContext, payload: dict) -> None:
        return None

    manager.register(HookEvent.BEFORE_LLM, hook)
    with pytest.raises(ValueError):
        manager.register(HookEvent.BEFORE_LLM, hook)
    with pytest.raises(ValueError):
        manager.register("unknown", hook)
    with pytest.raises(ValueError):
        manager.registered("unknown")
    with pytest.raises(ValueError):
        await manager.emit("unknown", make_ctx(settings), {})

    manager.clear()
    assert manager.registered(HookEvent.BEFORE_LLM) == ()


async def test_before_tool_deny_short_circuits(settings: Settings) -> None:
    manager = HookManager()
    calls: list[str] = []

    async def deny(ctx: HookContext, payload: dict) -> ToolDecision:
        return ToolDecision(deny="不允许执行")

    async def later(ctx: HookContext, payload: dict) -> None:
        calls.append("later")

    manager.register(HookEvent.BEFORE_TOOL, deny)
    manager.register(HookEvent.BEFORE_TOOL, later)
    payload = await manager.emit(
        HookEvent.BEFORE_TOOL,
        make_ctx(settings),
        {"tool": "run_bash", "args": {}},
    )
    assert payload["denied"] == "不允许执行"
    assert calls == []


async def test_before_tool_args_merge(settings: Settings) -> None:
    manager = HookManager()

    async def add_command(ctx: HookContext, payload: dict) -> ToolDecision:
        return ToolDecision(args={"command": "echo hello"})

    async def add_timeout(ctx: HookContext, payload: dict) -> ToolDecision:
        return ToolDecision(args={"timeout": 5})

    manager.register(HookEvent.BEFORE_TOOL, add_command)
    manager.register(HookEvent.BEFORE_TOOL, add_timeout)
    payload = await manager.emit(
        HookEvent.BEFORE_TOOL,
        make_ctx(settings),
        {"tool": "run_bash", "args": {}},
    )
    assert payload["args"] == {"command": "echo hello", "timeout": 5}
    assert "denied" not in payload


async def test_after_tool_replaces_result(settings: Settings) -> None:
    manager = HookManager()

    async def replace(ctx: HookContext, payload: dict) -> FakeResult:
        return FakeResult(ok=True, value="new")

    manager.register(HookEvent.AFTER_TOOL, replace)
    original = FakeResult(ok=False, value="old")
    payload = await manager.emit(
        HookEvent.AFTER_TOOL,
        make_ctx(settings),
        {"tool": "read_file", "args": {}, "result": original},
    )
    assert payload["result"].value == "new"
    assert payload["result"].ok is True


async def test_hook_exception_isolated(settings: Settings) -> None:
    manager = HookManager()
    calls: list[str] = []

    async def bad(ctx: HookContext, payload: dict) -> None:
        raise RuntimeError("boom")

    async def good(ctx: HookContext, payload: dict) -> dict:
        calls.append("good")
        return {"note": "ok"}

    manager.register(HookEvent.BEFORE_LLM, bad)
    manager.register(HookEvent.BEFORE_LLM, good)
    payload = await manager.emit(HookEvent.BEFORE_LLM, make_ctx(settings), {})
    assert calls == ["good"]
    assert payload["note"] == "ok"
    assert payload["hook_errors"][0]["hook"].endswith(".bad")
    assert payload["hook_errors"][0]["error"].startswith("RuntimeError")


async def test_context_emit(settings: Settings) -> None:
    manager = HookManager()
    emitted: list[tuple[str, dict]] = []

    async def hook(ctx: HookContext, payload: dict) -> None:
        await ctx.emit("skills.injected", {"skills": []})

    manager.register(HookEvent.USER_MESSAGE, hook)
    await manager.emit(HookEvent.USER_MESSAGE, make_ctx(settings, emitted), {})
    assert emitted == [("skills.injected", {"skills": []})]


def test_load_hook_modules(tmp_path: Path) -> None:
    directory = tmp_path / "hooks"
    directory.mkdir()
    (directory / "good.py").write_text(
        "from minillm.hooks import HookEvent\n"
        "\n"
        "async def hook(ctx, payload):\n"
        "    payload.setdefault('seen', []).append('good')\n"
        "\n"
        "def register(manager):\n"
        "    manager.register(HookEvent.BEFORE_LLM, hook)\n",
        encoding="utf-8",
    )
    (directory / "bad.py").write_text("raise RuntimeError('bad module')\n", encoding="utf-8")
    skip_source = "raise RuntimeError('should not load')\n"
    (directory / "_private.py").write_text(skip_source, encoding="utf-8")
    (directory / "__init__.py").write_text(skip_source, encoding="utf-8")

    manager = HookManager()
    loaded = load_hook_modules(manager, directory)
    assert loaded == ["good.py"]
    assert len(manager.registered(HookEvent.BEFORE_LLM)) == 1


def test_load_hook_modules_missing_dir(tmp_path: Path) -> None:
    manager = HookManager()
    assert load_hook_modules(manager, tmp_path / "missing") == []


def test_example_audit_hook_loads() -> None:
    manager = HookManager()
    loaded = load_hook_modules(manager, REPO_ROOT / "examples" / "hooks")
    assert "audit_hook.py" in loaded
    assert len(manager.registered(HookEvent.BEFORE_TOOL)) == 1
    assert len(manager.registered(HookEvent.AFTER_TOOL)) == 1


def _write_skill(
    directory: Path,
    slug: str,
    name: str,
    keywords: list[str],
    body: str,
) -> Path:
    path = directory / slug / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    keyword_text = ", ".join(keywords)
    path.write_text(
        f"---\nname: {name}\ndescription: demo\nkeywords: [{keyword_text}]\n---\n{body}\n",
        encoding="utf-8",
    )
    return path


async def test_skills_hook_injects_and_dedupes(settings: Settings) -> None:
    settings.skills_dir.mkdir(parents=True, exist_ok=True)
    _write_skill(
        settings.skills_dir,
        "pytest-debugging",
        "pytest-debugging",
        ["pytest", "测试失败"],
        "先读完整 traceback。",
    )
    emitted: list[tuple[str, dict]] = []
    manager = HookManager()
    skills_hook, _ = register_builtin_hooks(manager, settings, enable_audit=False)
    assert isinstance(skills_hook, SkillsHook)
    ctx = make_ctx(settings, emitted)
    payload = {"text": "pytest 报了断言失败，帮我看看"}

    payload = await manager.emit(HookEvent.USER_MESSAGE, ctx, payload)
    assert payload["active_skills"][0]["name"] == "pytest-debugging"
    assert "先读完整 traceback" in payload["system_parts"][0]
    assert emitted[0][0] == "skills.injected"
    assert emitted[0][1]["skills"][0]["name"] == "pytest-debugging"

    payload = await manager.emit(HookEvent.USER_MESSAGE, ctx, payload)
    assert len(payload["active_skills"]) == 1
    assert len(payload["system_parts"]) == 1
    assert len(emitted) == 1


async def test_audit_hook_redacts_and_writes(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    manager = HookManager()
    audit = AuditHook(settings)
    manager.register(HookEvent.BEFORE_TOOL, audit.before_tool)
    manager.register(HookEvent.AFTER_TOOL, audit.after_tool)
    ctx = make_ctx(settings)

    await manager.emit(
        HookEvent.BEFORE_TOOL,
        ctx,
        {"tool": "read_file", "args": {"path": "a.py", "api_key": "SECRET"}},
    )
    await manager.emit(
        HookEvent.AFTER_TOOL,
        ctx,
        {"tool": "read_file", "args": {}, "result": FakeResult(ok=True, value="ok")},
    )

    path = settings.data_dir / "audit.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["phase"] for line in lines] == ["before", "after"]
    assert lines[0]["tool"] == "read_file"
    assert lines[1]["ok"] is True
    assert isinstance(lines[1]["duration_ms"], int)
    assert "SECRET" not in path.read_text(encoding="utf-8")


async def test_audit_hook_write_failure_is_isolated(settings: Settings, tmp_path: Path) -> None:
    broken = tmp_path / "audit_dir"
    broken.mkdir()
    audit = AuditHook(settings, path=broken)
    await audit.before_tool(make_ctx(settings), {"tool": "read_file", "args": {}})


def test_register_builtin_hooks(settings: Settings) -> None:
    manager = HookManager()
    skills_hook, audit_hook = register_builtin_hooks(manager, settings)
    assert isinstance(skills_hook, SkillsHook)
    assert isinstance(audit_hook, AuditHook)
    assert len(manager.registered(HookEvent.SESSION_START)) == 1
    assert len(manager.registered(HookEvent.USER_MESSAGE)) == 1
    assert len(manager.registered(HookEvent.BEFORE_TOOL)) == 1
    assert len(manager.registered(HookEvent.AFTER_TOOL)) == 1

    manager2 = HookManager()
    _, audit_none = register_builtin_hooks(manager2, settings, enable_audit=False)
    assert audit_none is None
    assert manager2.registered(HookEvent.BEFORE_TOOL) == ()
