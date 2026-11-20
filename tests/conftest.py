"""基础测试 fixture；后续 WP 可在此基础上追加，但不要覆盖已有名字。"""

from __future__ import annotations

from pathlib import Path

import pytest

from minillm.artifacts import ArtifactStore
from minillm.config import Settings
from minillm.locks import FileLockManager
from minillm.persistence.store import SessionStore
from minillm.tools import build_core_registry
from minillm.tools.base import ToolContext


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """完全隔离的临时配置（不读取仓库根目录的 .env）。"""

    return Settings(
        _env_file=None,
        llm_api_key="test-key",
        small_llm_api_key="test-small-key",
        data_dir=tmp_path / "data",
        workspace_root=tmp_path / "workspace",
        hooks_dir=tmp_path / "hooks",
        skills_dir=tmp_path / "skills",
        mcp_config=tmp_path / "mcp.json",
        wal_fsync=False,
        llm_timeout=5.0,
    )


@pytest.fixture
def store(settings: Settings) -> SessionStore:
    return SessionStore(settings.data_dir, fsync=False)


class EventRecorder:
    """记录工具层 emit 的事件，供 todo 等测试断言。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def __call__(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))


@pytest.fixture
def emitter() -> EventRecorder:
    return EventRecorder()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture
def session_id(store: SessionStore, workspace: Path) -> str:
    meta = store.create_session(
        {"title": "test", "workspace": str(workspace), "model": "test-model"}
    )
    return str(meta["session_id"])


@pytest.fixture
def artifacts(store: SessionStore, session_id: str, settings: Settings) -> ArtifactStore:
    return ArtifactStore(store, session_id, settings)


@pytest.fixture
def locks() -> FileLockManager:
    return FileLockManager()


@pytest.fixture
def tool_ctx(
    settings: Settings,
    store: SessionStore,
    session_id: str,
    workspace: Path,
    artifacts: ArtifactStore,
    locks: FileLockManager,
) -> ToolContext:
    return ToolContext(
        session_id=session_id,
        workspace=workspace,
        data_dir=settings.data_dir,
        settings=settings,
        locks=locks,
        artifacts=artifacts,
        agent_id="test",
        kind="main",
        allow_subagents=False,
    )


@pytest.fixture
def small_ctx(
    settings: Settings,
    store: SessionStore,
    session_id: str,
    workspace: Path,
    emitter: EventRecorder,
) -> ToolContext:
    small = settings.model_copy(update={"max_tool_output_chars": 80})
    return ToolContext(
        session_id=session_id,
        workspace=workspace,
        data_dir=settings.data_dir,
        settings=small,
        locks=FileLockManager(),
        artifacts=ArtifactStore(store, session_id, small),
        agent_id="test",
        kind="main",
        allow_subagents=False,
        emit=emitter,
    )


@pytest.fixture
def registry():
    return build_core_registry()
