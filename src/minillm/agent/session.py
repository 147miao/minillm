"""会话创建与加载的轻量封装。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Settings
from ..errors import SessionError
from ..ids import new_id
from ..persistence.replay import SessionState
from ..persistence.store import SessionStore


def resolve_workspace(
    settings: Settings,
    session_id: str,
    workspace: str | Path | None,
) -> Path:
    root = Path(settings.workspace_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if workspace is None or str(workspace).strip() == "":
        target = root / session_id
    else:
        raw = Path(workspace).expanduser()
        target = (raw if raw.is_absolute() else Path.cwd() / raw).resolve()
        if not settings.allow_external_workspace and not target.is_relative_to(root):
            raise SessionError(
                f"workspace 必须位于 {root} 下",
                detail={"code": "invalid_workspace", "status_code": 400},
            )
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SessionError(f"workspace 不可创建：{target}") from exc
    if not target.is_dir():
        raise SessionError(f"workspace 不是目录：{target}")
    return target


def create_session(
    store: SessionStore,
    settings: Settings,
    *,
    title: str | None = None,
    workspace: str | Path | None = None,
    parent_id: str | None = None,
) -> dict[str, Any]:
    """创建会话目录、工作区与 session.created 事件。"""

    session_id = new_id("sess")
    target = resolve_workspace(settings, session_id, workspace)
    meta = store.create_session(
        {
            "session_id": session_id,
            "title": str(title or "新会话"),
            "workspace": str(target),
            "model": settings.llm_model,
            "parent_id": parent_id,
            "status": "idle",
        }
    )
    return meta


async def load_state(store: SessionStore, session_id: str) -> SessionState:
    return await store.recover(session_id)
