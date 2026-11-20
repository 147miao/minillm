"""FastAPI 路由：会话、事件、SSE 与 artifact。"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .. import __version__
from ..config import Settings
from ..skills.loader import load_skills
from .schemas import APIError, CreateSessionRequest, SendMessageRequest
from .sse import SSE_HEADERS, close_subscription, event_stream, event_to_dict

router = APIRouter()

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ARTIFACT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


async def _maybe(value: Any) -> Any:
    """兼容同步/异步服务方法，降低 WP5 联调成本。"""

    if inspect.isawaitable(value):
        return await value
    return value


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_service(request: Request) -> Any:
    service = request.app.state.service
    if service is None:
        raise APIError(503, "AgentService 尚未初始化")
    return service


def get_optional_service(request: Request) -> Any:
    return request.app.state.service


def _check_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or _SESSION_ID_RE.fullmatch(session_id) is None:
        raise APIError(400, f"非法会话 id：{session_id!r}")


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        data = value.model_dump()
        if isinstance(data, dict):
            return dict(data)
    if hasattr(value, "to_dict"):
        data = value.to_dict()
        if isinstance(data, dict):
            return dict(data)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise APIError(500, "服务返回的数据格式错误")


def _int_or(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _session_summary(raw: Any) -> dict[str, Any]:
    data = _to_mapping(raw)
    session_id = str(data.get("session_id") or data.get("id") or "").strip()
    if session_id == "":
        raise APIError(500, "会话数据缺少 id")
    created_at = str(data.get("created_at") or "")
    updated_at = str(data.get("updated_at") or created_at)
    parent_id = data.get("parent_id")
    return {
        "id": session_id,
        "session_id": session_id,
        "title": str(data.get("title") or ""),
        "workspace": str(data.get("workspace") or ""),
        "model": str(data.get("model") or ""),
        "parent_id": None if parent_id in (None, "") else str(parent_id),
        "created_at": created_at,
        "updated_at": updated_at,
        "status": str(data.get("status") or "idle"),
    }


def _public_message(raw: Any) -> dict[str, Any] | None:
    try:
        data = _to_mapping(raw)
    except APIError:
        return None
    return {key: value for key, value in data.items() if not str(key).startswith("_")}


def _agents_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        return []
    agents: list[dict[str, Any]] = []
    for item in raw:
        try:
            data = _to_mapping(item)
        except APIError:
            continue
        agent_id = data.get("agent_id") or data.get("id") or data.get("session_id")
        if not agent_id:
            continue
        parent = data.get("parent_session_id") or data.get("parent_id")
        agents.append(
            {
                "agent_id": str(agent_id),
                "parent_session_id": None if parent in (None, "") else str(parent),
                "task": str(data.get("task") or data.get("description") or ""),
                "status": str(data.get("status") or "unknown"),
                "result_preview": str(data.get("result_preview") or data.get("result") or ""),
                "updated_at": str(data.get("updated_at") or ""),
            }
        )
    agents.sort(key=lambda item: (item["updated_at"], item["agent_id"]))
    return agents


def _skills_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        return []
    skills: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            if item.strip():
                skills.append({"name": item.strip()})
            continue
        try:
            data = _to_mapping(item)
        except APIError:
            continue
        name = str(data.get("name") or data.get("id") or "").strip()
        if name == "":
            continue
        entry: dict[str, Any] = {"name": name}
        description = data.get("description")
        if description:
            entry["description"] = str(description)
        path = data.get("path")
        if path:
            entry["path"] = str(path)
        skills.append(entry)
    return skills


def _context_state(raw: Any, view: dict[str, Any], settings: Settings) -> dict[str, Any]:
    if raw:
        try:
            data = _to_mapping(raw)
        except APIError:
            data = {}
    else:
        data = {}
    tokens = _int_or(data.get("tokens"))
    if tokens == 0:
        usage = view.get("last_usage")
        if usage:
            try:
                usage_data = _to_mapping(usage)
            except APIError:
                usage_data = {}
            tokens = _int_or(usage_data.get("prompt_tokens"))
    budget = _int_or(data.get("budget")) or int(settings.context_budget)
    if data.get("ratio") is not None:
        ratio = _float_or(data.get("ratio"))
    else:
        ratio = (tokens / budget) if budget > 0 else 0.0
    compactions = data.get("compactions") or view.get("compactions") or []
    if not isinstance(compactions, list):
        compactions = []
    return {
        "tokens": tokens,
        "budget": budget,
        "ratio": ratio,
        "compactions": compactions,
    }


def _view_payload(raw: Any, settings: Settings) -> dict[str, Any]:
    data = _to_mapping(raw)
    meta = _session_summary(data.get("meta") or data)
    messages: list[dict[str, Any]] = []
    raw_messages = data.get("messages") or []
    if isinstance(raw_messages, list):
        for item in raw_messages:
            message = _public_message(item)
            if message is not None:
                messages.append(message)
    todos = data.get("todos") or []
    if not isinstance(todos, list):
        todos = []
    result = {
        "meta": meta,
        "messages": messages,
        "todos": todos,
        "agents": _agents_list(data.get("agents") or []),
        "context": _context_state(data.get("context"), data, settings),
        "skills": _skills_list(data.get("skills") or data.get("active_skills") or []),
        "last_seq": _int_or(data.get("last_seq")),
        "status": str(data.get("status") or meta["status"]),
    }
    return result


def _workspace_path(settings: Settings, raw: str | None) -> str | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
    except (OSError, ValueError) as exc:
        raise APIError(400, f"非法 workspace：{raw!r}") from exc
    if path.exists() and not path.is_dir():
        raise APIError(400, "workspace 必须是目录")
    root = settings.workspace_root.expanduser().resolve()
    if not settings.allow_external_workspace and not path.is_relative_to(root):
        raise APIError(400, f"workspace 必须位于 {root} 下")
    return str(path)


async def _service_mcp_servers(service: Any) -> list[str]:
    """MCP 服务名以 AgentService 持有的 MCPManager 为准，不直接读配置文件。"""

    if service is None:
        return []
    getter = getattr(service, "mcp_servers", None)
    if getter is None:
        return []
    try:
        raw = await _maybe(getter() if callable(getter) else getter)
    except Exception:  # noqa: BLE001 - 配置接口不能因 MCP 异常而 500
        return []
    if isinstance(raw, dict):
        if "servers" in raw or "names" in raw:
            raw = raw.get("servers") or raw.get("names") or []
        else:
            raw = list(raw.keys())
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, str):
            if item:
                names.append(item)
        elif isinstance(item, dict):
            name = item.get("name")
            if name:
                names.append(str(name))
    return names


def _config_skills(settings: Settings) -> list[dict[str, Any]]:
    try:
        skills = load_skills(settings.skills_dir)
    except Exception:  # noqa: BLE001 - 配置接口不能因技能目录坏掉而 500
        return []
    return [
        {
            "name": skill.name,
            "description": skill.description,
            "path": str(skill.path),
        }
        for skill in skills
    ]


def _public_event(raw: Any, session_id: str) -> dict[str, Any] | None:
    try:
        event = event_to_dict(raw)
    except TypeError:
        return None
    if event is None:
        return None
    event.setdefault("session_id", session_id)
    if not isinstance(event.get("data"), dict):
        event["data"] = {}
    return event


async def _load_events(service: Any, session_id: str, after: int) -> list[dict[str, Any]]:
    raw = await _maybe(service.events(session_id, after))
    if isinstance(raw, dict):
        raw = raw.get("events") or []
    if not isinstance(raw, list):
        return []
    events: list[dict[str, Any]] = []
    for item in raw:
        event = _public_event(item, session_id)
        if event is None:
            continue
        seq = _int_or(event.get("seq"), -1)
        if seq <= after:
            continue
        event["seq"] = seq
        events.append(event)
    events.sort(key=lambda item: int(item["seq"]))
    return events


def _header_after(request: Request, after: int) -> int:
    header = request.headers.get("last-event-id")
    if not header:
        return after
    try:
        return max(after, int(header))
    except ValueError:
        return after


SettingsDep = Annotated[Settings, Depends(get_settings)]
ServiceDep = Annotated[Any, Depends(get_service)]
OptionalServiceDep = Annotated[Any, Depends(get_optional_service)]
AfterQuery = Annotated[int, Query(ge=0)]


@router.get("/health")
async def health(settings: SettingsDep) -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "model": settings.llm_model}


@router.get("/config")
async def config(
    settings: SettingsDep,
    service: OptionalServiceDep,
) -> dict[str, Any]:
    return {
        "model": settings.llm_model,
        "small_model": settings.small_llm_model,
        "context_budget": settings.context_budget,
        "prune_threshold": settings.prune_threshold,
        "compact_threshold": settings.compact_threshold,
        "workspace_root": str(settings.workspace_root),
        "skills": _config_skills(settings),
        "mcp_servers": await _service_mcp_servers(service),
        "version": __version__,
    }


@router.get("/sessions")
async def list_sessions(service: ServiceDep) -> dict[str, Any]:
    raw = await _maybe(service.list_sessions())
    if isinstance(raw, dict):
        raw = raw.get("sessions") or []
    if not isinstance(raw, list):
        raw = []
    return {"sessions": [_session_summary(item) for item in raw]}


@router.post("/sessions", status_code=201)
async def create_session(
    payload: CreateSessionRequest,
    settings: SettingsDep,
    service: ServiceDep,
) -> dict[str, Any]:
    workspace = _workspace_path(settings, payload.workspace)
    meta = await _maybe(service.create_session(title=payload.title, workspace=workspace))
    return _session_summary(meta)


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    settings: SettingsDep,
    service: ServiceDep,
) -> dict[str, Any]:
    _check_session_id(session_id)
    view = await _maybe(service.get_view(session_id))
    return _view_payload(view, settings)


@router.get("/sessions/{session_id}/events")
async def get_events(
    session_id: str,
    service: ServiceDep,
    after: AfterQuery = 0,
) -> dict[str, Any]:
    _check_session_id(session_id)
    return {"events": await _load_events(service, session_id, after)}


@router.post("/sessions/{session_id}/messages", status_code=202)
async def post_message(
    session_id: str,
    payload: SendMessageRequest,
    service: ServiceDep,
) -> JSONResponse:
    _check_session_id(session_id)
    content = payload.content.strip()
    if content == "":
        raise APIError(400, "content 不能为空")
    turn_id = await _maybe(service.submit_message(session_id, content))
    if isinstance(turn_id, dict):
        turn_id = turn_id.get("turn_id") or turn_id.get("id")
    return JSONResponse(status_code=202, content={"turn_id": str(turn_id or "")})


@router.post("/sessions/{session_id}/stop")
async def stop_session(
    session_id: str,
    request: Request,
    service: ServiceDep,
) -> dict[str, Any]:
    _check_session_id(session_id)
    try:
        await request.json()
    except Exception:  # noqa: BLE001 - stop 允许空 body
        pass
    await _maybe(service.stop(session_id))
    return {"ok": True}


@router.get("/sessions/{session_id}/stream")
async def stream_session(
    session_id: str,
    request: Request,
    service: ServiceDep,
    after: AfterQuery = 0,
) -> StreamingResponse:
    _check_session_id(session_id)
    effective_after = _header_after(request, after)
    await _maybe(service.get_view(session_id))
    subscription: Any = None
    try:
        subscription = await _maybe(service.subscribe(session_id))
        replay = await _load_events(service, session_id, effective_after)
        stream = event_stream(subscription, replay, session_id, effective_after)
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
    except Exception:
        if subscription is not None:
            await close_subscription(subscription)
        raise


@router.get("/sessions/{session_id}/artifacts/{name}")
async def get_artifact(
    session_id: str,
    name: str,
    settings: SettingsDep,
) -> FileResponse:
    _check_session_id(session_id)
    if _ARTIFACT_RE.fullmatch(name) is None or name in {".", ".."}:
        raise APIError(404, "artifact 不存在")
    base = (settings.data_dir / "sessions" / session_id / "artifacts").resolve()
    target = (base / name).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise APIError(404, "artifact 不存在")
    return FileResponse(target)
