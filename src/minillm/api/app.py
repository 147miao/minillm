"""FastAPI 应用工厂：CORS、异常处理、静态托管与生命周期。"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import Settings
from ..errors import MiniLLMError, SessionError, ToolError
from .routes import router
from .schemas import APIError

logger = logging.getLogger("minillm.api")

_MISSING_HTML = """<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>MiniLLM</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 720px; margin: 10vh auto;">
  <h1>MiniLLM 前端尚未构建</h1>
  <p>API 已就绪。请在仓库根目录执行：</p>
  <pre>cd apps/web
pnpm install
pnpm build</pre>
  <p>然后刷新本页；开发模式可运行 <code>pnpm dev</code>（5173 端口）。</p>
</body>
</html>
"""


async def _call_optional(obj: Any, names: tuple[str, ...]) -> None:
    """按顺序调用第一个存在的生命周期方法，兼容同步/异步实现。"""

    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            result = fn()
            if inspect.isawaitable(result):
                await result
            return


def _build_default_service(settings: Settings) -> Any:
    """惰性构造 WP5 的 AgentService；未落地时给出清晰错误。"""

    try:
        module = importlib.import_module("minillm.agent.service")
    except ImportError as exc:
        raise RuntimeError(
            "AgentService 尚未实现（WP5），请注入 service 或先完成 agent/service.py"
        ) from exc
    builder = getattr(module, "build_service", None)
    if callable(builder):
        return builder(settings)
    service_cls = getattr(module, "AgentService", None)
    if service_cls is None:
        raise RuntimeError("minillm.agent.service 缺少 AgentService/build_service")
    try:
        return service_cls(settings)
    except TypeError as exc:
        raise RuntimeError("AgentService(settings) 构造签名与 API 约定不一致") from exc


def _status_for_error(exc: MiniLLMError) -> int:
    message = getattr(exc, "message", str(exc))
    detail = getattr(exc, "detail", {}) or {}
    raw_status = detail.get("status_code") or detail.get("status")
    if isinstance(raw_status, int) and 400 <= raw_status <= 599:
        return raw_status
    code = str(detail.get("code") or "").lower()
    if code in {"session_busy", "busy", "running", "conflict"}:
        return 409
    if code in {"not_found", "session_not_found"} or "不存在" in message:
        return 404
    if isinstance(exc, (SessionError, ToolError)):
        return 400
    return 500


def _error_body(exc: MiniLLMError) -> dict[str, Any]:
    body: dict[str, Any] = {"detail": getattr(exc, "message", str(exc))}
    detail = getattr(exc, "detail", {}) or {}
    code = detail.get("code")
    if code:
        body["code"] = str(code)
    errors = getattr(exc, "errors", None)
    if errors:
        body["errors"] = errors
    return body


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_request: Request, exc: APIError) -> JSONResponse:
        body: dict[str, Any] = {"detail": exc.message}
        if exc.code:
            body["code"] = exc.code
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(MiniLLMError)
    async def _domain_error(_request: Request, exc: MiniLLMError) -> JSONResponse:
        return JSONResponse(status_code=_status_for_error(exc), content=_error_body(exc))

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常：%s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})


def _register_static(app: FastAPI, web_dist: str | Path | None) -> None:
    dist = (
        Path(web_dist)
        if web_dist is not None
        else Path(__file__).resolve().parents[3] / "apps" / "web" / "dist"
    )
    try:
        dist = dist.expanduser().resolve()
    except OSError:
        dist = Path(web_dist) if web_dist is not None else Path("apps/web/dist")
    index = dist / "index.html"
    if not index.is_file():
        @app.get("/", include_in_schema=False)
        async def _missing_root() -> HTMLResponse:
            return HTMLResponse(_MISSING_HTML, status_code=200)

        return

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    async def _root() -> FileResponse:
        return FileResponse(index)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa(full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        target = (dist / full_path).resolve()
        if target.is_file() and target.is_relative_to(dist):
            return FileResponse(target)
        return FileResponse(index)


def create_app(
    settings: Settings | None = None,
    service: Any | None = None,
    *,
    web_dist: str | Path | None = None,
) -> FastAPI:
    """构造 FastAPI 应用；service 为空时在 lifespan 中惰性创建 AgentService。"""

    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if application.state.service is None:
            application.state.service = _build_default_service(settings)
            application.state.owns_service = True
        try:
            await _call_optional(application.state.service, ("start", "startup"))
            yield
        finally:
            try:
                await _call_optional(application.state.service, ("close", "shutdown"))
            except Exception:  # noqa: BLE001 - 关闭失败不能阻塞进程退出
                logger.exception("关闭 AgentService 失败")

    app = FastAPI(title="MiniLLM Coding Agent", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service
    app.state.owns_service = service is None

    if settings.enable_cors and settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            allow_credentials=False,
        )

    app.include_router(router, prefix="/api")
    _register_error_handlers(app)
    _register_static(app, web_dist)
    return app
