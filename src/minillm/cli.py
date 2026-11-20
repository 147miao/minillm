"""命令行入口：serve / run / chat / sessions / tools / doctor。

设计要点：
- 仅 argparse + 懒加载：--help / --version / sessions 不需要 API 服务或工具层就绪；
  后续 WP 尚未落地时，对应子命令给出清晰错误并 exit 2（而不是 ImportError 堆栈）。
- 与后续工作包的接口约定：
    minillm.api.app.create_app(settings) -> FastAPI
    minillm.agent.service.run_task(task, *, workspace=None, data_dir=None, settings=None) -> dict
    minillm.agent.service.chat(*, settings=None, workspace=None) -> int | None
    minillm.bootstrap.build_registry(settings) -> ToolRegistry（需提供 list_specs()/specs）
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from collections.abc import Callable
from typing import Any

from . import __version__
from .config import Settings
from .errors import MiniLLMError
from .persistence.store import SessionStore

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_READY = 2


def _settings(**overrides: Any) -> Settings:
    clean = {key: value for key, value in overrides.items() if value is not None}
    return Settings(**clean)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _load_symbol(module_name: str, symbol: str, command: str) -> Callable[..., Any] | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        print(
            f"错误：{command} 依赖模块 {module_name} 尚未就绪（可能对应工作包未完成）：{exc}",
            file=sys.stderr,
        )
        return None
    target = getattr(module, symbol, None)
    if target is None or not callable(target):
        print(
            f"错误：{module_name}.{symbol} 尚未实现，无法执行 {command}",
            file=sys.stderr,
        )
        return None
    return target


# ==== 子命令实现 ====
def _cmd_serve(args: argparse.Namespace) -> int:
    settings = _settings(
        data_dir=args.data_dir,
        workspace_root=args.workspace_root,
        host=args.host,
        port=args.port,
    )
    create_app = _load_symbol("minillm.api.app", "create_app", "serve")
    if create_app is None:
        return EXIT_NOT_READY
    import uvicorn

    app = create_app(settings)
    uvicorn.run(app, host=args.host or settings.host, port=args.port or settings.port)
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(data_dir=args.data_dir)
    run_task = _load_symbol("minillm.agent.service", "run_task", "run")
    if run_task is None:
        return EXIT_NOT_READY
    result = asyncio.run(
        run_task(
            args.task,
            workspace=args.workspace,
            data_dir=args.data_dir,
            settings=settings,
        )
    )
    if args.json:
        _print_json(result)
    elif isinstance(result, dict):
        print(
            result.get("final_answer")
            or result.get("final")
            or json.dumps(result, ensure_ascii=False)
        )
    else:
        print(result)
    return EXIT_OK


def _cmd_chat(args: argparse.Namespace) -> int:
    settings = _settings(data_dir=args.data_dir)
    chat = _load_symbol("minillm.agent.service", "chat", "chat")
    if chat is None:
        return EXIT_NOT_READY
    code = asyncio.run(chat(settings=settings, workspace=args.workspace))
    return int(code) if isinstance(code, int) else EXIT_OK


def _cmd_sessions(args: argparse.Namespace) -> int:
    settings = _settings(data_dir=getattr(args, "data_dir", None))
    store = SessionStore(settings.data_dir, fsync=settings.wal_fsync)
    action = getattr(args, "action", None)
    if action == "list":
        sessions = store.list_sessions()
        if args.json:
            _print_json(sessions)
        else:
            print(f"{'session_id':<28} {'status':<10} {'parent_id':<28} title")
            for item in sessions:
                print(
                    f"{str(item.get('session_id')):<28} {str(item.get('status')):<10} "
                    f"{str(item.get('parent_id') or '-'):<28} {item.get('title')}"
                )
        return EXIT_OK
    if action == "show":
        state = store.load_state(args.session_id)
        _print_json(
            {
                "meta": state.meta,
                "last_seq": state.last_seq,
                "message_count": len(state.messages),
                "messages": state.messages,
                "todos": state.todos,
                "agents": state.agents,
                "compactions": state.compactions,
                "last_usage": state.last_usage,
                "pending_runs": state.pending_runs,
            }
        )
        return EXIT_OK
    print("错误：未知 sessions 子命令", file=sys.stderr)
    return EXIT_NOT_READY


def _spec_to_dict(spec: Any) -> dict[str, Any]:
    if isinstance(spec, dict):
        return dict(spec)
    data: dict[str, Any] = {}
    for field in ("name", "kind", "description", "signature"):
        value = getattr(spec, field, None)
        if value is not None:
            data[field] = value
    params = getattr(spec, "params", None) or getattr(spec, "params_model", None)
    if params is not None:
        schema = getattr(params, "model_json_schema", None)
        if callable(schema):
            data["parameters"] = schema()
    return data


def _collect_specs(registry: Any) -> list[Any]:
    for attribute in ("list_specs", "specs", "all_specs"):
        value = getattr(registry, attribute, None)
        if value is None:
            continue
        specs = value() if callable(value) else value
        if isinstance(specs, dict):
            return list(specs.values())
        return list(specs)
    return []


def _cmd_tools(args: argparse.Namespace) -> int:
    settings = _settings(data_dir=getattr(args, "data_dir", None))
    build_registry = _load_symbol("minillm.bootstrap", "build_registry", "tools")
    if build_registry is None:
        return EXIT_NOT_READY
    registry = build_registry(settings)
    specs = [_spec_to_dict(spec) for spec in _collect_specs(registry)]
    if args.json:
        _print_json({"tools": specs})
        return EXIT_OK
    print(f"共 {len(specs)} 个工具")
    for spec in specs:
        kind = spec.get("kind", "?")
        name = spec.get("name", "?")
        description = spec.get("description", "")
        print(f"- {kind:<10} {name:<20} {description}")
    return EXIT_OK


def _cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings(data_dir=args.data_dir)
    import httpx

    config = settings.public_dict()
    print("== 配置 ==")
    _print_json(config)
    url = settings.llm_base_url.rstrip("/") + "/models"
    print(f"== 网关 ==\nGET {url}")
    try:
        client = httpx.Client(timeout=10.0)
    except Exception:  # noqa: BLE001 - 环境代理变量非法时兜底
        client = httpx.Client(timeout=10.0, trust_env=False)
    try:
        with client:
            response = client.get(
                url,
                headers={
                    "x-opencode-session": "minillm-doctor",
                    "User-Agent": "minillm/0.1.0",
                },
            )
    except Exception as exc:  # noqa: BLE001 - 网络/URL 配置错误统一提示
        print(f"连接失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        models = payload.get("data") or payload.get("models") or []
        if isinstance(models, list):
            names = [
                str(item.get("id") or item.get("name")) for item in models if isinstance(item, dict)
            ]
            print(f"可用模型：{', '.join(names) if names else '（空）'}")
    return EXIT_OK if response.status_code == 200 else EXIT_ERROR


# ==== 解析器 ====
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minillm", description="MiniLLM：ReAct Coding Agent")
    parser.add_argument("--version", action="version", version=f"minillm {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="启动 FastAPI + SSE 服务")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--data-dir")
    serve.add_argument("--workspace-root")
    serve.set_defaults(handler=_cmd_serve)

    run = subparsers.add_parser("run", help="执行一次性编码任务")
    run.add_argument("task", help="任务描述")
    run.add_argument("--workspace", help="任务工作目录")
    run.add_argument("--data-dir")
    run.add_argument("--json", action="store_true", help="输出完整 JSON 结果")
    run.set_defaults(handler=_cmd_run)

    chat = subparsers.add_parser("chat", help="交互式对话")
    chat.add_argument("--workspace", help="任务工作目录")
    chat.add_argument("--data-dir")
    chat.set_defaults(handler=_cmd_chat)

    sessions = subparsers.add_parser("sessions", help="查看本地会话")
    sessions.add_argument("--data-dir")
    sessions_sub = sessions.add_subparsers(dest="action", required=True)
    sessions_list = sessions_sub.add_parser("list", help="列出会话")
    sessions_list.add_argument("--data-dir", default=argparse.SUPPRESS)
    sessions_list.add_argument("--json", action="store_true")
    sessions_list.set_defaults(handler=_cmd_sessions)
    sessions_show = sessions_sub.add_parser("show", help="查看会话重放状态")
    sessions_show.add_argument("session_id")
    sessions_show.add_argument("--data-dir", default=argparse.SUPPRESS)
    sessions_show.set_defaults(handler=_cmd_sessions)

    tools = subparsers.add_parser("tools", help="查看工具注册表")
    tools.add_argument("--data-dir")
    tools_sub = tools.add_subparsers(dest="action", required=True)
    tools_list = tools_sub.add_parser("list", help="列出已注册工具")
    tools_list.add_argument("--data-dir", default=argparse.SUPPRESS)
    tools_list.add_argument("--json", action="store_true")
    tools_list.set_defaults(handler=_cmd_tools)

    doctor = subparsers.add_parser("doctor", help="检查配置与网关联通性（不打印密钥）")
    doctor.add_argument("--data-dir")
    doctor.set_defaults(handler=_cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_NOT_READY
    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    except MiniLLMError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
