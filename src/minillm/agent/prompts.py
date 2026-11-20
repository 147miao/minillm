"""System prompt 拼装。

设计：稳定前缀 + 动态段。
- 稳定前缀：角色、唯一工具 run_code、Python 程序约定、安全与输出纪律；
- 动态段：ToolRegistry 手册、技能正文、todos、上下文预算说明；
- 稳定前缀保持不变，便于网关 prompt cache 命中。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..config import Settings

PROMPT_VERSION = 3

RUN_CODE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_code",
        "description": (
            "执行一段 Python 程序。程序内可通过 tools.<name>(...) 调用 MiniLLM 工具；"
            "工具是同步函数，失败会抛 ToolCallError。这是你唯一可用的 function tool。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的完整 Python 源码（同步脚本，不要使用顶层 await）",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


def _section(title: str, body: str) -> str:
    body = body.strip()
    return f"## {title}\n{body}" if body else ""


def _render_skills(
    active_skills: Iterable[dict[str, Any]] | None,
    skill_parts: Iterable[str] | None,
) -> str:
    names: list[str] = []
    for item in active_skills or ():
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
    parts = [str(part).strip() for part in (skill_parts or ()) if str(part).strip()]
    if not names and not parts:
        return ""
    lines: list[str] = []
    if names:
        lines.append("已注入技能：" + "、".join(dict.fromkeys(names)))
    lines.extend(parts)
    return "\n\n".join(lines)


def _render_todos(todos: Iterable[dict[str, Any]] | None) -> str:
    rows: list[str] = []
    for item in todos or ():
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        status = str(item.get("status") or "pending")
        rows.append(f"- [{status}] {content}")
    return "\n".join(rows)


def _render_context(info: dict[str, Any] | None, settings: Settings) -> str:
    if not info:
        budget = int(settings.context_budget)
        return f"上下文预算约 {budget} tokens；接近阈值时系统会自动压缩，不要重复粘贴大段历史。"
    budget = int(info.get("budget") or settings.context_budget)
    tokens = int(info.get("tokens") or 0)
    level = int(info.get("level") or 0)
    ratio = (tokens / budget) if budget else 0.0
    text = (
        f"上下文预算约 {budget} tokens，当前约 {tokens}（{ratio:.0%}）。"
        "超过 50% 会剪枝旧思维链，超过 80% 会做全量压缩；压缩后以摘要为准。"
    )
    if level:
        text += f" 本轮已执行 level={level} 压缩。"
    return text


def build_system_prompt(
    settings: Settings,
    registry: Any,
    *,
    workspace: Path | str,
    active_skills: Iterable[dict[str, Any]] | None = None,
    todos: Iterable[dict[str, Any]] | None = None,
    is_subagent: bool = False,
    context_info: dict[str, Any] | None = None,
    skill_parts: Iterable[str] | None = None,
    context_summary: str | None = None,
) -> str:
    """拼装 system prompt；不写入密钥或绝对 data_dir。"""

    kinds = ["core", "mcp"] if is_subagent else None
    manual = registry.manuals(kinds=kinds) if hasattr(registry, "manuals") else ""
    if not manual:
        manual = "（工具注册表为空）"

    identity = (
        "你是 MiniLLM 的 Coding Agent，在用户给定的 workspace 内完成编码任务。"
        f"当前 workspace：{workspace}。"
    )
    if is_subagent:
        identity += " 你是子 agent，只完成父 agent 交给你的子任务；不能创建新的子 agent。"

    tool_contract = (
        "你只能通过唯一一个 function tool run_code 行动，不能调用其他 function tool。\n"
        "run_code 的参数是完整 Python 源码；源码在子进程里同步执行，"
        "通过 tools.<name>(...) 调用注册表中的工具。\n"
        "示例：\n"
        "files = tools.list_files(path='.')\n"
        "print(files)\n"
        "tools.write_file(path='hello.py', content='print(1)')\n"
        "tools.* 是同步调用，返回值是 Python 值；工具错误会抛 ToolCallError，"
        "可以 try/except 后打印错误再重试。"
    )

    rules = (
        "1. 先读相关文件，再改；改文件必须用 tools.write_file 或 tools.edit_file，"
        "不要用 open()/Path.write_text 直接写（会绕过原子替换与路径校验）。\n"
        "2. 改完要读回或运行测试验证；不要只声明完成。\n"
        "3. 只在 workspace 内操作；不要访问或修改系统目录。\n"
        "4. 不要安装未在 workspace 声明过的重依赖；优先用标准库。\n"
        "5. 工具输出可能被截断并落盘；需要全文时读取提示里的 artifact 路径。\n"
        "6. 最终回答用中文（用户用其他语言时跟随用户），总结改了什么、验证了什么、还剩什么。"
    )

    output_rules = (
        "最终回答必须包含：变更文件/命令、验证结果、未完成或风险。"
        "不要输出 API key、token、Authorization，也不要把 data_dir 绝对路径写进回答。"
    )

    skills = _render_skills(active_skills, skill_parts)
    todo_text = _render_todos(todos)
    context_text = _render_context(context_info, settings)
    if context_summary:
        context_text += "\n历史压缩摘要：\n" + context_summary.strip()

    sections = [
        identity,
        _section("唯一工具契约", tool_contract),
        _section("可用工具手册", manual),
        _section("执行纪律", rules),
        _section("输出纪律", output_rules),
        _section("技能", skills),
        _section("待办", todo_text),
        _section("上下文管理", context_text),
    ]
    if not is_subagent:
        sections.append(
            _section(
                "子 agent",
                "需要并行调研或隔离上下文时，用 tools.spawn_subagent(...) 创建子 agent；"
                "默认不继承上下文，等待用 tools.wait_agent(...)。",
            )
        )
    return "\n\n".join(part for part in sections if part)
