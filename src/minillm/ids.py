"""会话、agent、turn、工具调用等标识符与时间戳工具。

约定：
- 标识符形如 sess_1a2b3c4d5e6f，前缀表明类型，后缀 12 位十六进制随机串。
- 时间戳统一为 UTC ISO8601（毫秒精度，Z 结尾），便于 JSONL 排序与跨时区排查。
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

# 允许的 id 前缀（新增类型时在这里登记，便于审查与文档同步）
ID_PREFIXES: frozenset[str] = frozenset(
    {"sess", "agent", "turn", "call", "run", "msg", "hook", "skill"}
)


def new_id(prefix: str, *, length: int = 12) -> str:
    """生成带前缀的随机标识符，例如 new_id("sess") -> "sess_ab12cd34ef56"。"""

    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError(f"非法 id 前缀：{prefix!r}")
    if length <= 0:
        raise ValueError("id 长度必须为正数")
    return f"{prefix}_{secrets.token_hex((length + 1) // 2)[:length]}"


def utc_now_iso() -> str:
    """当前 UTC 时间，格式如 2026-11-20T07:30:00.123Z。"""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
