"""工具输出的 artifact 落盘与截断。

设计：
- 任何工具的大段输出都先写入会话 artifacts 目录，再返回相对路径；
- 返回值里只出现 artifacts/xxx 这样的相对路径，绝不泄露宿主绝对 data_dir；
- 截断提示固定为 [truncated; full output: artifacts/xxx]，方便模型与前端识别。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import Settings
from .ids import new_id
from .persistence.store import SessionStore

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _note(rel: str) -> str:
    return f"\n[truncated; full output: {rel}]"


class ArtifactStore:
    """按会话隔离的工具输出仓库。"""

    def __init__(self, store: SessionStore, session_id: str, settings: Settings) -> None:
        self.store = store
        self.session_id = session_id
        self.max_chars = max(1, int(settings.max_tool_output_chars))

    # ==== 基础读写 ====
    def save_text(self, tool: str, content: str, *, suffix: str = ".txt") -> str:
        """写入完整内容，返回相对路径（artifacts/...）。"""

        safe_tool = _SAFE_NAME.sub("_", tool).strip("._")[:40] or "tool"
        name = f"{safe_tool}_{new_id('art')}{suffix}"
        return self.store.write_artifact(self.session_id, name, content)

    def save_json(self, tool: str, value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, indent=2)
        return self.save_text(tool, payload, suffix=".json")

    def read_text(self, artifact: str) -> str:
        """按相对路径或文件名读回 artifact。"""

        name = artifact.rsplit("/", 1)[-1]
        return self.store.read_artifact(self.session_id, name)

    # ==== 截断 ====
    def cap_text(
        self,
        tool: str,
        text: str,
        *,
        max_chars: int | None = None,
        reason: str = "truncated",
    ) -> tuple[str, str | None]:
        """超限则落盘并截断，返回 (text, artifact_relative_path)。"""

        limit = self.max_chars if max_chars is None else max(1, int(max_chars))
        if len(text) <= limit:
            return text, None
        rel = self.save_text(tool, text)
        clipped = text[:limit].rstrip("\n") + _note(rel)
        if reason != "truncated":
            clipped += f"\n[reason: {reason}]"
        return clipped, rel

    def cap_streams(
        self, tool: str, stdout: str, stderr: str
    ) -> tuple[str, str, bool, str | None]:
        """run_bash 的 stdout/stderr 各自截断，完整两路共享一个 artifact。"""

        if len(stdout) <= self.max_chars and len(stderr) <= self.max_chars:
            return stdout, stderr, False, None
        full = f"$ stdout\n{stdout}\n\n$ stderr\n{stderr}\n"
        rel = self.save_text(tool, full)

        def clip(text: str) -> str:
            if len(text) <= self.max_chars:
                return text
            return text[: self.max_chars].rstrip("\n") + _note(rel)

        return clip(stdout), clip(stderr), True, rel
