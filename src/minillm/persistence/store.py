"""会话 JSONL 持久化（只追加 + fsync）与崩溃恢复。

目录结构：
    data/sessions/<session_id>/meta.json
    data/sessions/<session_id>/events.jsonl
    data/sessions/<session_id>/artifacts/

约束：
- events.jsonl 只 append，不重写历史；每行一条事件，flush + fsync 后才算落盘。
- 同一 session 的读写用可重入锁串行化（同进程多任务）。
- 恢复时只补 synthetic 结果事件，不改写旧事件（见 docs/EVENTS.md）。
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import SessionError
from ..events import Event, EventType
from ..ids import new_id, utc_now_iso

if TYPE_CHECKING:
    from .replay import SessionState

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SessionStore:
    """按会话隔离的事件存储。"""

    def __init__(self, data_dir: str | Path, *, fsync: bool = True) -> None:
        self.data_dir = Path(data_dir)
        self.sessions_dir = self.data_dir / "sessions"
        self.fsync = fsync
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._last_seq: dict[str, int] = {}

    # ==== 路径与锁 ====
    def _lock(self, session_id: str) -> threading.RLock:
        with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.RLock()
                self._locks[session_id] = lock
            return lock

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not _SESSION_ID_RE.match(session_id):
            raise SessionError(f"非法会话 id：{session_id!r}")
        return session_id

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / self._validate_session_id(session_id)

    def events_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "events.jsonl"

    def meta_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "meta.json"

    def artifacts_dir(self, session_id: str) -> Path:
        path = self.session_dir(session_id) / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ==== 创建 / 元数据 ====
    def create_session(self, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """创建会话目录、meta.json 并追加 session.created 事件。"""

        raw = dict(meta or {})
        session_id = str(raw.pop("session_id", "") or raw.pop("id", "") or new_id("sess"))
        self._validate_session_id(session_id)

        with self._lock(session_id):
            path = self.session_dir(session_id)
            if path.exists():
                raise SessionError(f"会话已存在：{session_id}")
            (path / "artifacts").mkdir(parents=True, exist_ok=False)
            meta_doc: dict[str, Any] = {
                "title": "未命名会话",
                "workspace": "",
                "model": "",
                "parent_id": None,
                "status": "idle",
            }
            meta_doc.update(raw)
            meta_doc["session_id"] = session_id
            meta_doc.setdefault("created_at", utc_now_iso())
            meta_doc["workspace"] = str(meta_doc.get("workspace") or "")
            self._write_meta(session_id, meta_doc)
            self._fsync_dir(path)
            self.append(
                session_id,
                EventType.SESSION_CREATED,
                {
                    "title": meta_doc.get("title"),
                    "workspace": meta_doc.get("workspace"),
                    "model": meta_doc.get("model"),
                    "parent_id": meta_doc.get("parent_id"),
                },
            )
            return dict(meta_doc)

    def meta(self, session_id: str) -> dict[str, Any]:
        path = self.meta_path(session_id)
        if not path.exists():
            raise SessionError(f"会话不存在：{session_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SessionError(f"会话 meta.json 不可读：{session_id}") from exc
        if not isinstance(data, dict):
            raise SessionError(f"会话 meta.json 格式错误：{session_id}")
        data.setdefault("session_id", session_id)
        return data

    def _write_meta(self, session_id: str, meta_doc: Mapping[str, Any]) -> None:
        path = self.meta_path(session_id)
        tmp = path.with_name(f".meta.{os.getpid()}.{new_id('msg')}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(dict(meta_doc), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover - 清理失败不影响主流程
                    pass

    def list_sessions(self) -> list[dict[str, Any]]:
        if not self.sessions_dir.exists():
            return []
        sessions: list[dict[str, Any]] = []
        for path in self.sessions_dir.iterdir():
            if not path.is_dir():
                continue
            meta_path = path / "meta.json"
            if not meta_path.exists():
                continue
            try:
                data = self.meta(path.name)
            except SessionError:
                continue
            sessions.append(data)
        sessions.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return sessions

    # ==== 追加 / 读取 ====
    def append(
        self, session_id: str, event_type: str, data: Mapping[str, Any] | None = None
    ) -> Event:
        """追加一条事件并返回；seq 在会话内从 1 单调递增。"""

        self._validate_session_id(session_id)
        if not isinstance(event_type, str) or not event_type:
            raise SessionError("事件类型不能为空")

        with self._lock(session_id):
            path = self.events_path(session_id)
            if not path.parent.exists():
                raise SessionError(f"会话不存在：{session_id}")
            seq = self._next_seq(session_id) + 1
            event = Event(
                seq=seq,
                ts=utc_now_iso(),
                session_id=session_id,
                type=event_type,
                data=dict(data or {}),
            )
            line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            self._last_seq[session_id] = seq
            return event

    def _next_seq(self, session_id: str) -> int:
        cached = self._last_seq.get(session_id)
        if cached is not None:
            return cached
        last = 0
        path = self.events_path(session_id)
        if path.exists():
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise SessionError(f"事件文件不可读：{session_id}") from exc
            # 崩溃可能留下没有换行的半行：首次追加前截断到最后一条完整事件，
            # 只删除从未成功落盘的部分，不改写已提交历史。
            if raw and not raw.endswith(b"\n"):
                cut = raw.rfind(b"\n") + 1
                try:
                    with path.open("r+b") as handle:
                        handle.truncate(cut)
                        handle.flush()
                        if self.fsync:
                            os.fsync(handle.fileno())
                except OSError as exc:
                    raise SessionError(f"事件文件修复失败：{session_id}") from exc
                raw = raw[:cut]
            for line in raw.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    last = max(last, int(json.loads(line).get("seq", 0)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        self._last_seq[session_id] = last
        return last

    def load_events(self, session_id: str, after: int = 0) -> list[Event]:
        """读取 seq > after 的事件；尾部未写完的半行会被忽略。"""

        with self._lock(session_id):
            path = self.events_path(session_id)
            if not path.exists():
                raise SessionError(f"会话不存在：{session_id}")
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise SessionError(f"事件文件不可读：{session_id}") from exc
            lines = raw.splitlines()
            ends_with_newline = raw.endswith("\n")
            events: list[Event] = []
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    event = Event.from_json(line)
                except (json.JSONDecodeError, ValueError) as exc:
                    is_partial_tail = index == len(lines) - 1 and not ends_with_newline
                    if is_partial_tail:
                        break
                    raise SessionError(f"事件文件第 {index + 1} 行不可解析：{session_id}") from exc
                if event.session_id != session_id:
                    raise SessionError(f"事件 session_id 不匹配：{session_id}")
                if event.seq > after:
                    events.append(event)
            return events

    # ==== artifact ====
    def write_artifact(self, session_id: str, name: str, content: str | bytes) -> str:
        """写入 artifacts/<name>，返回相对会话目录的路径。"""

        safe = Path(str(name)).name
        if safe in {"", ".", ".."}:
            raise SessionError(f"非法 artifact 名称：{name!r}")
        directory = self.artifacts_dir(session_id)
        target = directory / safe
        payload = content.encode("utf-8") if isinstance(content, str) else content
        tmp = directory / f".{safe}.{os.getpid()}.{new_id('msg')}.tmp"
        try:
            with tmp.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover
                    pass
        return f"artifacts/{safe}"

    def read_artifact(self, session_id: str, name: str) -> str:
        safe = Path(str(name)).name
        path = self.artifacts_dir(session_id) / safe
        if not path.exists():
            raise SessionError(f"artifact 不存在：{name}")
        return path.read_text(encoding="utf-8")

    # ==== 恢复 ====
    async def recover(self, session_id: str) -> SessionState:
        """补 synthetic 结果事件后重放状态（不自动续跑 turn）。"""

        with self._lock(session_id):
            events = self.load_events(session_id)
            finished_runs = {
                str(event.data.get("call_id"))
                for event in events
                if event.type == EventType.RUN_CODE_FINISHED
            }
            pending_runs = [
                event
                for event in events
                if event.type == EventType.RUN_CODE_STARTED
                and str(event.data.get("call_id")) not in finished_runs
            ]
            tool_results = {
                str(event.data.get("call_id"))
                for event in events
                if event.type == EventType.TOOL_RESULT
            }
            pending_tools = [
                event
                for event in events
                if event.type == EventType.TOOL_CALL
                and str(event.data.get("call_id")) not in tool_results
            ]

            for event in pending_runs:
                call_id = event.data.get("call_id")
                self.append(
                    session_id,
                    EventType.RUN_CODE_FINISHED,
                    {
                        "call_id": call_id,
                        "exit_code": -1,
                        "stdout_preview": "",
                        "stderr_preview": "服务重启，run_code 结果未知",
                        "content": "[synthetic] 服务重启，run_code 执行结果未知，可重新运行。",
                        "duration_ms": 0,
                        "tool_calls": [],
                        "synthetic": True,
                    },
                )
            for event in pending_tools:
                call_id = event.data.get("call_id")
                self.append(
                    session_id,
                    EventType.TOOL_RESULT,
                    {
                        "call_id": call_id,
                        "run_call_id": event.data.get("run_call_id"),
                        "tool": event.data.get("tool"),
                        "ok": False,
                        "preview": "",
                        "error": {
                            "kind": "interrupted",
                            "message": "服务重启，工具调用结果未知",
                        },
                        "synthetic": True,
                    },
                )

        from .replay import load_state

        return load_state(session_id, store=self)

    def load_state(self, session_id: str) -> SessionState:
        from .replay import load_state

        return load_state(session_id, store=self)

    def _fsync_dir(self, path: Path) -> None:
        if not self.fsync:
            return
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:  # pragma: no cover - 部分文件系统不支持目录 fsync
            pass
