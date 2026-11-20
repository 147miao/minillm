"""全局配置：pydantic-settings 从 .env 与环境变量读取。

约定：
- 所有可调参数集中在 Settings，禁止散落硬编码（统一配置约定）。
- 密钥使用 SecretStr，repr / 日志 / /api/config 都不会泄露明文。
- 小模型地址/密钥留空时自动回退到主模型配置。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError


class Settings(BaseSettings):
    """MiniLLM 运行配置。

    字段名全部小写，环境变量名自动映射为大写（case_sensitive=False）。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ==== LLM（OpenAI 兼容）====
    llm_base_url: str = "http://127.0.0.1:8788/inference/go/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "deepseek-v4.1-flash"
    small_llm_base_url: str = ""
    small_llm_api_key: SecretStr = SecretStr("")
    small_llm_model: str = "glm-5.3-flash"
    llm_extra_headers: dict[str, str] = Field(
        default_factory=lambda: {"x-opencode-session": "minillm"}
    )
    llm_timeout: float = 120.0
    llm_max_tokens: int = 16384
    include_reasoning: bool = True

    # ==== 上下文管理 ====
    context_budget: int = 200000
    prune_threshold: float = 0.5
    compact_threshold: float = 0.8
    keep_recent_turns: int = 6
    max_tool_output_chars: int = 20000
    run_code_output_chars: int = 8000

    # ==== 执行与运行 ====
    run_code_timeout: int = 120
    run_code_idle_timeout: int = 120
    run_code_max_timeout: int = 900
    bash_timeout: int = 60
    web_fetch_timeout: int = 20
    web_fetch_allow_private: bool = False
    max_steps: int = 40
    turn_timeout: int = 900

    # ==== 目录与工作区 ====
    data_dir: Path = Path("./data")
    workspace_root: Path = Path("./workspace")
    allow_external_workspace: bool = False
    hooks_dir: Path = Path("./hooks")
    skills_dir: Path = Path("./skills")
    mcp_config: Path = Path("./mcp.json")

    # ==== 子 agent ====
    subagent_max_depth: int = 1
    subagent_max_concurrent: int = 4

    # ==== 持久化 / Web ====
    wal_fsync: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    enable_cors: bool = True
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"

    # ==== 运行模式 ====
    minillm_llm: Literal["real", "scripted"] = "real"
    minillm_script: str | None = None

    @field_validator("llm_extra_headers", mode="before")
    @classmethod
    def _parse_extra_headers(cls, value: Any) -> Any:
        """允许 JSON 字符串（.env 中）或 dict；空值视为空表。"""

        if value is None or value == "":
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:  # pragma: no cover - 配置错误路径
                raise ValueError("LLM_EXTRA_HEADERS 必须是 JSON 对象") from exc
            if not isinstance(parsed, dict):
                raise ValueError("LLM_EXTRA_HEADERS 必须是 JSON 对象")
            return {str(k): str(v) for k, v in parsed.items()}
        return value

    # ==== 派生配置 ====
    @property
    def small_base_url(self) -> str:
        """小模型地址；留空回退主模型。"""

        return self.small_llm_base_url or self.llm_base_url

    @property
    def small_api_key(self) -> SecretStr:
        """小模型密钥；留空回退主模型。"""

        if self.small_llm_api_key.get_secret_value():
            return self.small_llm_api_key
        return self.llm_api_key

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS 白名单，支持逗号分隔或 JSON 数组字符串。"""

        raw = (self.cors_origins or "").strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return [part.strip() for part in raw.split(",") if part.strip()]

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def session_workspace(self, session_id: str) -> Path:
        """Web 模式下每个会话独立的工作区目录。"""

        return self.workspace_root / session_id

    def ensure_dirs(self) -> None:
        """创建运行期必需目录（可重复调用）。"""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def require_api_key(self) -> str:
        """取主模型密钥；缺失时给出可操作提示。"""

        key = self.llm_api_key.get_secret_value()
        if not key:
            raise ConfigError("未配置 LLM_API_KEY，请在 .env 中填写（不要提交到仓库）")
        return key

    def public_dict(self) -> dict[str, Any]:
        """供 /api/config 与 doctor 使用的脱敏配置。"""

        return {
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_api_key_set": bool(self.llm_api_key.get_secret_value()),
            "small_llm_base_url": self.small_base_url,
            "small_llm_model": self.small_llm_model,
            "small_llm_api_key_set": bool(self.small_api_key.get_secret_value()),
            "llm_timeout": self.llm_timeout,
            "llm_max_tokens": self.llm_max_tokens,
            "include_reasoning": self.include_reasoning,
            "context_budget": self.context_budget,
            "prune_threshold": self.prune_threshold,
            "compact_threshold": self.compact_threshold,
            "keep_recent_turns": self.keep_recent_turns,
            "max_tool_output_chars": self.max_tool_output_chars,
            "run_code_output_chars": self.run_code_output_chars,
            "run_code_timeout": self.run_code_timeout,
            "run_code_idle_timeout": self.run_code_idle_timeout,
            "run_code_max_timeout": self.run_code_max_timeout,
            "bash_timeout": self.bash_timeout,
            "web_fetch_timeout": self.web_fetch_timeout,
            "web_fetch_allow_private": self.web_fetch_allow_private,
            "max_steps": self.max_steps,
            "turn_timeout": self.turn_timeout,
            "data_dir": str(self.data_dir),
            "workspace_root": str(self.workspace_root),
            "allow_external_workspace": self.allow_external_workspace,
            "hooks_dir": str(self.hooks_dir),
            "skills_dir": str(self.skills_dir),
            "mcp_config": str(self.mcp_config),
            "subagent_max_depth": self.subagent_max_depth,
            "subagent_max_concurrent": self.subagent_max_concurrent,
            "wal_fsync": self.wal_fsync,
            "host": self.host,
            "port": self.port,
            "enable_cors": self.enable_cors,
            "cors_origins": self.cors_origin_list,
            "minillm_llm": self.minillm_llm,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程级单例；测试可用 reset_settings() 重新加载。"""

    return Settings()


def reset_settings() -> None:
    get_settings.cache_clear()
