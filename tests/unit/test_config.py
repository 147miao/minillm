"""Settings 默认值、环境覆盖、密钥脱敏与目录方法。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minillm.config import Settings, get_settings, reset_settings
from minillm.errors import ConfigError


def _clear_known_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_known_env(monkeypatch)
    settings = Settings(_env_file=None, llm_api_key="k")

    assert settings.llm_base_url == "http://127.0.0.1:8788/inference/go/v1"
    assert settings.llm_model == "deepseek-v4.1-flash"
    assert settings.small_llm_model == "glm-5.3-flash"
    assert settings.context_budget == 200000
    assert settings.prune_threshold == 0.5
    assert settings.compact_threshold == 0.8
    assert settings.keep_recent_turns == 6
    assert settings.max_tool_output_chars == 20000
    assert settings.run_code_output_chars == 8000
    assert settings.run_code_timeout == 120
    assert settings.run_code_idle_timeout == 120
    assert settings.run_code_max_timeout == 900
    assert settings.web_fetch_allow_private is False
    assert settings.allow_external_workspace is False
    assert settings.subagent_max_depth == 1
    assert settings.subagent_max_concurrent == 4
    assert settings.wal_fsync is True
    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.enable_cors is True
    assert settings.cors_origin_list == [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]
    assert settings.llm_extra_headers == {"x-opencode-session": "minillm"}
    assert settings.minillm_llm == "real"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_known_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("CONTEXT_BUDGET", "1234")
    monkeypatch.setenv("COMPACT_THRESHOLD", "0.75")
    monkeypatch.setenv("WEB_FETCH_ALLOW_PRIVATE", "true")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")

    settings = Settings(_env_file=None, llm_api_key="k")

    assert settings.llm_model == "env-model"
    assert settings.context_budget == 1234
    assert settings.compact_threshold == 0.75
    assert settings.web_fetch_allow_private is True
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_known_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_MODEL=file-model\n"
        "PRUNE_THRESHOLD=0.42\n"
        'LLM_EXTRA_HEADERS={"x-opencode-session":"from-file"}\n',
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file, llm_api_key="k")

    assert settings.llm_model == "file-model"
    assert settings.prune_threshold == 0.42
    assert settings.llm_extra_headers == {"x-opencode-session": "from-file"}


def test_small_model_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_known_env(monkeypatch)
    settings = Settings(
        _env_file=None,
        llm_api_key="main-key",
        small_llm_base_url="",
        small_llm_api_key="",
    )

    assert settings.small_base_url == settings.llm_base_url
    assert settings.small_api_key.get_secret_value() == "main-key"

    overridden = Settings(
        _env_file=None,
        llm_api_key="main-key",
        small_llm_base_url="http://small.test/v1",
        small_llm_api_key="small-key",
    )
    assert overridden.small_base_url == "http://small.test/v1"
    assert overridden.small_api_key.get_secret_value() == "small-key"


def test_secret_not_leaked(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_known_env(monkeypatch)
    secret = "super-secret-key"
    settings = Settings(_env_file=None, llm_api_key=secret)

    assert secret not in repr(settings)
    assert secret not in json.dumps(settings.public_dict(), default=str)
    assert settings.public_dict()["llm_api_key_set"] is True
    assert settings.llm_api_key.get_secret_value() == secret


def test_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_known_env(monkeypatch)
    settings = Settings(_env_file=None)
    with pytest.raises(ConfigError):
        settings.require_api_key()


def test_paths_and_ensure_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_known_env(monkeypatch)
    settings = Settings(
        _env_file=None,
        llm_api_key="k",
        data_dir=tmp_path / "data",
        workspace_root=tmp_path / "ws",
    )

    assert settings.sessions_dir == tmp_path / "data" / "sessions"
    assert settings.session_dir("sess_x") == tmp_path / "data" / "sessions" / "sess_x"
    assert settings.session_workspace("sess_x") == tmp_path / "ws" / "sess_x"

    settings.ensure_dirs()
    assert (tmp_path / "data" / "sessions").is_dir()
    assert (tmp_path / "ws").is_dir()


def test_extra_headers_accept_json_and_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_known_env(monkeypatch)
    settings = Settings(_env_file=None, llm_api_key="k", llm_extra_headers='{"a": "b"}')
    assert settings.llm_extra_headers == {"a": "b"}

    empty = Settings(_env_file=None, llm_api_key="k", llm_extra_headers="")
    assert empty.llm_extra_headers == {}


def test_cors_origins_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_known_env(monkeypatch)
    json_form = Settings(
        _env_file=None,
        llm_api_key="k",
        cors_origins='["http://x.test", "http://y.test"]',
    )
    assert json_form.cors_origin_list == ["http://x.test", "http://y.test"]
    empty = Settings(_env_file=None, llm_api_key="k", cors_origins="")
    assert empty.cors_origin_list == []


def test_get_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_known_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "cached-model")
    reset_settings()
    try:
        assert get_settings() is get_settings()
        assert get_settings().llm_model == "cached-model"
    finally:
        reset_settings()
