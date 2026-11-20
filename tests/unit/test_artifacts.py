"""ArtifactStore 截断与持久化测试。"""

from __future__ import annotations

import json

from minillm.artifacts import ArtifactStore
from minillm.config import Settings


def test_save_text_unique_and_readable(artifacts: ArtifactStore, store, session_id: str) -> None:
    first = artifacts.save_text("read_file", "hello")
    second = artifacts.save_text("read_file", "hello")
    assert first.startswith("artifacts/")
    assert second.startswith("artifacts/")
    assert first != second
    assert artifacts.read_text(first) == "hello"
    assert store.read_artifact(session_id, first.rsplit("/", 1)[-1]) == "hello"


def test_save_json(artifacts: ArtifactStore) -> None:
    rel = artifacts.save_json("search_files", [{"path": "a.py", "line": 1, "text": "x"}])
    assert rel.endswith(".json")
    assert json.loads(artifacts.read_text(rel))[0]["path"] == "a.py"


def test_cap_text_under_and_over(artifacts: ArtifactStore) -> None:
    text, rel = artifacts.cap_text("read_file", "short")
    assert text == "short"
    assert rel is None

    full = "x" * 300
    text, rel = artifacts.cap_text("read_file", full, max_chars=50)
    assert rel is not None
    assert text.endswith(f"full output: {rel}]")
    assert artifacts.read_text(rel) == full


def test_cap_streams_share_artifact(settings: Settings, store, session_id: str) -> None:
    small = settings.model_copy(update={"max_tool_output_chars": 10})
    artifacts = ArtifactStore(store, session_id, small)
    stdout, stderr, truncated, rel = artifacts.cap_streams(
        "run_bash", "a" * 30, "b" * 30
    )
    assert truncated is True
    assert rel is not None
    assert "full output" in stdout
    assert "full output" in stderr
    full = artifacts.read_text(rel)
    assert "$ stdout" in full
    assert "$ stderr" in full


def test_paths_are_relative(artifacts: ArtifactStore, settings: Settings) -> None:
    rel = artifacts.save_text("web_fetch", "body")
    assert rel.startswith("artifacts/")
    assert str(settings.data_dir) not in rel
