"""技能加载、选择与渲染单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from minillm.skills import (
    Skill,
    clear_skill_cache,
    load_skills,
    parse_skill_file,
    render_skill_prompt,
    select_skills,
)
from minillm.skills.loader import _parse_frontmatter_fallback


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_skill_cache()
    yield
    clear_skill_cache()


def write_skill(
    directory: Path,
    slug: str,
    frontmatter: str,
    body: str,
) -> Path:
    path = directory / slug / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")
    return path


def test_parse_skill_file(tmp_path: Path) -> None:
    path = write_skill(
        tmp_path,
        "demo",
        "name: demo-skill\ndescription: 演示\nkeywords: [Alpha, beta]",
        "# 正文\n\n按步骤执行。",
    )
    skill = parse_skill_file(path)
    assert skill.name == "demo-skill"
    assert skill.description == "演示"
    assert skill.keywords == ("Alpha", "beta")
    assert "按步骤执行" in skill.body
    assert skill.path == path


def test_parse_without_frontmatter_uses_dir_name(tmp_path: Path) -> None:
    path = tmp_path / "plain" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("没有任何 frontmatter 的正文。", encoding="utf-8")
    skill = parse_skill_file(path)
    assert skill.name == "plain"
    assert skill.keywords == ()
    assert skill.body == "没有任何 frontmatter 的正文。"


def test_keywords_string_and_chinese_comma(tmp_path: Path) -> None:
    path = write_skill(tmp_path, "kw", "keywords: pytest，测试失败; 断言", "body")
    skill = parse_skill_file(path)
    assert skill.keywords == ("pytest", "测试失败", "断言")


def test_fallback_frontmatter_parser() -> None:
    meta = _parse_frontmatter_fallback("name: x\ndescription: y\nkeywords: [a, b]\n")
    assert meta == {"name": "x", "description": "y", "keywords": ["a", "b"]}


def test_select_skills() -> None:
    with_keywords = Skill(
        name="pytest",
        description="",
        keywords=("Pytest", "测试失败"),
        body="",
        path=Path("skills/pytest/SKILL.md"),
    )
    no_keywords = Skill(
        name="plain",
        description="",
        keywords=(),
        body="",
        path=Path("skills/plain/SKILL.md"),
    )
    assert select_skills("PYTEST 失败了", [with_keywords, no_keywords]) == [with_keywords]
    assert select_skills("无关内容", [with_keywords, no_keywords]) == []
    assert select_skills("", [with_keywords]) == []


def test_render_skill_prompt_limits_length() -> None:
    skill = Skill(
        name="long",
        description="描述",
        keywords=("long",),
        body="内容" * 200,
        path=Path("skills/long/SKILL.md"),
    )
    text = render_skill_prompt([skill], max_chars=120)
    assert len(text) <= 120
    assert text.endswith("[技能内容已截断]")
    assert "## 技能：long" in text
    assert len(render_skill_prompt([skill], max_chars=5)) <= 5
    assert render_skill_prompt([]) == ""


def test_load_skills_sorted_and_skips_bad(tmp_path: Path) -> None:
    write_skill(tmp_path, "b", "name: beta\ndescription: b", "B")
    write_skill(tmp_path, "a", "name: alpha\ndescription: a", "A")
    bad = tmp_path / "bad" / "SKILL.md"
    bad.mkdir(parents=True)
    skills = load_skills(tmp_path)
    assert [skill.name for skill in skills] == ["alpha", "beta"]


def test_load_skills_cache_invalidated(tmp_path: Path) -> None:
    path = write_skill(tmp_path, "cache", "name: cache", "short")
    first = load_skills(tmp_path)
    assert first[0].body == "short"
    path.write_text("---\nname: cache\n---\n长一些的新正文\n", encoding="utf-8")
    second = load_skills(tmp_path)
    assert second[0].body == "长一些的新正文"
    assert load_skills(tmp_path / "missing") == []


def test_skill_to_dict() -> None:
    skill = Skill(
        name="demo",
        description="d",
        keywords=("k",),
        body="body",
        path=Path("skills/demo/SKILL.md"),
    )
    assert skill.to_dict() == {
        "name": "demo",
        "description": "d",
        "keywords": ["k"],
        "path": "skills/demo/SKILL.md",
    }
