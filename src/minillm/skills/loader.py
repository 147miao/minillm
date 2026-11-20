"""技能加载与注入文本渲染。

技能目录结构：skills/<slug>/SKILL.md，YAML frontmatter + Markdown 正文。
frontmatter 支持 name、description、keywords；无 frontmatter 时用目录名作为 name。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - pyyaml 是项目依赖，这里只做兜底
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger("minillm.skills")

_FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)$", re.S)
_CACHE: dict[str, tuple[tuple[tuple[str, int, int], ...], tuple[Skill, ...]]] = {}


@dataclass(frozen=True, slots=True)
class Skill:
    """一个可注入的技能。"""

    name: str
    description: str
    keywords: tuple[str, ...]
    body: str
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "keywords": list(self.keywords),
            "path": str(self.path),
        }


def _parse_frontmatter_fallback(raw: str) -> dict[str, Any]:
    """极简 YAML 兜底：key: value，value 支持 [a, b] 列表。"""

    meta: dict[str, Any] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [part.strip().strip("'\"") for part in value[1:-1].split(",")]
            meta[key] = [item for item in items if item]
        else:
            meta[key] = value.strip("'\"")
    return meta


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    text = text.lstrip("\ufeff")
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text.strip()
    raw_meta, body = match.group(1), match.group(2)
    meta: Any
    if yaml is not None:
        try:
            meta = yaml.safe_load(raw_meta)
        except Exception:  # noqa: BLE001 - 坏 YAML 退化为简单解析
            logger.warning("frontmatter YAML 解析失败，使用兜底解析")
            meta = _parse_frontmatter_fallback(raw_meta)
    else:  # pragma: no cover
        meta = _parse_frontmatter_fallback(raw_meta)
    if not isinstance(meta, dict):
        meta = {}
    return {str(key): value for key, value in meta.items()}, body.strip()


def _normalize_keywords(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = re.split(r"[,，;；]", value)
    elif isinstance(value, (list, tuple, set)):
        raw = [str(item) for item in value]
    else:
        raw = [str(value)]
    cleaned = [item.strip() for item in raw if item.strip()]
    return tuple(dict.fromkeys(cleaned))


def parse_skill_file(path: Path) -> Skill:
    """解析单个 SKILL.md；格式错误抛 ValueError。"""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    name = str(meta.get("name") or path.parent.name).strip()
    if not name:
        raise ValueError(f"技能缺少 name：{path}")
    description = str(meta.get("description") or "").strip()
    keywords = _normalize_keywords(meta.get("keywords"))
    return Skill(
        name=name,
        description=description,
        keywords=keywords,
        body=body,
        path=path,
    )


def load_skills(directory: Path) -> list[Skill]:
    """加载目录下所有 */SKILL.md，按 name 排序；结果按文件签名缓存。"""

    directory = Path(directory)
    if not directory.is_dir():
        return []

    files = sorted(directory.glob("*/SKILL.md"))
    signature: list[tuple[str, int, int]] = []
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    key = str(directory.resolve())
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == tuple(signature):
        return list(cached[1])

    skills: list[Skill] = []
    for path in files:
        try:
            skills.append(parse_skill_file(path))
        except Exception:  # noqa: BLE001 - 单个坏技能不影响其他技能
            logger.warning("跳过无法解析的技能：%s", path)
    skills.sort(key=lambda item: item.name.casefold())
    _CACHE[key] = (tuple(signature), tuple(skills))
    return list(skills)


def clear_skill_cache() -> None:
    _CACHE.clear()


def select_skills(text: str, skills: list[Skill] | tuple[Skill, ...]) -> list[Skill]:
    """按关键词做 casefold 包含匹配；空 keywords 不自动注入。"""

    haystack = (text or "").casefold()
    if not haystack:
        return []
    selected: list[Skill] = []
    for skill in skills:
        if not skill.keywords:
            continue
        if any(keyword.casefold() in haystack for keyword in skill.keywords):
            selected.append(skill)
    return selected


def render_skill_prompt(skills: list[Skill] | tuple[Skill, ...], max_chars: int = 8000) -> str:
    """把技能渲染为 system prompt 片段；超长时截断并标注。"""

    sections: list[str] = []
    for skill in skills:
        header = f"## 技能：{skill.name}"
        if skill.description:
            header += f"\n{skill.description}"
        header += f"\n（技能文件：{skill.path}）"
        sections.append(f"{header}\n\n{skill.body}".strip())
    text = "\n\n".join(section for section in sections if section)
    if max_chars > 0 and len(text) > max_chars:
        suffix = "\n\n[技能内容已截断]"
        if max_chars <= len(suffix):
            text = text[:max_chars]
        else:
            text = text[: max_chars - len(suffix)] + suffix
    return text
