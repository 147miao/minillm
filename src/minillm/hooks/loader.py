"""从目录加载用户 hook 插件。

约定：目录下每个 *.py 暴露同步的 register(manager) 函数。
单个模块失败只记日志，不阻塞其他模块或服务启动。
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path

from .manager import HookManager

logger = logging.getLogger("minillm.hooks.loader")


def load_hook_modules(manager: HookManager, directory: Path) -> list[str]:
    """加载 directory 下的 hook 模块，返回成功加载的文件名列表。"""

    directory = Path(directory)
    if not directory.is_dir():
        return []

    loaded: list[str] = []
    for path in sorted(directory.glob("*.py")):
        name = path.name
        if name.startswith("_") or name == "__init__.py":
            continue
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
        module_name = f"minillm_user_hooks_{path.stem}_{digest}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法为 {path} 创建 import spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            register = getattr(module, "register", None)
            if not callable(register):
                raise AttributeError("hook 模块缺少 register(manager)")
            result = register(manager)
            if inspect.isawaitable(result):
                raise TypeError("register 必须是同步函数")
            loaded.append(name)
            logger.info("已加载 hook 模块：%s", name)
        except Exception:  # noqa: BLE001 - 插件失败隔离
            logger.exception("hook 模块加载失败：%s", path)
            sys.modules.pop(module_name, None)
    return loaded
