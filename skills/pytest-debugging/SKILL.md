---
name: pytest-debugging
description: pytest 失败时的系统化定位与修复流程
keywords: [pytest, 测试失败, 断言失败, traceback, 单测]
---

# pytest 调试流程

1. 先完整读失败输出：区分收集错误、fixture 错误、断言失败与超时；不要只看最后一行。
2. 定位首个失败用例，读完整 traceback，找到第一个属于本项目的栈帧。
3. 用最小输入复现：只跑该用例（pytest -q path::test_name），必要时加 -x -vv。
4. 修改后重跑同一用例与整个相关文件；确认没有引入新的失败。
5. 如果测试环境缺依赖，先确认 pyproject 与服务端的 Python 版本，再安装最小依赖。
6. 报告时给出：失败用例、根因、修改文件、复跑命令与结果。
