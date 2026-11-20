# MiniLLM Web 控制台

React 19 + TypeScript + Vite 单页控制台，对接 MiniLLM FastAPI 服务：

- 会话列表（含子 agent 层级、运行状态、新建/切换）
- ReAct 对话流（流式文本、reasoning 折叠、run_code 代码块与嵌套工具卡片）
- run_code stdout/stderr/exit_code、artifact 链接
- 三级上下文压缩横幅与压缩历史
- Inspector：上下文仪表、todos、子 agent 树、已注入技能、模型与阈值配置
- SSE 事件流按 seq 去重、断线自动重连并从未处理位置续传
- 深色主题、中文界面、窄屏抽屉式侧栏

## 开发

先启动后端（默认 http://127.0.0.1:8765）：

    uv run minillm serve

再启动前端：

    cd apps/web
    pnpm install
    pnpm dev

打开 http://127.0.0.1:5173/ ，可直接用 ?session=<id> 打开指定会话。
Vite 会把 /api 代理到 http://127.0.0.1:8765。

## 构建与检查

    pnpm typecheck
    pnpm build

构建产物在 dist/（已由根目录 .gitignore 忽略），生产环境由 FastAPI 托管。

## 目录

    src/App.tsx            页面编排、会话加载、SSE 事件应用
    src/api.ts             REST/SSE 请求与错误处理
    src/types.ts           API 与事件类型
    src/normalize.ts       后端响应的容错归一化
    src/events.ts          事件 reducer、视图 reducer、流式缓冲
    src/hooks/             EventSource 与复制 hooks
    src/components/        会话、对话流、运行卡片、Inspector 等
    src/styles.css         全局变量与基础布局
    src/transcript.css     对话流与空状态
    src/runcode.css        run_code、工具卡片、压缩横幅、错误
    src/panels.css         Inspector 面板
    src/responsive.css     断点与抽屉布局

## 契约与约束

- 只访问 /api；不读取、不展示模型密钥。
- 事件 data 按 unknown 处理，逐字段守卫后使用；未知事件忽略。
- 首屏 GET /api/sessions/{id} 渲染，再订阅 /api/sessions/{id}/stream?after=<last_seq>；
  按 seq 去重，避免重复渲染回放事件。
- 单文件不超过 700 行；中文文案、英文标识符。
- pnpm-workspace.yaml 中的 allowBuilds 仅允许 esbuild 执行安装脚本，供冻结锁文件安装使用。

## 联调关注点

- 事件载荷按 plan.md 6.4 写入；tool.call/result 的 run_call_id 用于把工具卡片挂到 run_code 下。
- context.compacted 的 fallback 可能是布尔或字符串；前端对 false 不显示降级标签。
- GET /api/sessions/{id} 的 meta 字段若缺少 workspace/model，前端自动降级为空串。
- POST /api/sessions/{id}/messages 返回 202；运行中返回 409 时前端提示并保持运行状态。
