interface EmptyStateProps {
  loading: boolean;
  offline: boolean;
  error: string | null;
  creating: boolean;
  onCreate: () => void;
  onRetry: () => void;
  onClear: () => void;
}

const FEATURES: Array<[string, string]> = [
  ['run_code 单入口', '模型只看到一个 tool_call，Python 子进程经本地桥调用注册表中的核心工具'],
  ['三级上下文压缩', '工具输出截断落盘、小模型剪枝思维链、超过预算 80% 全量压缩'],
  ['JSONL 断点恢复', '每个会话独立追加事件日志，重启后重放 JSONL 恢复现场'],
  ['多 Agent 协作', '子会话隔离、按需继承上下文，提供 5 个子 agent 管理工具'],
];

export function EmptyState({
  loading,
  offline,
  error,
  creating,
  onCreate,
  onRetry,
  onClear,
}: EmptyStateProps) {
  if (loading) {
    return (
      <div className="empty-state">
        <span className="spinner" aria-hidden="true" />
        <p className="muted">正在连接后端…</p>
      </div>
    );
  }

  const failed = offline || error !== null;

  if (failed) {
    return (
      <div className="empty-state">
        <div className="empty-card empty-card-error">
          <span className="empty-icon" aria-hidden="true">!</span>
          <h1>{offline ? '无法连接后端' : '会话加载失败'}</h1>
          <p className="empty-desc">
            {error !== null
              ? error
              : '前端通过 /api 访问 FastAPI 服务。请确认后端已经启动。'}
          </p>
          <p className="empty-hint">
            启动命令：<code>uv run minillm serve</code>（默认 http://127.0.0.1:8765）
          </p>
          <div className="empty-actions">
            <button type="button" className="primary-button" onClick={onRetry}>
              重试连接
            </button>
            {error !== null && (
              <button type="button" className="ghost-button" onClick={onClear}>
                返回会话列表
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="empty-state">
      <div className="empty-hero">
        <span className="hero-mark" aria-hidden="true">M</span>
        <h1>MiniLLM Coding Agent</h1>
        <p className="empty-desc">
          ReAct 多轮工具调用 · run_code Python 桥 · 三级上下文压缩 · JSONL 断点恢复 · 子 Agent 协作
        </p>
        <div className="feature-grid">
          {FEATURES.map(([title, description]) => (
            <div className="feature-card" key={title}>
              <h2>{title}</h2>
              <p>{description}</p>
            </div>
          ))}
        </div>
        <div className="empty-actions">
          <button type="button" className="primary-button" onClick={onCreate} disabled={creating}>
            {creating ? '创建中…' : '新建会话'}
          </button>
          <span className="empty-hint">或从左侧选择已有会话</span>
        </div>
      </div>
    </div>
  );
}
