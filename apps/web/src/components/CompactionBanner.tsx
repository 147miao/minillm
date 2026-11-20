import type { CompactionRecord } from '../types';
import { formatClock, formatRatio, formatTokens } from '../utils';

export function compactionLevelLabel(level: number): string {
  if (level === 1) {
    return 'L1 工具输出截断落盘';
  }
  if (level === 2) {
    return 'L2 小模型思维链剪枝';
  }
  if (level === 3) {
    return 'L3 上下文全量压缩';
  }
  return '上下文压缩';
}

function savedRatio(record: CompactionRecord): number {
  if (record.before_tokens <= 0) {
    return 0;
  }
  return Math.max(0, 1 - record.after_tokens / record.before_tokens);
}

export function CompactionBanner({ record }: { record: CompactionRecord }) {
  const saved = savedRatio(record);
  return (
    <details className="compaction-banner">
      <summary className="compaction-head">
        <span className="compaction-level">L{record.level}</span>
        <span className="compaction-title">{compactionLevelLabel(record.level)}</span>
        <span className="compaction-stats">
          {formatTokens(record.before_tokens)} → {formatTokens(record.after_tokens)} tokens
        </span>
        {saved > 0 && <span className="compaction-saved">-{formatRatio(saved)}</span>}
        {record.fallback !== undefined && record.fallback !== false && (
          <span className="compaction-fallback">降级：{String(record.fallback)}</span>
        )}
        {record.ts !== undefined && <span className="compaction-time">{formatClock(record.ts)}</span>}
      </summary>
      <div className="compaction-body">
        {record.summary !== undefined && record.summary !== '' ? (
          <p>{record.summary}</p>
        ) : (
          <p className="muted">本次压缩没有摘要内容。</p>
        )}
        {record.pruned_upto_seq !== undefined && (
          <p className="compaction-meta">已处理到事件 #{record.pruned_upto_seq}</p>
        )}
      </div>
    </details>
  );
}
