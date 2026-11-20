import type { ContextState } from '../types';
import { formatRatio, formatTokens } from '../utils';

interface ContextMeterProps {
  context: ContextState;
  pruneThreshold: number;
  compactThreshold: number;
}

function clampRatio(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

export function ContextMeter({ context, pruneThreshold, compactThreshold }: ContextMeterProps) {
  const budget = context.budget > 0 ? context.budget : 0;
  const ratio = clampRatio(budget > 0 ? context.tokens / budget : context.ratio);
  const tone = ratio >= compactThreshold ? 'danger' : ratio >= pruneThreshold ? 'warn' : 'ok';
  const percent = Math.round(ratio * 1000) / 10;
  const prunePercent = clampRatio(pruneThreshold) * 100;
  const compactPercent = clampRatio(compactThreshold) * 100;

  return (
    <section className="panel-card context-meter">
      <header className="panel-head">
        <h3>上下文占用</h3>
        <span className={'meter-value tone-' + tone}>{formatRatio(ratio)}</span>
      </header>
      <div
        className="meter-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(ratio * 100)}
        aria-label="上下文占用"
      >
        <div className={'meter-fill tone-' + tone} style={{ width: String(percent) + '%' }} />
        <span
          className="meter-mark"
          style={{ left: String(prunePercent) + '%' }}
          title={'剪枝阈值 ' + formatRatio(pruneThreshold)}
        />
        <span
          className="meter-mark meter-mark-alert"
          style={{ left: String(compactPercent) + '%' }}
          title={'全量压缩阈值 ' + formatRatio(compactThreshold)}
        />
      </div>
      <div className="meter-foot">
        <span>
          {formatTokens(context.tokens)} / {formatTokens(budget)} tokens
        </span>
        <span>已压缩 {context.compactions.length} 次</span>
      </div>
      <p className="meter-hint">
        占用达到 {formatRatio(pruneThreshold)} 触发剪枝，{formatRatio(compactThreshold)} 触发全量压缩
      </p>
    </section>
  );
}
