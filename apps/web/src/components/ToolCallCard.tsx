import { artifactUrl } from '../api';
import type { ToolCallDetail, ToolErrorPayload } from '../types';
import { classNames, formatDuration, stringifyValue, truncate } from '../utils';

interface ToolCallCardProps {
  detail?: ToolCallDetail;
  fallbackName?: string;
  fallbackArgs?: string;
  fallbackPreview?: string;
  sessionId: string;
}

function toolErrorMessage(error: ToolErrorPayload): string {
  const parts: string[] = [];
  if (error.kind !== undefined) {
    parts.push('[' + error.kind + ']');
  }
  if (error.message !== undefined) {
    parts.push(error.message);
  }
  if (error.errors !== undefined) {
    parts.push(stringifyValue(error.errors));
  }
  return parts.join(' ');
}

export function ToolCallCard({
  detail,
  fallbackName,
  fallbackArgs,
  fallbackPreview,
  sessionId,
}: ToolCallCardProps) {
  const name = detail?.tool ?? fallbackName ?? 'tool';
  const args = detail?.argsPreview ?? fallbackArgs ?? '';
  const preview = detail?.preview ?? fallbackPreview ?? '';
  const pending = detail?.pending ?? false;
  const ok = detail?.ok;
  const tone = pending ? 'pending' : ok === false ? 'error' : ok === true ? 'ok' : 'neutral';
  const icon = pending ? '…' : ok === false ? '✗' : ok === true ? '✓' : '•';
  const duration = formatDuration(detail?.durationMs);
  const errorText = detail?.error !== undefined ? toolErrorMessage(detail.error) : '';
  const hasLinks =
    (detail?.argsArtifact !== undefined && detail.argsArtifact !== '') ||
    (detail?.artifact !== undefined && detail.artifact !== '');

  return (
    <details className={classNames('tool-card', 'tool-tone-' + tone)}>
      <summary className="tool-head">
        <span className={classNames('tool-icon', 'tool-icon-' + tone)}>{icon}</span>
        <code className="tool-name">{name}</code>
        {duration !== '' && <span className="tool-meta">{duration}</span>}
        {pending && <span className="tool-meta">执行中</span>}
        {args !== '' && <span className="tool-args">{truncate(args, 110)}</span>}
      </summary>
      <div className="tool-body">
        {args !== '' && (
          <section className="tool-section">
            <h4>参数</h4>
            <pre>{args}</pre>
          </section>
        )}
        {preview !== '' && (
          <section className="tool-section">
            <h4>结果</h4>
            <pre>{preview}</pre>
          </section>
        )}
        {errorText !== '' && <p className="tool-error">{errorText}</p>}
        {pending && preview === '' && <p className="muted">等待工具返回…</p>}
        {hasLinks && (
          <div className="tool-links">
            {detail?.argsArtifact !== undefined && detail.argsArtifact !== '' && (
              <a
                href={artifactUrl(sessionId, detail.argsArtifact)}
                target="_blank"
                rel="noreferrer"
              >
                参数详情
              </a>
            )}
            {detail?.artifact !== undefined && detail.artifact !== '' && (
              <a href={artifactUrl(sessionId, detail.artifact)} target="_blank" rel="noreferrer">
                完整结果
              </a>
            )}
          </div>
        )}
      </div>
    </details>
  );
}
