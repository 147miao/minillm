import { useState } from 'react';
import { artifactUrl } from '../api';
import type { ChatMessage, RunCodeDetail, ToolCallDetail, ToolCallRef } from '../types';
import { formatDuration, parseJsonObject } from '../utils';
import { CopyButton } from './CopyButton';
import { ToolCallCard } from './ToolCallCard';

interface RunCodeCardProps {
  sessionId: string;
  call: ToolCallRef;
  run?: RunCodeDetail;
  result?: ChatMessage;
  toolCalls: ToolCallDetail[];
}

type RunTone = 'pending' | 'ok' | 'error' | 'done';

function resolveTone(run: RunCodeDetail | undefined, result: ChatMessage | undefined): RunTone {
  if (run === undefined) {
    return result === undefined ? 'pending' : 'done';
  }
  if (!run.finished) {
    return 'pending';
  }
  if (run.exitCode !== undefined && run.exitCode !== null && run.exitCode !== 0) {
    return 'error';
  }
  return 'ok';
}

export function RunCodeCard({ sessionId, call, run, result, toolCalls }: RunCodeCardProps) {
  const [open, setOpen] = useState(true);
  const args = parseJsonObject(call.arguments);
  const codeFromArgs = args !== null && typeof args.code === 'string' ? args.code : '';
  const code = run?.code ?? codeFromArgs;
  const tone = resolveTone(run, result);
  const pending = tone === 'pending';

  let statusText = '执行中…';
  if (!pending) {
    if (run?.finished) {
      statusText = run.exitCode === undefined || run.exitCode === null
        ? '已完成'
        : 'exit ' + String(run.exitCode);
    } else {
      statusText = '已完成';
    }
  }

  const stdout = run?.stdout ?? '';
  const stderr = run?.stderr ?? '';
  const fallbackOutput = run === undefined && result !== undefined ? result.content : '';
  const summaries = toolCalls.length > 0 ? [] : run?.toolSummaries ?? [];
  const toolCount = toolCalls.length > 0 ? toolCalls.length : summaries.length;
  const duration = formatDuration(run?.durationMs);

  return (
    <article className="run-card">
      <header className="run-head">
        <button
          type="button"
          className="run-toggle"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          aria-label={open ? '收起代码' : '展开代码'}
        >
          <span className="chevron" aria-hidden="true">{open ? '▾' : '▸'}</span>
          <span className="run-name">run_code</span>
          <span className={'run-status run-status-' + tone}>{statusText}</span>
          {duration !== '' && <span className="run-meta">{duration}</span>}
          {toolCount > 0 && <span className="run-meta">{toolCount} 次工具调用</span>}
        </button>
        <div className="run-head-actions">
          <CopyButton text={code} label="复制代码" />
          {run?.artifact !== undefined && (
            <a
              className="artifact-link"
              href={artifactUrl(sessionId, run.artifact)}
              target="_blank"
              rel="noreferrer"
            >
              完整输出
            </a>
          )}
        </div>
      </header>
      {open && (
        <pre className="run-code">
          <code>{code !== '' ? code : '（模型未输出可执行代码）'}</code>
        </pre>
      )}
      {stdout !== '' && (
        <section className="run-stream">
          <h4>stdout</h4>
          <pre>{stdout}</pre>
        </section>
      )}
      {stderr !== '' && (
        <section className="run-stream run-stream-error">
          <h4>stderr</h4>
          <pre>{stderr}</pre>
        </section>
      )}
      {fallbackOutput !== '' && (
        <section className="run-stream">
          <h4>执行结果</h4>
          <pre>{fallbackOutput}</pre>
        </section>
      )}
      {!pending && stdout === '' && stderr === '' && fallbackOutput === '' && (
        <p className="run-empty muted">没有标准输出</p>
      )}
      {toolCalls.length > 0 && (
        <div className="tool-list">
          {toolCalls.map((tool) => (
            <ToolCallCard key={tool.callId} detail={tool} sessionId={sessionId} />
          ))}
        </div>
      )}
      {toolCalls.length === 0 && summaries.length > 0 && (
        <div className="tool-summaries">
          {summaries.map((summary, index) => (
            <span
              key={summary.name + '-' + String(index)}
              className={'tool-chip ' + (summary.ok === false ? 'is-error' : '')}
            >
              {summary.ok === false ? '✗' : '✓'} {summary.name}
              {summary.duration_ms !== undefined ? ' · ' + formatDuration(summary.duration_ms) : ''}
            </span>
          ))}
        </div>
      )}
      {pending && (
        <p className="run-pending">
          <span className="pulse-dot" aria-hidden="true" /> 正在执行程序…
        </p>
      )}
    </article>
  );
}
