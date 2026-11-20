import { useEffect, useMemo, useRef, useState } from 'react';
import { compactionRecords } from '../events';
import type { StreamStatus } from '../hooks/useSessionStream';
import type { ChatMessage, EventDetails, SessionView, StreamBuffer } from '../types';
import { shortId } from '../utils';
import { CompactionBanner } from './CompactionBanner';
import { MessageBubble, StreamingBubble } from './MessageBubble';
import { statusLabel, statusTone } from './StatusDot';

interface TranscriptProps {
  view: SessionView;
  details: EventDetails;
  stream: StreamBuffer;
  running: boolean;
  streamStatus: StreamStatus;
  syncing: boolean;
  onSync: () => void;
}

export function Transcript({
  view,
  details,
  stream,
  running,
  streamStatus,
  syncing,
  onSync,
}: TranscriptProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);
  const [atBottom, setAtBottom] = useState(true);

  const toolResults = useMemo(() => {
    const map = new Map<string, ChatMessage>();
    for (const message of view.messages) {
      if (message.role === 'tool' && message.tool_call_id !== undefined) {
        map.set(message.tool_call_id, message);
      }
    }
    return map;
  }, [view.messages]);

  const visibleMessages = useMemo(
    () => view.messages.filter((message) => message.role !== 'tool'),
    [view.messages],
  );

  const orphanTools = useMemo(() => {
    const knownIds = new Set<string>();
    for (const message of view.messages) {
      for (const call of message.tool_calls ?? []) {
        knownIds.add(call.id);
      }
    }
    return view.messages.filter(
      (message) =>
        message.role === 'tool' &&
        (message.tool_call_id === undefined || !knownIds.has(message.tool_call_id)),
    );
  }, [view.messages]);

  const compactions = useMemo(
    () => compactionRecords(view, details),
    [view, details],
  );

  const handleScroll = () => {
    const element = scrollRef.current;
    if (element === null) {
      return;
    }
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
    const next = distance < 90;
    atBottomRef.current = next;
    setAtBottom(next);
  };

  const scrollToBottom = (behavior: ScrollBehavior) => {
    const element = scrollRef.current;
    if (element !== null) {
      element.scrollTo({ top: element.scrollHeight, behavior });
    }
  };

  useEffect(() => {
    if (atBottomRef.current) {
      scrollToBottom('auto');
    }
  }, [view.messages.length, stream.text, stream.reasoning, running, details.lastSeq, compactions.length]);

  useEffect(() => {
    atBottomRef.current = true;
    setAtBottom(true);
    scrollToBottom('auto');
  }, [view.meta.id]);

  return (
    <section className="transcript-panel" aria-label="对话记录">
      <header className="transcript-head">
        <div className="transcript-title">
          <h1>{view.meta.title !== '' ? view.meta.title : '未命名会话'}</h1>
          <div className="transcript-meta">
            <span className={'status-chip status-chip-' + statusTone(view.status)}>
              {statusLabel(view.status)}
            </span>
            <span className="meta-item">会话 {shortId(view.meta.id)}</span>
            <span className="meta-item">模型 {view.meta.model !== '' ? view.meta.model : '—'}</span>
            {view.meta.workspace !== '' && (
              <span className="meta-item workspace" title={view.meta.workspace}>
                {view.meta.workspace}
              </span>
            )}
          </div>
        </div>
        <div className="transcript-actions">
          {streamStatus === 'reconnecting' && (
            <span className="stream-hint">
              <span className="pulse-dot" aria-hidden="true" /> 事件流重连中
            </span>
          )}
          <button type="button" className="ghost-button" onClick={onSync} disabled={syncing}>
            {syncing ? '同步中…' : '同步状态'}
          </button>
        </div>
      </header>
      <div
        className="transcript-scroll"
        ref={scrollRef}
        onScroll={handleScroll}
        role="log"
        aria-live="polite"
      >
        <div className="transcript-content">
          {visibleMessages.length === 0 && compactions.length === 0 && (
            <p className="muted transcript-empty">还没有消息。在下方输入任务，开始第一轮 ReAct 循环。</p>
          )}
          {visibleMessages.map((message, index) => (
            <MessageBubble
              key={message.role + '-' + String(index)}
              message={message}
              sessionId={view.meta.id}
              details={details}
              toolResults={toolResults}
            />
          ))}
          {compactions.map((record, index) => (
            <CompactionBanner
              key={
                'compaction-' +
                String(index) +
                '-' +
                String(record.seq ?? '') +
                '-' +
                String(record.before_tokens)
              }
              record={record}
            />
          ))}
          {orphanTools.map((message, index) => (
            <details className="orphan-tool" key={'orphan-' + String(index)}>
              <summary>
                工具结果 · {message.tool_call_id ?? message.name ?? '未知'}
              </summary>
              <pre>{message.content}</pre>
            </details>
          ))}
          {details.errors.map((error) => (
            <div className="error-banner" role="alert" key={'error-' + String(error.seq)}>
              <strong>执行错误{error.where !== undefined ? ' · ' + error.where : ''}</strong>
              <p>{error.message}</p>
              {error.traceback !== undefined && error.traceback !== '' && (
                <details>
                  <summary>堆栈</summary>
                  <pre>{error.traceback}</pre>
                </details>
              )}
            </div>
          ))}
          {running && <StreamingBubble stream={stream} />}
          {running && stream.text === '' && stream.reasoning === '' && (
            <div className="thinking-row">
              <span className="pulse-dot" aria-hidden="true" /> 模型思考中…
            </div>
          )}
        </div>
      </div>
      {!atBottom && (
        <button type="button" className="jump-bottom" onClick={() => scrollToBottom('smooth')}>
          回到底部 ↓
        </button>
      )}
    </section>
  );
}
