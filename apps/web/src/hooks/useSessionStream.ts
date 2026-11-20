import { useEffect, useRef, useState } from 'react';
import { sseUrl } from '../api';
import { normalizeEvent } from '../normalize';
import type { AgentEvent } from '../types';
import { firstString, isRecord } from '../utils';

export type StreamStatus = 'idle' | 'connecting' | 'open' | 'reconnecting';

const NAMED_EVENT_TYPES = [
  'user.message',
  'assistant.delta',
  'reasoning.delta',
  'assistant.message',
  'llm.usage',
  'run_code.started',
  'run_code.finished',
  'tool.call',
  'tool.result',
  'context.compacted',
  'skills.injected',
  'todo.updated',
  'subagent.spawned',
  'subagent.message',
  'subagent.completed',
  'turn.started',
  'turn.finished',
  'turn.cancelled',
  'agent.error',
  'session.created',
  'heartbeat',
  'ping',
];

export function useSessionStream(
  sessionId: string | null,
  afterRef: { current: number },
  onEvent: (event: AgentEvent) => void,
): StreamStatus {
  const [status, setStatus] = useState<StreamStatus>('idle');
  const handlerRef = useRef(onEvent);
  handlerRef.current = onEvent;

  useEffect(() => {
    if (sessionId === null) {
      setStatus('idle');
      return;
    }
    const id = sessionId;
    let source: EventSource | null = null;
    let retryTimer: number | null = null;
    let attempt = 0;
    let disposed = false;

    const handleRaw = (raw: string, fallbackType?: string, fallbackSeq?: string) => {
      if (raw === '') {
        return;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(raw);
      } catch {
        return;
      }
      if (
        isRecord(parsed) &&
        parsed.seq === undefined &&
        parsed.id === undefined &&
        fallbackSeq !== undefined &&
        fallbackSeq !== ''
      ) {
        const seq = Number(fallbackSeq);
        if (Number.isFinite(seq)) {
          parsed = { ...parsed, seq };
        }
      }
      const event = normalizeEvent(parsed, fallbackType);
      if (event === null) {
        const rawType = isRecord(parsed)
          ? firstString(parsed.type, parsed.event, fallbackType)
          : fallbackType;
        if (rawType === 'assistant.delta' || rawType === 'reasoning.delta') {
          const data = isRecord(parsed) && isRecord(parsed.data) ? parsed.data : parsed;
          handlerRef.current({ seq: 0, type: rawType, data: isRecord(data) ? data : {} });
        }
        return;
      }
      if (event.type === 'heartbeat' || event.type === 'ping') {
        return;
      }
      handlerRef.current(event);
    };

    const open = () => {
      if (disposed) {
        return;
      }
      setStatus(attempt === 0 ? 'connecting' : 'reconnecting');
      source = new EventSource(sseUrl(id, afterRef.current));
      source.onopen = () => {
        attempt = 0;
        setStatus('open');
      };
      source.onmessage = (message) => handleRaw(message.data, undefined, message.lastEventId);
      source.onerror = () => {
        if (disposed) {
          return;
        }
        source?.close();
        source = null;
        attempt += 1;
        setStatus('reconnecting');
        const delay = Math.min(1000 * Math.pow(2, Math.min(attempt - 1, 3)), 8000);
        retryTimer = window.setTimeout(open, delay);
      };
      for (const type of NAMED_EVENT_TYPES) {
        source.addEventListener(type, (message) => {
          const event = message as MessageEvent;
          if (typeof event.data === 'string') {
            handleRaw(event.data, type, event.lastEventId);
          }
        });
      }
    };

    open();

    return () => {
      disposed = true;
      if (retryTimer !== null) {
        window.clearTimeout(retryTimer);
      }
      source?.close();
      source = null;
    };
  }, [sessionId]);

  return status;
}
