import {
  normalizeConfig,
  normalizeEvent,
  normalizeHealth,
  normalizeSessionSummary,
  normalizeSessions,
  normalizeSessionView,
} from './normalize';
import type {
  AgentEvent,
  ConfigInfo,
  HealthInfo,
  SendMessageResult,
  SessionSummary,
  SessionView,
} from './types';
import { asString, firstString, isRecord } from './utils';

const API_BASE = '/api';
const GET_TIMEOUT_MS = 20000;

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

function timeoutSignal(ms: number): AbortSignal | undefined {
  if (typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function') {
    return AbortSignal.timeout(ms);
  }
  return undefined;
}

async function request(path: string, init?: RequestInit): Promise<unknown> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  if (init?.body !== undefined) {
    headers['Content-Type'] = 'application/json';
  }
  let response: Response;
  try {
    response = await fetch(API_BASE + path, { ...init, headers });
  } catch (error) {
    const reason = error instanceof Error ? error.message : '网络错误';
    throw new ApiError(0, '无法连接后端：' + reason);
  }
  const text = await response.text();
  const contentType = response.headers.get('content-type') ?? '';
  const isJson =
    contentType.includes('application/json') || contentType.includes('+json');
  if (response.ok && text !== '' && !isJson) {
    throw new ApiError(
      response.status,
      '后端返回了非 JSON 响应（' + contentType + '），请确认 API 服务已经启动',
    );
  }
  let payload: unknown = null;
  if (text !== '') {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }
  if (!response.ok) {
    const detail = isRecord(payload)
      ? firstString(payload.detail, payload.message, payload.error)
      : typeof payload === 'string' && payload !== ''
        ? payload
        : undefined;
    throw new ApiError(response.status, detail ?? '请求失败（HTTP ' + String(response.status) + '）');
  }
  return payload;
}

export async function fetchHealth(): Promise<HealthInfo> {
  return normalizeHealth(await request('/health', { signal: timeoutSignal(8000) }));
}

export async function fetchConfig(): Promise<ConfigInfo> {
  return normalizeConfig(await request('/config', { signal: timeoutSignal(GET_TIMEOUT_MS) }));
}

export async function fetchSessions(): Promise<SessionSummary[]> {
  return normalizeSessions(await request('/sessions', { signal: timeoutSignal(GET_TIMEOUT_MS) }));
}

export interface CreateSessionInput {
  title?: string;
  workspace?: string;
}

export async function createSession(input: CreateSessionInput = {}): Promise<SessionSummary> {
  const payload = await request('/sessions', {
    method: 'POST',
    body: JSON.stringify(input),
  });
  const record = isRecord(payload) && isRecord(payload.session) ? payload.session : payload;
  const session = normalizeSessionSummary(record);
  if (session.id === '') {
    throw new Error('创建会话失败：响应缺少会话 id');
  }
  return session;
}

export async function fetchSession(sessionId: string): Promise<SessionView> {
  const payload = await request('/sessions/' + encodeURIComponent(sessionId), {
    signal: timeoutSignal(GET_TIMEOUT_MS),
  });
  const view = normalizeSessionView(payload);
  if (view.meta.id === '') {
    throw new Error('会话数据缺少 meta.id');
  }
  return view;
}

export async function fetchEvents(sessionId: string, after = 0): Promise<AgentEvent[]> {
  const payload = await request(
    '/sessions/' + encodeURIComponent(sessionId) + '/events?after=' + String(after),
    { signal: timeoutSignal(GET_TIMEOUT_MS) },
  );
  const raw = Array.isArray(payload)
    ? payload
    : isRecord(payload) && Array.isArray(payload.events)
      ? payload.events
      : [];
  const seen = new Set<number>();
  const events: AgentEvent[] = [];
  for (const item of raw) {
    const event = normalizeEvent(item);
    if (event !== null && !seen.has(event.seq)) {
      seen.add(event.seq);
      events.push(event);
    }
  }
  events.sort((left, right) => left.seq - right.seq);
  return events;
}

export async function sendMessage(sessionId: string, content: string): Promise<SendMessageResult> {
  const payload = await request('/sessions/' + encodeURIComponent(sessionId) + '/messages', {
    method: 'POST',
    body: JSON.stringify({ content }),
  });
  if (!isRecord(payload)) {
    return {};
  }
  return { turn_id: asString(payload.turn_id) };
}

export async function stopSession(sessionId: string): Promise<void> {
  await request('/sessions/' + encodeURIComponent(sessionId) + '/stop', {
    method: 'POST',
    body: '{}',
  });
}

export function artifactUrl(sessionId: string, name: string): string {
  return (
    API_BASE +
    '/sessions/' +
    encodeURIComponent(sessionId) +
    '/artifacts/' +
    encodeURIComponent(name)
  );
}

export function sseUrl(sessionId: string, after: number): string {
  return (
    API_BASE +
    '/sessions/' +
    encodeURIComponent(sessionId) +
    '/stream?after=' +
    String(after)
  );
}
