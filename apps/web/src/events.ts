import {
  normalizeCompactionRecord,
  normalizeSkills,
  normalizeTodos,
  normalizeToolCalls,
  normalizeToolSummaries,
  normalizeUsage,
} from './normalize';
import type {
  AgentEvent,
  AgentInfo,
  ChatMessage,
  CompactionRecord,
  EventDetails,
  RunCodeDetail,
  SessionView,
  StreamBuffer,
  ToolCallDetail,
  ToolErrorPayload,
  UsageInfo,
} from './types';
import { asBoolean, asNumber, asString, firstString, isRecord, stringifyValue } from './utils';

export function createEmptyDetails(): EventDetails {
  return {
    lastSeq: 0,
    runs: {},
    tools: {},
    compactions: [],
    errors: [],
    skills: [],
    usage: null,
  };
}

export function clearStream(): StreamBuffer {
  return { text: '', reasoning: '' };
}

export function appendStreamDelta(stream: StreamBuffer, event: AgentEvent): StreamBuffer {
  const text = asString(event.data.text) ?? '';
  if (text === '') {
    return stream;
  }
  const kind = event.type === 'reasoning.delta' ? 'reasoning' : asString(event.data.kind) ?? 'text';
  if (kind === 'reasoning') {
    return { text: stream.text, reasoning: stream.reasoning + text };
  }
  return { text: stream.text + text, reasoning: stream.reasoning };
}

function normalizeToolError(raw: unknown): ToolErrorPayload | undefined {
  if (!isRecord(raw)) {
    return undefined;
  }
  return {
    kind: asString(raw.kind),
    tool: firstString(raw.tool, raw.name),
    message: firstString(raw.message, raw.detail),
    errors: raw.errors,
  };
}

function previewText(value: unknown): string | undefined {
  if (typeof value === 'string') {
    return value;
  }
  if (value === undefined || value === null) {
    return undefined;
  }
  return stringifyValue(value);
}

export function reduceDetails(previous: EventDetails, event: AgentEvent): EventDetails {
  if (event.seq <= previous.lastSeq) {
    return previous;
  }
  const next: EventDetails = {
    ...previous,
    lastSeq: event.seq,
  };
  const data = event.data;

  switch (event.type) {
    case 'run_code.started': {
      const callId = firstString(data.call_id, data.id) ?? 'run-' + String(event.seq);
      const prior = previous.runs[callId];
      const run: RunCodeDetail = {
        callId,
        started: true,
        finished: prior?.finished ?? false,
        code: asString(data.code) ?? prior?.code,
        exitCode: prior?.exitCode,
        stdout: prior?.stdout,
        stderr: prior?.stderr,
        artifact: prior?.artifact,
        durationMs: prior?.durationMs,
        toolSummaries: prior?.toolSummaries,
      };
      next.runs = { ...previous.runs, [callId]: run };
      break;
    }
    case 'run_code.finished': {
      const callId = firstString(data.call_id, data.id) ?? 'run-' + String(event.seq);
      const prior = previous.runs[callId];
      const run: RunCodeDetail = {
        callId,
        started: prior?.started ?? true,
        finished: true,
        code: prior?.code,
        exitCode: asNumber(data.exit_code) ?? prior?.exitCode ?? null,
        stdout: firstString(data.stdout_preview, data.stdout) ?? prior?.stdout,
        stderr: firstString(data.stderr_preview, data.stderr) ?? prior?.stderr,
        artifact: firstString(data.artifact) ?? prior?.artifact,
        durationMs: asNumber(data.duration_ms) ?? prior?.durationMs,
        toolSummaries: normalizeToolSummaries(data.tool_calls) ?? prior?.toolSummaries,
      };
      next.runs = { ...previous.runs, [callId]: run };
      break;
    }
    case 'tool.call': {
      const callId = firstString(data.call_id, data.id) ?? 'tool-' + String(event.seq);
      const prior = previous.tools[callId];
      const tool: ToolCallDetail = {
        callId,
        runCallId: firstString(data.run_call_id, data.run_id) ?? prior?.runCallId,
        tool: firstString(data.tool, data.name) ?? prior?.tool ?? 'tool',
        argsPreview: firstString(data.args_preview, data.arguments) ?? prior?.argsPreview,
        argsArtifact: firstString(data.args_artifact) ?? prior?.argsArtifact,
        ok: prior?.ok,
        preview: prior?.preview,
        artifact: prior?.artifact,
        error: prior?.error,
        durationMs: prior?.durationMs,
        pending: true,
      };
      next.tools = { ...previous.tools, [callId]: tool };
      break;
    }
    case 'tool.result': {
      const callId = firstString(data.call_id, data.id) ?? 'tool-' + String(event.seq);
      const prior = previous.tools[callId];
      const error = normalizeToolError(data.error) ?? prior?.error;
      const ok = asBoolean(data.ok) ?? (error !== undefined ? false : prior?.ok ?? true);
      const tool: ToolCallDetail = {
        callId,
        runCallId: firstString(data.run_call_id, data.run_id) ?? prior?.runCallId,
        tool: firstString(data.tool, data.name) ?? prior?.tool ?? 'tool',
        argsPreview: prior?.argsPreview,
        argsArtifact: prior?.argsArtifact,
        ok,
        preview: previewText(data.preview ?? data.result) ?? prior?.preview,
        artifact: firstString(data.artifact) ?? prior?.artifact,
        error,
        durationMs: asNumber(data.duration_ms) ?? prior?.durationMs,
        pending: false,
      };
      next.tools = { ...previous.tools, [callId]: tool };
      break;
    }
    case 'context.compacted': {
      const record = normalizeCompactionRecord(data);
      record.seq = event.seq;
      record.ts = firstString(event.ts, record.ts);
      next.compactions = [...previous.compactions, record];
      break;
    }
    case 'skills.injected': {
      const skills = normalizeSkills(data.skills);
      if (skills.length > 0) {
        next.skills = skills;
      }
      break;
    }
    case 'llm.usage': {
      next.usage = normalizeUsage(data);
      break;
    }
    case 'agent.error': {
      const entry = {
        seq: event.seq,
        ts: event.ts,
        message: firstString(data.message, data.error, data.detail) ?? '未知错误',
        where: asString(data.where),
        traceback: asString(data.traceback),
      };
      next.errors = [...previous.errors, entry];
      break;
    }
    default:
      break;
  }
  return next;
}

export function detailsFromEvents(events: AgentEvent[]): EventDetails {
  return events.reduce(reduceDetails, createEmptyDetails());
}

function assistantMessageFromEvent(data: Record<string, unknown>): ChatMessage | null {
  const content = asString(data.content) ?? '';
  const reasoning = firstString(data.reasoning, data.reasoning_content);
  const toolCalls = normalizeToolCalls(data.tool_calls);
  if (content === '' && reasoning === undefined && toolCalls.length === 0) {
    return null;
  }
  const message: ChatMessage = { role: 'assistant', content };
  if (reasoning !== undefined) {
    message.reasoning = reasoning;
  }
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }
  return message;
}

function contextWithTokens(view: SessionView, tokens: number): SessionView['context'] {
  const budget = view.context.budget;
  return {
    ...view.context,
    tokens,
    ratio: budget > 0 ? tokens / budget : 0,
  };
}

function upsertAgent(agents: AgentInfo[], data: Record<string, unknown>, eventType: string): AgentInfo[] {
  const id = firstString(data.agent_id, data.agent_session_id, data.id, data.session_id);
  if (id === undefined) {
    return agents;
  }
  const index = agents.findIndex((agent) => agent.id === id || agent.session_id === id);
  const existing = index >= 0 ? agents[index] : undefined;
  const status =
    asString(data.status) ??
    (eventType === 'subagent.spawned'
      ? 'running'
      : eventType === 'subagent.completed'
        ? 'completed'
        : existing?.status ?? 'running');
  const updated: AgentInfo = {
    id: existing?.id ?? id,
    status,
    session_id: firstString(data.session_id, data.agent_session_id) ?? existing?.session_id,
    parent_session_id:
      firstString(data.parent_session_id, data.parent_id) ?? existing?.parent_session_id ?? null,
    task: firstString(data.task, data.description) ?? existing?.task,
    result_preview: firstString(data.result_preview, data.result) ?? existing?.result_preview,
    last_message:
      firstString(data.content, data.message, data.last_message) ?? existing?.last_message,
  };
  if (index >= 0) {
    const next = [...agents];
    next[index] = updated;
    return next;
  }
  return [...agents, updated];
}

export function applyViewEvent(view: SessionView, event: AgentEvent): SessionView {
  const data = event.data;
  const base: SessionView = { ...view, last_seq: Math.max(view.last_seq, event.seq) };

  switch (event.type) {
    case 'user.message': {
      const content = asString(data.content) ?? '';
      return { ...base, messages: [...view.messages, { role: 'user', content }] };
    }
    case 'assistant.message': {
      const message = assistantMessageFromEvent(data);
      if (message === null) {
        return base;
      }
      return { ...base, messages: [...view.messages, message] };
    }
    case 'llm.usage': {
      const usage: UsageInfo = normalizeUsage(data);
      const purpose = asString(data.purpose);
      if (
        purpose === 'small' ||
        purpose === 'summary' ||
        purpose === 'compact' ||
        purpose === 'prune'
      ) {
        return base;
      }
      const tokens = usage.prompt_tokens ?? usage.total_tokens;
      if (tokens === undefined) {
        return base;
      }
      return { ...base, context: contextWithTokens(view, tokens) };
    }
    case 'context.compacted': {
      const record = normalizeCompactionRecord(data);
      record.seq = event.seq;
      record.ts = firstString(event.ts, record.ts);
      const after = asNumber(data.after_tokens);
      const context = after !== undefined ? contextWithTokens(view, after) : { ...view.context };
      return {
        ...base,
        context: { ...context, compactions: [...view.context.compactions, record] },
      };
    }
    case 'todo.updated': {
      return { ...base, todos: normalizeTodos(data.items) };
    }
    case 'skills.injected': {
      const skills = normalizeSkills(data.skills);
      return { ...base, skills: skills.length > 0 ? skills : view.skills };
    }
    case 'subagent.spawned':
    case 'subagent.message':
    case 'subagent.completed': {
      return { ...base, agents: upsertAgent(view.agents, data, event.type) };
    }
    case 'turn.started': {
      return { ...base, status: 'running' };
    }
    case 'turn.finished': {
      const status = asString(data.status) ?? (view.status === 'error' ? 'error' : 'idle');
      return { ...base, status };
    }
    case 'turn.cancelled': {
      return { ...base, status: 'cancelled' };
    }
    case 'agent.error': {
      return { ...base, status: 'error' };
    }
    case 'session.created': {
      return {
        ...base,
        meta: {
          ...view.meta,
          title: asString(data.title) ?? view.meta.title,
          status: asString(data.status) ?? view.meta.status,
        },
      };
    }
    default:
      return base;
  }
}

export function compactionRecords(view: SessionView, details: EventDetails): CompactionRecord[] {
  if (details.compactions.length > 0) {
    return details.compactions;
  }
  return view.context.compactions;
}

