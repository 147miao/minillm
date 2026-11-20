import type {
  AgentEvent,
  AgentInfo,
  ChatMessage,
  CompactionRecord,
  ConfigInfo,
  ContextState,
  HealthInfo,
  SessionSummary,
  SessionView,
  SkillInfo,
  TodoItem,
  ToolCallRef,
  ToolRunSummary,
  UsageInfo,
} from './types';
import { asBoolean, asNumber, asString, firstString, isRecord } from './utils';

export function normalizeHealth(raw: unknown): HealthInfo {
  const record = isRecord(raw) ? raw : {};
  return {
    status: asString(record.status) ?? 'unknown',
    version: asString(record.version) ?? '',
    model: asString(record.model) ?? '',
  };
}

export function normalizeConfig(raw: unknown): ConfigInfo {
  const record = isRecord(raw) ? raw : {};
  return {
    model: asString(record.model) ?? '',
    small_model: asString(record.small_model) ?? '',
    context_budget: asNumber(record.context_budget) ?? 0,
    prune_threshold: asNumber(record.prune_threshold) ?? 0.5,
    compact_threshold: asNumber(record.compact_threshold) ?? 0.8,
    workspace_root: asString(record.workspace_root) ?? '',
    skills: normalizeSkills(record.skills),
    mcp_servers: Array.isArray(record.mcp_servers)
      ? record.mcp_servers.map((item) => asString(item) ?? '').filter((item) => item !== '')
      : [],
    version: asString(record.version) ?? '',
  };
}

export function normalizeSessionSummary(raw: unknown): SessionSummary {
  const record = isRecord(raw) ? raw : {};
  return {
    id: asString(record.id) ?? '',
    title: asString(record.title) ?? '',
    workspace: asString(record.workspace) ?? '',
    model: asString(record.model) ?? '',
    parent_id: firstString(record.parent_id) ?? null,
    created_at: asString(record.created_at) ?? '',
    updated_at: firstString(record.updated_at, record.created_at) ?? '',
    status: asString(record.status) ?? 'idle',
  };
}

export function normalizeSessions(raw: unknown): SessionSummary[] {
  if (Array.isArray(raw)) {
    return raw.map(normalizeSessionSummary).filter((session) => session.id !== '');
  }
  const record = isRecord(raw) ? raw : {};
  const list = Array.isArray(record.sessions) ? record.sessions : [];
  return list.map(normalizeSessionSummary).filter((session) => session.id !== '');
}

function normalizeContent(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  if (Array.isArray(value)) {
    return value
      .map((part) => {
        if (isRecord(part)) {
          return asString(part.text) ?? asString(part.content) ?? '';
        }
        return asString(part) ?? '';
      })
      .filter((part) => part !== '')
      .join('\n');
  }
  return asString(value) ?? '';
}

export function normalizeToolCalls(raw: unknown): ToolCallRef[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const calls: ToolCallRef[] = [];
  raw.forEach((item, index) => {
    if (!isRecord(item)) {
      return;
    }
    const fn = isRecord(item.function) ? item.function : undefined;
    const name = firstString(item.name, fn?.name) ?? '';
    const argsValue = item.arguments ?? fn?.arguments;
    let args = '';
    if (typeof argsValue === 'string') {
      args = argsValue;
    } else if (argsValue !== undefined) {
      try {
        args = JSON.stringify(argsValue);
      } catch {
        args = '';
      }
    }
    calls.push({
      id: firstString(item.id, item.call_id) ?? 'call-' + String(index),
      name,
      arguments: args,
    });
  });
  return calls;
}

function normalizeMessage(raw: unknown): ChatMessage | null {
  if (!isRecord(raw)) {
    return null;
  }
  const role = asString(raw.role) ?? 'assistant';
  const message: ChatMessage = {
    role,
    content: normalizeContent(raw.content),
  };
  const reasoning = firstString(raw.reasoning, raw.reasoning_content);
  if (reasoning !== undefined) {
    message.reasoning = reasoning;
  }
  const toolCalls = normalizeToolCalls(raw.tool_calls);
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }
  const toolCallId = asString(raw.tool_call_id);
  if (toolCallId !== undefined) {
    message.tool_call_id = toolCallId;
  }
  const name = asString(raw.name);
  if (name !== undefined) {
    message.name = name;
  }
  return message;
}

export function normalizeMessages(raw: unknown): ChatMessage[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.map(normalizeMessage).filter((message): message is ChatMessage => message !== null);
}

export function normalizeTodos(raw: unknown): TodoItem[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const items: TodoItem[] = [];
  raw.forEach((item) => {
    if (typeof item === 'string') {
      items.push({ content: item, status: 'pending' });
      return;
    }
    if (!isRecord(item)) {
      return;
    }
    const content = firstString(item.content, item.title, item.text) ?? '';
    if (content === '') {
      return;
    }
    const todo: TodoItem = {
      content,
      status: asString(item.status) ?? 'pending',
    };
    const id = asString(item.id);
    if (id !== undefined) {
      todo.id = id;
    }
    items.push(todo);
  });
  return items;
}

export function normalizeAgents(raw: unknown): AgentInfo[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const agents: AgentInfo[] = [];
  raw.forEach((item, index) => {
    if (!isRecord(item)) {
      return;
    }
    const id = firstString(item.id, item.agent_id, item.session_id);
    const task = firstString(item.task, item.description, item.prompt);
    if (id === undefined && task === undefined) {
      return;
    }
    const agent: AgentInfo = {
      id: id ?? 'agent-' + String(index),
      status: asString(item.status) ?? 'unknown',
    };
    const sessionId = firstString(item.session_id, item.agent_session_id);
    if (sessionId !== undefined) {
      agent.session_id = sessionId;
    }
    agent.parent_session_id = firstString(item.parent_session_id, item.parent_id) ?? null;
    if (task !== undefined) {
      agent.task = task;
    }
    const resultPreview = firstString(item.result_preview, item.result);
    if (resultPreview !== undefined) {
      agent.result_preview = resultPreview;
    }
    const lastMessage = firstString(item.last_message, item.message);
    if (lastMessage !== undefined) {
      agent.last_message = lastMessage;
    }
    agents.push(agent);
  });
  return agents;
}

export function normalizeSkills(raw: unknown): SkillInfo[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const skills: SkillInfo[] = [];
  for (const item of raw) {
    if (typeof item === 'string') {
      if (item !== '') {
        skills.push({ name: item });
      }
      continue;
    }
    if (!isRecord(item)) {
      continue;
    }
    const name = firstString(item.name, item.id);
    if (name === undefined) {
      continue;
    }
    const skill: SkillInfo = { name };
    const description = asString(item.description);
    if (description !== undefined) {
      skill.description = description;
    }
    const path = asString(item.path);
    if (path !== undefined) {
      skill.path = path;
    }
    skills.push(skill);
  }
  return skills;
}

export function normalizeCompactionRecord(raw: unknown): CompactionRecord {
  const record = isRecord(raw) ? raw : {};
  const normalized: CompactionRecord = {
    level: asNumber(record.level) ?? 0,
    before_tokens: asNumber(record.before_tokens) ?? 0,
    after_tokens: asNumber(record.after_tokens) ?? 0,
  };
  const summary = asString(record.summary);
  if (summary !== undefined) {
    normalized.summary = summary;
  }
  const fallbackBool = asBoolean(record.fallback);
  const fallbackText = asString(record.fallback);
  if (fallbackBool !== undefined) {
    normalized.fallback = fallbackBool;
  } else if (fallbackText !== undefined) {
    normalized.fallback = fallbackText;
  }
  const prunedUpto = asNumber(record.pruned_upto_seq);
  if (prunedUpto !== undefined) {
    normalized.pruned_upto_seq = prunedUpto;
  }
  const ts = asString(record.ts);
  if (ts !== undefined) {
    normalized.ts = ts;
  }
  const seq = asNumber(record.seq);
  if (seq !== undefined) {
    normalized.seq = seq;
  }
  return normalized;
}

export function normalizeContext(raw: unknown): ContextState {
  const record = isRecord(raw) ? raw : {};
  const budget = asNumber(record.budget) ?? 0;
  const tokens = asNumber(record.tokens) ?? 0;
  const ratio = asNumber(record.ratio) ?? (budget > 0 ? tokens / budget : 0);
  const compactions = Array.isArray(record.compactions)
    ? record.compactions.map(normalizeCompactionRecord)
    : [];
  return { tokens, budget, ratio, compactions };
}

export function normalizeUsage(raw: unknown): UsageInfo {
  const record = isRecord(raw) ? raw : {};
  return {
    prompt_tokens: asNumber(record.prompt_tokens),
    completion_tokens: asNumber(record.completion_tokens),
    reasoning_tokens: asNumber(record.reasoning_tokens),
    total_tokens: asNumber(record.total_tokens),
    model: asString(record.model),
    purpose: asString(record.purpose),
  };
}

export function normalizeSessionView(raw: unknown): SessionView {
  const record = isRecord(raw) ? raw : {};
  const metaSource = isRecord(record.meta) ? record.meta : record;
  const meta = normalizeSessionSummary(metaSource);
  const status = firstString(record.status, meta.status) ?? 'idle';
  return {
    meta: { ...meta, status },
    messages: normalizeMessages(record.messages),
    todos: normalizeTodos(record.todos),
    agents: normalizeAgents(record.agents),
    context: normalizeContext(record.context),
    skills: normalizeSkills(record.skills),
    last_seq: asNumber(record.last_seq) ?? 0,
    status,
  };
}

export function normalizeToolSummaries(raw: unknown): ToolRunSummary[] | undefined {
  if (!Array.isArray(raw)) {
    return undefined;
  }
  const summaries: ToolRunSummary[] = [];
  for (const item of raw) {
    if (!isRecord(item)) {
      continue;
    }
    const name = firstString(item.name, item.tool);
    if (name === undefined) {
      continue;
    }
    const summary: ToolRunSummary = { name };
    const ok = asBoolean(item.ok);
    if (ok !== undefined) {
      summary.ok = ok;
    }
    const duration = asNumber(item.duration_ms);
    if (duration !== undefined) {
      summary.duration_ms = duration;
    }
    summaries.push(summary);
  }
  return summaries.length > 0 ? summaries : undefined;
}

export function normalizeEvent(raw: unknown, fallbackType?: string): AgentEvent | null {
  if (!isRecord(raw)) {
    return null;
  }
  const type = firstString(raw.type, raw.event, fallbackType);
  const seq = asNumber(raw.seq) ?? asNumber(raw.id);
  if (type === undefined || seq === undefined) {
    return null;
  }
  return {
    seq,
    ts: firstString(raw.ts, raw.timestamp),
    session_id: asString(raw.session_id),
    type,
    data: isRecord(raw.data) ? raw.data : raw,
  };
}
