export interface ToolCallRef {
  id: string;
  name: string;
  arguments: string;
}

export interface ChatMessage {
  role: string;
  content: string;
  reasoning?: string;
  tool_calls?: ToolCallRef[];
  tool_call_id?: string;
  name?: string;
}

export interface SessionSummary {
  id: string;
  title: string;
  workspace: string;
  model: string;
  parent_id: string | null;
  created_at: string;
  updated_at: string;
  status: string;
}

export interface SessionMeta extends SessionSummary {}

export interface TodoItem {
  content: string;
  status: string;
  id?: string;
}

export interface AgentInfo {
  id: string;
  session_id?: string;
  parent_session_id?: string | null;
  task?: string;
  description?: string;
  status: string;
  result_preview?: string;
  last_message?: string;
}

export interface CompactionRecord {
  level: number;
  before_tokens: number;
  after_tokens: number;
  summary?: string;
  fallback?: string | boolean;
  pruned_upto_seq?: number;
  ts?: string;
  seq?: number;
}

export interface ContextState {
  tokens: number;
  budget: number;
  ratio: number;
  compactions: CompactionRecord[];
}

export interface SkillInfo {
  name: string;
  description?: string;
  path?: string;
}

export interface SessionView {
  meta: SessionMeta;
  messages: ChatMessage[];
  todos: TodoItem[];
  agents: AgentInfo[];
  context: ContextState;
  skills: SkillInfo[];
  last_seq: number;
  status: string;
}

export interface HealthInfo {
  status: string;
  version: string;
  model: string;
}

export interface ConfigInfo {
  model: string;
  small_model: string;
  context_budget: number;
  prune_threshold: number;
  compact_threshold: number;
  workspace_root: string;
  skills: SkillInfo[];
  mcp_servers: string[];
  version: string;
}

export interface UsageInfo {
  prompt_tokens?: number;
  completion_tokens?: number;
  reasoning_tokens?: number;
  total_tokens?: number;
  model?: string;
  purpose?: string;
}

export interface AgentEvent {
  seq: number;
  ts?: string;
  session_id?: string;
  type: string;
  data: Record<string, unknown>;
}

export interface ToolRunSummary {
  name: string;
  ok?: boolean;
  duration_ms?: number;
}

export interface ToolErrorPayload {
  kind?: string;
  tool?: string;
  message?: string;
  errors?: unknown;
}

export interface RunCodeDetail {
  callId: string;
  code?: string;
  exitCode?: number | null;
  stdout?: string;
  stderr?: string;
  artifact?: string;
  durationMs?: number;
  started: boolean;
  finished: boolean;
  toolSummaries?: ToolRunSummary[];
}

export interface ToolCallDetail {
  callId: string;
  runCallId?: string;
  tool: string;
  argsPreview?: string;
  argsArtifact?: string;
  ok?: boolean;
  preview?: string;
  artifact?: string;
  error?: ToolErrorPayload;
  durationMs?: number;
  pending: boolean;
}

export interface ErrorEntry {
  seq: number;
  ts?: string;
  message: string;
  where?: string;
  traceback?: string;
}

export interface EventDetails {
  lastSeq: number;
  runs: Record<string, RunCodeDetail>;
  tools: Record<string, ToolCallDetail>;
  compactions: CompactionRecord[];
  errors: ErrorEntry[];
  skills: SkillInfo[];
  usage: UsageInfo | null;
}

export interface StreamBuffer {
  text: string;
  reasoning: string;
}

export interface SendMessageResult {
  turn_id?: string;
}
