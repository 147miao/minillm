import type { AgentInfo } from '../types';
import { shortId, truncate } from '../utils';
import { StatusDot, statusLabel } from './StatusDot';

interface AgentNode {
  agent: AgentInfo;
  children: AgentNode[];
}

function buildAgentTree(agents: AgentInfo[], currentSessionId: string): AgentNode[] {
  const nodes = new Map<string, AgentNode>();
  for (const agent of agents) {
    nodes.set(agent.id, { agent, children: [] });
  }
  const roots: AgentNode[] = [];
  for (const agent of agents) {
    const node = nodes.get(agent.id);
    if (node === undefined) {
      continue;
    }
    let parentNode: AgentNode | undefined;
    const parentSession = agent.parent_session_id ?? null;
    if (parentSession !== null && parentSession !== '' && parentSession !== currentSessionId) {
      for (const candidate of nodes.values()) {
        if (
          candidate.agent.id === parentSession ||
          candidate.agent.session_id === parentSession
        ) {
          parentNode = candidate;
          break;
        }
      }
    }
    if (parentNode !== undefined && parentNode !== node) {
      parentNode.children.push(node);
    } else {
      roots.push(node);
    }
  }
  return roots.length > 0 ? roots : agents.map((agent) => ({ agent, children: [] }));
}

interface AgentRowProps {
  node: AgentNode;
  depth: number;
  currentSessionId: string;
  onOpenSession: (sessionId: string) => void;
}

function AgentRow({ node, depth, currentSessionId, onOpenSession }: AgentRowProps) {
  if (depth > 3) {
    return null;
  }
  const { agent } = node;
  const target = agent.session_id ?? agent.id;
  const isCurrent = target === currentSessionId;
  const title = agent.task ?? agent.last_message ?? agent.id;
  return (
    <li className="agent-item">
      <div className="agent-row">
        <StatusDot status={agent.status} />
        <div className="agent-main">
          <button
            type="button"
            className="agent-task"
            onClick={() => onOpenSession(target)}
            disabled={isCurrent}
            title={isCurrent ? '当前会话' : '打开子会话 ' + target}
          >
            {truncate(title, 80)}
          </button>
          <span className="agent-meta">
            {statusLabel(agent.status)} · {shortId(target)}
          </span>
          {agent.result_preview !== undefined && agent.result_preview !== '' && (
            <span className="agent-result">{truncate(agent.result_preview, 140)}</span>
          )}
        </div>
      </div>
      {node.children.length > 0 && (
        <ul className="agent-children">
          {node.children.map((child) => (
            <AgentRow
              key={child.agent.id}
              node={child}
              depth={depth + 1}
              currentSessionId={currentSessionId}
              onOpenSession={onOpenSession}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

interface SubagentTreeProps {
  agents: AgentInfo[];
  currentSessionId: string;
  onOpenSession: (sessionId: string) => void;
}

export function SubagentTree({ agents, currentSessionId, onOpenSession }: SubagentTreeProps) {
  const roots = buildAgentTree(agents, currentSessionId);
  return (
    <section className="panel-card">
      <header className="panel-head">
        <h3>子 Agent</h3>
        <span className="count">{agents.length}</span>
      </header>
      {agents.length === 0 ? (
        <p className="muted">暂无子 agent。主 agent 调用 spawn_subagent 后会出现在这里。</p>
      ) : (
        <ul className="agent-list">
          {roots.map((node) => (
            <AgentRow
              key={node.agent.id}
              node={node}
              depth={0}
              currentSessionId={currentSessionId}
              onOpenSession={onOpenSession}
            />
          ))}
        </ul>
      )}
    </section>
  );
}
