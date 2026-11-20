import { useMemo } from 'react';
import type { SessionSummary } from '../types';
import { classNames, formatRelativeTime, shortId } from '../utils';
import { StatusDot, statusTone } from './StatusDot';

interface SessionNode {
  session: SessionSummary;
  children: SessionNode[];
}

function timeValue(session: SessionSummary): number {
  const parsed = Date.parse(session.updated_at);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function compareSessions(left: SessionSummary, right: SessionSummary): number {
  const diff = timeValue(right) - timeValue(left);
  if (diff !== 0) {
    return diff;
  }
  return left.id.localeCompare(right.id);
}

function hasCycle(session: SessionSummary, nodes: Map<string, SessionNode>): boolean {
  const seen = new Set<string>([session.id]);
  let current: SessionSummary | undefined = session;
  while (current !== undefined && current.parent_id !== null && current.parent_id !== '') {
    if (seen.has(current.parent_id)) {
      return true;
    }
    seen.add(current.parent_id);
    current = nodes.get(current.parent_id)?.session;
  }
  return false;
}

function buildTree(sessions: SessionSummary[]): SessionNode[] {
  const nodes = new Map<string, SessionNode>();
  for (const session of sessions) {
    nodes.set(session.id, { session, children: [] });
  }
  const roots: SessionNode[] = [];
  for (const session of sessions) {
    const node = nodes.get(session.id);
    if (node === undefined) {
      continue;
    }
    const parent =
      session.parent_id !== null && !hasCycle(session, nodes)
        ? nodes.get(session.parent_id)
        : undefined;
    if (parent !== undefined && parent !== node) {
      parent.children.push(node);
    } else {
      roots.push(node);
    }
  }
  const sortNodes = (list: SessionNode[]) => {
    list.sort((left, right) => compareSessions(left.session, right.session));
    for (const node of list) {
      sortNodes(node.children);
    }
  };
  sortNodes(roots);
  return roots;
}

interface SessionRowProps {
  node: SessionNode;
  depth: number;
  currentId: string | null;
  onSelect: (id: string) => void;
}

function SessionRow({ node, depth, currentId, onSelect }: SessionRowProps) {
  const { session } = node;
  const active = session.id === currentId;
  return (
    <li className="session-item">
      <button
        type="button"
        className={classNames('session-row', active && 'is-active')}
        style={{ paddingLeft: 10 + Math.min(depth, 4) * 16 }}
        onClick={() => onSelect(session.id)}
        title={session.id}
        aria-current={active ? 'page' : undefined}
      >
        <StatusDot status={session.status} />
        <span className="session-main">
          <span className="session-title">{session.title || '未命名会话'}</span>
          <span className="session-sub">
            {session.parent_id !== null ? '子 agent · ' : ''}
            {formatRelativeTime(session.updated_at) || shortId(session.id)}
          </span>
        </span>
        {statusTone(session.status) === 'running' && (
          <span className="session-running">运行中</span>
        )}
      </button>
      {node.children.length > 0 && (
        <ul className="session-children">
          {node.children.map((child) => (
            <SessionRow
              key={child.session.id}
              node={child}
              depth={depth + 1}
              currentId={currentId}
              onSelect={onSelect}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

interface SessionListProps {
  sessions: SessionSummary[];
  currentId: string | null;
  loading: boolean;
  creating: boolean;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRefresh: () => void;
}

export function SessionList({
  sessions,
  currentId,
  loading,
  creating,
  onSelect,
  onCreate,
  onRefresh,
}: SessionListProps) {
  const tree = useMemo(() => buildTree(sessions), [sessions]);

  return (
    <aside className="sidebar" aria-label="会话列表">
      <div className="sidebar-head">
        <div className="sidebar-title">
          <h2>会话</h2>
          <span className="count">{sessions.length}</span>
        </div>
        <div className="sidebar-actions">
          <button
            type="button"
            className="icon-button"
            onClick={onRefresh}
            title="刷新会话列表"
            aria-label="刷新会话列表"
          >
            ↻
          </button>
          <button type="button" className="new-session" onClick={onCreate} disabled={creating}>
            {creating ? '创建中…' : '＋ 新建'}
          </button>
        </div>
      </div>
      <div className="sidebar-scroll">
        {loading ? (
          <div className="skeleton-list" aria-label="正在加载会话">
            <div className="skeleton-row" />
            <div className="skeleton-row" />
            <div className="skeleton-row" />
          </div>
        ) : tree.length === 0 ? (
          <div className="sidebar-empty">
            <p>还没有会话</p>
            <button type="button" className="ghost-button" onClick={onCreate} disabled={creating}>
              新建第一个会话
            </button>
          </div>
        ) : (
          <ul className="session-list">
            {tree.map((node) => (
              <SessionRow
                key={node.session.id}
                node={node}
                depth={0}
                currentId={currentId}
                onSelect={onSelect}
              />
            ))}
          </ul>
        )}
      </div>
      <div className="sidebar-foot">
        <span>ReAct · run_code 桥 · 三级压缩</span>
      </div>
    </aside>
  );
}
