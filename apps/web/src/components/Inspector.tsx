import { compactionRecords } from '../events';
import type { CompactionRecord, ConfigInfo, EventDetails, SessionView, SkillInfo } from '../types';
import { formatClock, formatTokens, shortId } from '../utils';
import { compactionLevelLabel } from './CompactionBanner';
import { ContextMeter } from './ContextMeter';
import { SubagentTree } from './SubagentTree';
import { TodoPanel } from './TodoPanel';

function compactionKey(record: CompactionRecord, index: number): string {
  return (
    'history-' +
    String(index) +
    '-' +
    String(record.seq ?? '') +
    '-' +
    String(record.level)
  );
}

function CompactionHistory({ records }: { records: CompactionRecord[] }) {
  const recent = records.slice(-6).reverse();
  return (
    <section className="panel-card">
      <header className="panel-head">
        <h3>压缩记录</h3>
        <span className="count">{records.length}</span>
      </header>
      {recent.length === 0 ? (
        <p className="muted">尚未触发压缩。</p>
      ) : (
        <ul className="compaction-list">
          {recent.map((record, index) => (
            <li key={compactionKey(record, index)}>
              <details>
                <summary className="compaction-mini-head">
                  <span className="compaction-level">L{record.level}</span>
                  <span>{compactionLevelLabel(record.level)}</span>
                  <span className="muted">
                    {formatTokens(record.before_tokens)} → {formatTokens(record.after_tokens)}
                  </span>
                  {record.ts !== undefined && (
                    <span className="muted">{formatClock(record.ts)}</span>
                  )}
                </summary>
                <p className="compaction-mini-body">
                  {record.summary !== undefined && record.summary !== ''
                    ? record.summary
                    : '（无摘要）'}
                </p>
              </details>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function SkillsPanel({ skills }: { skills: SkillInfo[] }) {
  return (
    <section className="panel-card">
      <header className="panel-head">
        <h3>已注入技能</h3>
        <span className="count">{skills.length}</span>
      </header>
      {skills.length === 0 ? (
        <p className="muted">本轮未注入技能。</p>
      ) : (
        <ul className="skill-list">
          {skills.map((skill) => (
            <li key={skill.name}>
              <span className="skill-name">{skill.name}</span>
              {skill.description !== undefined && skill.description !== '' && (
                <span className="skill-desc">{skill.description}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function ConfigPanel({ config, online }: { config: ConfigInfo | null; online: boolean }) {
  const rows: Array<[string, string]> = config === null
    ? []
    : [
        ['主模型', config.model],
        ['小模型', config.small_model],
        ['上下文预算', formatTokens(config.context_budget) + ' tokens'],
        ['工作目录', config.workspace_root],
        ['MCP', config.mcp_servers.length > 0 ? config.mcp_servers.join('、') : '未配置'],
        ['版本', config.version],
      ];
  return (
    <section className="panel-card config-card">
      <header className="panel-head">
        <h3>配置</h3>
        <span className={online ? 'config-health is-online' : 'config-health is-offline'}>
          {online ? '已连接' : '离线'}
        </span>
      </header>
      {config === null ? (
        <p className="muted">配置尚未加载。</p>
      ) : (
        <dl className="config-list">
          {rows.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd title={value}>{value !== '' ? value : '—'}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}

interface InspectorProps {
  view: SessionView | null;
  details: EventDetails;
  config: ConfigInfo | null;
  online: boolean;
  onClose: () => void;
  onOpenSession: (sessionId: string) => void;
}

export function Inspector({
  view,
  details,
  config,
  online,
  onClose,
  onOpenSession,
}: InspectorProps) {
  const compactions = view !== null ? compactionRecords(view, details) : [];
  return (
    <aside className="inspector" aria-label="运行面板">
      <div className="inspector-head">
        <h2>运行面板</h2>
        {view !== null && (
          <span className="inspector-session" title={view.meta.id}>
            {shortId(view.meta.id)}
          </span>
        )}
        <button
          type="button"
          className="icon-button inspector-close"
          onClick={onClose}
          aria-label="关闭运行面板"
        >
          ×
        </button>
      </div>
      <div className="inspector-scroll">
        {view === null ? (
          <p className="muted panel-empty">
            选择或新建会话后，这里显示上下文占用、压缩记录、任务清单与子 agent 树。
          </p>
        ) : (
          <>
            <ContextMeter
              context={view.context}
              pruneThreshold={config?.prune_threshold ?? 0.5}
              compactThreshold={config?.compact_threshold ?? 0.8}
            />
            <CompactionHistory records={compactions} />
            <TodoPanel todos={view.todos} />
            <SubagentTree
              agents={view.agents}
              currentSessionId={view.meta.id}
              onOpenSession={onOpenSession}
            />
            <SkillsPanel skills={view.skills.length > 0 ? view.skills : config?.skills ?? []} />
          </>
        )}
        <ConfigPanel config={config} online={online} />
      </div>
    </aside>
  );
}
