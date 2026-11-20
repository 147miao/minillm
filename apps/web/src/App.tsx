import { useCallback, useEffect, useRef, useState } from 'react';
import * as api from './api';
import { Composer } from './components/Composer';
import { EmptyState } from './components/EmptyState';
import { Inspector } from './components/Inspector';
import { SessionList } from './components/SessionList';
import { Transcript } from './components/Transcript';
import { Toasts } from './components/Toasts';
import type { ToastItem } from './components/Toasts';
import {
  appendStreamDelta,
  applyViewEvent,
  clearStream,
  createEmptyDetails,
  detailsFromEvents,
  reduceDetails,
} from './events';
import { useSessionStream } from './hooks/useSessionStream';
import type {
  AgentEvent,
  ConfigInfo,
  EventDetails,
  HealthInfo,
  SessionSummary,
  SessionView,
  StreamBuffer,
} from './types';
import { classNames, errorText } from './utils';

const SESSION_REFRESH_EVENTS = new Set([
  'turn.finished',
  'turn.cancelled',
  'subagent.spawned',
  'subagent.completed',
  'session.created',
]);

const EMPTY_STREAM: StreamBuffer = { text: '', reasoning: '' };

function readSessionParam(): string | null {
  const params = new URLSearchParams(window.location.search);
  const value = params.get('session');
  if (value === null || value.trim() === '') {
    return null;
  }
  return value.trim();
}

function setSessionParam(sessionId: string | null) {
  const url = new URL(window.location.href);
  if (sessionId === null) {
    url.searchParams.delete('session');
  } else {
    url.searchParams.set('session', sessionId);
  }
  window.history.replaceState(null, '', url.toString());
}

export function App() {
  const [config, setConfig] = useState<ConfigInfo | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [online, setOnline] = useState(true);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [activeId, setActiveId] = useState<string | null>(() => readSessionParam());
  const [view, setView] = useState<SessionView | null>(null);
  const [details, setDetails] = useState<EventDetails>(createEmptyDetails);
  const [stream, setStream] = useState<StreamBuffer>(EMPTY_STREAM);
  const [loadedId, setLoadedId] = useState<string | null>(null);
  const [loadingSession, setLoadingSession] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [reloadNonce, setReloadNonce] = useState(0);
  const [creating, setCreating] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [sending, setSending] = useState(false);
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);

  const seqRef = useRef(0);
  const liveEventsRef = useRef<AgentEvent[]>([]);
  const activeIdRef = useRef<string | null>(activeId);
  activeIdRef.current = activeId;
  const refreshTimerRef = useRef<number | null>(null);

  const notify = useCallback((text: string, tone: 'info' | 'error' = 'info') => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { id, text, tone }]);
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== id));
    }, 4500);
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      const next = await api.fetchHealth();
      setHealth(next);
      setOnline(true);
    } catch {
      setOnline(false);
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const list = await api.fetchSessions();
      setSessions(list);
      setOnline(true);
    } catch {
      setOnline(false);
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  const scheduleSessionsRefresh = useCallback(() => {
    if (refreshTimerRef.current !== null) {
      return;
    }
    refreshTimerRef.current = window.setTimeout(() => {
      refreshTimerRef.current = null;
      void refreshSessions();
    }, 600);
  }, [refreshSessions]);

  const handleEvent = useCallback(
    (event: AgentEvent) => {
      if (event.type === 'assistant.delta' || event.type === 'reasoning.delta') {
        setStream((prev) => appendStreamDelta(prev, event));
        return;
      }
      setDetails((prev) => reduceDetails(prev, event));
      if (event.seq <= seqRef.current) {
        return;
      }
      seqRef.current = event.seq;
      liveEventsRef.current = [...liveEventsRef.current, event].slice(-200);
      if (
        event.type === 'turn.started' ||
        event.type === 'assistant.message' ||
        event.type === 'turn.finished' ||
        event.type === 'turn.cancelled' ||
        event.type === 'agent.error'
      ) {
        setStream(clearStream());
      }
      setView((prev) => (prev === null ? prev : applyViewEvent(prev, event)));
      if (SESSION_REFRESH_EVENTS.has(event.type)) {
        scheduleSessionsRefresh();
      }
      if (event.type === 'agent.error') {
        const message = event.data.message;
        notify(typeof message === 'string' ? message : '本轮执行出错', 'error');
      }
    },
    [notify, scheduleSessionsRefresh],
  );

  const streamStatus = useSessionStream(
    loadedId !== null && loadedId === activeId ? loadedId : null,
    seqRef,
    handleEvent,
  );

  useEffect(() => {
    let cancelled = false;
    seqRef.current = 0;
    setLoadedId(null);
    setView(null);
    setDetails(createEmptyDetails());
    setStream(EMPTY_STREAM);
    setLoadError(null);
    if (activeId === null) {
      setLoadingSession(false);
      return;
    }
    setLoadingSession(true);
    void (async () => {
      try {
        const [nextView, events] = await Promise.all([
          api.fetchSession(activeId),
          api.fetchEvents(activeId),
        ]);
        if (cancelled) {
          return;
        }
        let patched = nextView;
        let maxSeq = nextView.last_seq;
        for (const event of events) {
          maxSeq = Math.max(maxSeq, event.seq);
          if (event.seq > patched.last_seq) {
            patched = applyViewEvent(patched, event);
          }
        }
        seqRef.current = maxSeq;
        setView(patched);
        setDetails(detailsFromEvents(events));
        setLoadedId(activeId);
        setOnline(true);
        setLoadingSession(false);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setLoadingSession(false);
        setLoadError(errorText(error));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeId, reloadNonce]);

  useEffect(() => {
    void refreshHealth();
    void api
      .fetchConfig()
      .then(setConfig)
      .catch(() => {
        setOnline(false);
      });
    void refreshSessions();
    const interval = window.setInterval(() => {
      void refreshHealth();
    }, 20000);
    const onFocus = () => {
      void refreshHealth();
      void refreshSessions();
    };
    window.addEventListener('focus', onFocus);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener('focus', onFocus);
    };
  }, [refreshHealth, refreshSessions]);

  useEffect(() => {
    const onPopState = () => {
      setActiveId(readSessionParam());
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setSidebarOpen(false);
        setInspectorOpen(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  useEffect(() => {
    return () => {
      if (refreshTimerRef.current !== null) {
        window.clearTimeout(refreshTimerRef.current);
      }
    };
  }, []);

  const selectSession = useCallback((id: string) => {
    setActiveId(id);
    setSidebarOpen(false);
    setInspectorOpen(false);
    setSessionParam(id);
  }, []);

  const handleCreate = useCallback(async () => {
    setCreating(true);
    try {
      const session = await api.createSession();
      setSessions((prev) => [session, ...prev.filter((item) => item.id !== session.id)]);
      selectSession(session.id);
      setOnline(true);
      void refreshSessions();
    } catch (error) {
      notify('新建会话失败：' + errorText(error), 'error');
    } finally {
      setCreating(false);
    }
  }, [notify, refreshSessions, selectSession]);

  const handleSend = useCallback(
    async (content: string): Promise<boolean> => {
      const id = activeIdRef.current;
      if (id === null) {
        return false;
      }
      setSending(true);
      try {
        await api.sendMessage(id, content);
        setView((prev) => (prev === null ? prev : { ...prev, status: 'running' }));
        setStream(clearStream());
        setOnline(true);
        return true;
      } catch (error) {
        if (error instanceof api.ApiError && error.status === 409) {
          notify('当前轮次仍在运行，请先停止或等待完成', 'error');
          setView((prev) => (prev === null ? prev : { ...prev, status: 'running' }));
        } else {
          notify('发送失败：' + errorText(error), 'error');
        }
        return false;
      } finally {
        setSending(false);
      }
    },
    [notify],
  );

  const handleStop = useCallback(async () => {
    const id = activeIdRef.current;
    if (id === null) {
      return;
    }
    try {
      await api.stopSession(id);
      notify('已请求停止当前轮次', 'info');
    } catch (error) {
      notify('停止失败：' + errorText(error), 'error');
    }
  }, [notify]);

  const handleSync = useCallback(async () => {
    const id = activeIdRef.current;
    if (id === null) {
      return;
    }
    setSyncing(true);
    try {
      const [nextView, events] = await Promise.all([
        api.fetchSession(id),
        api.fetchEvents(id),
      ]);
      if (activeIdRef.current !== id) {
        return;
      }
      let patched = nextView;
      let maxSeq = nextView.last_seq;
      for (const event of events) {
        maxSeq = Math.max(maxSeq, event.seq);
        if (event.seq > patched.last_seq) {
          patched = applyViewEvent(patched, event);
        }
      }
      let nextDetails = detailsFromEvents(events);
      for (const event of liveEventsRef.current) {
        maxSeq = Math.max(maxSeq, event.seq);
        if (event.seq > patched.last_seq) {
          patched = applyViewEvent(patched, event);
        }
        nextDetails = reduceDetails(nextDetails, event);
      }
      seqRef.current = Math.max(seqRef.current, maxSeq);
      setView(patched);
      setDetails(nextDetails);
      if (patched.status !== 'running') {
        setStream(clearStream());
      }
      setLoadedId(id);
      setOnline(true);
    } catch (error) {
      notify('同步失败：' + errorText(error), 'error');
    } finally {
      setSyncing(false);
    }
  }, [notify]);

  const handleRetry = useCallback(() => {
    void refreshHealth();
    void api
      .fetchConfig()
      .then(setConfig)
      .catch(() => undefined);
    void refreshSessions();
    setReloadNonce((value) => value + 1);
  }, [refreshHealth, refreshSessions]);

  const handleClearSession = useCallback(() => {
    setActiveId(null);
    setView(null);
    setLoadedId(null);
    setLoadError(null);
    setSessionParam(null);
  }, []);

  const activeView = view !== null && loadedId === activeId ? view : null;
  const sessionPending =
    activeId !== null && loadedId !== activeId && loadError === null;
  const showLoading = loadingSession || sessionPending;
  const running = activeView !== null && activeView.status === 'running';

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">M</span>
          <span className="brand-name">MiniLLM</span>
          <span className="brand-sub">Coding Agent</span>
        </div>
        <div className="topbar-actions">
          <button
            type="button"
            className="icon-button mobile-only"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="打开会话列表"
            aria-expanded={sidebarOpen}
          >
            ☰
          </button>
          <span className="model-chip" title="主模型">
            {config?.model !== undefined && config.model !== ''
              ? config.model
              : health?.model ?? '未连接'}
          </span>
          <button
            type="button"
            className={classNames('health-chip', online ? 'is-online' : 'is-offline')}
            onClick={() => {
              void refreshHealth();
            }}
            title={online ? '后端连接正常，点击重新检查' : '后端未连接，点击重试'}
          >
            <span className="health-dot" aria-hidden="true" />
            {online ? '已连接' : '后端离线'}
          </button>
          <button
            type="button"
            className="icon-button mobile-only"
            onClick={() => setInspectorOpen(!inspectorOpen)}
            aria-label="打开运行面板"
            aria-expanded={inspectorOpen}
          >
            ▣
          </button>
        </div>
      </header>
      <div className="app-body">
        <div className={classNames('sidebar-wrap', sidebarOpen && 'is-open')}>
          <SessionList
            sessions={sessions}
            currentId={activeId}
            loading={sessionsLoading}
            creating={creating}
            onSelect={selectSession}
            onCreate={() => {
              void handleCreate();
            }}
            onRefresh={() => {
              void refreshSessions();
            }}
          />
        </div>
        <main className="main-wrap">
          {showLoading ? (
            <div className="loading-state">
              <span className="spinner" aria-hidden="true" />
              <p>正在载入会话…</p>
            </div>
          ) : activeView !== null ? (
            <>
              <Transcript
                view={activeView}
                details={details}
                stream={stream}
                running={running}
                streamStatus={streamStatus}
                syncing={syncing}
                onSync={() => {
                  void handleSync();
                }}
              />
              <Composer
                running={running}
                sending={sending}
                status={activeView.status}
                onSend={handleSend}
                onStop={() => {
                  void handleStop();
                }}
              />
            </>
          ) : (
            <EmptyState
              loading={sessionsLoading && activeId === null}
              offline={!online}
              error={loadError}
              creating={creating}
              onCreate={() => {
                void handleCreate();
              }}
              onRetry={handleRetry}
              onClear={handleClearSession}
            />
          )}
        </main>
        <div className={classNames('inspector-wrap', inspectorOpen && 'is-open')}>
          <Inspector
            view={activeView}
            details={details}
            config={config}
            online={online}
            onClose={() => setInspectorOpen(false)}
            onOpenSession={selectSession}
          />
        </div>
        {(sidebarOpen || inspectorOpen) && (
          <button
            type="button"
            className="drawer-backdrop"
            aria-label="关闭面板"
            onClick={() => {
              setSidebarOpen(false);
              setInspectorOpen(false);
            }}
          />
        )}
      </div>
      <Toasts
        items={toasts}
        onDismiss={(id) => {
          setToasts((prev) => prev.filter((toast) => toast.id !== id));
        }}
      />
    </div>
  );
}
