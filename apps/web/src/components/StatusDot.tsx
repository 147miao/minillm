export type StatusTone = 'running' | 'idle' | 'error' | 'cancelled' | 'unknown';

const TONE_BY_STATUS: Record<string, StatusTone> = {
  running: 'running',
  working: 'running',
  started: 'running',
  in_progress: 'running',
  queued: 'running',
  idle: 'idle',
  ready: 'idle',
  done: 'idle',
  completed: 'idle',
  success: 'idle',
  error: 'error',
  failed: 'error',
  failure: 'error',
  cancelled: 'cancelled',
  canceled: 'cancelled',
  stopped: 'cancelled',
};

const LABEL_BY_STATUS: Record<string, string> = {
  running: '运行中',
  working: '运行中',
  started: '运行中',
  in_progress: '进行中',
  queued: '排队中',
  idle: '空闲',
  ready: '就绪',
  done: '已完成',
  completed: '已完成',
  success: '成功',
  error: '错误',
  failed: '失败',
  cancelled: '已停止',
  canceled: '已停止',
  stopped: '已停止',
};

export function statusTone(status: string): StatusTone {
  return TONE_BY_STATUS[status] ?? 'unknown';
}

export function statusLabel(status: string): string {
  return LABEL_BY_STATUS[status] ?? (status === '' ? '未知' : status);
}

interface StatusDotProps {
  status: string;
  title?: string;
}

export function StatusDot({ status, title }: StatusDotProps) {
  const tone = statusTone(status);
  const label = title ?? statusLabel(status);
  return (
    <span
      className={'status-dot status-dot-' + tone}
      title={label}
      aria-label={label}
      role="img"
    />
  );
}
