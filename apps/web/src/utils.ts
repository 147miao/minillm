export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function asString(value: unknown): string | undefined {
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  return undefined;
}

export function asNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

export function asBoolean(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') {
    return value;
  }
  if (value === 'true') {
    return true;
  }
  if (value === 'false') {
    return false;
  }
  return undefined;
}

export function firstString(...values: unknown[]): string | undefined {
  for (const value of values) {
    const text = asString(value);
    if (text !== undefined && text !== '') {
      return text;
    }
  }
  return undefined;
}

export function classNames(...parts: Array<string | false | null | undefined>): string {
  return parts.filter((part): part is string => Boolean(part)).join(' ');
}

export function formatTokens(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) {
    return '—';
  }
  if (value >= 1000000) {
    return (value / 1000000).toFixed(value >= 10000000 ? 0 : 1) + 'M';
  }
  if (value >= 1000) {
    return (value / 1000).toFixed(value >= 10000 ? 0 : 1) + 'k';
  }
  return String(Math.round(value));
}

export function formatRatio(ratio: number): string {
  if (!Number.isFinite(ratio)) {
    return '0%';
  }
  return Math.round(Math.max(0, Math.min(1, ratio)) * 100) + '%';
}

export function formatDuration(ms: number | undefined): string {
  if (ms === undefined || !Number.isFinite(ms) || ms < 0) {
    return '';
  }
  if (ms < 1000) {
    return Math.round(ms) + 'ms';
  }
  if (ms < 60000) {
    return (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + 's';
  }
  return Math.floor(ms / 60000) + 'm ' + Math.round((ms % 60000) / 1000) + 's';
}

export function formatClock(ts: string | undefined): string {
  if (!ts) {
    return '';
  }
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) {
    return '';
  }
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

export function formatRelativeTime(ts: string | undefined): string {
  if (!ts) {
    return '';
  }
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) {
    return '';
  }
  const diff = Date.now() - date.getTime();
  if (diff < 60000) {
    return '刚刚';
  }
  if (diff < 3600000) {
    return Math.floor(diff / 60000) + ' 分钟前';
  }
  if (diff < 86400000) {
    return Math.floor(diff / 3600000) + ' 小时前';
  }
  if (diff < 7 * 86400000) {
    return Math.floor(diff / 86400000) + ' 天前';
  }
  return date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' });
}

export function truncate(text: string, max: number): string {
  const normalized = text.trim();
  if (normalized.length <= max) {
    return normalized;
  }
  return normalized.slice(0, Math.max(0, max - 1)) + '…';
}

export function parseJsonObject(text: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(text);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function stringifyValue(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  if (value === undefined || value === null) {
    return '';
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function shortId(id: string): string {
  return id.length > 10 ? id.slice(0, 8) : id;
}

export async function copyText(text: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', 'true');
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  try {
    document.execCommand('copy');
  } finally {
    document.body.removeChild(textarea);
  }
}

export function errorText(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return '未知错误';
}
