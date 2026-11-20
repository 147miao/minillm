export interface ToastItem {
  id: number;
  text: string;
  tone: 'info' | 'error';
}

interface ToastsProps {
  items: ToastItem[];
  onDismiss: (id: number) => void;
}

export function Toasts({ items, onDismiss }: ToastsProps) {
  return (
    <div className="toasts" aria-live="polite">
      {items.map((item) => (
        <div
          key={item.id}
          className={'toast toast-' + item.tone}
          role={item.tone === 'error' ? 'alert' : 'status'}
        >
          <span>{item.text}</span>
          <button
            type="button"
            className="toast-close"
            onClick={() => onDismiss(item.id)}
            aria-label="关闭提示"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
