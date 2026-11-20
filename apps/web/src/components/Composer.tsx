import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { classNames } from '../utils';

interface ComposerProps {
  running: boolean;
  sending: boolean;
  status: string;
  onSend: (content: string) => Promise<boolean>;
  onStop: () => void;
}

export function Composer({ running, sending, status, onSend, onStop }: ComposerProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const element = textareaRef.current;
    if (element === null) {
      return;
    }
    element.style.height = 'auto';
    element.style.height = Math.min(element.scrollHeight, 200) + 'px';
  }, [value]);

  const submit = async () => {
    const content = value.trim();
    if (content === '' || running || sending) {
      return;
    }
    const accepted = await onSend(content);
    if (accepted) {
      setValue('');
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submit();
    }
  };

  const hint = running
    ? 'MiniLLM 正在执行，可点击停止中断本轮'
    : status === 'cancelled'
      ? '上一轮已停止，可以继续输入'
      : sending
        ? '正在提交…'
        : 'Enter 发送 · Shift+Enter 换行';

  return (
    <div className="composer">
      <div className={classNames('composer-box', running && 'is-running')}>
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          disabled={running || sending}
          placeholder={running ? '当前轮次运行中…' : '描述任务，例如：为 greet.py 写单元测试并运行'}
          aria-label="消息输入框"
        />
        {running ? (
          <button type="button" className="stop-button" onClick={onStop}>
            ■ 停止
          </button>
        ) : (
          <button
            type="button"
            className="send-button"
            onClick={() => {
              void submit();
            }}
            disabled={sending || value.trim() === ''}
          >
            {sending ? '发送中' : '发送'}
          </button>
        )}
      </div>
      <div className="composer-hint">
        <span>{hint}</span>
        <span className="composer-count">{value.length > 0 ? value.length + ' 字' : ''}</span>
      </div>
    </div>
  );
}
