import { useCopy } from '../hooks/useCopy';
import { classNames } from '../utils';

interface CopyButtonProps {
  text: string;
  label?: string;
  className?: string;
}

export function CopyButton({ text, label = '复制', className }: CopyButtonProps) {
  const { copied, copy } = useCopy();
  return (
    <button
      type="button"
      className={classNames('copy-button', className)}
      onClick={() => copy(text)}
      title="复制到剪贴板"
      aria-label={copied ? '已复制' : label}
    >
      {copied ? '已复制' : label}
    </button>
  );
}
