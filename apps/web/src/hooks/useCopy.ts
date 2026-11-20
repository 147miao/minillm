import { useCallback, useEffect, useRef, useState } from 'react';
import { copyText } from '../utils';

export interface CopyState {
  copied: boolean;
  copy: (text: string) => void;
}

export function useCopy(): CopyState {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, []);

  const copy = useCallback((text: string) => {
    void copyText(text)
      .then(() => {
        setCopied(true);
        if (timerRef.current !== null) {
          window.clearTimeout(timerRef.current);
        }
        timerRef.current = window.setTimeout(() => setCopied(false), 1600);
      })
      .catch(() => undefined);
  }, []);

  return { copied, copy };
}
