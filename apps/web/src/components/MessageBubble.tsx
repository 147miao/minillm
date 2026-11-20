import type { ChatMessage, EventDetails, StreamBuffer } from '../types';
import { Markdown } from './Markdown';
import { RunCodeCard } from './RunCodeCard';
import { ToolCallCard } from './ToolCallCard';

interface MessageBubbleProps {
  message: ChatMessage;
  sessionId: string;
  details: EventDetails;
  toolResults: Map<string, ChatMessage>;
}

export function MessageBubble({ message, sessionId, details, toolResults }: MessageBubbleProps) {
  if (message.role === 'user') {
    return (
      <div className="message message-user">
        <div className="message-role">你</div>
        <div className="user-text">{message.content}</div>
      </div>
    );
  }

  if (message.role === 'system') {
    return (
      <div className="message message-system">
        <div className="message-role">上下文摘要 · system</div>
        <Markdown text={message.content} />
      </div>
    );
  }

  if (message.role !== 'assistant') {
    return null;
  }

  const toolCalls = message.tool_calls ?? [];

  return (
    <div className="message message-assistant">
      <div className="message-role">MiniLLM</div>
      {message.reasoning !== undefined && message.reasoning !== '' && (
        <details className="reasoning">
          <summary>思考过程</summary>
          <div className="reasoning-body">{message.reasoning}</div>
        </details>
      )}
      {message.content !== '' && <Markdown text={message.content} />}
      {toolCalls.map((call) => {
        const run = details.runs[call.id];
        const result = toolResults.get(call.id);
        const relatedTools = Object.values(details.tools).filter(
          (tool) => tool.runCallId === call.id,
        );
        const isRunCode = call.name === 'run_code' || run !== undefined;
        if (isRunCode) {
          return (
            <RunCodeCard
              key={call.id}
              sessionId={sessionId}
              call={call}
              run={run}
              result={result}
              toolCalls={relatedTools}
            />
          );
        }
        return (
          <ToolCallCard
            key={call.id}
            detail={details.tools[call.id]}
            fallbackName={call.name}
            fallbackArgs={call.arguments}
            fallbackPreview={result?.content}
            sessionId={sessionId}
          />
        );
      })}
      {message.content === '' && toolCalls.length === 0 && (
        <p className="muted message-empty">（空响应）</p>
      )}
    </div>
  );
}

export function StreamingBubble({ stream }: { stream: StreamBuffer }) {
  if (stream.text === '' && stream.reasoning === '') {
    return null;
  }
  return (
    <div className="message message-assistant is-streaming">
      <div className="message-role">
        MiniLLM
        <span className="live-badge">生成中</span>
      </div>
      {stream.reasoning !== '' && (
        <details className="reasoning">
          <summary>思考过程</summary>
          <div className="reasoning-body">{stream.reasoning}</div>
        </details>
      )}
      {stream.text !== '' && <Markdown text={stream.text} />}
      <span className="stream-cursor" aria-hidden="true" />
    </div>
  );
}
