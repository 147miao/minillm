import type { TodoItem } from '../types';

const STATUS_ICON: Record<string, string> = {
  pending: '○',
  in_progress: '◐',
  completed: '✓',
};

export function TodoPanel({ todos }: { todos: TodoItem[] }) {
  const done = todos.filter((todo) => todo.status === 'completed').length;
  return (
    <section className="panel-card">
      <header className="panel-head">
        <h3>任务清单</h3>
        <span className="count">
          {done}/{todos.length}
        </span>
      </header>
      {todos.length === 0 ? (
        <p className="muted">暂无任务。模型调用 todo_write 后会出现在这里。</p>
      ) : (
        <ul className="todo-list">
          {todos.map((todo, index) => {
            const status =
              todo.status === 'completed'
                ? 'done'
                : todo.status === 'in_progress'
                  ? 'active'
                  : 'pending';
            return (
              <li key={todo.id ?? String(index)} className={'todo todo-' + status}>
                <span className="todo-icon" aria-hidden="true">
                  {STATUS_ICON[todo.status] ?? '○'}
                </span>
                <span className="todo-text">{todo.content}</span>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
