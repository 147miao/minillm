#!/usr/bin/env bash
# P0 C13：真实网关最小编码任务。
# 创建 greet.py + test_greet.py，跑 pytest，并断言 JSONL 里有 write_file 与 run_code.finished。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [ ! -f .env ]; then
  echo "live-smoke: .env 不存在，无法连接真实网关" >&2
  exit 2
fi

TMP="$(mktemp -d /tmp/minillm-live-smoke.XXXXXX)"
cleanup() {
  echo "live-smoke: 临时目录 $TMP" >&2
  rm -rf "$TMP"
}
trap cleanup EXIT

TASK="在当前 workspace 创建 greet.py（函数 greet(name) 返回 Hello, <name>!，例如 greet('MiniLLM') 得到 Hello, MiniLLM!）和 test_greet.py（用 pytest 断言 greet('MiniLLM') == 'Hello, MiniLLM!'），然后用 pytest 运行测试并总结。请用 tools.write_file 创建文件，用 tools.run_bash 运行 pytest。"
TASK_RETRY="$TASK 再次强调：必须调用 tools.write_file 创建 greet.py 与 test_greet.py，不要用 shell 重定向或 heredoc 创建文件。"

run_attempt() {
  local attempt="$1"
  local task="$2"
  local ws="$TMP/ws-$attempt"
  local data="$TMP/data-$attempt"
  local json="$TMP/out-$attempt.json"
  local log="$TMP/run-$attempt.log"
  local status=""
  local sid=""
  local events=""

  mkdir -p "$ws" "$data"
  echo "live-smoke: attempt $attempt 开始（timeout 300s）"
  set +e
  timeout 300 uv run minillm run --workspace "$ws" --data-dir "$data" --json "$task" >"$json" 2>"$log"
  status=$?
  set -e

  if [ "$status" -ne 0 ]; then
    echo "live-smoke: attempt $attempt 命令失败 (exit $status)" >&2
    tail -n 40 "$log" || true
    return 1
  fi

  if ! python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("ok") is True, d' "$json"; then
    echo "live-smoke: result.ok 不为 true" >&2
    cat "$json" || true
    return 1
  fi

  sid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("session_id") or "")' "$json")"
  if [ -z "$sid" ]; then
    echo "live-smoke: 结果缺少 session_id" >&2
    cat "$json" || true
    return 1
  fi

  echo "live-smoke: session=$sid workspace=$ws"
  if [ ! -f "$ws/greet.py" ] || [ ! -f "$ws/test_greet.py" ]; then
    echo "live-smoke: workspace 缺少 greet.py 或 test_greet.py" >&2
    find "$ws" -maxdepth 1 -type f -print || true
    return 1
  fi

  if ! ( cd "$ws" && "$REPO/.venv/bin/python" -m pytest -q ); then
    echo "live-smoke: pytest 未通过" >&2
    return 1
  fi

  events="$data/sessions/$sid/events.jsonl"
  if [ ! -f "$events" ]; then
    echo "live-smoke: events.jsonl 不存在：$events" >&2
    return 1
  fi

  if ! python3 -c 'import json,sys
es=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert any(e.get("type")=="tool.call" and (e.get("data") or {}).get("tool")=="write_file" for e in es), "missing tool.call(write_file)"
assert any(e.get("type")=="run_code.finished" for e in es), "missing run_code.finished"
print("events ok", len(es))' "$events"; then
    echo "live-smoke: JSONL 事件断言失败" >&2
    tail -n 8 "$events" || true
    return 1
  fi

  echo "$sid" >"$TMP/session-id"
  echo "live-smoke: attempt $attempt 成功"
  return 0
}

START="$(date +%s)"
if run_attempt 1 "$TASK"; then
  :
else
  echo "live-smoke: 第一次失败，重试一次（模型行为差异）" >&2
  if ! run_attempt 2 "$TASK_RETRY"; then
    echo "live-smoke: FAILED" >&2
    exit 1
  fi
fi
END="$(date +%s)"
SID="$(cat "$TMP/session-id")"
echo "live-smoke ok session=$SID elapsed=$((END-START))s"
