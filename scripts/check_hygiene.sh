#!/usr/bin/env bash
# P0 C14：仓库卫生（无密钥、单文件 <=700 行、运行期目录不入库）。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

fail() {
  echo "hygiene FAIL: $*" >&2
  exit 1
}

git check-ignore -q .env || fail ".env 没有被 .gitignore 忽略"
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  fail ".env 已被 git 跟踪"
fi

if grep -rIn --binary-files=without-match --exclude='check_hygiene.sh' -E 'oc_sk_|LLM_API_KEY=' src tests apps/web/src scripts docs README.md 2>/dev/null; then
  fail "扫描到密钥或 sk 前缀"
fi

FILES="$(for ext in py ts tsx css sh; do find src tests apps/web/src scripts -type f -name "*.$ext"; done | sort -u)"
if [ -n "$FILES" ]; then
  if ! echo "$FILES" | xargs -r wc -l | awk '$2 != "total" && $1 > 700 {print "hygiene FAIL: " $2 " (" $1 " lines)"; bad=1} END {exit bad}'; then
    exit 1
  fi
fi

if git ls-files | grep -Eq '(^|/)(data|workspace|node_modules|apps/web/dist|__pycache__)(/|$)'; then
  fail "git 跟踪了 data/workspace/node_modules/dist/__pycache__"
fi

COUNT="$(echo "$FILES" | wc -l)"
echo "hygiene ok ($COUNT files checked)"
