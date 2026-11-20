#!/usr/bin/env bash
# 把 D 盘规范根目录的测试数据同步到投稿包/发行包的只读快照。
# 方向永远是：规范根目录 -> 快照；绝不反向。
# 用法: bash scripts/sync_release_data.sh
set -euo pipefail

# Locate AIJ_Submission by walking up from this script, so the sync still works
# when AIJ_Submission/ is moved under the model project directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$SCRIPT_DIR")"
GRAND="$(dirname "$PARENT")"
if [ -d "$PARENT/AIJ_Submission/github_release" ]; then
  SUB="$PARENT/AIJ_Submission"   # <model>/scripts/sync_release_data.sh
elif [ -d "$SCRIPT_DIR/github_release" ]; then
  SUB="$SCRIPT_DIR"              # AIJ_Submission/sync_release_data.sh
elif [ -d "$GRAND/github_release" ]; then
  SUB="$GRAND"                   # AIJ_Submission/github_release/scripts/
else
  echo "ERROR: cannot locate AIJ_Submission from $SCRIPT_DIR" >&2
  exit 1
fi
MODEL="$(dirname "$SUB")"
SRC="$MODEL/data/processed"
REL="$SUB/github_release"

echo "=== 规范数据源 (D 盘) ==="
echo "  $SRC"

for name in synthetic_contracts.jsonl synthetic_meta.json; do
  if [ ! -f "$SRC/$name" ]; then
    echo "ERROR: 规范数据源缺失 $SRC/$name" >&2
    exit 1
  fi
done

# 同步到 GitHub 发行包（必须存在）
mkdir -p "$REL/data/processed"
for name in synthetic_contracts.jsonl synthetic_meta.json; do
  cp "$SRC/$name" "$REL/data/processed/$name"
  echo "  快照 <- $name  ($(wc -c < "$SRC/$name") bytes)"
done

# 同步到 AIJ_Submission/data（如果目录存在则一并同步，保持单一事实来源）
if [ -d "$SUB/data" ]; then
  mkdir -p "$SUB/data/processed"
  for name in synthetic_contracts.jsonl synthetic_meta.json; do
    cp "$SRC/$name" "$SUB/data/processed/$name"
    echo "  投稿目录快照 <- $name"
  done
fi

echo "=== 校验 sha256 一致 ==="
sha256sum "$SRC/synthetic_contracts.jsonl" "$REL/data/processed/synthetic_contracts.jsonl" | awk '{print $1, $2}'
echo "同步完成：所有 data/processed 快照均来自 D 盘规范根目录。"
