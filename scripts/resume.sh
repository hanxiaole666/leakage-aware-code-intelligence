#!/usr/bin/env bash
# 断点续跑专用：不传 --fresh，直接续跑上次中断后未完成的 run。
# 场景：测试脚本被不小心关闭 → 重新执行本脚本 → 已完成 run 跳过，未完成继续。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/apple/anaconda3/python.exe"
[ -x "$PY" ] || PY="python"

echo "==> 当前状态："
"$PY" -m experiment_suite.runner --status

echo "==> 续跑（保留已完成进度）："
"$PY" -m experiment_suite.runner --all

echo "==> 刷新汇总："
"$PY" -m experiment_suite.reporting
