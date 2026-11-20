#!/usr/bin/env bash
# 一键运行全部增补实验（R001-R061 可由 runner 执行的部分），随后生成汇总报告。
# 断点续跑：已完成的 run 自动跳过；中途 Ctrl+C 关闭后再跑本脚本继续。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/apple/anaconda3/python.exe"
[ -x "$PY" ] || PY="python"

echo "==> 运行实验（断点续跑：done 的自动跳过）"
"$PY" -m experiment_suite.runner --all

echo "==> 生成结果汇总"
"$PY" -m experiment_suite.reporting
