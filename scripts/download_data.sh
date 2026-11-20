#!/usr/bin/env bash
# 数据下载脚本：幂等——本地有数据则跳过，无数据自动下载，失败回退合成集。
# 用法：
#   bash scripts/download_data.sh            # 尝试真实下载 + 保证合成集就绪
#   SKIP_DOWNLOAD=1 bash scripts/download_data.sh   # 仅用合成集（离线）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/apple/anaconda3/python.exe"
[ -x "$PY" ] || PY="python"

echo "==> 1) 真实数据集：本地缺失则自动下载（含断点续传），存在则跳过"
"$PY" -c "from experiment_suite.data.download import download_all; \
import json; print(json.dumps({k:str(v) if v else None for k,v in download_all().items()}, indent=2))"

echo "==> 2) 保证 M0/M1 管道数据就绪（合成克隆族集，已存在则复用）"
"$PY" -c "from experiment_suite.data.synthesize import generate_dataset; \
p=generate_dataset(); print('dataset:', p)"

echo "==> 完成。查看下载日志：results/logs/download_*.json"
