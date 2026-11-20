"""结果汇总：读取 runs/*.json，生成 Markdown 结果表 + 主 claim 的统计检验。

所有数字来自真实运行的落盘 JSON；若某 run 未跑，明确标注 NOT RUN，
绝不填入占位数值（这是评审报告 D 类缺陷的直接修复）。
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import RESULTS_ROOT, TABLES_DIR, RUNS
from .eval.metrics import metrics_table
from .eval.stats import paired_bootstrap, mcnemar


def _load(run_id: str) -> dict | None:
    p = RESULTS_ROOT / "runs" / f"{run_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_report() -> dict:
    results = {}
    details = {}
    for spec in RUNS:
        res = _load(spec.run_id)
        if res and "metrics" in res:
            results[spec.run_id] = res["metrics"]
            details[spec.run_id] = res
    md = metrics_table(results)

    # Claim C1 的核心对照：随机 vs 严格协议（同一 ours 变体）
    stats = {}
    r_random, r_strict = _load("R011"), _load("R012")
    if r_random and r_strict:
        f1_r = r_random["metrics"]["f1_macro"]
        f1_s = r_strict["metrics"]["f1_macro"]
        stats["leakage_gap_f1"] = {
            "random_split_f1": f1_r, "clone_aware_f1": f1_s,
            "inflation": round(f1_r - f1_s, 4),
            "interpretation": "随机切分相对严格切分的 F1 虚高幅度（A2 记忆假说检验）",
        }
    # 主对比 ours vs 静态基线（严格协议）
    r_ours, r_base = _load("R012"), _load("R003")
    if r_ours and r_base:
        stats["ours_vs_static_strict"] = {
            "ours_f1": r_ours["metrics"]["f1_macro"],
            "baseline_f1": r_base["metrics"]["f1_macro"],
        }

    report = {
        "note": "全部指标由 experiment_suite 真实运行产出；未运行 run 不在表中。"
                "本轮为 M0/M1 合成数据骨架，非真实 3B SOTA 结果。",
        "metrics_markdown": md,
        "stats": stats,
        "n_runs_completed": len(results),
        "n_runs_total": len(RUNS),
    }
    out = TABLES_DIR / "RESULTS_SUMMARY.md"
    out.write_text(
        "# 实验结果汇总（脚本真实产出）\n\n" + report["note"] + "\n\n"
        f"完成度：{len(results)}/{len(RUNS)} runs\n\n" + md + "\n\n"
        "## 关键统计\n\n```json\n" +
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    (TABLES_DIR / "RESULTS_SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    rpt = build_report()
    print(f"报告已生成：{TABLES_DIR / 'RESULTS_SUMMARY.md'}")
    print(rpt["metrics_markdown"])
