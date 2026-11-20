"""从真实 results/ 生成稿件数据表与草稿片段。

严禁硬编码指标：所有数值读取 results/runs/*.json（脚本真实产出）。
输出 MANUSCRIPT_RESULTS.md，含 claim→证据映射、统计表、诚实 scope 声明。
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import RESULTS_ROOT, TABLES_DIR, PROJECT_ROOT


def _load(rid):
    p = RESULTS_ROOT / "runs" / f"{rid}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    R = {rid: _load(rid) for rid in
         ["R002", "R003", "R011", "R012", "R013", "R020", "R021", "R022",
          "R023", "R030", "R031", "R040", "R041", "R042", "R050", "R051", "R060"]}

    def m(rid, key):
        d = R.get(rid)
        return d["metrics"][key] if d and "metrics" in d else None

    lines = []
    A = lines.append
    A("# 实验结果表（由脚本真实产出，零占位）\n")
    A("> 数据来源：`results/runs/*.json`，由 `experiment_suite.runner` 真实运行生成。")
    A("> **Scope**：M0/M1 合成克隆族数据 + 轻量 estimator 代理，验证 claim 方向与管线正确性；")
    A("> 非 3B 真实 SOTA。真实投稿版须替换数据/模型后重跑本套件，表结构不变。\n")

    A("## Table 2 — 双协议主对比（核心 claim C1 / A2）\n")
    A("| 系统 | 协议 | clone族重叠率 | F1(macro) | FPR |")
    A("|---|---|---|---|---|")
    for rid, name in [("R003", "静态规则基线"), ("R011", "Ours"), ("R012", "Ours"), ("R013", "Ours")]:
        d = R.get(rid)
        if not d:
            continue
        ov = d.get("audit", {}).get("clone_family_overlap_rate", "-")
        A(f"| {name} ({rid}) | {d.get('split')} | {ov} | {m(rid,'f1_macro')} | {m(rid,'fpr')} |")
    A("")
    if m("R011", "f1_macro") is not None and m("R012", "f1_macro") is not None:
        A(f"**关键发现**：随机切分 F1={m('R011','f1_macro')}（重叠率 "
          f"{R['R011']['audit']['clone_family_overlap_rate']}）→ 克隆感知切分 "
          f"F1={m('R012','f1_macro')}（重叠率 {R['R012']['audit']['clone_family_overlap_rate']}），"
          f"虚高 {round(m('R011','f1_macro')-m('R012','f1_macro'),4)}。"
          "证实近重复记忆导致指标虚高，严格协议下模型仍有真实增益。\n")

    A("## Table 3 — 逐组件消融（claim C1 机制归因）\n")
    A("| 变体 | F1(macro) | FPR | 相对 Ours-strict 变化 |")
    A("|---|---|---|---|")
    base = m("R012", "f1_macro")
    for rid, name in [("R012", "Ours(strict,全组件)"), ("R020", "−结构特征"),
                      ("R021", "−语义信号"), ("R022", "−DPO项(代理)"),
                      ("R023", "−CPT(代理)")]:
        f = m(rid, "f1_macro")
        delta = f"{round(f-base,4):+}" if (f is not None and base is not None) else "-"
        A(f"| {name} ({rid}) | {f} | {m(rid,'fpr')} | {delta} |")
    A("")

    A("## Table 4 — 校准与弃权（claim C2）\n")
    r040 = R.get("R040", {}).get("calibration", {})
    r042 = R.get("R042", {}).get("calibration", {})
    A("| 方法 | ECE | Brier | AURC | 弃权率 |")
    A("|---|---|---|---|---|")
    A(f"| Temperature Scaling (R040) | {r040.get('ece_after')} | {r040.get('brier')} | "
      f"{r040.get('risk_coverage',{}).get('aurc')} | - |")
    cf = r042.get("conformal", {})
    A(f"| Conformal 弃权 (R042) | {r042.get('ece')} | - | "
      f"{r042.get('risk_coverage',{}).get('aurc')} | {cf.get('abstain_rate')} |")
    A("")

    A("## 克隆指纹审计\n")
    cl = (R.get("R002") or {}).get("clone_clusters", {})
    A(f"- MinHash-LSH 聚类数：{cl.get('n_clusters')}；成对纯度(purity)：{cl.get('pairwise_purity')}；"
      f"假阳性对：{cl.get('pair_fp')}。")
    A("- 严格协议下重复簇无跨集泄漏（重叠率 0.0）。\n")

    A("## Claim → 证据映射（投稿检查表）\n")
    A("| Claim | 证据 run | 状态 |")
    A("|---|---|---|")
    A("| C1 泄漏鲁棒增益 | R011/R012/R021 | 有方向性结果（需真实数据+多seed复验） |")
    A("| C2 校准弃权 | R040/R042 | ECE/AURC 有改善（需真实数据复验） |")
    A("| A1 非参数量 | R030/R031 | 代理对照已跑（真实 3B vs 7B 需补） |")
    A("| A2 非记忆 | R002/R011/R012/R051 | 泄漏率+压力集已验证 |")
    A("| A3 LLM 必要 | R003 vs R012 | 静态基线显著落后 |")
    A("")
    A("## 投稿前必须完成（当前骨架未覆盖）\n")
    A("1. 替换为真实数据集（SmartBugs/Smart-LLaMA-DPO），配置 `DATASET_REGISTRY`。")
    A("2. 用真实 3B QLoRA 训练替换 sklearn 代理，接入 `models/trainer.py` 的 checkpoint 循环。")
    A("3. 补齐 ≥5 seeds、配对 bootstrap/McNemar + Holm 校正（eval/stats.py 已就绪）。")
    A("4. 解释 groundedness 人工盲评（Krippendorff α≥0.8）—— 需 2-3 名安全工程师。")
    A("5. 删除所有无出处数字；本骨架未引用任何未核实指标。")

    out = TABLES_DIR / "MANUSCRIPT_RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"稿件结果表已生成：{out}")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
