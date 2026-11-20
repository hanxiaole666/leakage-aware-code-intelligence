"""experiment_suite — CCF-A 可复现实验框架（区块链风控可信小模型）.

设计目标
--------
1. 声明驱动：所有 claim / run / 指标在配置中冻结，实验由配置生成。
2. 幂等数据：``ensure_data()`` 在本地缺失时自动下载，存在则直接复用。
3. 断点续跑：pipeline 级（run 状态）+ 训练级（epoch/step/RNG）双层 checkpoint。
4. 统计严谨：多 seed、配对 bootstrap、McNemar、Holm-Bonferroni 全部内置。
5. 诚实：所有数值由脚本真实产出并落盘，禁止硬编码占位指标。

本包是 **M0/M1 级可运行骨架**：默认使用带克隆族结构的合成数据真实演示
"随机切分虚高 vs 泄漏感知切分"这一核心 claim。接入真实数据集与 3B 模型
只需替换 ``data.download`` 的数据源与 ``models`` 的 estimator，其余不变。
"""

__version__ = "0.1.0"
