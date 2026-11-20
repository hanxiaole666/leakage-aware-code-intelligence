"""可运行 estimator 工厂 + 变体克隆。

代理映射（诚实声明）：
* ours           -> LogisticRegression（可解释线性，代表"结构+语义信号"可学习）
* rule_baseline  -> DummyClassifier（静态规则代理，仅用先验，最弱）
* large_proxy    -> GradientBoosting（更大容量，代表 7B 贴边对照）
* complex_proxy  -> GradientBoosting + 更多树（花哨变体，验证无增益）
* ablate_*       -> ours 去掉对应特征族
* calib_*        -> ours + 不同校准策略（在 runner 里处理）
* robust_*       -> ours + 扰动/压力评估（在 runner 里处理）
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_estimator(variant: str, seed: int = 42, n_classes: int | None = None):
    # 训练单类时 LogisticRegression/GBDT 会报错，优雅降级为常数预测器
    if n_classes is not None and n_classes < 2:
        return DummyClassifier(strategy="most_frequent")
    if variant == "rule_baseline":
        return DummyClassifier(strategy="prior")
    if variant == "toy_lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    if variant in ("large_proxy", "complex_proxy"):
        n = 200 if variant == "complex_proxy" else 80
        return GradientBoostingClassifier(n_estimators=n, max_depth=3, random_state=seed)
    if variant == "rf_proxy":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    # ours 及消融默认线性模型
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0, random_state=seed))


def clone_with_variant(variant: str, seed: int = 42):
    return build_estimator(variant, seed)


VARIANTS = {
    "toy_lr", "audit_only", "rule_baseline", "ours", "large_proxy", "complex_proxy",
    "ablate_structure", "ablate_semantic", "ablate_shortcut", "ablate_dpo", "ablate_cpt",
    "calib_ts", "calib_mcd", "calib_conformal", "calib_conformal_split", "robust_semantic", "robust_neardup",
    "generalize_loo",
}
