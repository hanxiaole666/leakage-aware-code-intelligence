"""集中配置：路径、随机种子、数据 URL、实验矩阵。

所有可变参数集中于此，禁止在实验脚本中散落硬编码。环境变量可覆盖：
  EXP_ROOT   实验产物根目录（默认 <project>/results）
  DATA_ROOT  数据缓存根目录（默认 <project>/data）
  SKIP_DOWNLOAD=1  跳过真实数据下载，强制使用合成 fallback

数据存放位置契约（唯一事实来源 / single source of truth）
--------------------------------------------------------
本机所有测试数据只落在 D 盘，规范根目录为：

    D:\\Financial Micro-Modelling Project\\Blockchain Risk Management Model\\data

即本模块 PROJECT_ROOT 下的 data/。禁止把合成集、下载缓存或任何 run 数据写到 C 盘。
AIJ 投稿包与 GitHub 发行包中的 data/ 只是由此规范根目录同步出的**只读快照**，
用 scripts/sync_release_data.sh 生成，绝不反过来作为数据源。
若 DATA_ROOT 被环境变量改到 C 盘，describe_data_root() 会给出明确警告。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).resolve() if raw else default


DATA_ROOT = _env_path("DATA_ROOT", PROJECT_ROOT / "data")
RESULTS_ROOT = _env_path("EXP_ROOT", PROJECT_ROOT / "results")
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
CKPT_DIR = RESULTS_ROOT / "checkpoints"
TABLES_DIR = RESULTS_ROOT / "tables"
FIGURES_DIR = RESULTS_ROOT / "figures"
LOGS_DIR = RESULTS_ROOT / "logs"
STATE_FILE = RESULTS_ROOT / "pipeline_state.json"

for _p in (DATA_ROOT, RESULTS_ROOT, RAW_DIR, PROCESSED_DIR, CKPT_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 43, 44, 123, 2024)  # >=5 seeds（评审要求）

# 真实数据源注册表。每条: name -> (url, sha256或None, 本地相对路径, 说明)
# 说明：URL 为公开数据集地址；下载失败自动回退到合成 fallback，并在 provenance.json 记录原因。
DATASET_REGISTRY: dict[str, dict] = {
    "smartbugs_curated": {
        "url": "https://github.com/smartbugs/smartbugs-curated/archive/refs/heads/main.tar.gz",
        "sha256": None,
        "path": "smartbugs_curated.tar.gz",
        "kind": "real_solidity",
        "desc": "SmartBugs Curated 漏洞合约（多类型），仅用于真实数据接入演示",
    },
    "self_attacked_attack": {
        "url": "https://github.com/smartbugs/SmartBugs_Curated/raw/main/README.md",
        "sha256": None,
        "path": "self_attacked_readme.md",
        "kind": "real_solidity",
        "desc": "占位真实源（轻量 README，避免大下载）；生产环境替换为完整数据集",
    },
}

# 主实验配置：锚定单一任务 = 智能合约漏洞检测
TASK_LABELS = ("reentrancy", "timestamp", "overflow", "access_control", "safe")
PRIMARY_METRIC = "f1_macro"
SAFETY_METRIC = "fpr"  # 工业第一痛点

MODEL_SMALL = "qwen2.5-coder-3b-qlora-nf4"  # 逻辑基座名（本骨架用可运行 estimator 代理）
MODEL_MEDIUM = "qwen2.5-coder-7b-qlora-nf4"


@dataclass(frozen=True)
class SplitSpec:
    protocol: str  # random | temporal | clone_aware
    train_frac: float = 0.7
    val_frac: float = 0.15
    # clone_aware 强制 group 不重叠；temporal 用 deployed_at 排序


@dataclass
class RunSpec:
    run_id: str
    milestone: str
    block: str
    description: str
    variant: str
    split: str
    priority: str  # MUST | NICE
    params: dict = field(default_factory=dict)


# 增补实验清单 R001-R061（来自 EXPERIMENT_TRACKER.md，全部可由 runner 执行）
RUNS: list[RunSpec] = [
    RunSpec("R001", "M0", "B0", "sanity/toy 数据过拟合检查", "toy_lr", "random", "MUST"),
    RunSpec("R002", "M0", "B0", "泄漏审计：克隆族重叠率", "audit_only", "clone_aware", "MUST"),
    RunSpec("R003", "M0", "B1", "静态基线（多数类参照，非真实分析器）", "rule_baseline", "random", "MUST"),
    RunSpec("R004", "M0", "B1", "E1：检测器簇分组切分（残余真值族重叠审计）", "ours", "detector_aware", "MUST"),
    RunSpec("R005", "M3", "B2", "消融：去掉捷径特征（与 split 无关）", "ablate_shortcut", "random", "MUST"),
    RunSpec("R043", "M4", "B4", "真正的 split conformal（独立校准集）", "calib_conformal_split", "clone_aware", "MUST"),
    RunSpec("R011", "M2", "B1", "ours 主方法（LLM 代理）随机协议", "ours", "random", "MUST"),
    RunSpec("R012", "M2", "B1", "ours 主方法 严格协议", "ours", "clone_aware", "MUST"),
    RunSpec("R013", "M2", "B1", "ours 主方法 时序协议", "ours", "temporal", "MUST"),
    RunSpec("R020", "M3", "B2", "消融：无结构特征", "ablate_structure", "clone_aware", "MUST"),
    RunSpec("R021", "M3", "B2", "消融：无语义对抗增强", "ablate_semantic", "clone_aware", "MUST"),
    RunSpec("R022", "M3", "B2", "消融：无 DPO 偏好项（FPR 退化）", "ablate_dpo", "clone_aware", "MUST"),
    RunSpec("R023", "M3", "B2", "消融：无 CPT 预训练", "ablate_cpt", "clone_aware", "MUST"),
    RunSpec("R030", "M3", "B3", "简洁性：大基座代理 vs 小基座", "large_proxy", "clone_aware", "MUST"),
    RunSpec("R031", "M3", "B3", "简洁性：复杂变体（更多特征）无增益检查", "complex_proxy", "clone_aware", "NICE"),
    RunSpec("R040", "M4", "B4", "校准：Temperature Scaling", "calib_ts", "clone_aware", "MUST"),
    RunSpec("R041", "M4", "B4", "校准：MC Dropout 代理", "calib_mcd", "clone_aware", "MUST"),
    RunSpec("R042", "M4", "B4", "校准：ours conformal 弃权", "calib_conformal", "clone_aware", "MUST"),
    RunSpec("R050", "M4", "B5", "鲁棒：语义等价扰动保持率", "robust_semantic", "clone_aware", "MUST"),
    RunSpec("R051", "M4", "B5", "鲁棒：近重复压力集衰减", "robust_neardup", "clone_aware", "MUST"),
    RunSpec("R060", "M4", "B5", "泛化：跨漏洞类型 leave-one-out", "generalize_loo", "clone_aware", "NICE"),
]
RUN_BY_ID = {r.run_id: r for r in RUNS}


# 规范数据根目录（绝对路径，写死为本机 D 盘项目位置）。
# 用途：向读者/复现者声明唯一事实来源，并供发布快照同步脚本引用。
CANONICAL_DATA_ROOT = Path(r"D:\Financial Micro-Modelling Project\Blockchain Risk Management Model\data")


def describe_data_root() -> dict:
    """返回当前数据路径契约，并在数据落到 C 盘时给出明确警告。

    返回字典可直接写入 provenance，供论文复现者核对数据位置。
    """
    on_c_drive = str(DATA_ROOT.drive).upper() == "C"
    if on_c_drive:
        import warnings
        warnings.warn(
            f"数据根目录当前位于 C 盘 ({DATA_ROOT})。按项目约定，测试数据应放在 "
            f"{CANONICAL_DATA_ROOT}；请设置 DATA_ROOT 或移动数据。",
            stacklevel=2,
        )
    return {
        "canonical_data_root": str(CANONICAL_DATA_ROOT),
        "active_data_root": str(DATA_ROOT),
        "processed_dir": str(PROCESSED_DIR),
        "raw_dir": str(RAW_DIR),
        "on_c_drive": on_c_drive,
        "contract": "测试数据只在 D 盘规范根目录；投稿/发行包 data/ 为只读同步快照",
    }


def provenance() -> dict:
    return {
        "framework_version": "0.1.0",
        "data_root": str(DATA_ROOT),
        "results_root": str(RESULTS_ROOT),
        "seeds": list(SEEDS),
        "note": "M0/M1 骨架：默认合成带克隆族数据；真实 3B SOTA 需替换 estimator 并接入真实集",
        "data_location": describe_data_root(),
    }
