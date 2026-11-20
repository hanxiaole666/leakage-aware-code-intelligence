"""实验框架冒烟/单元测试：确保每个模块可运行、指标正确、续跑有效。

运行： .venv/Scripts/python.exe -m pytest tests/ -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiment_suite.data.synthesize import generate_dataset, load_records
from experiment_suite.data.splits import make_split, audit_overlap
from experiment_suite.data.fingerprints import normalize_source, cluster_clones
from experiment_suite.models.checkpoint import Checkpoint, signature
from experiment_suite.eval.metrics import compute_metrics
from experiment_suite.eval.calibration import ece, brier, auroc, temperature_scale, conformal_accept
from experiment_suite.eval.stats import paired_bootstrap, mcnemar, holm_bonferroni, mean_std
from experiment_suite.runner import run_one


def test_dataset_and_leakage():
    generate_dataset()
    recs = load_records()
    assert len(recs) == 5 * 40 * 6
    rnd = make_split(recs, "random")
    strict = make_split(recs, "clone_aware")
    a_r, a_s = audit_overlap(rnd), audit_overlap(strict)
    # 随机切分必然泄漏（同族跨集），严格切分重叠率应为 0
    assert a_r["clone_family_overlap_rate"] > 0.5
    assert a_s["clone_family_overlap_rate"] == 0.0


def test_normalization_removes_identifiers():
    s1 = "contract A { function f(uint x) public { x = x + 1; } }"
    s2 = "contract B { function g(uint y) public { y = y + 9; } }"
    assert normalize_source(s1) == normalize_source(s2)


def test_clone_clustering_purity():
    recs = load_records()[:60]
    cl = cluster_clones(recs)
    # 同族变体应聚类正确（无假阳性），纯度接近 1
    assert cl["pairwise_purity"] >= 0.9
    assert cl["pair_fp"] == 0
    assert cl["n_clusters"] >= 1


def test_metrics_perfect_and_random():
    y = [0, 1, 0, 1]
    m = compute_metrics(y, y)
    assert m["f1_macro"] == 1.0 and m["fpr"] == 0.0
    m2 = compute_metrics(y, [1, 1, 1, 1])
    assert m2["fpr"] > 0.0


def test_calibration_bounds():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    assert 0.0 <= ece(y, s) <= 1.0
    assert 0.0 <= brier(y, s) <= 1.0
    assert 0.0 <= auroc(y, s) <= 1.0
    assert 0.0 <= temperature_scale(y, s)["T"] <= 5.0
    cf = conformal_accept(y, s, alpha=0.5)
    assert 0.0 <= cf["accept_rate"] <= 1.0


def test_statistics():
    d = [0.01, 0.02, -0.005, 0.03, 0.015]
    r = paired_bootstrap(d)
    assert r["ci95"][0] <= r["ci95"][1]
    m = mcnemar([1, 1, 0, 1], [1, 0, 0, 1])
    assert 0.0 <= m["p_value"] <= 1.0
    h = holm_bonferroni([0.01, 0.04, 0.2])
    assert all(x["p_adj"] >= x["p"] for x in h)
    assert mean_std([1, 2, 3])[0] == 2.0


def test_checkpoint_resume(tmp_path):
    sig = signature("X", [1, 2])
    cp = Checkpoint.__new__(Checkpoint)
    cp.run_id, cp.sig = "TEST", sig
    cp.path = tmp_path / "TEST.json"
    cp.tmp = tmp_path / "TEST.json.tmp"
    assert cp.load() is None
    cp.save({"done": [1]})
    assert cp.load()["payload"]["done"] == [1]
    cp2 = Checkpoint.__new__(Checkpoint)
    cp2.run_id, cp2.sig = "TEST", signature("Y", [9])
    cp2.path, cp2.tmp = cp.path, cp.tmp
    assert cp2.load() is None  # 签名不符→作废旧 ckpt


def test_run_one_executes():
    recs = load_records()
    # 分层采样：每类取前 4 个克隆族，避免切片落在单一类别
    from collections import defaultdict
    by_lab = defaultdict(list)
    for r in recs:
        by_lab[r["label"]].append(r)
    small = []
    for lab, rows in by_lab.items():
        fams = sorted({r["family"] for r in rows})[:4]
        small += [r for r in rows if r["family"] in fams]
    res = run_one("R012", small, seed=42)
    assert "metrics" in res and 0.0 <= res["metrics"]["f1_macro"] <= 1.0
