"""数据层：幂等下载、克隆族感知合成、泄漏审计与切分。"""
from .download import ensure_data, download_all
from .synthesize import generate_dataset, synthesize_if_needed
from .fingerprints import normalize_source, minhash_signature, cluster_clones
from .splits import make_split, audit_overlap

__all__ = [
    "ensure_data", "download_all", "generate_dataset", "synthesize_if_needed",
    "normalize_source", "minhash_signature", "cluster_clones", "make_split", "audit_overlap",
]
