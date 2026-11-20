"""Generate all paper figures from the REAL run JSON files (no hardcoded numbers).

Source of truth:
  ../Blockchain Risk Management Model/results/runs/*.json
  ../Blockchain Risk Management Model/results/revision_evidence.json
  ../Blockchain Risk Management Model/results/round2_evidence.json
Outputs: figures/*.pdf (vector) + figures/*.png (preview) + figure_provenance.json.

Final proof fixes: all figure text is plain ASCII (no mathtext, which collided
when extracted) and each annotation is placed at a distinct, non-overlapping
vertical position. Underlying values are also written to figure_provenance.json
so the editor can check every printed number without reading the figure.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "results"
if not (BASE / "runs").is_dir():
    BASE = HERE.parent / "Blockchain Risk Management Model" / "results"
RUNS = BASE / "runs"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def load(rid: str) -> dict:
    with open(RUNS / f"{rid}.json", encoding="utf-8") as f:
        return json.load(f)


with open(BASE / "revision_evidence.json", encoding="utf-8") as f:
    rev = json.load(f)
with open(BASE / "round2_evidence.json", encoding="utf-8") as f:
    r2 = json.load(f)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

# ---------- Fig 1: single axis; F1 inside bars, overlaps above ----------
r11, r12, r04, r13 = load("R011"), load("R012"), load("R004"), load("R013")
names = ["Random\nR011", "Oracle family\nR012", "Detector-aware\nR004", "Temporal\nR013"]
f1 = [r11["metrics"]["f1_macro"], r12["metrics"]["f1_macro"],
      r04["metrics"]["f1_macro"], r13["metrics"]["f1_macro"]]
ov = rev["E1_dual_overlap"]["overlap_by_protocol"]
rho_g = [ov["random"]["true_family_overlap_rate"],
         ov["clone_aware"]["true_family_overlap_rate"],
         ov["detector_aware"]["true_family_overlap_rate"],
         ov["temporal"]["true_family_overlap_rate"]]
rho_h = [ov["random"]["detected_cluster_overlap_rate"],
         ov["clone_aware"]["detected_cluster_overlap_rate"],
         ov["detector_aware"]["detected_cluster_overlap_rate"],
         ov["temporal"]["detected_cluster_overlap_rate"]]

fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(names))
bars = ax.bar(x, f1, width=0.56, color=["#c0392b", "#2471a3", "#e67e22", "#1e8449"], alpha=0.9)
ax.set_ylabel("Macro-F1")
ax.set_ylim(0, 1.30)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8)
for b, v in zip(bars, f1):
    ax.text(b.get_x() + b.get_width()/2, v/2, f"F1 {v:.4f}",
            ha="center", va="center", fontsize=7, color="white", fontweight="bold")
for i, (g, h) in enumerate(zip(rho_g, rho_h)):
    ax.text(x[i], 1.06, f"rho_g {g:.4f}", ha="center", fontsize=7, color="#7d3c98")
    ax.text(x[i], 1.14, f"rho_hat {h:.4f}", ha="center", fontsize=7, color="#148f77")
ax.set_title("Macro-F1 by protocol (overlap values printed above bars)")
fig.tight_layout()
fig.savefig(FIG / "fig1_split_protocol.pdf")
fig.savefig(FIG / "fig1_split_protocol.png")
plt.close(fig)

# ---------- Fig 2: risk-coverage with endpoint and budget ----------
rc = load("R040")["calibration"]["risk_coverage"]
cov = rc["coverage_grid"] + [rc["coverage_endpoint_1p0"]]
risk = rc["risk_grid"] + [rc["risk_at_full_coverage"]]

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(cov, risk, "s-", color="#1a5276", lw=1.6, ms=5, label="Selective risk = 1 minus accuracy")
ax.axhline(0.1, color="#c0392b", ls="--", lw=1.2, label="Risk budget 0.10")
ax.text(0.52, rc["risk_grid"][5] + 0.01, f"R(0.51)={rc['risk_grid'][5]:.3f}",
        fontsize=7, color="#c0392b")
ax.set_xlabel("Coverage")
ax.set_ylabel("Selective risk")
ax.set_ylim(0, max(risk) * 1.18)
ax.set_xlim(0, 1.02)
ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.tick_params(axis="both", labelsize=8)
ax.set_title(f"Risk-coverage, AURC {rc['aurc']:.4f}, endpoint {rc['risk_at_full_coverage']:.4f}")
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "fig2_risk_coverage.pdf")
fig.savefig(FIG / "fig2_risk_coverage.png")
plt.close(fig)

# ---------- Fig 3: leakage dose-response ----------
dr = r2["D4_dose_response"]
xs = [a["rho_g_mean"] for a in dr["aggregate"]]
ys = [a["macro_f1_mean"] for a in dr["aggregate"]]
yerr = [a["macro_f1_sd"] for a in dr["aggregate"]]

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color="#1a5276", lw=1.6, ms=5,
            capsize=3, label="Observed mean and SD")
slope = dr["slope_mean"]
grid = np.linspace(0, 1, 50)
ax.plot(grid, ys[0] + slope*grid, ls="--", color="#c0392b", lw=1.2,
        label=f"All-point fit slope {slope:.3f}")
ax.set_xlabel("True-family overlap rho_g")
ax.set_ylabel("Macro-F1")
ax.set_xlim(-0.03, 1.03)
ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.tick_params(axis="both", labelsize=8)
ax.set_title("Leakage dose-response, fixed test set and training size")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "fig3_dose_response.pdf")
fig.savefig(FIG / "fig3_dose_response.png")
plt.close(fig)

# ---------- machine-readable provenance for every printed number ----------
prov = {
    "fig1": {"source_runs": ["R011", "R012", "R004", "R013"],
             "source_revision": "E1_dual_overlap",
             "macro_f1": dict(zip(["R011", "R012", "R004", "R013"], f1)),
             "rho_g": dict(zip(["R011", "R012", "R004", "R013"], rho_g)),
             "rho_hat": dict(zip(["R011", "R012", "R004", "R013"], rho_h))},
    "fig2": {"source_run": "R040", "aurc": rc["aurc"],
             "coverage": cov, "risk": risk,
             "endpoint": rc["risk_at_full_coverage"],
             "peak_risk": max(rc["risk_grid"]),
             "budget_violated_in_middle": True},
    "fig3": {"source": "round2_evidence.json D4_dose_response",
             "rho_g": xs, "macro_f1_mean": ys, "macro_f1_sd": yerr,
             "all_point_slope": slope,
             "note": "all-point slope reported for completeness; interval slopes are the conclusion"},
}
(FIG / "figure_provenance.json").write_text(
    json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

print("Figures generated from real runs:")
for p in sorted(FIG.glob("*.pdf")):
    print(" ", p.name, os.path.getsize(p), "bytes")
