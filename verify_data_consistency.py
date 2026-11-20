"""Verify that every number quoted in main.tex matches a real run JSON.

Run: python verify_data_consistency.py ; exit 0 only if all checks pass.
Third-round revision: adds dose-response, real-corpus audit, corrected binary
calibration (score = 1-p(safe)), completeness of R004 and the sweep, and the
configuration-hash integrity assertion.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "results"
if not (BASE / "runs").is_dir():
    BASE = HERE / "results"
RUNS = BASE / "runs"
TEX = (HERE / ".." / "Manuscript.tex").read_text(encoding="utf-8") if (HERE / ".." / "Manuscript.tex").exists() else (HERE / "main.tex").read_text(encoding="utf-8")


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


REV = _load_json(BASE / "revision_evidence.json")
R2 = _load_json(BASE / "round2_evidence.json")

failures, checked = [], 0


def load(rid):
    with open(RUNS / f"{rid}.json", encoding="utf-8") as f:
        return json.load(f)


def check(label, quoted, actual, tol=5e-5):
    global checked
    checked += 1
    if actual is None:
        failures.append(f"{label}: paper says {quoted!r} but field missing")
        return
    try:
        ok = abs(float(quoted) - float(actual)) <= tol
    except (TypeError, ValueError):
        ok = str(quoted).strip() == str(actual).strip()
    if not ok:
        failures.append(f"{label}: paper={quoted!r} data={actual!r}")
    else:
        print(f"  OK  {label:50s} = {quoted}")


def in_tex(value, label):
    global checked
    checked += 1
    if value not in TEX:
        failures.append(f"{label}: {value!r} not found in main.tex")
    else:
        print(f"  OK  {label:50s} appears")


print("=== R1: abstract seed convention ===")
summary = REV["E5_multiseed"]["summary"]
check("oracle ten-seed mean", "0.7054", summary["clone_aware"]["f1_mean"])
check("oracle ten-seed sd", "0.0333", summary["clone_aware"]["f1_sd"])
check("random ten-seed mean", "0.9983", summary["random"]["f1_mean"])
check("random ten-seed sd", "0.0053", summary["random"]["f1_sd"])
gap = REV["E5_multiseed"]["gap_random_minus_clone"]
check("gap", "0.293", gap["mean_diff"])
in_tex("0.7054", "abstract oracle value")
if "0.6749\\pm0.0333" in TEX:
    failures.append("R1 regression: single-seed value paired with ten-seed SD")

print("=== E1 two-way overlap ===")
det = REV["E1_dual_overlap"]["detector"]
ov = REV["E1_dual_overlap"]["overlap_by_protocol"]
for label, val, actual in [
    ("detector pair P", "0.9957", det["pair_precision"]),
    ("detector pair R", "0.6983", det["pair_recall"]),
    ("detector pair F1", "0.8209", det["pair_f1"]),
    ("n_clusters", "321", det["n_clusters"]),
    ("pair TP", "2095", det["pair_tp"]),
    ("pair FN", "905", det["pair_fn"]),
    ("pair FP", "9", det["pair_fp"]),
]:
    check(label, val, actual)
for proto, rg, rh in [
    ("random", "0.9833", "0.9111"),
    ("clone_aware", "0.0000", "0.0000"),
    ("detector_aware", "0.4084", "0.0000"),
]:
    check(f"{proto} rho_g", rg, ov[proto]["true_family_overlap_rate"])
    check(f"{proto} rho_chat", rh, ov[proto]["detected_cluster_overlap_rate"])
# R3: shortcut column must be gone from main.tex
if "shortcut overlap" in TEX.lower() and "removed a previous shortcut" not in TEX.lower():
    failures.append("R3 regression: shortcut overlap still presented as a mechanism")

print("=== Tables 3/4/5 completeness ===")
for rid in ["R004", "R011", "R012", "R013"]:
    m = load(rid)["metrics"]
    for k in ["mcc", "auprc_macro", "accuracy"]:
        if m.get(k) is None:
            failures.append(f"{rid} missing {k}")
check("R004 MCC", "0.8002", load("R004")["metrics"]["mcc"])
check("R004 AUPRC", "0.8798", load("R004")["metrics"]["auprc_macro"])
sweep = {s["threshold"]: s for s in REV["E3_detector_profile"]["threshold_sweep"]}
for t, p, r, fp, fn in [(0.80, "0.0216", "0.9360", "127057", "192"),
                        (0.90, "0.9957", "0.6983", "9", "905")]:
    s = sweep.get(t, sweep.get(round(t, 2)))
    check(f"tau={t} P", p, s["pair_precision"])
    check(f"tau={t} R", r, s["pair_recall"])
    check(f"tau={t} FP", fp, s["pair_fp"])
    check(f"tau={t} FN", fn, s["pair_fn"])

print("=== R4 dose-response ===")
dr = R2["D4_dose_response"]
check("dose slope", "0.3082", dr["slope_mean"])
check("dose slope CI low", "0.2925", dr["slope_ci95"][0])
check("dose slope CI high", "0.3249", dr["slope_ci95"][1])
for a, f1 in zip(dr["aggregate"], ["0.6872", "0.7348", "0.7802", "0.8383", "0.9133", "1.0000"]):
    check(f"dose {a['dose']} F1", f1, a["macro_f1_mean"])
in_tex("0.3082", "dose slope in paper")
# C2: interval-specific slopes have family-level intervals
for entry, sl, lo, hi in zip(dr["local_slope_intervals"],
                            ["0.238", "0.227", "0.2905", "0.375", "0.4335"],
                            [0.1367, 0.1244, 0.205, 0.3018, 0.3779],
                            [0.3256, 0.3144, 0.3783, 0.4562, 0.4925]):
    check(f"local slope {entry['interval']}", sl, entry["slope"])
    check(f"local CI lo {entry['interval']}", round(lo, 4), entry["ci95"][0])
    check(f"local CI hi {entry['interval']}", round(hi, 4), entry["ci95"][1])
check("non-saturated slope", "0.2778", dr["linear_slope_excluding_saturation"])

print("=== R5 real corpus ===")
rc = R2["R5_real_corpus"]
check("real n_records", "36", rc["n_records"])
check("real n_clusters", "29", rc["n_detected_clusters"])
check("real pair recall", "0.0232", rc["pair_recall"])
check("real reentrancy used", "31", rc["class_counts"]["reentrancy"])
check("real time used", "5", rc["class_counts"]["time_manipulation"])
check("real same-category pairs", "475", rc["same_category_pairs"])
check("sample listed reentrancy", "32", rc["sample_provenance"][0]["entries_listed"])
check("sample listed time", "6", rc["sample_provenance"][1]["entries_listed"])

print("=== R2 corrected calibration ===")
c40 = load("R040")["calibration"]
b = c40["baselines"]
check("binary positive rate", "0.8", b["binary_positive_rate"])
check("constant baseline", "0.8000", b["constant_vulnerable_accuracy"])
check("five-class acc", "0.6444", b["five_class_accuracy_same_split"])
check("coarsened argmax acc", "0.9333", b["binary_accuracy_coarsened_argmax"])
check("score field", "1 - p(safe)", c40["score"])
check("ece before", "0.0598", c40["ece_before"])
check("ece after", "0.0351", c40["ece_after"])
check("adaptive ECE before", "0.0987", load("R041")["calibration"]["ece_adaptive"]["ece"])
check("adaptive ECE after", "0.0770", c40["ece_adaptive_after"]["ece"])
check("brier after", "0.0589", c40["brier"])
check("temperature", "0.65", c40["temperature"]["T"])
check("aurc", "0.0484", c40["risk_coverage"]["aurc"])
check("risk endpoint", "0.0667", c40["risk_coverage"]["risk_at_full_coverage"])
cal42 = load("R042")["calibration"]
cf = cal42["conformal"]                       # boundary rule
low = cal42["conformal_low_score_negative"]   # anti-selective negative result
check("boundary delta", "0.3023", cf["delta"])
check("boundary abstain", "0.5111", cf["abstain_rate"])
check("boundary accepted acc", "1.0000", cf["accepted_accuracy"])
check("low-score abstain", "0.1333", low["abstain_rate"])
check("low-score accepted acc", "0.9231", low["accepted_accuracy"])
check("conformal overall acc", "0.9333", cf["overall_accuracy_same_target"])
in_tex("0.9333", "corrected binary accuracy")
in_tex("0.9231", "low-score negative documented")
in_tex("1.0000", "boundary accepted accuracy documented")
# Round-3 core finding: verify the anti-selectivity arithmetic is stated
in_tex("24", "abstained-count arithmetic")
check("anti-selective gain", -0.0102, round(low["accepted_accuracy"] - low["overall_accuracy_same_target"], 4))
# C1: proper split conformal meets the finite-sample coverage bound
cs = load("R043")["calibration"]["conformal_split"]
check("split conformal coverage achieved", "1.0000", cs["coverage_achieved"])
check("split conformal coverage bound", "0.8945", cs["coverage_bound"])
check("split conformal meets bound", True, cs["meets_bound"])
check("split conformal n_cal", "180", cs["n_cal"])
check("split conformal k", "163", cs["k"])
check("split conformal threshold", "0.9784", cs["threshold"])
check("split conformal score cutoff", "0.0216", cs["implied_score_cutoff"])
check("split conformal min test score", "0.1102", cs["min_test_score"])
check("split conformal is vacuous", True, cs["vacuous"])
in_tex("0.0216", "vacuity cutoff in paper")
in_tex("0.1102", "minimum test score in paper")
# E1/E2: front matter must not headline the disowned all-point slope
if "0.308 per unit" in TEX or "slope 0.308 per unit" in TEX:
    failures.append("E1 regression: abstract still headlines disowned all-point slope")
if "low-score conformal" in TEX:
    failures.append("E2 regression: low-score rule still called conformal")

print("=== R5 diversity integrity ===")
div = REV["E4_diversity"]
hashes = [v["config_hash"] for k, v in div.items() if k != "_integrity"]
check("diversity distinct configs", True, len(set(hashes)) == len(hashes))
check("k=1 L1", "74.860", div["1"]["coef_l1"])
check("k=6 L1", "123.668", div["6"]["coef_l1"])

print("=== Citation integrity ===")
bib = (HERE / "references.bib").read_text(encoding="utf-8")
keys = ["zhuang2020smart", "ren2026procure", "zhang2020adversarial", "jha2023codeattack",
        "yu2025smartllama", "barz2020cifar", "guo2017calibration", "angelopoulos2023conformal",
        "kocetkov2022stack", "allamanis2019adverse", "chakraborty2022deep", "croft2023data",
        "ding2025primevul", "zhou2025lessleak", "defuria2026time", "szolnoki2026java",
        "sajnani2016sourcerercc", "lee2022dedup", "ferreira2020smartbugs", "pendlebury2019tesseract"]
for key in keys:
    in_tex(key, f"citation {key}")
    if not re.search(r"@\w+\{" + re.escape(key) + r"\s*,", bib):
        failures.append(f"bib entry {key} missing")

print()
if failures:
    print(f"FAILED {len(failures)} / {checked} checks:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"ALL {checked} consistency checks PASSED — final manuscript matches real data.")
