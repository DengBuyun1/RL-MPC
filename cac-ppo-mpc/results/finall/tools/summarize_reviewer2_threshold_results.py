"""Create reviewer-facing BH-FDR tables for the completed threshold study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = (
    "cgm_tir", "cgm_tbr70", "cgm_tbr54", "cgm_tar180", "cgm_tar250",
    "cgm_mean_bg", "total_insulin_u",
)


def bh_fdr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def paired_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, zero_method="wilcox", method="auto").pvalue)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def summarize(root: Path) -> dict:
    root = Path(root)
    summary = pd.read_csv(root / "threshold_setting_summary.csv")
    paired = pd.read_csv(root / "threshold_patient_paired.csv")

    versus_ppo = []
    for metric in METRICS:
        p_values = summary[f"delta_{metric}_wilcoxon_p"].to_numpy(float)
        q_values = bh_fdr(p_values)
        for (_, row), q_value in zip(summary.iterrows(), q_values):
            versus_ppo.append({
                "metric": metric,
                "setting": row["setting"],
                "ppo_mean": row[f"ppo_{metric}"],
                "switch_mean": row[f"switch_{metric}"],
                "mean_effect_switch_minus_ppo": row[f"delta_{metric}"],
                "bootstrap_ci95_low": row[f"delta_{metric}_ci95_low"],
                "bootstrap_ci95_high": row[f"delta_{metric}_ci95_high"],
                "wilcoxon_p": row[f"delta_{metric}_wilcoxon_p"],
                "bh_fdr_q_within_metric": q_value,
            })
    versus_ppo = pd.DataFrame(versus_ppo)

    current = paired.loc[paired["setting"].eq("current")].set_index("patient")
    alternatives = sorted(set(paired["setting"]) - {"current"})
    versus_current = []
    for metric_index, metric in enumerate(METRICS):
        metric_rows = []
        for setting_index, setting in enumerate(alternatives):
            candidate = paired.loc[paired["setting"].eq(setting)].set_index("patient")
            delta = (
                candidate[f"switch_{metric}"] - current[f"switch_{metric}"]
            ).to_numpy(float)
            low, high = bootstrap_ci(delta, 20260920 + 100 * metric_index + setting_index)
            metric_rows.append({
                "metric": metric,
                "setting": setting,
                "mean_change_from_current": float(delta.mean()),
                "median_change_from_current": float(np.median(delta)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "wilcoxon_p": paired_pvalue(delta),
                "patients_increased": int((delta > 0).sum()),
                "patients_decreased": int((delta < 0).sum()),
            })
        q_values = bh_fdr(np.array([row["wilcoxon_p"] for row in metric_rows]))
        for row, q_value in zip(metric_rows, q_values):
            row["bh_fdr_q_within_metric"] = float(q_value)
            versus_current.append(row)
    versus_current = pd.DataFrame(versus_current)

    metric_ranges = []
    for metric in METRICS:
        means = paired.groupby("setting")[f"switch_{metric}"].mean()
        metric_ranges.append({
            "metric": metric,
            "minimum_setting": means.idxmin(),
            "minimum_mean": float(means.min()),
            "maximum_setting": means.idxmax(),
            "maximum_mean": float(means.max()),
            "range_across_settings": float(means.max() - means.min()),
        })
    metric_ranges = pd.DataFrame(metric_ranges)

    versus_ppo.to_csv(root / "threshold_vs_ppo_bh_fdr.csv", index=False)
    versus_current.to_csv(root / "threshold_vs_current_bh_fdr.csv", index=False)
    metric_ranges.to_csv(root / "threshold_metric_ranges.csv", index=False)

    primary = versus_ppo.loc[versus_ppo["metric"].eq("cgm_tir")]
    current_row = summary.loc[summary["setting"].eq("current")].iloc[0]
    report = {
        "schema": "reviewer2-threshold-sensitivity-interpretation-v1",
        "analysis_status": "complete",
        "statistical_unit": "patient_mean_over_three_repeats",
        "multiplicity": "BH-FDR across threshold settings within each endpoint family",
        "current_setting": {
            "delta_cgm_tir_pp": float(current_row["delta_cgm_tir"]),
            "delta_cgm_tbr70_pp": float(current_row["delta_cgm_tbr70"]),
            "delta_cgm_tbr54_pp": float(current_row["delta_cgm_tbr54"]),
            "delta_cgm_tar180_pp": float(current_row["delta_cgm_tar180"]),
            "delta_cgm_tar250_pp": float(current_row["delta_cgm_tar250"]),
            "ppo_early_terminations": int(current_row["ppo_early_terminations"]),
            "switch_early_terminations": int(current_row["switch_early_terminations"]),
            "switch_only_early_terminations": int(current_row["switch_only_early_terminations"]),
        },
        "robustness": {
            "all_settings_positive_tir_effect": bool((primary["mean_effect_switch_minus_ppo"] > 0).all()),
            "all_settings_tir_q_below_0_05": bool((primary["bh_fdr_q_within_metric"] < 0.05).all()),
            "tir_effect_min_pp": float(primary["mean_effect_switch_minus_ppo"].min()),
            "tir_effect_max_pp": float(primary["mean_effect_switch_minus_ppo"].max()),
            "tir_switch_mean_range_pp": float(
                metric_ranges.loc[metric_ranges["metric"].eq("cgm_tir"), "range_across_settings"].iloc[0]
            ),
            "any_alternative_differs_from_current_tir_after_fdr": bool((
                versus_current.loc[versus_current["metric"].eq("cgm_tir"), "bh_fdr_q_within_metric"] < 0.05
            ).any()),
            "any_alternative_differs_from_current_tbr70_after_fdr": bool((
                versus_current.loc[versus_current["metric"].eq("cgm_tbr70"), "bh_fdr_q_within_metric"] < 0.05
            ).any()),
        },
        "claim_boundary": (
            "The sensitivity study supports robustness within the tested local threshold ranges; "
            "it does not establish global parameter insensitivity or improved hypoglycaemia safety."
        ),
    }
    (root / "threshold_sensitivity_interpretation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
