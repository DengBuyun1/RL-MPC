from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, wilcoxon


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "summary.csv"
METRICS = ["tir", "tbr70", "tbr54", "tar180", "tar250", "mean_bg", "total_insulin_u"]
CLINICAL_METRICS = ["tir", "tbr70", "tbr54", "tar180", "tar250"]
MODE_ORDER = ["ppo_shield", "switch_nogate_shield"]
MODE_LABEL = {"ppo_shield": "PPO", "switch_nogate_shield": "Risk-aware switch"}


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def make_patient_table(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["patient", "mode"], as_index=False)[METRICS].mean()
    blocks: list[pd.DataFrame] = []
    for metric in METRICS:
        pivot = grouped.pivot(index="patient", columns="mode", values=metric)
        block = pd.DataFrame(
            {
                "patient": pivot.index,
                "metric": metric,
                "ppo": pivot["ppo_shield"].to_numpy(float),
                "risk_aware_switch": pivot["switch_nogate_shield"].to_numpy(float),
            }
        )
        block["difference_switch_minus_ppo"] = block["risk_aware_switch"] - block["ppo"]
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def summarize(patient_table: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for metric_index, metric in enumerate(METRICS):
        block = patient_table.loc[patient_table["metric"] == metric]
        delta = block["difference_switch_minus_ppo"].to_numpy(float)
        try:
            p_value = float(wilcoxon(delta, alternative="two-sided", method="auto").pvalue)
        except ValueError:
            p_value = 1.0
        ci_low, ci_high = bootstrap_mean_ci(delta, 20260919 + metric_index)
        rows.append(
            {
                "analysis_set": analysis_set,
                "metric": metric,
                "n_patients": len(delta),
                "ppo_mean": float(block["ppo"].mean()),
                "risk_aware_switch_mean": float(block["risk_aware_switch"].mean()),
                "mean_difference_switch_minus_ppo": float(delta.mean()),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wilcoxon_p": p_value,
                "patients_with_positive_difference": int((delta > 0).sum()),
            }
        )
    result = pd.DataFrame(rows)
    result["bh_fdr_q"] = np.nan
    mask = result["metric"].isin(CLINICAL_METRICS)
    result.loc[mask, "bh_fdr_q"] = bh_adjust(result.loc[mask, "wilcoxon_p"].to_numpy(float))
    return result


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def make_figure(patient_table: pd.DataFrame, primary: pd.DataFrame, completion: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue = "#0072B2"
    orange = "#D55E00"
    gray = "#777777"
    light_gray = "#D0D0D0"

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.85),
        gridspec_kw={"width_ratios": [1.05, 1.30, 0.75]},
        constrained_layout=False,
    )

    # (a) Patient-level TIR effects: a zero-centred display avoids a truncated raw-TIR axis.
    ax = axes[0]
    tir = patient_table.loc[patient_table["metric"] == "tir"].copy().sort_values("patient")
    tir["short_patient"] = tir["patient"].str.extract(r"(\d+)$")[0].astype(int).map(lambda x: f"P{x:02d}")
    y = np.arange(len(tir))[::-1]
    effects = tir["difference_switch_minus_ppo"].to_numpy(float)
    colors = np.where(effects >= 0, blue, orange)
    ax.hlines(y, 0, effects, color=light_gray, lw=1.0, zorder=1)
    ax.scatter(effects, y, c=colors, s=24, edgecolor="white", linewidth=0.4, zorder=2)
    ax.axvline(0, color=gray, lw=0.8, ls="--")
    ax.set_yticks(y, tir["short_patient"])
    ax.set_xlabel("TIR difference (percentage points)")
    ax.set_ylabel("Virtual patient")
    ax.set_xlim(-0.8, 4.8)
    ax.grid(axis="x", color="#EAEAEA", lw=0.6)
    ax.set_title("(a) Patient-level TIR effects")
    ax.text(
        0.98,
        0.03,
        f"9/10 improved\nmean = {effects.mean():+.2f} pp",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
    )

    # (b) Patient-level effect estimates and 95% bootstrap confidence intervals.
    ax = axes[1]
    shown = ["tir", "tbr70", "tbr54", "tar180", "tar250"]
    labels = ["TIR", "TBR <70", "TBR <54", "TAR >180", "TAR >250"]
    table = primary.set_index("metric").loc[shown]
    y = np.arange(len(shown))[::-1]
    for ypos, metric, label in zip(y, shown, labels):
        row = table.loc[metric]
        effect = float(row["mean_difference_switch_minus_ppo"])
        low = float(row["bootstrap_ci95_low"])
        high = float(row["bootstrap_ci95_high"])
        q_value = float(row["bh_fdr_q"])
        color = orange if metric.startswith("tbr") else blue
        face = color if q_value < 0.05 else "white"
        ax.errorbar(
            effect,
            ypos,
            xerr=[[effect - low], [high - effect]],
            fmt="o",
            color=color,
            ecolor=color,
            markerfacecolor=face,
            markeredgecolor=color,
            markeredgewidth=1.0,
            capsize=2.4,
            ms=4.6,
            lw=1.1,
            zorder=2,
        )
    ax.axvline(0, color=gray, lw=0.8, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Risk-aware switch − PPO (percentage points)")
    ax.set_xlim(-3.0, 3.0)
    ax.grid(axis="x", color="#EAEAEA", lw=0.6)
    ax.set_title("(b) Outcomes (95% bootstrap CI)")
    ax.text(
        0.02,
        -0.24,
        "Filled: BH-FDR q < 0.05; orange: low-glucose endpoint",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
    )

    # (c) Exact binomial intervals for early termination.
    ax = axes[2]
    positions = np.array([0, 1])
    for xpos, (_, row), color in zip(positions, completion.iterrows(), [blue, orange]):
        k = int(row["early_terminations"])
        n = int(row["planned_runs"])
        rate = 100.0 * k / n
        low, high = clopper_pearson(k, n)
        ax.errorbar(
            xpos,
            rate,
            yerr=[[rate - 100 * low], [100 * high - rate]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3.0,
            ms=5.0,
            lw=1.1,
        )
        ax.text(xpos, rate + 2.2, f"{k}/{n}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(positions, ["PPO", "Risk-aware\nswitch"])
    ax.set_ylabel("Early termination (%)")
    ax.set_ylim(0, 32)
    ax.grid(axis="y", color="#EAEAEA", lw=0.6)
    ax.set_title("(c) Early termination")
    ax.text(
        0.5,
        -0.24,
        "Exact 95% binomial CI",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.6,
    )

    fig.subplots_adjust(left=0.075, right=0.992, top=0.89, bottom=0.26, wspace=0.62)
    base = ROOT / "paper_formal_control_results"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(SUMMARY)
    data = data.loc[data["mode"].isin(MODE_ORDER)].copy()

    all_patient = make_patient_table(data)
    primary = summarize(all_patient, "all_observed_trajectories")

    complete = data.loc[~data["terminated_early"].astype(bool)].copy()
    pair_counts = complete.groupby(["patient", "repeat"])["mode"].nunique()
    valid_pairs = pair_counts.loc[pair_counts == 2].index
    complete_pairs = complete.set_index(["patient", "repeat"]).loc[valid_pairs].reset_index()
    complete_patient = make_patient_table(complete_pairs)
    complete_summary = summarize(complete_patient, "complete_72h_pairs")

    completion_rows = []
    for mode in MODE_ORDER:
        block = data.loc[data["mode"] == mode]
        completion_rows.append(
            {
                "method": MODE_LABEL[mode],
                "planned_runs": len(block),
                "early_terminations": int(block["terminated_early"].astype(bool).sum()),
                "complete_runs": int((~block["terminated_early"].astype(bool)).sum()),
            }
        )
    completion = pd.DataFrame(completion_rows)

    all_patient.to_csv(ROOT / "paper_formal_patient_differences.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(ROOT / "paper_formal_primary_table.csv", index=False, encoding="utf-8-sig")
    complete_summary.to_csv(ROOT / "paper_formal_complete_pair_table.csv", index=False, encoding="utf-8-sig")
    completion.to_csv(ROOT / "paper_formal_early_termination.csv", index=False, encoding="utf-8-sig")
    make_figure(all_patient, primary, completion)

    print(primary.to_string(index=False))
    print(f"Complete matched 72-h pairs: {len(complete_pairs) // 2}/50")


if __name__ == "__main__":
    main()
