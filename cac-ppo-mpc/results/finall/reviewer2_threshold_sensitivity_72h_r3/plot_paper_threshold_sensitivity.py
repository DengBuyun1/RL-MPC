from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
SETTINGS = [
    "current",
    "low_threshold_120",
    "low_threshold_140",
    "fast_fall_m1p5",
    "fast_fall_m2p5",
    "high_threshold_150_170",
    "high_threshold_170_190",
    "iob_guard_120",
    "iob_guard_140",
]
SETTING_LABELS = {
    "current": "Current",
    "low_threshold_120": "Low threshold 120",
    "low_threshold_140": "Low threshold 140",
    "fast_fall_m1p5": "Fast-fall −1.5",
    "fast_fall_m2p5": "Fast-fall −2.5",
    "high_threshold_150_170": "High thresholds 150/170",
    "high_threshold_170_190": "High thresholds 170/190",
    "iob_guard_120": "IOB guard 120",
    "iob_guard_140": "IOB guard 140",
}
METRICS = ["tir", "tbr70", "tbr54", "tar180", "tar250"]


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


def derive_bg_table() -> pd.DataFrame:
    reference = pd.read_csv(ROOT / "ppo_reference" / "summary.csv")
    ppo = reference.groupby("patient")[METRICS].mean()
    rows: list[dict[str, float | int | str]] = []
    for setting_index, setting in enumerate(SETTINGS):
        data = pd.read_csv(ROOT / setting / "summary.csv")
        switch = data.groupby("patient")[METRICS].mean()
        for metric_index, metric in enumerate(METRICS):
            delta = (switch[metric] - ppo[metric]).to_numpy(float)
            try:
                p_value = float(wilcoxon(delta, alternative="two-sided", method="auto").pvalue)
            except ValueError:
                p_value = 1.0
            low, high = bootstrap_mean_ci(delta, 20260920 + setting_index * 20 + metric_index)
            rows.append(
                {
                    "metric_source": "BG",
                    "metric": metric,
                    "setting": setting,
                    "setting_label": SETTING_LABELS[setting],
                    "n_patients": len(delta),
                    "ppo_mean": float(ppo[metric].mean()),
                    "switch_mean": float(switch[metric].mean()),
                    "mean_effect_switch_minus_ppo": float(delta.mean()),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "wilcoxon_p": p_value,
                    "patients_with_positive_difference": int((delta > 0).sum()),
                }
            )
    result = pd.DataFrame(rows)
    result["bh_fdr_q_within_metric"] = np.nan
    for metric, indices in result.groupby("metric").groups.items():
        result.loc[indices, "bh_fdr_q_within_metric"] = bh_adjust(
            result.loc[indices, "wilcoxon_p"].to_numpy(float)
        )
    return result


def load_cgm_table() -> pd.DataFrame:
    source = pd.read_csv(ROOT / "threshold_vs_ppo_bh_fdr.csv")
    result = source.rename(
        columns={
            "mean_effect_switch_minus_ppo": "mean_effect_switch_minus_ppo",
            "bootstrap_ci95_low": "bootstrap_ci95_low",
            "bootstrap_ci95_high": "bootstrap_ci95_high",
        }
    ).copy()
    result["metric"] = result["metric"].str.replace("cgm_", "", regex=False)
    result["metric_source"] = "CGM"
    result["setting_label"] = result["setting"].map(SETTING_LABELS)
    result["n_patients"] = 10
    return result


def make_figure(table: pd.DataFrame, metric_source: str, output_stem: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.titlesize": 8.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue = "#0072B2"
    orange = "#D55E00"
    gray = "#777777"

    shown_metrics = ["tir", "tbr70", "tar180", "tar250"]
    titles = [
        "(a) TIR\nhigher is better",
        "(b) TBR <70\nlower is better",
        "(c) TAR >180\nlower is better",
        "(d) TAR >250\nlower is better",
    ]
    x_limits = [(-0.3, 2.9), (-0.8, 1.2), (-3.3, 0.4), (-2.1, 0.4)]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.25), sharey=True)
    y_positions = np.arange(len(SETTINGS))[::-1]
    for panel_index, (ax, metric, title, xlim) in enumerate(zip(axes, shown_metrics, titles, x_limits)):
        block = table.loc[table["metric"] == metric].set_index("setting").loc[SETTINGS]
        for ypos, setting in zip(y_positions, SETTINGS):
            row = block.loc[setting]
            effect = float(row["mean_effect_switch_minus_ppo"])
            low = float(row["bootstrap_ci95_low"])
            high = float(row["bootstrap_ci95_high"])
            q_value = float(row["bh_fdr_q_within_metric"])
            color = orange if setting == "current" else blue
            marker = "D" if setting == "current" else "o"
            face = color if q_value < 0.05 else "white"
            ax.errorbar(
                effect,
                ypos,
                xerr=[[effect - low], [high - effect]],
                fmt=marker,
                color=color,
                ecolor=color,
                markerfacecolor=face,
                markeredgecolor=color,
                markeredgewidth=0.9,
                ms=4.0,
                capsize=2.0,
                lw=0.9,
                zorder=2,
            )
        ax.axvline(0, color=gray, lw=0.8, ls="--")
        ax.set_xlim(*xlim)
        ax.grid(axis="x", color="#EAEAEA", lw=0.55)
        ax.set_title(title)
        ax.set_xlabel("Switch − PPO (pp)")
        ax.tick_params(axis="y", length=0)
        if panel_index == 0:
            ax.set_yticks(y_positions, [SETTING_LABELS[s] for s in SETTINGS])
        else:
            ax.tick_params(labelleft=False)

    fig.text(
        0.995,
        0.015,
        f"{metric_source} endpoints; points are patient-level mean effects, bars are 95% bootstrap CIs (n=10). "
        "Orange diamond: current setting; filled markers: BH-FDR q<0.05 within endpoint.",
        ha="right",
        va="bottom",
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.235, right=0.995, top=0.84, bottom=0.23, wspace=0.34)
    base = ROOT / output_stem
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
    bg = derive_bg_table()
    cgm = load_cgm_table()
    bg.to_csv(ROOT / "paper_threshold_sensitivity_bg_table.csv", index=False, encoding="utf-8-sig")
    cgm.to_csv(ROOT / "supp_threshold_sensitivity_cgm_table.csv", index=False, encoding="utf-8-sig")
    compact_rows = []
    for metric in METRICS:
        block = bg.loc[bg["metric"] == metric]
        current = block.loc[block["setting"] == "current"].iloc[0]
        compact_rows.append(
            {
                "metric": metric,
                "current_effect_pp": current["mean_effect_switch_minus_ppo"],
                "current_ci95_low": current["bootstrap_ci95_low"],
                "current_ci95_high": current["bootstrap_ci95_high"],
                "minimum_effect_across_settings_pp": block["mean_effect_switch_minus_ppo"].min(),
                "maximum_effect_across_settings_pp": block["mean_effect_switch_minus_ppo"].max(),
                "settings_with_bh_fdr_q_below_0_05": int((block["bh_fdr_q_within_metric"] < 0.05).sum()),
                "settings_tested": len(block),
            }
        )
    pd.DataFrame(compact_rows).to_csv(
        ROOT / "paper_threshold_sensitivity_compact_table.csv", index=False, encoding="utf-8-sig"
    )
    make_figure(bg, "BG", "paper_threshold_sensitivity_bg")
    make_figure(cgm, "CGM", "supp_threshold_sensitivity_cgm")
    print("Wrote BG and CGM threshold-sensitivity tables and figures.")


if __name__ == "__main__":
    main()
