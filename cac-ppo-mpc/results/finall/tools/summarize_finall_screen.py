from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "screen_v6_all_adolescents_24h_r1" / "summary.csv"
PAIRED_OUT = ROOT / "screen_v6_all_adolescents_24h_r1_paired.csv"
SUMMARY_OUT = ROOT / "screen_v6_all_adolescents_24h_r1_summary.csv"


def main() -> None:
    frame = pd.read_csv(SOURCE)
    wanted = frame[frame["mode"].isin(["ppo_shield", "switch_nogate_shield"])].copy()
    metrics = ["tir", "tbr70", "tar180", "tar250", "mean_bg", "total_insulin_u"]
    wide = wanted.pivot(index="patient", columns="mode", values=metrics)

    paired = pd.DataFrame(index=wide.index)
    for metric in metrics:
        paired[f"ppo_{metric}"] = wide[(metric, "ppo_shield")]
        paired[f"switch_{metric}"] = wide[(metric, "switch_nogate_shield")]
        paired[f"delta_{metric}"] = (
            paired[f"switch_{metric}"] - paired[f"ppo_{metric}"]
        )
    paired.reset_index().to_csv(PAIRED_OUT, index=False)

    rows = []
    for metric in metrics:
        delta = paired[f"delta_{metric}"]
        rows.append(
            {
                "metric": metric,
                "ppo_mean": paired[f"ppo_{metric}"].mean(),
                "switch_mean": paired[f"switch_{metric}"].mean(),
                "mean_paired_delta_switch_minus_ppo": delta.mean(),
                "patients_delta_positive": int((delta > 1e-12).sum()),
                "patients_delta_zero": int((delta.abs() <= 1e-12).sum()),
                "patients_delta_negative": int((delta < -1e-12).sum()),
                "n_patients": int(len(delta)),
            }
        )
    pd.DataFrame(rows).to_csv(SUMMARY_OUT, index=False)


if __name__ == "__main__":
    main()
