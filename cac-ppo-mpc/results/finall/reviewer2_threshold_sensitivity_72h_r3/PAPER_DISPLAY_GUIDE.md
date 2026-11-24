# Threshold sensitivity: recommended paper display

## Recommended placement

For a short conference paper, put the compact values from
`paper_threshold_sensitivity_compact_table.csv` in the main text and use
`paper_threshold_sensitivity_bg` in the response letter or supplementary
material.  If the full sensitivity evidence must appear in the paper, use the
figure and omit a redundant full numerical table.

The BG figure is the recommended version for `PPO_MPC1.pdf`, because the
original paper defines TIR/TBR/TAR using simulator BG.  The parallel
`supp_threshold_sensitivity_cgm` figure preserves the experiment's original
CGM analysis and can be supplied as supplementary evidence.

## Suggested English caption

**Fig. Y. Local sensitivity of the enhanced switch to its decision
thresholds.** Patient-level mean differences between each switch setting and
the shared PPO reference are shown for ten adolescent virtual patients and
three paired 72-h repeats per patient.  Error bars are 95% bootstrap confidence
intervals.  The orange diamond denotes the current setting; filled markers
denote BH-FDR-adjusted q<0.05 across the nine settings within each endpoint.
All tested settings retained positive TIR and negative TAR effects, whereas
TBR70 did not differ significantly from PPO.  The analysis supports local
robustness within the tested ranges, not global parameter insensitivity.

## Statistical scope

- Statistical unit: virtual patient (n=10), after averaging three repeats.
- Paired test: two-sided Wilcoxon signed-rank test.
- Uncertainty: 20,000 patient-level bootstrap resamples.
- Multiplicity: BH-FDR across the nine threshold settings separately within
  each endpoint family.
- The BG table is derived directly from each setting's `summary.csv`; the CGM
  table uses the experiment's existing `threshold_vs_ppo_bh_fdr.csv`.

## Claims that the display supports

- Across all nine settings, the direction of TIR and TAR changes was stable.
- For BG endpoints, TIR effects ranged from +1.18 to +1.53 percentage points;
  all nine settings had q<0.05 within the TIR family.
- TAR180 and TAR250 decreased under all nine settings.
- The small spread across settings suggests local threshold robustness.

## Claims that should not be made

- Do not call the method globally insensitive to parameter choice; only local,
  one-factor perturbations were tested.
- Do not present the positive TBR70 differences as an improvement.  They are a
  safety trade-off and were not significant after correction.
- Do not compare the absolute BG and CGM values as if they came from the same
  measurement source.
