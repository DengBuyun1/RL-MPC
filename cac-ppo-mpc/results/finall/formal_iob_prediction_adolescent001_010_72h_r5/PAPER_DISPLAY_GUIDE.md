# Formal experiment: recommended paper display

## Recommended placement

Use `paper_formal_control_results` as the main new result figure after the
enhanced risk-state and IOB-prediction switching method has been introduced.
Use `paper_formal_primary_table.csv` for the exact result table.  The figure
answers whether the enhanced switch changes final control outcomes; the table
provides the values needed for verification.

## Suggested English caption

**Fig. X. Control outcomes of PPO and the enhanced risk-aware switch.**
(a) Patient-level changes in BG time in range (TIR), after averaging five
repeats for each of ten adolescent virtual patients.  Nine of ten patients
showed higher TIR.  (b) Mean patient-level differences between the switch and
PPO; error bars are 95% bootstrap confidence intervals.  Filled markers denote
BH-FDR-adjusted q<0.05 across the five glucose endpoints.  Orange markers denote
low-glucose endpoints and are not interpreted as a safety advantage.  (c)
Early-termination rates with exact 95% binomial confidence intervals.  Glucose
metrics use the simulator's internal BG, consistent with the original paper.

## Statistical scope

- Statistical unit: virtual patient (n=10), after averaging five repeats.
- Primary display: all observed trajectory segments; no post-termination
  values are imputed.
- Sensitivity set: `paper_formal_complete_pair_table.csv`, retaining 44 matched
  pairs that completed the full 72 h.
- Paired test: two-sided Wilcoxon signed-rank test.
- Uncertainty: 20,000 patient-level bootstrap resamples.
- Multiplicity: BH-FDR across TIR, TBR70, TBR54, TAR180 and TAR250.

## Claims that the display supports

- TIR increased by 1.37 percentage points and improved in 9/10 patients.
- TAR180 and TAR250 decreased by 1.66 and 1.19 points, respectively.
- The complete-pair analysis retained a 1.57-point TIR increase.

## Claims that should not be made

- Do not claim that early termination improved: 6/50 versus 5/50 is
  descriptive and its intervals overlap widely.
- Do not claim a hypoglycaemia-safety advantage.  TBR70 and TBR54 point
  estimates increased; the BG differences did not remain significant after
  BH-FDR correction.
- Do not attribute these outcome improvements to the prediction-error gate
  alone.  They apply to the enhanced risk-state and IOB-prediction switch.
