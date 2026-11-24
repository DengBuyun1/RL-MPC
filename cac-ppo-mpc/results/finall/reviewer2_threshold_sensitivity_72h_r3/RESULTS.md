# Reviewer-2 threshold sensitivity results

## Integrity and analysis unit

The matrix contains one PPO reference and nine switch settings.  Each cell
contains ten adolescent virtual patients, 72 hours and three paired repeats.
All 300 planned runs completed.  The patient, after averaging its three
repeats, is the statistical unit.  BH-FDR is applied across threshold settings
within each endpoint family.

Every directory has `claim_valid=false` because the evaluator marks any early
termination as invalid.  This is not hidden: PPO had three severe-hyperglycaemia
terminations and every switch setting had two.  No switch-only early
termination occurred.

## Current setting versus PPO

With the new seed family, the current setting produced:

- CGM-TIR: 72.32% to 73.88%, a change of +1.56 percentage points
  (95% patient bootstrap CI +0.56 to +2.42; Wilcoxon p=0.037,
  BH-FDR q=0.037);
- CGM-TAR >180 mg/dL: -1.93 points (q=0.0098);
- CGM-TAR >250 mg/dL: -1.06 points (q=0.0417);
- CGM-TBR <70 mg/dL: +0.37 points (q=0.297; no statistically significant
  difference after correction);
- CGM-TBR <54 mg/dL: +0.19 points (q=0.080; no statistically significant
  difference after correction).

Nine of ten patients had higher TIR.  The directions reproduce the earlier
five-repeat formal experiment: TIR and TAR improve, while hypoglycaemia
exposure shifts upward and cannot be claimed to improve.

## Sensitivity across thresholds

Across all nine settings:

- TIR improvement versus PPO ranged from +1.37 to +1.69 points.  Every setting
  remained significant after within-endpoint BH-FDR correction, and every
  setting improved TIR in nine of ten patients.
- The switch-setting mean TIR range was only 0.32 points.
- TAR >180 improvement ranged from -1.70 to -2.08 points; TAR >250 improvement
  ranged from -0.89 to -1.16 points.  Both remained significant for every
  tested setting after correction.
- TBR <70 changes ranged from +0.33 to +0.39 points.  None was significant
  after correction, but all estimates were positive, so the study does not
  support a hypoglycaemia-safety improvement claim.
- Relative to the current threshold set, none of the eight alternatives
  changed TIR or TBR <70 significantly after BH-FDR correction.

The largest local response came from the IOB projection guard.  Lowering it
from 130 to 120 mg/dL increased mean TIR by 0.13 points relative to the current
setting; raising it to 140 mg/dL reduced mean TIR by 0.19 points.  Neither
difference remained statistically significant after correction.  The
fast-fall settings -1.5, -2.0 and -2.5 mg/dL/min were practically identical in
their final clinical metrics.  The corresponding fast-fall reason appeared in
5.72%, 2.79% and 1.44% of policy-history rows, respectively, so the threshold
did change the risk label frequency.  Its small outcome effect instead suggests
that the lower-insulin candidate comparison and the shared outer safety
governor often produced the same effective action after that label changed.

## Reviewer-facing conclusion

Within the tested local ranges, the central result is not caused by selecting
one precise threshold: all one-factor perturbations retain positive TIR and
negative TAR effects, and no alternative differs significantly from the
current setting on TIR or TBR <70 after multiplicity correction.  This supports
local threshold robustness, not global parameter insensitivity.  The safety
trade-off remains and must be reported together with the separate empirical
safety/closed-loop regularity analysis.
