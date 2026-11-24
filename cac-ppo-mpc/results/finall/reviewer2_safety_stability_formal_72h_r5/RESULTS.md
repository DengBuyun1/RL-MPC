# Empirical safety and closed-loop regularity results

This analysis uses all 100 formal runs.  The patient is the statistical unit;
five repeats are averaged within each patient and controller.  A second table
uses the 44 disturbance pairs for which both controllers completed 72 hours.

## Safety

- CGM-TIR increased from 75.02% to 76.14% (+1.11 percentage points,
  patient-paired Wilcoxon p=0.0039).
- CGM-TAR >180 mg/dL decreased by 1.56 points and CGM-TAR >250 mg/dL
  decreased by 1.10 points.
- CGM-TBR <70 mg/dL increased from 0.72% to 1.16% (+0.44 points,
  p=0.028).  Mean low-event exposure increased by 17.6 min per 72-hour run,
  and mean maximum low-event duration increased by 14.3 min.
- CGM-TBR <54 mg/dL increased by 0.10 points; the paired test was not
  statistically significant (p=0.116).
- Early termination occurred in 6 PPO runs and 5 switch runs.  No
  switch-only termination occurred.  All terminations were caused by severe
  hyperglycaemia rather than hypoglycaemia.
- The same directions remained after restricting the analysis to 44 complete
  disturbance pairs.

These results support improved hyperglycaemia control but do not support a
claim of improved hypoglycaemia safety.

## Empirical boundedness and input regularity

- All stored CGM values were finite.
- The mean patient-level maximum CGM decreased by 4.47 mg/dL and the minimum
  CGM decreased by 3.48 mg/dL, consistent with the observed downward shift in
  glucose.
- The switch selected a different controller about 2.20 times per hour.
- Basal total variation increased by 0.00587 units per hour (p=0.002), but the
  95th percentile one-step basal change did not increase (p=0.557), and the
  direction-reversal rate did not differ (p=0.695).

Thus the switching logic increases the accumulated amount of input movement,
as expected, but the available finite-horizon trajectories do not show larger
extreme step changes or more frequent direction reversals.  This is empirical
regularity evidence only; it is not a Lyapunov, ISS, or formal closed-loop
stability proof.
