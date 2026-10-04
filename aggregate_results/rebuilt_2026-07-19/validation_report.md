# CHARLS rebuilt-analysis validation report

**Overall status:** PASS WITH DOCUMENTED WARNINGS

The program completed 35 checks. It found 0 failures and 1 warnings.

| Check | Status | Evidence |
|---|---|---|
| Three analysis windows | PASS | found 3 |
| Window interval counts | PASS | [9921, 8273, 8380] |
| Window event counts | PASS | [1489, 1064, 1430] |
| Primary intervals | PASS | 26574 |
| Primary events | PASS | 3983 |
| Unique participants | PASS | 11947 |
| Baseline communities | PASS | 445 |
| Sixteen transition cells | PASS | found 16 |
| Transition-cell intervals | PASS | 26574 |
| Transition-cell events | PASS | 3983 |
| Thirty imputations | PASS | 30 |
| Fixed seed sequence | PASS | 20260718–20260747 |
| All pooled model estimates are finite | PASS | 60 pooled coefficients |
| All pooled confidence intervals contain their estimates | PASS | all pooled terms |
| Model 1 persistent-multisite estimate is numerically equivalent | PASS | rebuilt 3.417525 (3.055237–3.822774); archived 3.415457 (3.053332–3.820530) |
| Model 2 persistent-multisite estimate is numerically equivalent | PASS | rebuilt 1.651856 (1.452649–1.878380); archived 1.652184 (1.452213–1.879692) |
| All IPCW values are positive | PASS | all imputation-window summaries are positive |
| All effective sample sizes are positive | PASS | minimum 4569.0 |
| Eight standardized risks | PASS | found 8 |
| Standardized risks are probabilities | PASS | range 0.090–0.306 |
| Persistent-multisite risk rounds to 30.6% | PASS | 30.591% |
| Persistent-multisite risk difference rounds to 21.6 pp | PASS | 21.640 pp |
| Exclude proxy interviews sample count | PASS | found 25554; expected 25554 |
| Exclude proxy interviews event count | PASS | found 3706; expected 3706 |
| Severe BADL sample count | PASS | found 26574; expected 26574 |
| Severe BADL event count | PASS | found 969; expected 969 |
| BADL or death sample count | PASS | found 27261; expected 27261 |
| BADL or death event count | PASS | found 4670; expected 4670 |
| Pain threshold ≥ Some/Somewhat sample count | PASS | found 26773; expected 26773 |
| Pain threshold ≥ Some/Somewhat event count | PASS | found 4006; expected 4006 |
| Next-wave IADL limitation sample count | PASS | found 25403; expected 25403 |
| Next-wave IADL limitation event count | PASS | found 4147; expected 4147 |
| Community bootstrap has 500 replicates | PASS | found 500 |
| Community bootstrap estimates are finite | PASS | all 500 replicates |
| Exploratory PHC estimate matches archived output | WARN | rebuilt 0.919; archived 0.831; the revised manuscript must use or explain one transparent implementation |

## Interpretation

The exact cohort flow, transition counts, sensitivity-analysis denominators, and headline Model 1 and Model 2 estimates were reproduced. The PHC analysis remains exploratory. Its rebuilt estimate differs from the archived exploratory output, although both estimates are imprecise and support the same conclusion of no clear effect modification.

The cumulative chronic-condition routing also does not reproduce every archived diagnostic count. The primary Model 1 estimate does not use those variables. The Model 2 headline contrast still matches after rounding. A user who needs bit-for-bit replication of all Model 2 nuisance coefficients would need the original pre-reconstruction script, which was not available.
