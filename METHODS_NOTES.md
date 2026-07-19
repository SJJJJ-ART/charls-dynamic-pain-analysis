# Reproducible methods notes

## Estimand

Each interval includes two observed pain states, BADL independence at the
second pain assessment, and a next-wave BADL outcome. The three outcome windows
span about 2, 3, and 2 years. The pooled absolute risks therefore represent an
average across those windows. They are not fixed two-year risks.

## Multiple imputation

The program creates 30 completed data sets. The seeds are 20260718 through
20260747. The program uses stochastic chained regression with posterior
sampling for 15 iterations. The program then bounds continuous variables and
rounds categorical variables to valid codes. The program does not impute pain,
BADL, death, identifiers, community identifiers, or survey weights.

Rubin's total covariance is

`T = U_bar + (1 + 1 / m) B`, where `m = 30`.

## IPCW

The program fits a separate weighted logistic response model in each window.
The numerator is the survey-weighted response probability in that window. The
denominator is the fitted probability conditional on the pain transition and
all Model 1 and Model 2 covariates. The program limits fitted probabilities to
0.02 through 0.98.

For participant `i` and window `w`, the stabilized weight is

`SW_iw = P(R_iw = 1 | W = w) / P(R_iw = 1 | A_iw, X_iw, W = w)`.

The final weight multiplies `SW_iw` by the 2011 individual survey weight after
normalization to a mean of one within the window.

## Regression

The program fits log-link modified-Poisson models. The main covariance matrix
uses baseline community clusters. Every repeated interval for one participant
has the same baseline community identifier. A participant-clustered analysis
checks this choice.

Model 1 includes window, landmark age, sex, education, marital status, rural
residence, agricultural hukou, and 2011 BMI. Model 2 also includes diagnosed
conditions, arthritis, stroke, CES-D-10, sleep, smoking, drinking, self-rated
health, mobility difficulty, and IADL difficulty.

## Interpretation

The study estimates prospective associations. The analysis does not identify
causal effects. The analysis also does not validate a clinical prediction
tool.
