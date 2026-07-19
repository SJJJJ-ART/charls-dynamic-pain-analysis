# CHARLS dynamic pain transitions and BADL limitation

Repository: https://github.com/SJJJJ-ART/charls-dynamic-pain-analysis

This repository contains reproducible Python code for a repeated-landmark
study of pain-distribution transitions and subsequent basic activities of
daily living (BADL) limitation in the China Health and Retirement
Longitudinal Study (CHARLS).

CHARLS individual-level data are not redistributed here. Researchers must
obtain the 2011, 2013, 2015, 2018, and 2020 releases directly from the CHARLS
data portal and accept the applicable data-use terms. The scripts expect the
original release files listed in `config/data_manifest.yml`.

The code was rebuilt from the original CHARLS release files and archived
aggregate outputs. The validation report distinguishes exact checks,
rounding-level agreement, and remaining implementation differences. The
repository does not claim bit-for-bit identity where that identity was not
demonstrated.

## Verified headline result

The rebuilt data pipeline produced 26,574 person-intervals, 3,983 incident
BADL events, and 11,947 participants. The rebuilt primary Model 1 estimate for
persistent axial multisite pain was RR 3.42 (95% CI 3.06–3.82). The rebuilt
Model 2 estimate was RR 1.65 (95% CI 1.45–1.88). Both estimates match the
archived analysis within 0.005 on the RR and both confidence limits. The
archived Model 1 lower limit rounds to 3.05. The rebuilt lower limit rounds to
3.06.

## Run order

The full workflow can be run with:

```bash
bash run_all.sh /path/to/harmonized-charls /private/output/root
```

The same steps can be run separately as follows:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python code/01_build_analysis_data.py \
  --raw-dir /path/to/charls \
  --output-dir /private/path/derived

python code/02_run_analysis.py \
  --raw-dir /path/to/charls \
  --derived-dir /private/path/derived \
  --output-dir /private/path/results \
  --imputations 30 --jobs 4

python code/03_run_secondary.py \
  --primary-results-dir /private/path/results \
  --output-dir /private/path/secondary \
  --imputations 30 --jobs 4

python code/04_run_full_sensitivities.py \
  --raw-dir /path/to/charls \
  --primary-results-dir /private/path/results \
  --output-dir /private/path/full-sensitivities \
  --imputations 30 --jobs 4 --bootstrap-replicates 500

python code/05_validate_results.py \
  --derived-dir /private/path/derived \
  --primary-results-dir /private/path/results \
  --secondary-results-dir /private/path/secondary \
  --full-sensitivity-dir /private/path/full-sensitivities \
  --report validation_report.md

python code/06_make_figures.py \
  --secondary-results-dir /private/path/secondary \
  --full-sensitivity-dir /private/path/full-sensitivities \
  --output-dir figures
```

The scripts require Python 3.12 and the packages listed in
`requirements-lock.txt`.

## Analysis map

- `01_build_analysis_data.py` harmonizes identifiers, pain states, BADL, and
  the three repeated-landmark risk sets.
- `02_run_analysis.py` creates 30 stochastic chained-regression imputations,
  estimates stabilized IPCW, fits Models 0–2, and applies Rubin's rules.
- `03_run_secondary.py` estimates standardized risks, risk differences,
  direct contrasts, PHC interactions, and core weight/cluster sensitivities.
- `04_run_full_sensitivities.py` estimates outcome, threshold, proxy, window,
  complete-case, and 500-replicate community-bootstrap analyses.
- `05_validate_results.py` checks sample counts, model estimates, weights,
  uncertainty intervals, and aggregate output integrity.
- `06_make_figures.py` creates Figures 2–4 from disclosure-safe aggregate
  source files.

## Restricted-data policy

Do not commit `.dta`, `.sav`, `.csv`, `.parquet`, `.zip`, or `.rar` files that
contain CHARLS participant records. The `.gitignore` file blocks common raw and
derived data formats. The repository includes only aggregate tables whose cells
have already appeared in the manuscript or its supplement. See
`DATA_ACCESS.md` for the data-access and rerun procedure.

## Known reconstruction limits

The rebuilt cumulative chronic-condition routing does not reproduce every
archived missingness count or completed-value mean. The main Model 2 contrast
still matches after rounding, but the repository records this difference. The
rebuilt PHC interaction also differs from the archived exploratory estimate.
The revised manuscript uses the transparent rebuilt estimate. These limits do
not affect the exact sample flow, transition counts, primary Model 1 estimate,
or the paper's main conclusion.

## Citation

The manuscript citation will be added after publication. Users should also
cite CHARLS and the relevant CHARLS user guides.
