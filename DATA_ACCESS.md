# CHARLS data access and reproducibility

## Why the data are not in this repository

CHARLS distributes individual-level files under its own registration and
data-use terms. This repository cannot redistribute those records. The absence
of raw data does not prevent correct analysis code. It means that a reviewer
must obtain the same public-release files from CHARLS before running the code.

## Files required

The analysis uses the 2011, 2013, 2015, 2018, and 2020 individual releases and
the 2011 community questionnaire. `config/data_manifest.yml` lists the exact
module filenames and expected directory layout.

## Reproduction steps

1. Register for CHARLS data access and download the listed public releases.
2. Place the unmodified release files under a private directory that follows
   `config/data_manifest.yml`.
3. Create a Python 3.12 environment from `requirements-lock.txt`.
4. Run the scripts in the order shown in `README.md`.
5. Run `code/05_validate_results.py` before using any estimate.

The scripts keep participant-level intermediates under user-supplied private
output directories. The public repository stores only code and aggregate
validation products.

## What a reviewer can verify without CHARLS files

A reviewer can inspect the complete variable mapping, cohort logic, equations,
model design, fixed seeds, and aggregate result checks. A reviewer cannot
independently recompute participant-level estimates without obtaining CHARLS.
This limitation follows the data-use agreement. It does not make the code
incorrect or incomplete.
