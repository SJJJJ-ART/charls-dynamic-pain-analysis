# Manual GitHub upload instructions

This package is prepared for manual upload to:

https://github.com/SJJJJ-ART/charls-dynamic-pain-analysis

## What to upload

Upload the contents of this folder to the repository root:

- `README.md`
- `DATA_ACCESS.md`
- `METHODS_NOTES.md`
- `requirements-lock.txt`
- `run_all.sh`
- `config/`
- `code/`
- `aggregate_results/`

The `aggregate_results/` folder contains only disclosure-safe aggregate outputs,
figure source tables, and rendered figure files. It does not contain CHARLS
participant-level records.

## What not to upload

Do not upload any original or derived participant-level CHARLS data. Do not
upload files with these extensions if they contain individual records:

- `.dta`
- `.sav`
- `.sas7bdat`
- `.parquet`
- `.feather`
- `.pkl`
- `.pickle`
- `.zip`
- `.rar`
- `.csv.gz`

Do not upload folders named `raw/`, `data/`, `derived/`, or `private/` if they
contain original CHARLS files or derived individual-level files.

## Web upload steps

1. Open the repository page.
2. If the repository is empty, create a small `README.md` first.
3. Click `Add file` and then `Upload files`.
4. Drag the extracted package contents into the upload page.
5. Confirm that no raw CHARLS files are listed.
6. Commit the upload.

## After upload

Open the repository in a browser and check that:

- `README.md` is visible on the repository homepage.
- `code/02_run_analysis.py` exists.
- `aggregate_results/rebuilt_2026-07-19/validation_report.md` exists.
- No raw CHARLS data files are visible.

After these checks pass, the manuscript can state that the code and
disclosure-safe aggregate outputs are available at the repository URL.
