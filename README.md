# Advanced Machine Learning - Project 1

This repository contains Project 1 for the Advanced Machine Learning course (WUT MiNI).
The project studies binary logistic regression when training labels are partially missing.

The implementation includes:
- custom Logistic Lasso trained with FISTA,
- multiple missing-label mechanisms,
- semi-supervised label completion methods,
- benchmark comparisons and visual analysis in a notebook.

## Repository layout

- `src/data_prep.py` - dataset loading and preprocessing.
- `src/missingness.py` - MCAR, MAR1, MAR2, MNAR label-missingness generators.
- `src/fista.py` - custom Logistic Lasso (FISTA) implementation.
- `src/unlabeled.py` - `UnlabeledLogReg` with pseudo-labeling and EM.
- `src/benchmarks.py` - Naive and Oracle training helpers.
- `src/main.ipynb` - end-to-end experiments and plots.
- `tests/` - unit and smoke tests.

## Task coverage

### Task 1: data preparation and missingness schemes

Implemented real binary datasets:
- Spambase (OpenML)
- Sonar (OpenML)
- Breast Cancer Wisconsin (scikit-learn)
- Ionosphere (OpenML)

Optional synthetic dataset:
- Artificial binary dataset (for additional stress testing)

Preprocessing pipeline:
- numeric conversion,
- mean imputation,
- removal of highly collinear features (correlation threshold).

Missing-label generators return `(X, y_obs)` with missing labels encoded as `-1`:
- MCAR,
- MAR1,
- MAR2,
- MNAR.

### Task 2: custom logistic regression with FISTA

Implemented in `src/fista.py`:
- `fit(X_train, y_train)`
- `validate(X_valid, y_valid, measure)`
- `predict_proba(X_test)`
- `predict(X_test)`
- `plot(measure)`
- `plot_coefficients()`

Supported validation measures:
- recall,
- precision,
- F1,
- balanced accuracy,
- ROC AUC,
- PR AUC.

### Task 3: learning with missing labels

Implemented in `src/unlabeled.py`:
- `UnlabeledLogReg(..., method="pseudo_labeling")`
- `UnlabeledLogReg(..., method="em")`

Baselines:
- Naive (only observed labels),
- Oracle (fully observed labels).

Comparisons and analyses are run in `src/main.ipynb`:
- methods under MCAR/MAR1/MAR2/MNAR,
- MCAR sensitivity analysis across different `c` values,
- custom FISTA vs sklearn L1 comparison,
- prediction agreement and confusion-matrix diagnostics.

## Setup and run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run:

`src/main.ipynb`

4. Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Running on custom data

The notebook includes a helper function `run_on_custom_data(csv_path, target_column)`.

Expected input:
- CSV file with numeric features,
- one binary target column (0/1) provided via `target_column`.

The helper applies imputation + scaling and trains `UnlabeledLogReg`.

## Reproducibility notes

- Default seed used across core experiments: `67`.
- Some sections use repeated random splits for averaged metrics.
- Current workflow is notebook-driven (`src/main.ipynb`); there is no standalone CLI pipeline for automatic CSV export at this moment.

## Authors

- Anna Ostrowska
- Gabriela Majstrak
- Igor Lechoszest
