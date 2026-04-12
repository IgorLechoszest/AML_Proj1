# Advanced Machine Learning, Project 1

This project is developed for the Advanced Machine Learning course at Warsaw University of Technology (WUT MiNI). The aim of the project is to analyze a logistic regression model in a situation where the training dataset contains observations with missing labels. 

This repository includes a custom implementation of the **FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm) for Logistic Regression with L1 regularization (Lasso) and methods for handling missing labels.

# Code description
The code is organized into focused modules for each project task.

## Task 1
- Data loading and preparation: `src/data_prep.py`
- Missing-label generation mechanisms (MCAR, MAR1, MAR2, MNAR): `src/missingness.py`
- Data preparation includes:
	- numeric conversion,
	- mean imputation,
	- removal of highly collinear features using a correlation threshold.

Implemented datasets:
- Required real datasets (4):
	- Spambase (OpenML)
	- Sonar (OpenML)
	- Breast Cancer Wisconsin (scikit-learn)
	- Ionosphere (OpenML)
- Additional synthetic dataset (optional, for stress testing/sanity checks):
	- Artificial binary dataset

## Task 2
- Custom Logistic Lasso with FISTA: `src/fista.py`
- Validation over lambda grid with selectable metric:
	- recall,
	- precision,
	- F1,
	- balanced accuracy,
	- ROC AUC,
	- PR AUC.
- Visualization methods:
	- `plot(measure)`
	- `plot_coefficients()`

## Task 3
- Semi-supervised model on partially labeled data: `src/unlabeled.py`
- Baselines:
	- Naive (`S=0` only),
	- Oracle (`Y` fully observed),
	- helpers in `src/benchmarks.py`.
- End-to-end experiments and comparisons: `src/main.ipynb`

# How to run
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run experiments in:

`src/main.ipynb`

## Authors
* Anna Ostrowska
* Gabriela Majstrak
* Igor Lechoszest
