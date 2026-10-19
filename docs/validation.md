# Reproducibility and verification

## Current portfolio analysis

The complete command `python src/model.py --output-dir results/portfolio --jobs 1` ran successfully in the existing Python 3.14 environment. It evaluated nine model/feature combinations across five state-held-out outer folds. Every ElasticNet outer fit performed a three-fold grouped hyperparameter search.

The committed output includes pooled and per-fold metrics, paired feature-set differences, all held-out predictions, fold-specific coefficients, subgroup errors, state membership audits, two figures, and a run manifest with dependency versions and the input SHA-256 hash.

Six automated checks passed: the input data contract; rejection of duplicate counties and missing measurements; outer and inner geographic separation and complete held-out coverage; exclusion of target columns and redundant feature combinations; agreement of published predictions and summary metrics; and the published inner/outer state-membership audit. These checks run in the included GitHub Actions workflow, but the hosted workflow has not yet run because the repository has not been published.

The notebook’s code cells were executed sequentially using the same Python environment and their actual outputs saved. The current comparison and error-analysis figures were visually reviewed. This was a local run, not a fresh dependency installation or a raw-source rebuild. Both remain additional reproducibility checks for another environment.

## Earlier analysis

Before the evaluation revision, all four unweighted legacy variants reproduced the original RMSE, MAE, and R² within 1e-10. The old figures and selected metrics remain under `results/reference/` solely for traceability. Their values and feature sets should not be mixed with the current `results/portfolio/` results.
