# Validation performed for the showcase

The complete default modeling command was run successfully against the included 3,110-county dataset using the existing Python 3.14 environment and the recorded modeling-library versions, with `--jobs 1` and Matplotlib’s noninteractive Agg backend.

All four unweighted model variants completed: ElasticNet and XGBoost, each with the log-ratio and difference-based swamp measure. Their RMSE, MAE, and R² matched the original saved metrics to a tolerance of 1e-10. The scripts generated prediction CSVs, metric tables, coefficients or feature importance, and plots.

Python syntax, unique five-character FIPS codes, complete dataset values, and local Markdown links were checked. The selected SHAP figure was visually inspected. A fresh virtual-environment installation and rebuilding the dataset from raw sources were not tested during this showcase preparation. Raw files remain in the original workspace; their hashes are recorded in the source manifest.
