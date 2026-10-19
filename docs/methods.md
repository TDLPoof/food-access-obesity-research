# Experimental design and limitations

## Question and outcome

The study asks whether food-environment measurements add predictive information beyond socioeconomic and rural–urban context. The target is 2019 age-adjusted adult obesity percentage (`obesity_adj_pct`). Each of the 3,110 rows is a county, not a person. Counties receive equal weight.

## Controlled feature comparisons

| Feature set | Inputs |
| --- | --- |
| `controls` | Poverty, high-school and bachelor’s attainment, unemployment, RUCC 2–9 indicators |
| `access` | Controls + low-access population share |
| `swamp` | Controls + access + smoothed unhealthy/healthy outlet log ratio |
| `outlets` | Controls + access + separate healthy and unhealthy outlet densities |

The composite and its component densities are not included together. The outcome fraction (`y_prop`), FIPS, state identifier, and diagnostic group labels are never predictors. FIPS supplies grouping information only.

## Evaluation

Five deterministic GroupKFold outer folds hold out entire states. All nine experiments use identical outer folds. A training-mean DummyRegressor establishes a naive baseline. Each county receives exactly one held-out prediction per experiment.

For ElasticNet, each outer training set runs a three-fold state-grouped GridSearchCV over alpha ∈ {0.01, 0.1, 1.0} and L1 ratio ∈ {0.2, 0.5, 0.8}. Selection minimizes mean inner-fold RMSE. Scaling occurs inside the pipeline, independently within each inner fit, and the selected pipeline is refitted using only the outer training set before prediction. RUCC indicator variables are passed through unscaled. Fold-specific coefficients therefore use each outer training set’s own scaling.

XGBoost uses fixed settings inherited from the prior analysis: 300 trees, depth 3, learning rate 0.05, subsample 0.8, column sampling 0.8, L2 regularization 5, squared-error objective, and random seed 42. No XGBoost tuning or early stopping uses outer test data. These settings have a history with this dataset; the revised evaluation is not a preregistered or untouched external test.

`fold_audit.json` records outer and inner state membership. `fold_metrics.csv` records selected ElasticNet settings per outer fold. `predictions.csv` supports recomputing all metrics.

## Reporting

- Summary RMSE, MAE, R², and bias use pooled out-of-fold predictions; errors are in percentage points.
- Fold RMSE mean and sample standard deviation are reported separately. Standard deviations are not confidence intervals; fold training sets overlap.
- Paired feature-set comparisons subtract controls-only RMSE within the same fold. Negative values indicate improvement.
- Model rankings are descriptive. Selecting the best of these experiments and reporting its score can introduce selection optimism; independent validation remains necessary.
- A positive bias means overprediction. State, RUCC metro/nonmetro (codes 1–3 versus 4–9), and fixed observed-prevalence bands support error analysis. Outcome bands are retrospective; subgroup R² may be unstable with few counties or restricted outcome ranges.

## Findings

The nested ElasticNet swamp model achieves pooled RMSE 2.9346 pp and R² 0.5250; the fixed XGBoost swamp model achieves RMSE 2.9442 pp and R² 0.5218. Controls-only RMSEs are 2.9757 and 3.0009 pp, respectively. Limited access alone contributes little incremental predictive value under these specifications. These small differences do not establish significance or causality.

Both models regress toward the middle of the observed outcome distribution. Similar overall accuracy can hide geographic errors; the subgroup tables make this visible.

## Data and interpretation limits

- **Mixed vintages:** 2019 obesity, 2016 retail-density proxies, 2013 RUCC, and a 2010 population denominator for access. Exact ACS education vintage remains unverified.
- **Coverage:** Complete-case merging reduces the earlier 3,122-row table to 3,110 rows. Earlier notes attribute exclusions to Connecticut geography mismatches and missing records; a row-level exclusion audit has not been reconstructed.
- **Missing outlets:** Preparation fills some missing densities with zero, potentially conflating missing information and absent outlets.
- **Geography:** State-held-out validation reduces within-state overlap but does not eliminate cross-border spatial dependence or measurement dependencies.
- **Source outcome:** The CDC target is an estimated county prevalence; this exercise predicts those estimates rather than individual diagnoses.
- **Association:** Controls and predictive models do not remove unmeasured confounding, identify intervention effects, or justify individual-level conclusions from county aggregates.

## Earlier analysis

`archive/legacy_model.py` and `results/reference/` preserve the earlier workflow and selected outputs. That workflow tuned ElasticNet before out-of-fold evaluation and combined outlet components with their derived swamp measure. It also contained optional outcome-derived weighting, which is not population weighting. Those results and the full-fit SHAP figure are historical artifacts, not results for the current feature sets or validation protocol. The current portfolio workflow uses unweighted models and nested ElasticNet tuning.
