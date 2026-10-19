# Methods and limitations

## Research design

This is a cross-sectional county-level analysis. The target is CDC PLACES age-adjusted adult obesity prevalence for 2019 (`obesity_adj_pct`). The included dataset contains 3,110 unique county FIPS codes with 20 columns.

Both models use limited-access share, healthy and unhealthy outlet densities, one food-swamp measure, poverty, high-school and bachelor’s attainment, unemployment, and rural–urban indicator variables. The baseline swamp measure is a smoothed log ratio; a difference-based index is evaluated as a sensitivity check.

ElasticNet standardizes continuous predictors inside its training pipeline and searches alpha values 0.01, 0.1, 1.0 and L1 ratios 0.2, 0.5, 0.8. XGBoost uses 300 trees, depth 3, learning rate 0.05, subsampling and column sampling of 0.8, and L2 regularization of 5. Both use a random seed of 42.

Five-fold GroupKFold groups counties by the first two digits of their FIPS code. Entire states are held out together. Reported RMSE, MAE, and R² are calculated over pooled out-of-fold predictions. They are not fold means with confidence intervals.

## Reading the results

The baseline reference metrics are copied from the existing analysis outputs, not reconstructed from narrative notes. ElasticNet’s standardized coefficients include bachelor’s attainment −1.912, poverty +1.382, and swamp log ratio +0.331 percentage points per standard-deviation increase, conditional on the other predictors. These are regularized associations, not treatment effects or significance tests.

The saved XGBoost gain importance ranks bachelor’s attainment and poverty highest. Gain and SHAP characterize fitted model behavior; correlated inputs can share or redistribute importance. The SHAP reference plot uses the full fitted dataset.

## Limitations to retain when presenting this work

- **Model selection:** ElasticNet selects hyperparameters on the full dataset using grouped cross-validation, then evaluates the selected settings on the same grouped folds. This is not nested cross-validation, so its performance estimate may be optimistic.
- **Geography:** Holding out states reduces within-state overlap but does not remove every form of spatial dependence.
- **Mixed vintages:** The outcome is from 2019, retail-density proxies from 2016, rural–urban codes from 2013, and the access-share denominator from the 2010 census population. Source publication dates differ from measurement years.
- **Coverage and missing data:** The preparation code performs complete-case deletion after merging. The supplied cleaned data have 3,110 rows versus 3,122 in the earlier workspace. Original notes attribute exclusions to Connecticut geographic mismatches and other missing records; a row-level exclusion audit has not been reconstructed.
- **Feature redundancy:** Both outlet densities and a derived swamp measure are used together. The difference-based swamp index is an exact linear combination of the two densities. This is not a clean test of a composite measure versus its components.
- **Missing outlets:** Preparation fills some missing outlet densities with zero. This can conflate missing information with absence of stores.
- **No causal identification:** County associations cannot establish individual relationships or intervention effects. Controls do not eliminate unmeasured confounding.
- **Legacy weights:** Earlier exploratory code calls `y_prop` sample weights “population weights” in its narrative. `y_prop` is obesity prevalence divided by 100. These outcome-derived weighted runs are excluded from the showcase’s default workflow and claims.

## Changes for this showcase

The modeling logic for the unweighted runs is preserved. The command-line interface adds an output directory and CPU-job control. XGBoost prediction plots now use out-of-fold predictions, and those predictions are saved to CSV. Historical reference plots include only the correctly labeled ElasticNet out-of-fold figure and the full-fit SHAP figure. Exploratory state fixed-effects and policy-ranking scripts are outside this repository’s core scope.
