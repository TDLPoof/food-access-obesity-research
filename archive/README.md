# Earlier modeling workflow

`legacy_model.py` preserves the previous unweighted analysis workflow. Its selected historical outputs live in `../results/reference/`. The current portfolio analysis is `../src/model.py`, with results in `../results/portfolio/`.

The current workflow adds nested ElasticNet selection, a training-mean baseline, distinct feature-set comparisons, fold variability, and subgroup errors. The older full-fit SHAP plot describes a different feature specification and should not be used to interpret the current models.
