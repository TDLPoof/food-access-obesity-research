"""State-held-out experiments for county obesity prediction.

ElasticNet tuning and scaling occur exclusively inside each outer training fold.
XGBoost uses fixed, pre-specified parameters. All experiments share outer folds.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from importlib.metadata import version

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TARGET = 'obesity_adj_pct'
CONTROLS = ['poverty_pct', 'hs_plus_pct', 'ba_plus_pct', 'unemp_rate']
RUCC = [f'rucc_{i}' for i in range(2, 10)]
FEATURE_SETS = {
    'controls': CONTROLS + RUCC,
    'access': CONTROLS + RUCC + ['low_access_share'],
    'swamp': CONTROLS + RUCC + ['low_access_share', 'swamp_logratio'],
    'outlets': CONTROLS + RUCC + ['low_access_share', 'healthy_density', 'unhealthy_density'],
}


def load_data(path):
    """Fail explicitly on malformed IDs, missing values, or invalid measurements."""
    df = pd.read_csv(path, dtype={'fips': str})
    required = {'fips', TARGET, *sum(FEATURE_SETS.values(), [])}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if not df.fips.str.fullmatch(r'\d{5}').all() or df.fips.duplicated().any():
        raise ValueError('FIPS must be unique five-digit strings.')
    numeric = df[sorted(required - {'fips'})].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Model inputs must be finite and complete.')
    if not df[TARGET].between(0, 100).all() or not df.low_access_share.between(0, 1).all():
        raise ValueError('Outcome or food-access share is outside its valid range.')
    if not df[RUCC].isin([0, 1, False, True]).all().all() or (df[RUCC].astype(int).sum(axis=1) > 1).any():
        raise ValueError('RUCC indicators must encode at most one category.')
    df['state'] = df.fips.str[:2]
    df['rurality'] = np.where(df[RUCC[2:]].astype(int).sum(axis=1) > 0, 'Nonmetro', 'Metro')
    df['outcome_band'] = pd.cut(df[TARGET], [0, 30, 40, 100], right=False,
                                labels=['Below 30%', '30–40%', '40% or higher']).astype(str)
    return df


def grouped_splits(groups, n_splits):
    """Return deterministic folds and assert geographic separation."""
    groups = np.asarray(groups)
    splits = list(GroupKFold(n_splits).split(np.zeros(len(groups)), groups=groups))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError('State leakage detected.')
    return splits


def make_elasticnet(features, jobs):
    continuous = [f for f in features if f not in RUCC]
    prep = ColumnTransformer([('continuous', StandardScaler(), continuous),
                              ('rurality', 'passthrough', RUCC)])
    pipeline = Pipeline([('prep', prep), ('model', ElasticNet(max_iter=10000, random_state=42))])
    return GridSearchCV(pipeline,
                        {'model__alpha': [0.01, 0.1, 1.0], 'model__l1_ratio': [0.2, 0.5, 0.8]},
                        cv=GroupKFold(3), scoring='neg_root_mean_squared_error', n_jobs=jobs)


def metrics(y, pred):
    return {'rmse': root_mean_squared_error(y, pred), 'mae': mean_absolute_error(y, pred),
            'r2': r2_score(y, pred), 'bias': float(np.mean(np.asarray(pred) - np.asarray(y)))}


def run_experiments(df, output, jobs=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    folds = grouped_splits(df.state, 5)
    specs = [('Mean baseline', 'controls')] + [(m, f) for m in ['ElasticNet', 'XGBoost'] for f in FEATURE_SETS]
    fold_rows, prediction_rows, coefficient_rows, audit = [], [], [], []
    for fold, (train, test) in enumerate(folds, 1):
        inner = grouped_splits(df.iloc[train].state, 3)
        audit.append({'fold': fold, 'train_states': sorted(df.iloc[train].state.unique()),
                      'test_states': sorted(df.iloc[test].state.unique()),
                      'inner': [{'train_states': sorted(df.iloc[train].iloc[a].state.unique()),
                                 'validation_states': sorted(df.iloc[train].iloc[b].state.unique())}
                                for a, b in inner]})
    for name, feature_set in specs:
        features = FEATURE_SETS[feature_set]
        print(f'Evaluating {name}: {feature_set}', flush=True)
        for fold, (train, test) in enumerate(folds, 1):
            X_train, X_test = df.iloc[train][features].astype(float), df.iloc[test][features].astype(float)
            y_train, y_test = df.iloc[train][TARGET], df.iloc[test][TARGET]
            params = {}
            if name == 'ElasticNet':
                model = make_elasticnet(features, jobs)
                model.fit(X_train, y_train, groups=df.iloc[train].state)
                params = model.best_params_
                fitted = model.best_estimator_
                names = fitted.named_steps['prep'].get_feature_names_out()
                coefficient_rows.extend({'feature_set': feature_set, 'fold': fold,
                                         'feature': n.split('__', 1)[1], 'coefficient': float(c)}
                                        for n, c in zip(names, fitted.named_steps['model'].coef_))
            else:
                model = (DummyRegressor() if name == 'Mean baseline' else
                         XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                                      objective='reg:squarederror', random_state=42, n_jobs=jobs))
                model.fit(X_train, y_train)
            pred = model.predict(X_test)
            fold_rows.append({'model': name, 'feature_set': feature_set, 'fold': fold,
                              'n_test': len(test), **metrics(y_test, pred), 'parameters': json.dumps(params)})
            part = df.iloc[test][['fips', 'state', 'rurality', 'outcome_band', TARGET]].copy()
            part = part.rename(columns={TARGET: 'observed'})
            part['predicted'], part['model'], part['feature_set'], part['fold'] = pred, name, feature_set, fold
            prediction_rows.append(part)
    scores, predictions = pd.DataFrame(fold_rows), pd.concat(prediction_rows, ignore_index=True)
    summary = []
    for (name, feature_set), part in predictions.groupby(['model', 'feature_set'], sort=False):
        foldscores = scores[(scores.model == name) & (scores.feature_set == feature_set)]
        summary.append({'model': name, 'feature_set': feature_set, **metrics(part.observed, part.predicted),
                        'fold_rmse_mean': foldscores.rmse.mean(), 'fold_rmse_sd': foldscores.rmse.std()})
    summary = pd.DataFrame(summary)
    errors = []
    for (name, feature_set), part in predictions.groupby(['model', 'feature_set'], sort=False):
        for dimension in ['state', 'rurality', 'outcome_band']:
            for group, subset in part.groupby(dimension):
                result = metrics(subset.observed, subset.predicted) if len(subset) > 1 else {
                    'rmse': abs(float(subset.predicted.iloc[0] - subset.observed.iloc[0])),
                    'mae': abs(float(subset.predicted.iloc[0] - subset.observed.iloc[0])),
                    'r2': np.nan, 'bias': float(subset.predicted.iloc[0] - subset.observed.iloc[0])}
                errors.append({'model': name, 'feature_set': feature_set, 'dimension': dimension,
                               'group': group, 'n': len(subset), **result})
    deltas = []
    for name in ['ElasticNet', 'XGBoost']:
        controls = scores[(scores.model == name) & (scores.feature_set == 'controls')].set_index('fold')
        for f in ['access', 'swamp', 'outlets']:
            compare = scores[(scores.model == name) & (scores.feature_set == f)].set_index('fold')
            for fold in controls.index:
                deltas.append({'model': name, 'feature_set': f, 'fold': fold,
                               'rmse_delta_vs_controls': compare.loc[fold, 'rmse'] - controls.loc[fold, 'rmse']})
    for filename, frame in [('summary', summary), ('fold_metrics', scores), ('predictions', predictions),
                             ('subgroup_errors', pd.DataFrame(errors)), ('paired_deltas', pd.DataFrame(deltas)),
                             ('elasticnet_coefficients', pd.DataFrame(coefficient_rows))]:
        frame.to_csv(output / f'{filename}.csv', index=False)
    (output / 'fold_audit.json').write_text(json.dumps(audit, indent=2))
    plot_results(summary, predictions, output)
    return summary


def plot_results(summary, predictions, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    data = summary.iloc[::-1]
    colors = ['#64748b' if m == 'Mean baseline' else '#0891b2' if m == 'ElasticNet' else '#7c3aed' for m in data.model]
    ax.barh([f'{m} · {f}' for m, f in zip(data.model, data.feature_set)], data.fold_rmse_mean,
            xerr=data.fold_rmse_sd, color=colors, capsize=3)
    ax.set(xlabel='Held-out-state RMSE (percentage points) · lower is better',
           title='Does food-environment information improve prediction?')
    ax.text(0, -0.19, 'Bars: mean across five outer folds. Error bars: fold standard deviation, not confidence intervals.',
            transform=ax.transAxes, fontsize=8)
    fig.savefig(output / 'model_comparison.png', dpi=180, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for name, color in [('ElasticNet', '#0891b2'), ('XGBoost', '#7c3aed')]:
        part = predictions[(predictions.model == name) & (predictions.feature_set == 'swamp')]
        axes[0].scatter(part.observed, part.predicted, s=9, alpha=0.2, color=color, label=name)
        groups = list(part.groupby('outcome_band'))
        groups.sort(key=lambda x: {'Below 30%': 0, '30–40%': 1, '40% or higher': 2}[x[0]])
        axes[1].plot([g for g, _ in groups], [(v.predicted-v.observed).mean() for _, v in groups],
                     'o-', color=color, label=name)
    axes[0].plot([10, 55], [10, 55], '--', color='#64748b')
    axes[0].set(xlabel='Observed obesity (%)', ylabel='Out-of-fold prediction (%)', title='Predictions on unseen states')
    axes[1].axhline(0, color='#64748b', linestyle='--')
    axes[1].set(xlabel='Observed county obesity prevalence', ylabel='Mean prediction − observation (pp)',
                title='Errors at the extremes')
    axes[1].legend(frameon=False)
    fig.savefig(output / 'error_analysis.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/processed/county_obesity_2019_model_data.csv')
    parser.add_argument('--output-dir', default='results/generated')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    data = load_data(args.data)
    summary = run_experiments(data, args.output_dir, args.jobs)
    manifest = {'python': platform.python_version(), 'rows': len(data), 'states': data.state.nunique(),
                'data_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                'seed': 42, 'outer_folds': 5, 'inner_folds': 3,
                'dependencies': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'matplotlib']}}
    (Path(args.output_dir) / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
