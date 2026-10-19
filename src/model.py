# repro_pipeline.py
import argparse, warnings, os
from pathlib import Path

JOBS = 2
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import shap

warnings.filterwarnings("ignore", category=UserWarning)

def derive_state_fips(df):
    fips_str = df["fips"].astype(str).str.zfill(5)
    df["state_fips"] = fips_str.str[:2]
    return df

def run_elasticnet(df, swamp_var="swamp_logratio", weighted=False, tag="elasticnet"):
    continuous = ["low_access_share","unhealthy_density","healthy_density",
                  swamp_var,"poverty_pct","hs_plus_pct","ba_plus_pct","unemp_rate"]
    rucc_cols = [c for c in df.columns if c.startswith("rucc_")]
    features = [c for c in continuous if c in df.columns] + [c for c in rucc_cols if c in df.columns]
    d = df.loc[:, ["fips","state_fips","obesity_adj_pct"] + features + (["y_prop"] if "y_prop" in df.columns else [])].dropna().copy()

    X = d[features].copy(); y = d["obesity_adj_pct"].astype(float).copy()
    groups = d["state_fips"]; sw = d["y_prop"].values if (weighted and "y_prop" in d.columns) else None

    preproc = ColumnTransformer(
        [("num", StandardScaler(), [c for c in continuous if c in X.columns]),
         ("rucc","passthrough",[c for c in rucc_cols if c in X.columns])],
        remainder="drop"
    )
    enet = ElasticNet(max_iter=2000, random_state=42)
    pipe = Pipeline([("prep", preproc), ("model", enet)])
    grid = GridSearchCV(
        pipe, {"model__alpha":[0.01,0.1,1.0], "model__l1_ratio":[0.2,0.5,0.8]},
        cv=GroupKFold(5), n_jobs=JOBS, scoring="neg_mean_squared_error", refit=True
    )
    fit_kwargs = {"model__sample_weight": sw} if sw is not None else {}
    grid.fit(X, y, groups=groups, **fit_kwargs)

    best = clone(grid.best_estimator_)
    oof = cross_val_predict(
        best, X, y, cv=GroupKFold(5), groups=groups, method="predict", n_jobs=JOBS,
        params=fit_kwargs if sw is not None else None
    )
    rmse = root_mean_squared_error(y, oof); mae = mean_absolute_error(y, oof); r2 = r2_score(y, oof)

    final = grid.best_estimator_.fit(X, y, **fit_kwargs)
    fnames = [c for c in continuous if c in X.columns] + [c for c in rucc_cols if c in X.columns]
    coefs = final.named_steps["model"].coef_

    pd.DataFrame({"OOF_RMSE_pp":[rmse],"OOF_MAE_pp":[mae],"OOF_R2":[r2],
                  "best_alpha":[grid.best_params_["model__alpha"]],
                  "best_l1_ratio":[grid.best_params_["model__l1_ratio"]],
                  "weights":[bool(weighted)],"swamp_var":[swamp_var]}).to_csv(f"{tag}_metrics.csv", index=False)

    pd.DataFrame({"feature": fnames, "coef_std": coefs}).to_csv(f"{tag}_coefficients.csv", index=False)
    pd.DataFrame({"fips": d["fips"].values, "obs": y.values, "pred_oof": oof}).to_csv(f"{tag}_oof_predictions.csv", index=False)

    plt.figure()
    plt.scatter(y, oof, alpha=0.6)
    lims=[min(y.min(),oof.min()), max(y.max(),oof.max())]; plt.plot(lims, lims)
    plt.xlabel("Observed obesity_adj_pct"); plt.ylabel("Predicted (OOF)"); plt.title(f"ElasticNet ({tag})")
    plt.tight_layout(); plt.savefig(f"{tag}_pred_vs_obs.png", dpi=160); plt.close()

def run_xgb(df, swamp_var="swamp_logratio", weighted=False, tag="xgb"):
    from xgboost import XGBRegressor
    rucc_cols = [c for c in df.columns if c.startswith("rucc_")]
    feats = ["low_access_share","unhealthy_density","healthy_density", swamp_var,
             "poverty_pct","hs_plus_pct","ba_plus_pct","unemp_rate"]
    features = [c for c in feats if c in df.columns] + [c for c in rucc_cols if c in df.columns]
    d = df.loc[:, ["fips","state_fips","obesity_adj_pct"] + features + (["y_prop"] if "y_prop" in df.columns else [])].dropna().copy()

    X = d[features].astype(float).values; y = d["obesity_adj_pct"].astype(float).values; groups = d["state_fips"].values
    sw = d["y_prop"].values if (weighted and "y_prop" in d.columns) else None
    fit_params = {"sample_weight": sw} if sw is not None else {}

    gkf = GroupKFold(5)
    model = XGBRegressor(objective="reg:squarederror", n_estimators=300, learning_rate=0.05, max_depth=3,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=1, n_jobs=JOBS, random_state=42)
    oof = cross_val_predict(model, X, y, cv=gkf, groups=groups, method="predict", n_jobs=JOBS, params=fit_params)
    rmse = root_mean_squared_error(y, oof); mae = mean_absolute_error(y, oof); r2 = r2_score(y, oof)

    model.fit(X, y, **fit_params)

    # 1. Create the explainer
    explainer = shap.TreeExplainer(model)

    # 2. Calculate SHAP values for the dataset (X)
    shap_values = explainer.shap_values(X)

    # 3. Create the plot
    plt.figure(figsize=(10, 6))  # Make it large enough to read
    shap.summary_plot(
        shap_values,
        X,
        feature_names=features,
        show=False,  # Don't show interactively
        plot_type="dot"  # "dot" creates the beeswarm style
    )

    # 4. Save the plot
    plt.title(f"SHAP Summary (Beeswarm) - {tag}")
    plt.tight_layout()
    plt.savefig(f"{tag}_shap_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()


    booster = model.get_booster(); score = booster.get_score(importance_type="gain")
    fnames = [c for c in feats if c in df.columns] + [c for c in rucc_cols if c in df.columns]
    rows = [{"feature": name, "gain": score.get(f"f{i}", 0.0)} for i,name in enumerate(fnames)]
    imp = pd.DataFrame(rows).sort_values("gain", ascending=False)
    if imp["gain"].sum()>0: imp["share_%"]=100*imp["gain"]/imp["gain"].sum()

    pd.DataFrame({"model":[tag],"OOF_RMSE_pp":[rmse],"OOF_MAE_pp":[mae],"OOF_R2":[r2],
                  "weighted":[weighted],"swamp_var":[swamp_var]}).to_csv(f"{tag}_metrics.csv", index=False)
    imp.to_csv(f"{tag}_feature_importance_gain.csv", index=False)
    pd.DataFrame({"fips": d["fips"].values, "obs": y, "pred_oof": oof}).to_csv(f"{tag}_oof_predictions.csv", index=False)

    plt.figure()
    plt.scatter(y, oof, alpha=0.6)
    lims=[min(y.min(),oof.min()), max(y.max(),oof.max())]; plt.plot(lims, lims)
    plt.xlabel("Observed obesity_adj_pct"); plt.ylabel("Predicted (OOF)"); plt.title(f"{tag} (out-of-fold)")
    plt.tight_layout(); plt.savefig(f"{tag}_pred_vs_obs.png", dpi=160); plt.close()

def main():
    global JOBS
    ap = argparse.ArgumentParser(description="Evaluate county obesity models with state-grouped cross-validation.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", default="results/generated")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    JOBS = args.jobs
    df = derive_state_fips(pd.read_csv(Path(args.data).resolve(), dtype={"fips": str}))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    for swamp_var, suffix in [("swamp_logratio", ""), ("swamp_index", "_swamp_index")]:
        run_elasticnet(df, swamp_var, False, "elasticnet" + suffix)
        run_xgb(df, swamp_var, False, "xgb" + suffix)
    print(f"Results written to {output}")

if __name__ == "__main__":
    main()
