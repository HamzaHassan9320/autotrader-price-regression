"""
One‑shot training script.  Usage:
    python src/train.py --csv data/Adverts.csv
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in sqrt",
    category=RuntimeWarning,
    module="sklearn.feature_selection",
)

from argparse import ArgumentParser
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
import pandas as pd
from src.data import load_raw, clean
from src.features import add_engineered, split_features
from src.models import (
    pipe_lr,
    pipe_rfr,
    pipe_gbr,
    pipe_ensemble,
    grid_search,
    GRID_RFR,
    GRID_GBR,
)
from .visualise import mae_bar


def main(csv_path: str, quick: bool = False):
    df = load_raw(csv_path)
    if quick:
        df = df.sample(n=60_000, random_state=42)
    df = add_engineered(clean(df))

    numeric, categorical, X, y = split_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = {
        "lr": pipe_lr(numeric, categorical),
        "rfr": pipe_rfr(numeric, categorical),
        "gbr": pipe_gbr(numeric, categorical),
        "ensemble": pipe_ensemble(numeric, categorical),
    }

    metrics: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        metrics[name] = {
            "mae": mean_absolute_error(y_te, y_pred),
            "r2": r2_score(y_te, y_pred),
        }
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, f"models/{name}.pkl")

    # Optional grid search on RF & GBR
    grid_search(models["rfr"], GRID_RFR, X_tr, y_tr)
    grid_search(models["gbr"], GRID_GBR, X_tr, y_tr)
    if not quick:
        gs_rfr = grid_search(models["rfr"], GRID_RFR, X_tr, y_tr)
        gs_gbr = grid_search(models["gbr"], GRID_GBR, X_tr, y_tr)
        models["rfr"] = gs_rfr.best_estimator_
        models["gbr"] = gs_gbr.best_estimator_
        cv_models = {k: v for k, v in models.items() if k != "ensemble"}
        rows = []
        for k, mdl in cv_models.items():
            cv_res = cross_validate(
                mdl, X, y, cv=5,
                scoring="neg_mean_absolute_error",
                return_train_score=True
            )
            rows.append([
                -cv_res["test_score"].mean(), cv_res["test_score"].std(),
                -cv_res["train_score"].mean(), cv_res["train_score"].std()
            ])
        cv_df = pd.DataFrame(
            rows,
            columns=["test_mae_mean","test_mae_std","train_mae_mean","train_mae_std"],
            index=cv_models.keys()
        )
        print(cv_df)

    mae_bar(metrics)  # writes docs/images/mae_bar.png
    print("Training complete. Metrics:", metrics)


if __name__ == "__main__":
     p = ArgumentParser()
     p.add_argument("--csv", default="data/Adverts.csv")
     p.add_argument("--quick", action="store_true", help="sample 60k rows & skip grid-search (CI mode)")
     args = p.parse_args()
     main(args.csv, quick=args.quick)
