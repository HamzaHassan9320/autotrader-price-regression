"""
Pipeline builders, hyper‑parameter grids, and a helper for grid search.
"""
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from src.features import make_preprocessor


def build_pipeline(
    estimator,
    numeric,
    categorical,
    *,
    k_best: int | None = None,
    poly: bool = False,
    use_pca: bool = False,
) -> Pipeline:
    steps: list[tuple] = [("prep", make_preprocessor(numeric, categorical, poly=poly))]
    if k_best:
        steps.append(("select", SelectKBest(f_regression, k=k_best)))
    if use_pca:
        steps.append(("pca", PCA()))
    steps.append(("model", estimator))
    return Pipeline(steps)

def pipe_lr(numeric, categorical):
    return build_pipeline(
        LinearRegression(), numeric, categorical, poly=True, k_best=10, use_pca=True
    )


def pipe_rfr(numeric, categorical):
    return build_pipeline(
        RandomForestRegressor(max_depth=10),
        numeric,
        categorical,
        k_best=10,
        use_pca=True,
    )


def pipe_gbr(numeric, categorical):
    return build_pipeline(
        GradientBoostingRegressor(max_depth=7, learning_rate=0.1),
        numeric,
        categorical,
        k_best=10,
        use_pca=True,
    )


def pipe_ensemble(numeric, categorical):
    return VotingRegressor(
        estimators=[
            ("gbr", pipe_gbr(numeric, categorical)),
            ("rfr", pipe_rfr(numeric, categorical)),
            ("lr", pipe_lr(numeric, categorical)),
        ]
    )

GRID_RFR = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [5, 8, 10],
}

GRID_GBR = {
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__max_depth": [3, 5, 7],
}


def grid_search(pipe, grid, X, y, cv: int = 5):
    gs = GridSearchCV(
        pipe, param_grid=grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    gs.fit(X, y)
    return gs
