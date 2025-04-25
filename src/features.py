"""
Feature engineering + preprocessing utilities.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer          
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder



def add_engineered(df: pd.DataFrame, current_year: int = 2024) -> pd.DataFrame:
    """Add `vehicle_age` and `mileage_to_age_ratio`."""
    df = df.copy()
    df["vehicle_age"] = current_year - df["year_of_registration"]
    df["mileage_to_age_ratio"] = df["mileage"] / df["vehicle_age"]
    return df


def split_features(df: pd.DataFrame, target: str = "price"):
    """Return (numeric_cols, categorical_cols, X, y)."""
    numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    # remove target and obvious IDs
    for col in ("price", "public_reference", "year_of_registration"):
        numeric = [n for n in numeric if n != col]
    categorical = [c for c in categorical if c != "reg_code"]

    X = df[numeric + categorical]
    y = df[target]
    return numeric, categorical, X, y


def make_preprocessor(numeric, categorical, *, poly: bool = False):
    num_steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler()),
    ]
    if poly:
        num_steps.append(("poly", PolynomialFeatures(degree=3, include_bias=False)))
    num_pipe = Pipeline(num_steps)            

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("enc", TargetEncoder()),
    ]
    cat_pipe = Pipeline(cat_steps)            

    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
