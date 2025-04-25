"""
Data loading and basic cleansing for the AutoTrader dataset.
"""
from pathlib import Path
import pandas as pd

RAW_CSV = Path("data/Adverts.csv")


def load_raw(path: str | Path = RAW_CSV) -> pd.DataFrame:
    """
    Read the raw CSV (not in repo – user must copy it to `data/`).
    """
    return pd.read_csv(path)


def _remove_outlier(df: pd.DataFrame, column: str, whisker: float = 1.5) -> pd.DataFrame:
    q1, q3 = df[column].quantile([0.25, 0.75])
    upper = q3 + whisker * (q3 - q1)
    return df[df[column] <= upper]


def _fill_missing_mode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.assign(**{column: df[column].fillna(df[column].mode()[0])})


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
      · Trim outliers in mileage & price
      · Drop cars registered before 1975
      · Mode‑impute three categoricals
    """
    for col in ("mileage", "price"):
        df = _remove_outlier(df, col)

    df = df[df["year_of_registration"] > 1975]

    for col in ("fuel_type", "body_type", "standard_colour"):
        df = _fill_missing_mode(df, col)

    return df.reset_index(drop=True)
