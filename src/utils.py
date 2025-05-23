# src/utils.py
from pathlib import Path
import pandas as pd

_CACHE = {}      

_LOOKUP_CSV = Path("data/makes_models.csv")

def _load_df(csv_path: str | Path = _LOOKUP_CSV) -> pd.DataFrame:
    if "df" not in _CACHE:
        _CACHE["df"] = pd.read_csv(csv_path, usecols=["standard_make","standard_model"])
    return _CACHE["df"]


def list_makes(csv_path: str | Path = _LOOKUP_CSV) -> list[str]:
    """All unique makes, sorted A-Z."""
    df = _load_df(csv_path)
    return sorted(df["standard_make"].dropna().unique())


def list_models(make: str, csv_path: str | Path = _LOOKUP_CSV) -> list[str]:
    """Models for the chosen make."""
    df = _load_df(csv_path)
    models = (df.loc[df["standard_make"] == make, "standard_model"]
                .dropna().unique())
    return sorted(models)
