"""Модуль проверок качества данных для датасета FishGrow."""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "fish.csv"

KNOWN_SPECIES = {"Bream", "Roach", "Whitefish", "Parkki", "Perch", "Pike", "Smelt"}
NUMERIC_COLS = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert not df.empty, "Пустая таблица"
    assert df.columns.is_unique, "Дубликаты в названиях столбцов"
    return df


def quality_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "unique": df.nunique(dropna=False),
    })


def physical_checks(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "check": [
            "Weight > 0",
            "Length1 > 0", "Length2 > 0", "Length3 > 0",
            "Height > 0", "Width > 0",
            "Length1 <= Length2", "Length2 <= Length3",
            "Species in known set",
        ],
        "violations": [
            (df["Weight"] <= 0).sum(),
            (df["Length1"] <= 0).sum(),
            (df["Length2"] <= 0).sum(),
            (df["Length3"] <= 0).sum(),
            (df["Height"] <= 0).sum(),
            (df["Width"] <= 0).sum(),
            (df["Length1"] > df["Length2"]).sum(),
            (df["Length2"] > df["Length3"]).sum(),
            (~df["Species"].isin(KNOWN_SPECIES)).sum(),
        ],
    })


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def mad_z(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (series - med) / mad


def outliers_report(df: pd.DataFrame, cols: list = NUMERIC_COLS) -> pd.DataFrame:
    rows = []
    for col in cols:
        lo, hi = iqr_bounds(df[col])
        iqr_mask = (df[col] < lo) | (df[col] > hi)
        mad_mask = mad_z(df[col]).abs() > 3.5
        rows.append({
            "column": col,
            "iqr_count": int(iqr_mask.sum()),
            "mad_count": int(mad_mask.sum()),
            "iqr_only": sorted(set(df.index[iqr_mask]) - set(df.index[mad_mask])),
            "mad_only": sorted(set(df.index[mad_mask]) - set(df.index[iqr_mask])),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_data()
    print("Размер:", df.shape)
    print("\nКачество:")
    print(quality_table(df))
    print("\nПроверки физсмысла:")
    print(physical_checks(df))
    print("\nВыбросы:")
    print(outliers_report(df))