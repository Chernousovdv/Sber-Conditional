import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from utils.constants import (
    CATEGORY_MAP
)


def normalize_series(s: pd.Series,
                     method: str = "index"):
    """
    Normalize a Series. method in {"index", "minmax", "zscore"}.
    - "index": value / first_non_null_value * 100
    - "minmax": (value - min) / (max - min)
    - "zscore": (value - mean) / std
    """
    s = s.copy()
    if method == "index":
        # use first non-null value as base
        first = s.dropna().iloc[0] if not s.dropna().empty else None
        if first is None or first == 0:
            return s * float("nan")
        return s / first * 100.0
    elif method == "minmax":
        lo = s.min()
        hi = s.max()
        if hi == lo:
            return (s - lo)  # all zeros
        return (s - lo) / (hi - lo)
    elif method == "zscore":
        mu = s.mean()
        sigma = s.std()
        if sigma == 0:
            return s - mu
        return (s - mu) / sigma
    else:
        raise ValueError("Unknown normalization method")


def aggregate_ts(ts_values: list[pd.Series],
                 agg_method_name: str = "sum"):
    array_data = np.array([s.values for s in ts_values])
    if agg_method_name == "sum":
        return np.sum(array_data, axis=0)
    elif agg_method_name == "mean":
        return np.mean(array_data, axis=0)
    else:
        raise ValueError(f"There are no aggregation mthod by name {agg_method_name}")

    
def transform_categories(df: pd.DataFrame,
                         norm_method: str = "index",
                         ts_aggregation_funct: str = "sum",
                         features_to_cat: dict[str, str] = CATEGORY_MAP,
                         plot_graphics: bool = True):
    categoris_by_normalized_ts = {}
    print(df.columns)
    for ts_name in df.columns:
        ts_name_true = ts_name.replace("/", " ")
        if ts_name_true not in features_to_cat.keys():
            print(f"{ts_name_true} not in columns df")
            continue
        category_name = features_to_cat.get(ts_name_true)
        df[f"{ts_name}_norm"] = normalize_series(df[f"{ts_name}"], method=norm_method)
        cat_norm_name = f"{category_name}_norm"
        if cat_norm_name not in categoris_by_normalized_ts:
            categoris_by_normalized_ts[cat_norm_name] = [df[f"{ts_name}_norm"]]
        else:
            categoris_by_normalized_ts[cat_norm_name].append(df[f"{ts_name}_norm"])
    for idx, category_name in enumerate(features_to_cat.values()):
        if f"{category_name}_norm" in categoris_by_normalized_ts:
            df[f"{category_name}_norm_{ts_aggregation_funct}"] = aggregate_ts(categoris_by_normalized_ts[f"{category_name}_norm"], ts_aggregation_funct)
        
    return df
