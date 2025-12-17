# forecasting/utils.py
from typing import Optional, List
import pandas as pd
import numpy as np
import warnings


def _ensure_series_datetime_index(series: pd.DataFrame) -> pd.DataFrame:
    """Ensure `series` has a DateTimeIndex. Convert if possible, else raise."""
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series = series.copy()
            series.index = pd.to_datetime(series.index)
        except Exception as e:
            raise ValueError("`series` index could not be converted to DatetimeIndex.") from e
    return series.sort_index()

def _align_exog_to_series(series: pd.DataFrame, exog: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Make exog index aligned with series index.
    - If exog is None, return None.
    - If exog.index equals series.index -> fine.
    - If exog can be converted to DatetimeIndex and matches -> reindex/align.
    - If len(exog) == len(series) but indices differ -> align by position (set exog.index = series.index) and warn.
    - else raise informative error.
    """
    if exog is None:
        return None

    exog = exog.copy()

    # Try converting exog index to datetime if not already
    if not isinstance(exog.index, pd.DatetimeIndex):
        try:
            exog.index = pd.to_datetime(exog.index)
        except Exception:
            # If conversion fails but lengths match, align by position
            if len(exog) == len(series):
                exog.index = series.index.copy()
                warnings.warn("exog index was not datetime and conversion failed — aligned by position to series index.")
                return exog
            raise ValueError("exog index is not datetime and cannot be converted; lengths differ from series.")

    # Now we have DatetimeIndex; try to align
    if exog.index.equals(series.index):
        return exog.sort_index()
    # If same length but different dates (e.g. off-by-one or different anchor), align by position
    if len(exog) == len(series):
        exog.index = series.index.copy()
        warnings.warn("exog has same length as series but different dates -> aligning by position (setting exog.index = series.index).")
        return exog
    # Otherwise try reindex (if exog covers superset or subset of series dates)
    try:
        exog_reindexed = exog.reindex(series.index)
        if exog_reindexed.isna().any().any():
            # some missing values after reindex -> user must handle
            raise ValueError("Reindexing exog to series index created NaNs. Provide properly aligned exog or fill/forecast exog.")
        return exog_reindexed
    except Exception as e:
        raise ValueError("Could not align exog to series index. Provide exog with the same DatetimeIndex as series or matching length.") from e


def _to_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    try:
        df.index = pd.DatetimeIndex(df.index, freq='MS')
    except Exception:
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index).to_period('M').to_timestamp(), freq='MS')
    return df

def _safe_get_exog_future(exog_full: Optional[pd.DataFrame], future_index: pd.DatetimeIndex, fallback_repeat_last: bool = True):
    if exog_full is None:
        return None
    exog_full = exog_full.copy()
    if not isinstance(exog_full.index, pd.DatetimeIndex):
        exog_full.index = pd.to_datetime(exog_full.index)
    # Try to slice exact dates
    exog_future = exog_full.reindex(future_index)
    if exog_future.isna().any().any():
        # If exog_full contains the exact dates -> take them
        if future_index.isin(exog_full.index).all():
            exog_future = exog_full.loc[future_index]
        else:
            if fallback_repeat_last:
                last = exog_full.iloc[-1:].copy()
                exog_future = pd.concat([last] * len(future_index))
                exog_future.index = future_index
            else:
                raise ValueError("exog_future missing and fallback_repeat_last is False.")
    exog_future = exog_future[exog_full.columns]
    return exog_future


def _get_trained_regressor(forecaster, step: int):
    """
    Try several ways to obtain the trained regressor for a given step from a skforecast Forecaster.
    Returns regressor object or None if not found.
    """
    # Preferred official method (if present)
    try:
        if hasattr(forecaster, "get_trained_regressor"):
            return forecaster.get_trained_regressor(step=step)
    except Exception:
        pass

    # Common internal names (try these heuristics)
    for attr in ("regressors_", "regressors", "steps_regressors", "_regressors"):
        reglist = getattr(forecaster, attr, None)
        if reglist is None:
            continue
        # if dict-like keyed by step
        if isinstance(reglist, dict):
            if step in reglist:
                return reglist[step]
            # sometimes keys are strings
            if str(step) in reglist:
                return reglist[str(step)]
        # if list-like
        try:
            if len(reglist) >= step:
                # step is 1-indexed in skforecast API; python list is 0-indexed
                return reglist[step - 1]
        except Exception:
            pass

    # fallback: maybe single regressor used (for 1-step) at attribute 'regressor'
    if hasattr(forecaster, "regressor"):
        return getattr(forecaster, "regressor")

    return None


def _build_feature_names_from_inputs(train_series: pd.DataFrame, train_exog: Optional[pd.DataFrame], lags: int):
    """
    Build feature name list consistent with what the forecaster generally constructs:
      - For each series column: <col>_lag_<i> for i=1..lags
      - For exog columns: prefix with EXOG_ (or keep names as-is)
    This is a heuristic fallback when the forecaster can't provide feature names.
    """
    feature_names = []
    # series lag features
    series_cols = list(train_series.columns)
    for col in series_cols:
        for lag in range(1, lags + 1):
            feature_names.append(f"{col}_lag_{lag}")

    # exog features (assume not lagged here)
    if train_exog is not None:
        exog_cols = list(train_exog.columns)
        # disambiguate if exog columns overlap with series names
        for c in exog_cols:
            feature_names.append(f"EXOG_{c}")

    return feature_names
