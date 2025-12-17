import pandas as pd
import warnings
from typing import Optional, Tuple, Any
import pickle
import numpy as np


def _ensure_monthly_index_and_align_exog(
    series: pd.DataFrame,
    exog: Optional[pd.DataFrame],
    freq: str = "MS",
    reindex_if_irregular: bool = False,
    fill_method: Optional[str] = None  # None, 'ffill', 'bfill', or 'interpolate'
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Ensure `series` has a DatetimeIndex with monthly frequency (freq != None).
    Aligns `exog` to `series` index (by index if possible, or by position if same length).
    
    Parameters
    ----------
    series: pd.DataFrame
        Time series dataframe with DatetimeIndex or index convertible to datetime.
    exog: pd.DataFrame or None
        Exogenous dataframe to align to series (index must match after alignment).
    freq: str
        Desired frequency, default 'MS' (month start). Use 'M' for month end if needed.
    reindex_if_irregular: bool
        If True and index is irregular (missing months), reindex `series` to full monthly range
        from first to last date and fill using `fill_method`. If False, raise informative error.
    fill_method: None | 'ffill' | 'bfill' | 'interpolate'
        Method to fill values if reindexing introduces NaNs.
    
    Returns
    -------
    (series_aligned, exog_aligned)
    """
    # 1) ensure datetime index
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series = series.copy()
            series.index = pd.to_datetime(series.index)
        except Exception as e:
            raise ValueError("`series.index` is not a DatetimeIndex and couldn't be converted.") from e

    series = series.sort_index()

    # 2) try to infer existing frequency
    inferred = pd.infer_freq(series.index)
    if inferred:
        # Accept many monthly-like inferred frequencies, but we convert to the user freq for consistency
        # If inferred differs from requested freq, we still set the desired freq (these are compatible monthly types)
        try:
            # Create a new DatetimeIndex with requested freq but same values
            series.index = pd.DatetimeIndex(series.index, freq=freq)
        except Exception:
            # If setting freq fails, fall back to use the inferred freq
            series.index = pd.DatetimeIndex(series.index, freq=inferred)
    else:
        # no freq inferred -> check whether the index looks like monthly sequence (consecutive months)
        # build expected monthly index from first to last and compare length
        first = series.index[0].to_period('M').to_timestamp()
        last  = series.index[-1].to_period('M').to_timestamp()
        expected_index = pd.date_range(start=first, end=last, freq=freq)

        if len(expected_index) == len(series):
            # same number of months -> assume monthly but freq metadata missing -> set freq
            series.index = pd.DatetimeIndex(series.index, freq=freq)
            warnings.warn(f"No frequency inferred but index appears monthly — set freq='{freq}'.")
        else:
            # irregular: missing or duplicate months
            msg = (
                "series DatetimeIndex has no frequency (freq=None) and it is irregular "
                "(missing/duplicate months)."
            )
            if not reindex_if_irregular:
                raise ValueError(
                    msg + " Set `reindex_if_irregular=True` to reindex to a full monthly range "
                    "or fix your index so it is a complete monthly sequence."
                )
            # reindex to expected_index and fill if requested
            series = series.reindex(expected_index)
            if fill_method is None:
                # leave NaNs, but warn
                warnings.warn("Reindexed series to full monthly range and created NaNs. Fill them or pass fill_method.")
            elif fill_method in ("ffill", "bfill"):
                series = getattr(series, fill_method)()
            elif fill_method == "interpolate":
                series = series.interpolate()
            else:
                raise ValueError("Unsupported fill_method. Choose None, 'ffill', 'bfill', or 'interpolate'.")
            # set freq
            series.index = pd.DatetimeIndex(series.index, freq=freq)
            warnings.warn(f"Series reindexed to monthly freq='{freq}' and filled with method={fill_method}.")

    # 3) align exog to the series index
    if exog is None:
        return series, None

    exog = exog.copy()
    # If exog has non-datetime index try convert
    if not isinstance(exog.index, pd.DatetimeIndex):
        try:
            exog.index = pd.to_datetime(exog.index)
        except Exception:
            # if cannot convert but lengths match -> align by position
            if len(exog) == len(series):
                exog.index = series.index.copy()
                warnings.warn("exog index not datetime and couldn't be converted — aligned to series by position.")
                # ensure columns order preserved
                return series, exog
            raise ValueError("exog index couldn't be converted to datetime. Provide exog with datetimelike index.")

    # If exog index equals series.index -> good
    if exog.index.equals(series.index):
        exog.index = pd.DatetimeIndex(exog.index, freq=series.index.freq)
        return series, exog

    # If same length, align by position
    if len(exog) == len(series):
        exog.index = series.index.copy()
        warnings.warn("exog length equals series but index differs -> aligned by position (set exog.index = series.index).")
        return series, exog

    # If exog covers the same time span but with a superset/subset of dates, try reindex
    try:
        exog_reindexed = exog.reindex(series.index)
        if exog_reindexed.isna().any().any():
            raise ValueError(
                "Reindexing exog to series.index created NaNs. Provide exog with exact same DatetimeIndex as series "
                "or ensure it covers the same dates without gaps."
            )
        exog_reindexed.index = pd.DatetimeIndex(exog_reindexed.index, freq=series.index.freq)
        return series, exog_reindexed
    except Exception as e:
        raise ValueError("Could not align exog to series index. Provide exog with same DatetimeIndex or matching length.") from e
