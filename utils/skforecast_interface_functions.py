import os
import warnings
from typing import Dict, List, Optional, Tuple
from .model_operations import plot_forecaster_feature_importance
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skforecast.direct import ForecasterDirectMultiVariate

import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from skforecast.model_selection import TimeSeriesFold
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# skforecast imports
from skforecast.model_selection  import (
    backtesting_forecaster_multiseries,
    grid_search_forecaster_multiseries,
    random_search_forecaster_multiseries
)

# -------------------------
# Helper: build forecaster
# -------------------------
def build_forecaster(
    regressor,
    target_level: str,
    lags: int | dict = 12,
    steps: int = 12,
    transformer_series = StandardScaler(),
    transformer_exog = StandardScaler(),
    n_jobs = 'auto',
    forecaster_id: Optional[str] = None
) -> ForecasterDirectMultiVariate:
    """
    Create a ForecasterDirectMultiVariate ready to fit.
    - regressor: any sklearn-compatible regressor or a Pipeline
    - target_level: name of the series (column) you want to predict, e.g. 'A'
    - lags: int or dict (different lags per series) as allowed by skforecast
    - steps: forecast horizon (12 for next 12 months)
    """
    forecaster = ForecasterDirectMultiVariate(
        regressor = regressor,
        level = target_level,
        lags = lags,
        steps = steps,
        transformer_series = transformer_series,
        transformer_exog = transformer_exog,
        n_jobs = n_jobs,
        forecaster_id = forecaster_id
    )
    return forecaster

# ---------------------------------------
# Backtest / evaluate a single forecaster
# ---------------------------------------
def evaluate_forecaster_backtest(
    forecaster,
    series: pd.DataFrame,
    exog: Optional[pd.DataFrame],
    steps: int = 12,
    initial_train_size: Optional[int] = None,
    metric: str = "mean_absolute_percentage_error",
    levels: Optional[str | List[str]] = None,
    refit: bool = False,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run multiseries backtesting focusing on the `levels` (target series).
    Returns (metrics_df, predictions_df).
    """
    if initial_train_size is None:
        # default: use first 70% of data as initial train
        initial_train_size = int(len(series) * 0.7)
    
    cv = TimeSeriesFold(
        initial_train_size = 60,   # e.g. first 60 months
        steps               = 12,   # forecast horizon
        gap                 = 0,
        fixed_train_size    = False,
        allow_incomplete_fold = True,
        refit = refit
    )
    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster = forecaster,
        series = series,
        exog = exog,
        cv = cv,
        metric = metric,
        levels = levels,
        verbose = verbose,
        show_progress = verbose
    )
    return metrics, predictions


def compare_models_backtest(
    series: pd.DataFrame,
    exog: Optional[pd.DataFrame],
    target_level: str,
    regressors: Dict[str, object],
    lags_to_try: List[int] = [6, 12, 24],
    steps: int = 12,
    initial_train_size: Optional[int] = None,
    metric: str = "mean_absolute_error",
    n_jobs = 1,
) -> pd.DataFrame:
    """
    For a list/dict of regressors, run backtesting and return a DataFrame
    with aggregate metric for target_level.
    regressors: {'rf': RandomForestRegressor(...), 'ridge': Ridge(...)}
    """
    records = []
    for name, reg in regressors.items():
        for l in lags_to_try:
            forecaster = build_forecaster(
                regressor = reg,
                target_level = target_level,
                lags = l,
                steps = steps,
                transformer_series = StandardScaler(),
                transformer_exog = StandardScaler(),
                n_jobs = n_jobs,
                forecaster_id = f"{name}_lags{l}"
            )

            warnings.filterwarnings("ignore")
            try:
                metrics, preds = evaluate_forecaster_backtest(
                    forecaster = forecaster,
                    series = series,
                    exog = exog,
                    steps = steps,
                    initial_train_size = initial_train_size,
                    metric = metric,
                    levels = target_level,
                    refit = False,
                    verbose = False
                )
                # metrics is a DataFrame: get the metric value for our level
                metric_val = None
                if isinstance(metrics, pd.DataFrame):
                    # try typical layout: metrics.loc[target_level, metric]
                    if target_level in metrics.index:
                        metric_val = metrics.loc[target_level, metric]
                    else:
                        # aggregated metric case (weighted_average / average)
                        metric_val = metrics.iloc[0][metric]
                else:
                    metric_val = float(metrics)
            except Exception as e:
                metric_val = np.nan
                preds = pd.DataFrame()
                print(f"[WARN] model {name} lags {l} failed: {e}")
            records.append({
                "model": name,
                "lags": l,
                "metric": metric,
                "metric_value": metric_val
            })

    results_df = pd.DataFrame.from_records(records)
    results_df = results_df.sort_values("metric_value").reset_index(drop=True)
    return results_df


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


def fit_and_forecast(
    forecaster,
    series: pd.DataFrame,
    exog: Optional[pd.DataFrame],
    steps: int = 12,
    exog_future: Optional[pd.DataFrame] = None,
    fallback_repeat_last_exog: bool = True
) -> Tuple[object, pd.DataFrame]:
    """
    Fit forecaster on full history (series + exog) then forecast next `steps`.
    Robust index alignment and clear error messages.
    """
    # 1) Ensure datetime index on series
    series = _ensure_series_datetime_index(series)

    # 2) Align exog to series index (or None)
    exog_aligned = _align_exog_to_series(series, exog)

    # 3) Fit forecaster
    forecaster.fit(series=series, exog=exog_aligned)

    # 4) Build future index (monthly start frequency)
    last_index = series.index[-1]
    start = (pd.to_datetime(last_index) + pd.DateOffset(months=1)).to_period('M').to_timestamp()
    future_index = pd.date_range(start=start, periods=steps, freq='MS')

    # 5) Build or validate exog_future
    if exog_future is None:
        if exog_aligned is None:
            exog_future = None
            warnings.warn("No exogenous provided at fit time and exog_future is None — predicting without exog.")
        else:
            if fallback_repeat_last_exog:
                last_row = exog_aligned.iloc[-1:].copy()
                exog_future = pd.concat([last_row] * steps, ignore_index=False)
                exog_future.index = future_index
                # ensure column order same as fit-time exog
                exog_future = exog_future[exog_aligned.columns]
                warnings.warn("No exog_future provided — repeating last observed exog row for the forecast horizon.")
            else:
                raise ValueError("exog_future is required because exog was provided at fit time.")
    else:
        # If exog_future provided, ensure index is DatetimeIndex and matches future_index
        exog_future = exog_future.copy()
        if not isinstance(exog_future.index, pd.DatetimeIndex):
            try:
                exog_future.index = pd.to_datetime(exog_future.index)
            except Exception:
                # If conversion fails but lengths match, set index by future_index
                if len(exog_future) == steps:
                    exog_future.index = future_index
                    warnings.warn("exog_future index wasn't datetime — set to forecast future_index by position.")
                else:
                    raise ValueError("exog_future index is not datetime and length != steps; cannot assign proper index.")
        # if index doesn't match required future_index but lengths equal -> reindex by position
        if not exog_future.index.equals(future_index):
            if len(exog_future) == steps:
                exog_future = exog_future.reindex(future_index)
            else:
                raise ValueError("exog_future index doesn't match forecast horizon. Provide exog_future indexed by the target future dates.")
        # Ensure same columns as exog_aligned
        if exog_aligned is not None:
            if set(exog_future.columns) != set(exog_aligned.columns):
                raise ValueError("exog_future columns do not match the exog columns used during fit.")
            exog_future = exog_future[exog_aligned.columns]

    # 6) Predict
    preds = forecaster.predict(steps=steps, exog=exog_future)

    # 7) Make sure preds index is the future_index
    try:
        preds.index = future_index
    except Exception:
        pass

    return forecaster, preds


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

# ---------- Helper utilities ----------

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


def extract_feature_importances(trained_regressor, feature_names: List[str]) -> np.ndarray:
    """
    Return ndarray of importances aligned with feature_names length.
    Handles:
      - sklearn-like .feature_importances_
      - linear .coef_ (abs value)
      - pipeline wrappers by unwrapping common attributes.
    """
    # unwrap sklearn Pipeline-like
    try:
        # If it's a Pipeline, get the final estimator
        if hasattr(trained_regressor, "named_steps"):
            trained_regressor = list(trained_regressor.named_steps.values())[-1]
    except Exception:
        pass

    imp = None
    if hasattr(trained_regressor, "feature_importances_"):
        try:
            imp = np.array(trained_regressor.feature_importances_, dtype=float)
        except Exception:
            imp = None
    if imp is None and hasattr(trained_regressor, "coef_"):
        try:
            coef = np.array(trained_regressor.coef_, dtype=float)
            # For multi-output, average absolute coefficients
            if coef.ndim == 1:
                imp = np.abs(coef)
            else:
                imp = np.mean(np.abs(coef), axis=0)
        except Exception:
            imp = None

    # fallback: zeros
    if imp is None:
        imp = np.zeros(len(feature_names), dtype=float)

    # ensure length matches feature_names; if not, try resize (best-effort)
    if imp.shape[0] != len(feature_names):
        imp = np.resize(imp, len(feature_names))

    # ensure numeric
    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    return imp


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


# ---------- Modified function (returns feature importance df as 4th value) ----------

def evaluate_models_with_macros_kfolds_with_preds(
    series_full: pd.DataFrame,
    exog_full: Optional[pd.DataFrame],
    target_level: str,
    regressors: Dict[str, object],
    predicted_by: str = "",
    lags_to_try: List[int] = [6, 12, 24],
    horizons: List[int] = [12, 24, 36],
    n_folds: int = 3,
    fold_stride_months: Optional[int] = None,
    fallback_repeat_last_exog: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rolling-origin k-fold evaluation that also returns predictions per fold and a single
    feature importance dataframe with one row per (feature, step, fold, model...).
    Returns: (summary_df, detailed_df, preds_df, feature_imp_df)
    """
    # Prepare
    series_full = _to_monthly_index(series_full)
    if exog_full is not None:
        exog_full = _to_monthly_index(exog_full)

    last_date = series_full.index.max()
    detailed_results = []
    preds_records = []
    feature_imp_records = []   # <-- store per-feature importance rows here
    idx_feature_imp = 0

    for horizon in horizons:
        if horizon <= 0:
            continue
        stride = horizon if fold_stride_months is None else fold_stride_months
        last_cutoff = (last_date - pd.DateOffset(months=horizon)).to_period('M').to_timestamp()
        # compute cutoffs (n_folds cutoffs ending at last_cutoff, spaced by stride)
        cutoffs = [(last_cutoff - pd.DateOffset(months=stride * (n_folds - 1 - i))).to_period('M').to_timestamp()
                   for i in range(n_folds)]
        cutoffs = [c for c in cutoffs if c > series_full.index.min()]
        if len(cutoffs) == 0:
            warnings.warn(f"No valid cutoffs for horizon={horizon}. Skipping.")
            continue

        for name, reg in regressors.items():
            for l in lags_to_try:
                fold_mapes = []
                for fold_idx, cutoff in enumerate(cutoffs, start=1):
                    if verbose:
                        print(f"[INFO] model={name}, lags={l}, horizon={horizon}, fold={fold_idx}/{len(cutoffs)}, cutoff={cutoff.date()}")

                    train_series = series_full.loc[series_full.index <= cutoff]
                    train_exog = exog_full.loc[exog_full.index <= cutoff] if exog_full is not None else None

                    if len(train_series) <= l:
                        warnings.warn(f"Not enough training history for lags={l} at cutoff={cutoff}. Skipping fold.")
                        continue
                    
                    start = (cutoff + pd.DateOffset(months=1)).to_period('M').to_timestamp()
                    future_index = pd.date_range(start=start, periods=horizon, freq='MS')

                    actual = series_full.reindex(future_index)[target_level]
                    print(actual)
                    forecaster = ForecasterDirectMultiVariate(
                        regressor = reg,
                        level     = target_level,
                        lags      = l,
                        steps     = horizon,
                        transformer_series = StandardScaler(),
                        transformer_exog   = StandardScaler()
                    )
                    try:
                        forecaster.fit(series = train_series, exog = train_exog)
                    except Exception as e:
                        warnings.warn(f"Fit failed for model={name}, lags={l}, cutoff={cutoff}: {e}")
                        continue

                    exog_future = _safe_get_exog_future(exog_full, future_index, fallback_repeat_last_exog)

                    # If you still want the existing plotting calls, keep them:
                    try:
                        plot_forecaster_feature_importance(forecaster,
                                                          iteration=idx_feature_imp,
                                                          target_level=target_level,
                                                          predicted_by=predicted_by,
                                                          series=train_series,
                                                          exog=train_exog,
                                                          average_across_steps=True,
                                                          top_n=30)
                    except Exception as e:
                        print(f"Failed to plot feature imortance: {e}")
                    try:
                        preds = forecaster.predict(steps = horizon, exog = exog_future)
                    except Exception as e:
                        warnings.warn(f"Predict failed for model={name}, lags={l}, cutoff={cutoff}: {e}")
                        continue

                    # normalize preds to Series indexed by future_index
                    print(preds)
                    if isinstance(preds, pd.DataFrame):
                        if target_level in preds.columns:
                            preds_series = preds[target_level].copy()
                        elif "pred" in preds.columns:
                            preds_series = preds["pred"].copy()
                        else:
                            preds_series = preds.iloc[:, 0].copy()
                    else:
                        preds_series = pd.Series(preds).copy()

                    try:
                        preds_series.index = future_index
                    except Exception:
                        preds_series = preds_series.reindex(future_index)

                    # compute MAPE
                    actual_nonnull = actual.dropna()
                    mask = (actual_nonnull != 0)
                    if mask.sum() == 0:
                        mape = np.nan
                        warnings.warn("All actuals are zero or NaN for this fold; MAPE undefined.")
                    else:
                        common_idx = actual_nonnull.index[mask]
                        mape = ((actual_nonnull.loc[common_idx] - preds_series.loc[common_idx]).abs() / actual_nonnull.loc[common_idx].abs()).mean() * 100.0

                    fold_mapes.append(mape)
                    detailed_results.append({
                        'model': name,
                        'lags': l,
                        'horizon_months': horizon,
                        'fold_index': fold_idx,
                        'cutoff': cutoff,
                        'mape': mape
                    })

                    # store predictions and actuals for plotting later
                    preds_records.append({
                        'model': name,
                        'lags': l,
                        'horizon_months': horizon,
                        'fold_index': fold_idx,
                        'cutoff': cutoff,
                        'preds_series': preds_series,
                        'actual_series': actual
                    })

                    # -------------------------
                    # EXTRACT FEATURE IMPORTANCES
                    # -------------------------
                    # We'll try to get trained regressor for each step (1..horizon)
                    for step in range(1, horizon + 1):
                        trained_reg = _get_trained_regressor(forecaster, step)
                        if trained_reg is None:
                            # skip if not retrievable
                            continue

                        # try to get feature names from forecaster if available
                        feature_names = None
                        try:
                            # some forecaster implementations expose get_feature_names
                            if hasattr(forecaster, "get_feature_names"):
                                feature_names = forecaster.get_feature_names(step=step)
                        except Exception:
                            feature_names = None

                        # fallback: build heuristic feature names
                        if feature_names is None:
                            feature_names = _build_feature_names_from_inputs(train_series, train_exog, lags=l)

                        # now extract importances aligned with feature_names
                        importances_raw = extract_feature_importances(trained_reg, feature_names)
                        sum_imp = importances_raw.sum()
                        if sum_imp == 0:
                            importances_norm = np.zeros_like(importances_raw)
                        else:
                            importances_norm = importances_raw / sum_imp

                        # append per-feature rows
                        for fname, raw, norm in zip(feature_names, importances_raw, importances_norm):
                            feature_imp_records.append({
                                'target_series': target_level,
                                'predictor_series': predicted_by,
                                'model_family': name,
                                'horizon': horizon,
                                'step': step,
                                'lags': l,
                                'fold_index': fold_idx,
                                'cutoff': cutoff,
                                'mape': mape,
                                'feature_name': fname,
                                'importance_raw': float(raw),
                                'importance_norm': float(norm)
                            })

                    idx_feature_imp += 1
                # (fold aggregation done later from detailed_results)

    detailed_df = pd.DataFrame(detailed_results)
    if detailed_df.empty:
        summary_df = pd.DataFrame(columns=['model','lags','horizon_months','mean_mape','std_mape','n_valid_folds'])
        preds_df = pd.DataFrame(preds_records)
        feature_imp_df = pd.DataFrame(feature_imp_records)  # may be empty
        return summary_df, detailed_df, preds_df, feature_imp_df

    summary_df = (
        detailed_df
        .groupby(['model','lags','horizon_months'], as_index=False)
        .agg(mean_mape=('mape','mean'),
             std_mape=('mape','std'),
             n_valid_folds=('mape','count'))
        .sort_values(['horizon_months','mean_mape'])
        .reset_index(drop=True)
    )

    preds_df = pd.DataFrame(preds_records)
    feature_imp_df = pd.DataFrame(feature_imp_records)

    return summary_df, detailed_df, preds_df, feature_imp_df
