# forecasting/evaluator.py
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np
import warnings
import logging
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries
from .builder import ForecasterFactory
from .utils import (
    _ensure_series_datetime_index,
    _align_exog_to_series,
    _to_monthly_index,
    _safe_get_exog_future,
    _get_trained_regressor
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ForecasterEvaluator:
    """Encapsulates evaluation routines for multivariate forecasters."""

    def __init__(self):
        pass

    def evaluate_forecaster_backtest(
        self,
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
        self,
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
                forecaster = ForecasterFactory.build_forecaster(
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
                    metrics, preds = self.evaluate_forecaster_backtest(
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

    def fit_and_forecast(
        self,
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

    def evaluate_models_with_macros_kfolds_with_preds(
        self,
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
        Rolling-origin evaluation with sanitization of column names so LightGBM (and others)
        do not receive problematic feature names. Returns DataFrames with feature names
        mapped back to their original (Russian) names in the feature_imp_df and preds_df.
        """
        # --- 1) Ensure monthly index and make copies ---
        series_full = _to_monthly_index(series_full)
        if exog_full is not None:
            exog_full = _to_monthly_index(exog_full)
    
        # --- 2) Sanitize column names (create local renamed DataFrames) ---
        series_s, map_series, inv_series = sanitize_df_columns(series_full)
        exog_s = None
        map_exog = {}
        inv_exog = {}
        if exog_full is not None:
            exog_s, map_exog, inv_exog = sanitize_df_columns(exog_full)
    
        # sanitize target_level and predicted_by using map (they should be original names)
        if target_level not in map_series:
            raise ValueError(f"target_level '{target_level}' not found in series columns.")
        target_level_s = map_series[target_level]
        # predicted_by may refer to series column name (a second series) -> sanitize if present
        predicted_by_s = map_series.get(predicted_by, predicted_by) if predicted_by else predicted_by
    
        last_date = series_s.index.max()
        detailed_results = []
        preds_records = []
        feature_imp_records = []
        idx_feature_imp = 0
    
        for horizon in horizons:
            if horizon <= 0:
                continue
            stride = horizon if fold_stride_months is None else fold_stride_months
            last_cutoff = (last_date - pd.DateOffset(months=horizon)).to_period('M').to_timestamp()
    
            cutoffs = [(last_cutoff - pd.DateOffset(months=stride * (n_folds - 1 - i))).to_period('M').to_timestamp()
                       for i in range(n_folds)]
            cutoffs = [c for c in cutoffs if c > series_s.index.min()]
            if len(cutoffs) == 0:
                warnings.warn(f"No valid cutoffs for horizon={horizon}. Skipping.")
                continue
    
            for name, reg in regressors.items():
                if entry.is_file() and entry.name.endswith('.xlsx'):  # check if it's a file
                    # print(entry.name)
                    d[entry.name] = pd.read_excel(f"{directory}{entry.name}")
                for l in lags_to_try:
                    fold_mapes = []
                    for fold_idx, cutoff in enumerate(cutoffs, start=1):
                        if verbose:
                            print(f"[INFO] model={name}, lags={l}, horizon={horizon}, fold={fold_idx}/{len(cutoffs)}, cutoff={cutoff.date()}")
    
                        train_series = series_s.loc[series_s.index <= cutoff]
                        train_exog = exog_s.loc[exog_s.index <= cutoff] if exog_s is not None else None
    
                        if len(train_series) <= l:
                            warnings.warn(f"Not enough training history for lags={l} at cutoff={cutoff}. Skipping fold.")
                            continue
    
                        start = (cutoff + pd.DateOffset(months=1)).to_period('M').to_timestamp()
                        future_index = pd.date_range(start=start, periods=horizon, freq='MS')
    
                        # build regressor instance (callable or clone)
                        try:
                            if callable(reg):
                                reg_instance = reg(horizon, l)
                            else:
                                from sklearn.base import clone
                                reg_instance = clone(reg)
                        except Exception:
                            reg_instance = reg  # fallback
    
                        forecaster = ForecasterDirectMultiVariate(
                            regressor = reg_instance,
                            level     = target_level_s,
                            lags      = l,
                            steps     = horizon,
                            transformer_series = StandardScaler(),
                            transformer_exog   = StandardScaler()
                        )
                        try:
                            forecaster.fit(series = train_series, exog = train_exog)
                            # # 1) Are trained regressors different objects per step?
                            # trained_objs = []
                            # print(forecaster.steps)
                            # for step in range(1, len(forecaster.steps) + 1):
                            #     tr = _get_trained_regressor(forecaster, step)
                            #     trained_objs.append(tr)
                            #     print(f"step={step}  type={type(tr)}  id={id(tr)} repr={getattr(tr,'__class__', None)}")
    
                        except Exception as e:
                            print("?"*100)
                            print(f"Fit failed for model={name}, lags={l}, cutoff={cutoff}: {e}")
                            continue
    
                        exog_future = _safe_get_exog_future(exog_s, future_index, fallback_repeat_last_exog)
    
                        try:
                            preds = forecaster.predict(steps = horizon, exog = exog_future)
                        except Exception as e:
                            warnings.warn(f"Predict failed for model={name}, lags={l}, cutoff={cutoff}: {e}")
                            continue
                        # if isinstance(preds, pd.DataFrame):
                        #     print("Prediction columns:", preds.columns.tolist())
                        #     # compute pairwise correlations between step columns
                        #     corr = preds["pred"].corr()
                        #     print("Correlation between step predictions:\n", corr)
                        #     # quick check: are columns almost identical?
                        #     max_offdiag = corr.where(~np.eye(corr.shape[0],dtype=bool)).max().max()
                        #     if max_offdiag > 0.98:
                        #         print("!"*100)
                        #         print("Predictions for different steps are extremely highly correlated (>0.98).")
    
                        # Normalize preds to Series indexed by future_index and set original series name
                        if isinstance(preds, pd.DataFrame):
                            if target_level_s in preds.columns:
                                preds_series = preds[target_level_s].copy()
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
    
                        # set series names back to original Russian names for downstream use
                        preds_series.name = inv_series.get(target_level_s, target_level_s)
                        actual = series_s.reindex(future_index)[target_level_s]
                        actual.name = inv_series.get(target_level_s, target_level_s)
    
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
                        for step in range(1, horizon + 1):
                            trained_reg = _get_trained_regressor(forecaster, step)
                            if trained_reg is None:
                                print(f"TRAINED REG is none")
                                print("="*100)
                                continue
    
                            feature_names = None
                            try:
                                if hasattr(forecaster, "get_feature_names"):
                                    feature_names = forecaster.get_feature_names(step=step)
                            except Exception:
                                feature_names = None
    
                            if feature_names is None:
                                feature_names = _build_feature_names_from_inputs(train_series, train_exog, lags=l)
    
                            importances_raw = extract_feature_importances_lgb(trained_reg, feature_names)
                            sum_imp = importances_raw.sum()
                            if sum_imp == 0:
                                importances_norm = np.zeros_like(importances_raw)
                            else:
                                importances_norm = importances_raw / sum_imp
    
                            for fname, raw, norm in zip(feature_names, importances_raw, importances_norm):
                                feature_imp_records.append({
                                    'target_series': inv_series.get(target_level_s, target_level_s),
                                    'predictor_series': predicted_by if predicted_by else inv_series.get(predicted_by_s, predicted_by_s),
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
    
        detailed_df = pd.DataFrame(detailed_results)
        if detailed_df.empty:
            summary_df = pd.DataFrame(columns=['model','lags','horizon_months','mean_mape','std_mape','n_valid_folds'])
            preds_df = pd.DataFrame(preds_records)
            feature_imp_df = pd.DataFrame(feature_imp_records)
            # map feature names back if any
            if not feature_imp_df.empty:
                feature_imp_df['feature_name'] = feature_imp_df['feature_name'].apply(lambda f: map_back_feature_name(f, inv_series, inv_exog))
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
    
        # Map feature names back to original Russian names (best-effort heuristic)
        if not feature_imp_df.empty:
            feature_imp_df['feature_name'] = feature_imp_df['feature_name'].apply(lambda f: map_back_feature_name(f, inv_series, inv_exog))
            # ensure target_series / predictor_series show original names
            feature_imp_df['target_series'] = feature_imp_df['target_series'].apply(lambda x: x if x is None else x)
            feature_imp_df['predictor_series'] = feature_imp_df['predictor_series'].apply(lambda x: x if x is None else x)
    
        # Also ensure preds_df entries have Series named with original target name (we set earlier)
        return summary_df, detailed_df, preds_df, feature_imp_df
