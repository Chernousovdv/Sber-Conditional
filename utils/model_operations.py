import pickle
from pathlib import Path
import pandas as pd
import warnings
from typing import Optional, Tuple, Any
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.pipeline import Pipeline


def save_shap(model,
              train_data: pd.DataFrame,
              model_name: str):
    explainer = shap.TreeExplainer(model)
    shap_values = np.array(explainer.shap_values(train_data))
    os.makedirs(f"results/{save_dir}/{model_name}_shaps", exist_ok=True)
    shap.summary_plot(shap_values,
                      X_train,
                      max_display=10,
                      show=False)
    plt.savefig(f"results/{save_dir}/{model_name}_shaps/{model_name}_shap.png") #.png,.pdf will also support here


def save_model(model,
               train_data: pd.DataFrame,
               save_dir: str,
               model_name: str,
               is_shap: bool = True):
    os.makedirs(f"results/{save_dir}", exist_ok=True)
    prev_results = Path(f"results/{save_dir}/{model_name}.pkl")
    if prev_results.is_file():
        print(f"Skipping {prev_results}\texperiment was made.")
        return
    with open(prev_results, 'wb') as file:
        pickle.dump(model, file)
    if is_shap:
        save_shap(model, train_data, model_name)


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


def _get_final_estimator(estimator):
    """
    If estimator is a sklearn Pipeline, return the last estimator else return estimator.
    """
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator

def _extract_feature_importances_from_estimator(estimator) -> Optional[np.ndarray]:
    """
    Return feature importances array from estimator if possible.
    Supports tree-based estimators (.feature_importances_) and linear (.coef_).
    Returns None if estimator doesn't expose importances.
    """
    est = _get_final_estimator(estimator)
    # Tree-based
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_)
    # Linear / coef-based
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        # coef may be 1d or 2d (multioutput). If 2d, average absolute across outputs.
        if coef.ndim == 1:
            return np.abs(coef)
        else:
            return np.mean(np.abs(coef), axis=0)
    # sklearn newer attribute feature_importances_in_? not standard — fallback None
    return None


def plot_forecaster_feature_importance(
    forecaster,
    series: pd.DataFrame,
    iteration: int = None,
    target_level: str = None,
    predicted_by: str = None,
    exog: Optional[pd.DataFrame] = None,
    step: Optional[int] = None,
    top_n: Optional[int] = 25,
    average_across_steps: bool = False,
    steps_to_use: Optional[list[int]] = None,
    figsize=(10,6),
    ascending=False,
    title: Optional[str] = None
) -> dict[str, Any]:
    """
    Plot feature importances for a trained ForecasterDirectMultiVariate.

    Parameters
    ----------
    forecaster : fitted ForecasterDirectMultiVariate
        The forecaster must already be fitted (forecaster.fit(...)).
    series : pd.DataFrame
        The series DataFrame used for training (wide format).
    exog : pd.DataFrame or None
        Exogenous DataFrame used for training (aligned with series).
    step : int or None
        If specified, show importances for that specific step (1..steps). If None and average_across_steps==False
        the function will plot importances for the first step.
    top_n : int or None
        How many top features to display (None -> all).
    average_across_steps : bool
        If True, compute mean importance across the selected steps and plot a single aggregated ranking.
    steps_to_use : list[int] or None
        List of steps to include when averaging. If None, use all `forecaster.regressors_.keys()`.
    Returns
    -------
    dict with keys:
      - 'features' : pd.Index of feature names
      - 'importances' : numpy array of importances (same length)
      - 'fig' : matplotlib Figure
    """
    # 1) basic checks
    if not hasattr(forecaster, "regressors_"):
        raise ValueError("Forecaster does not appear to be fitted or does not have regressors_. Fit it first.")

    # 2) get training matrices that skforecast used (this gives correct feature names/order).
    # For multivariate direct forecaster, use create_train_X_y(series, exog)
    X_train, y_train = forecaster.create_train_X_y(series, exog=exog)
    # X_train is the full matrix used to train all step models (columns include suffixes like '_step_i')
    # For a particular step, we need to filter the columns needed for that step.
    # If user asked for a specific step:
    if step is None:
        # default step = first step
        step = list(forecaster.regressors_.keys())[0] if hasattr(forecaster, "regressors_") else 1

    # If averaging across steps, decide which steps to use
    if average_across_steps:
        if steps_to_use is None:
            steps_to_use = sorted(list(forecaster.regressors_.keys()))
        # accumulate importances for each step
        importance_list = []
        feature_names_list = []
        for s in steps_to_use:
            X_step, y_step = forecaster.filter_train_X_y_for_step(s, X_train, y_train, remove_suffix=True)
            est = forecaster.regressors_.get(s, None)
            if est is None:
                warnings.warn(f"No trained estimator for step {s} found in forecaster.regressors_. Skipping step.")
                continue
            imp = _extract_feature_importances_from_estimator(est)
            if imp is None:
                warnings.warn(f"Estimator for step {s} exposes no feature importances. Skipping step.")
                continue
            # imp must correspond to X_step.columns order
            feature_names_list.append(X_step.columns)
            importance_list.append(pd.Series(imp, index=X_step.columns))
        if len(importance_list) == 0:
            raise ValueError("No feature importances found for any steps.")
        # align series and compute mean importance across steps
        imp_df = pd.concat(importance_list, axis=1).fillna(0)
        mean_imp = imp_df.mean(axis=1)
        feat_names = mean_imp.index
        importances = mean_imp.values
    else:
        # single step case
        X_step, y_step = forecaster.filter_train_X_y_for_step(step, X_train, y_train, remove_suffix=True)
        est = forecaster.regressors_.get(step, None)
        if est is None:
            raise ValueError(f"No trained estimator for step={step} found in forecaster.regressors_.")
        imp = _extract_feature_importances_from_estimator(est)
        if imp is None:
            raise ValueError("Estimator does not expose feature importances (not a tree or linear model).")
        # imp should match X_step.columns order
        feat_names = X_step.columns
        # If imp length differs from number of features, attempt to use estimator.feature_names_in_ if available
        if len(imp) != len(feat_names):
            # try to use feature_names_in_ if present
            est_final = _get_final_estimator(est)
            if hasattr(est_final, "feature_names_in_"):
                feat_names = pd.Index(est_final.feature_names_in_)
            # else: best-effort: trim or pad imp
            if len(imp) > len(feat_names):
                imp = imp[:len(feat_names)]
            elif len(imp) < len(feat_names):
                # pad with zeros
                imp = np.concatenate([imp, np.zeros(len(feat_names) - len(imp))])
        importances = np.asarray(imp)

    # 3) build dataframe and plot
    imp_df = pd.Series(importances, index=feat_names).sort_values(ascending=ascending)
    if top_n is not None:
        imp_df = imp_df.iloc[-top_n:] if not ascending else imp_df.iloc[:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    imp_df.plot(kind='barh', ax=ax)
    ax.set_xlabel("Importance (abs or tree feature_importances_)")
    t = title or f"Feature importances (model id={getattr(forecaster, 'forecaster_id', None)}, step={step})"
    ax.set_title(t)
    plt.tight_layout()
    os.makedirs(f"results/{target_level}_{predicted_by}_featureimportance", exist_ok=True)
    plt.savefig(f"results/{target_level}_{predicted_by}_featureimportance/{target_level}_{predicted_by}_{iteration}.png")

    # return {'features': feat_names, 'importances': importances, 'fig': fig}