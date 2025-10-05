import numpy as np
import os
from collections import defaultdict
from typing import List, Dict, Any, Type
import pandas as pd
import pmdarima as pm
from pathlib import Path
import statsmodels.api as sm
import itertools


def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return np.nan

    y_true_safe = y_true[non_zero_mask]
    y_pred_safe = y_pred[non_zero_mask]

    return np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100


def cross_validate_model(
    model_class: Type,
    df: pd.DataFrame,
    horizon: int,
    stride: int,
    start_window: int,
    target_columns: List[str],
    model_params: Dict[str, Any],
) -> Dict[str, List[float]]:
    errors = defaultdict(list)
    max_train_end_index = len(df) - horizon
    current_train_end_index = start_window

    fold_count = 0
    while current_train_end_index <= max_train_end_index:
        fold_count += 1

        df_train = df.iloc[:current_train_end_index]

        model = model_class(df_train, **model_params)

        for conditioning_col in target_columns:
            conditioning_idx = current_train_end_index + horizon - 1
            conditioning_val = df.loc[df.index[conditioning_idx], conditioning_col]

            predicted_cols, predicted_series = model.make_prediction(
                horizon=horizon, columns=[conditioning_col], values=[conditioning_val]
            )

            for i, pred_col in enumerate(predicted_cols):
                start_val_idx = current_train_end_index
                end_val_idx = current_train_end_index + horizon
                actual_vals = df[pred_col].iloc[start_val_idx:end_val_idx].values

                predicted_vals = predicted_series[i]

                mape = _calculate_mape(actual_vals, predicted_vals)
                if not np.isnan(mape):
                    errors[pred_col].append(mape)

        current_train_end_index += stride

    return dict(errors)


def calc_avg_error(cv_errors):
    ans = {}
    for column, error_list in cv_errors.items():
        avg_mape = np.mean(error_list)
        ans[column] = avg_mape

    return ans


def build_short_error_matrix(
    model_class: Type,
    df: pd.DataFrame,
    horizon: int,
    stride: int,
    start_window: int,
    model_params: Dict[str, Any],
) -> pd.DataFrame:

    all_columns = df.columns.tolist()

    error_matrix = pd.DataFrame(index=all_columns, columns=all_columns, dtype=float)
    error_matrix.index.name = "Conditioning Variable"
    error_matrix.columns.name = "Predicted Variable"

    avg_errors = 0
    for conditioning_col in all_columns[:1]:
        cv_errors = cross_validate_model(
            model_class=model_class,
            df=df,
            horizon=horizon,
            stride=stride,
            start_window=start_window,
            target_columns=[conditioning_col],
            model_params=model_params,
        )

        avg_errors = calc_avg_error(cv_errors)

        for predicted_col, avg_error in avg_errors.items():
            error_matrix.loc[conditioning_col, predicted_col] = avg_error

    for conditioning_col in all_columns[1:]:
        for predicted_col, avg_error in avg_errors.items():
            error_matrix.loc[conditioning_col, predicted_col] = avg_error

    return error_matrix


class DynamicFactorModel:
    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.fitted_model = None

        k_factors = kwargs.get("k_factors", 1)
        factor_order = kwargs.get("factor_order", 1)
        error_order = kwargs.get("error_order", 1)
        maxiter = kwargs.get("maxiter", 30)

        try:

            model = sm.tsa.DynamicFactor(
                df,
                k_factors=k_factors,
                factor_order=factor_order,
                error_order=error_order,
                enforce_stationarity=False,
            )
            self.fitted_model = model.fit(disp=False, maxiter=maxiter)
        except Exception as e:
            print(f"Warning: Dynamic Factor model could not be fitted. Error: {e}")

    def make_prediction(self, horizon: int, columns: List[str], values: List[float]):
        if not self.fitted_model:
            return [], []

        forecast = self.fitted_model.forecast(steps=horizon)

        predictable_cols = self.all_columns
        predicted_series_list = forecast.T.values.tolist()

        return predictable_cols, predicted_series_list


# 1. Load Data
data_path = "/app/Data/apk_nona.csv"  # ============================================================
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index)

# 2. Define Hyperparameter Grid
param_grid = {  # =================================================================================
    "k_factors": [3, 4],
    "factor_order": [2, 3, 4],
    "error_order": [2, 3],
    "enforce_stationarity": [False],
    "maxiter": [50],  # Can be fixed if not tuning
}

# Generate all unique combinations of parameters
param_keys = list(param_grid.keys())
param_combinations = list(itertools.product(*param_grid.values()))

# 3. Define Cross-Validation Parameters
horizon_list = [
    12
]  # Smaller list for example  =====================================================
stride = 5  # ========================================================================================
start_window = 60  # =================================================================================

# 4. Run the Grid Search
all_results = []

for horizon in horizon_list:
    print(f"\n--- Starting Hyperparameter Tuning for Horizon: {horizon} ---")

    for combo in param_combinations:
        model_params = dict(zip(param_keys, combo))

        # Evaluate this combination
        err_mat = build_short_error_matrix(
            model_class=DynamicFactorModel,
            df=df,
            horizon=horizon,
            stride=stride,
            start_window=start_window,
            model_params=model_params,
        )

        # Store results
        output_dir = "/app/results/apk_dfm_test1"  # =====================
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/dfm_apk_{horizon}_{combo}.csv"  # ==============
        err_mat.to_csv(output_path, index=True)


print("Finished")
