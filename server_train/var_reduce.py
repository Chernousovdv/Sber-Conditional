import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Type
import pandas as pd
import pmdarima as pm
from pathlib import Path
import os


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


def build_error_matrix(
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

    for conditioning_col in all_columns:
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

    return error_matrix


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


class SARIMAModel:
    """
    A model that fits an individual SARIMA model for each time series.
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.sarima_params = {
            "m": kwargs.get("m", 12),
            "seasonal": kwargs.get("seasonal", True),
            "stepwise": kwargs.get("stepwise", True),
            "suppress_warnings": kwargs.get("suppress_warnings", True),
            "error_action": kwargs.get("error_action", "ignore"),
        }
        self.fitted_models = {}

        # Fit a separate auto_arima model for each column
        for col in self.all_columns:
            model = pm.auto_arima(df[col], **self.sarima_params)
            self.fitted_models[col] = model

    def make_prediction(self, horizon: int, columns: List[str], values: List[float]):
        """
        Unconditional forecast.
        """
        predictable_cols = []
        predicted_series_list = []

        for col, model in self.fitted_models.items():
            forecast = model.predict(n_periods=horizon)
            predicted_series_list.append(forecast.tolist())
            predictable_cols.append(col)

        return predictable_cols, predicted_series_list


# ==================================================================================================
# ==================================================================================================

# ==================================================================================================
# ==================================================================================================

# Read data
data_path = "/app/Data/apk_nona.csv"
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index)

# Parameters
matrix_params = {
    "model_class": SARIMAModel,
    "df": df,
    "horizon": 12,
    "stride": 10,
    "start_window": 60,
    "model_params": {},  # seasonality is 12 by default
}
sarima_model_matrices = {}
horizon_list = [24, 36, 48]
output_dir = "/app/code/"  # Проще без вложенности
output_path = f"{output_dir}/test_worKED.csv"
df.to_csv(output_path, index=True)
# Perform calculations
for horizon in horizon_list:
    matrix_params["horizon"] = horizon
    err_mat = build_short_error_matrix(**matrix_params)

    # Используйте уже смонтированную директорию

    output_path = f"{output_dir}/sarima_apk_{horizon}.csv"
    err_mat.to_csv(output_path, index=True)


# Chemicals

# Read data
data_path = "/app/Data/chemicals_nona.csv"
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index)

# Parameters
matrix_params = {
    "model_class": SARIMAModel,
    "df": df,
    "horizon": 12,
    "stride": 10,
    "start_window": 60,
    "model_params": {},  # seasonality is 12 by default
}
sarima_model_matrices = {}
horizon_list = [24, 36, 48]

output_dir = "/app/code/"  # Проще без вложенности

# Perform calculations
for horizon in horizon_list:
    matrix_params["horizon"] = horizon
    err_mat = build_short_error_matrix(**matrix_params)

    # Используйте уже смонтированную директорию

    output_path = f"{output_dir}/sarima_chemicals_{horizon}.csv"
    err_mat.to_csv(output_path, index=True)
