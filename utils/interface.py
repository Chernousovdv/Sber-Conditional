import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Type
from tqdm import tqdm


def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: Numpy array of true values.
        y_pred: Numpy array of predicted values.

    Returns:
        The MAPE value as a percentage. Returns NaN if all true values are zero.
    """
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
    """
    Performs walk-forward cross-validation for a conditional time series model.

    In each step, it trains the model on a window of past data, makes a conditional
    prediction for a future `horizon`, and evaluates the prediction against the
    actual data.

    Args:
        model_class: The class of the model to be cross-validated. The class
                     must have an __init__(df, **kwargs) method and a
                     make_prediction(horizon, columns, values) method.
        df: A pandas DataFrame with a datetime index and series as columns.
        horizon: The number of future time steps to predict.
        stride: The number of time steps to move the training window forward
                in each cross-validation fold.
        start_window: The size of the initial training window.
        target_columns: A list of column names to use for conditioning. In each fold,
                        the function will iterate through these columns, using the
                        actual future value of each as a condition.
        model_params: A dictionary of parameters to pass to the model's constructor.

    Returns:
        A dictionary where keys are the names of the predicted columns and
        values are lists of MAPE scores from each validation fold.
    """
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
    """
    Builds a matrix of forecast errors for a conditional model.

    The matrix shows the average prediction error (MAPE) for each variable (columns)
    when conditioned on the future value of another variable (rows).

    Args:
        model_class: The class of the model to be cross-validated.
        df: A pandas DataFrame with a datetime index and series as columns.
        horizon: The number of future time steps to predict.
        stride: The number of time steps to move the window forward in each fold.
        start_window: The size of the initial training window.
        model_params: A dictionary of parameters for the model's constructor.

    Returns:
        A pandas DataFrame representing the error matrix.
    """
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
