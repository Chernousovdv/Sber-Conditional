import os
import pandas as pd
import matplotlib.pyplot as plt
from .constants import (
    CATEGORY_MAP,
    BEST_PREDICTORS_FOR_INDEX
)
from typing import Optional, Tuple, Any, Union
from .cluster_forecast import ClusterForecaster
from datetime import datetime, timedelta
import plotly.graph_objects as go


def plot_graphics_for_each_ts_plt(
    df_true: pd.DataFrame,
    df_preds: pd.DataFrame,
    df_preds_naive: pd.DataFrame,
    goods_name: str,
    mape_val: float,
    mape_val_naive: float,
    predictor_name: str,
    save_dir: str,
    forecasting_horizon: int = 12,
    is_backtest: bool = True,
    up_to_date: str = ""
):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_facecolor('lightgray')
    ax.set_facecolor('lightgray')
    
    # Plot original series
    ax.plot(df_true.index,
            df_true,
            label=f"original {goods_name}",
            color="black",
            linestyle='-',  # Changed to solid line for connection
            marker='o')
    
    # Combine the last point of original series with predictions for continuous line
    # Get the last date and value from original series
    last_true_date = df_true.index[-1]
    last_true_value = df_true.iloc[-1]

    if is_backtest:
        combined_dates = list(df_preds.index) + [last_true_date]
        combined_values = list(df_preds.values) + [last_true_value]  # Assuming single column
    else:
        # Create combined series for continuous plotting
        combined_dates = [last_true_date] + list(df_preds.index)
        combined_values = [last_true_value] + list(df_preds.values)  # Assuming single column
    
    # Plot the prediction line connecting original to predictions
    ax.plot(combined_dates,
            combined_values,
            label=f"preds {goods_name}; with mape: {mape_val*100:.2f}",
            color="maroon",
            linestyle='-',  # Solid line for connection
            marker='o')
    
    if df_preds_naive is not None and len(df_preds_naive) > 0:
        # Similarly for naive predictions
        naive_combined_values = [last_true_value] + list(df_preds_naive.iloc[:, 0])
        ax.plot(combined_dates,
                naive_combined_values,
                label=f"preds naive {goods_name}; with mape: {mape_val_naive*100:.2f}",
                color="darkblue",
                linestyle='-',  # Solid line for connection
                marker='o')
    
    ax.set_xlabel("Date with month frequent")
    ax.set_ylabel("Price")
    if is_backtest:
        ax.set_title(f"TS for {goods_name} predicted by {predictor_name} with forecasting horizon of {forecasting_horizon} months")
    else:
        ax.set_title(f"TS for {goods_name} predicted by {predictor_name} up to date {up_to_date}")
    
    ax.legend()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{goods_name}_{predictor_name}_ts_preds.png")
    
    return fig, ax


def plot_graphics_for_each_ts(
    df_true: pd.DataFrame,
    df_preds: pd.DataFrame,
    df_preds_naive: pd.DataFrame,
    goods_name: str,
    mape_val: float,
    mape_val_naive: float,
    predictor_name: str,
    save_dir: str,
    forecasting_horizon: int = 12,
    is_backtest: bool = True,
    up_to_date: str = ""
):
    """
    Plot time series with predictions using Plotly
    """
    fig = go.Figure()
    
    # Get the last point from original series
    last_true_date = df_true.index[-1]
    last_true_value = df_true.iloc[-1]

    goods_name_short = goods_name if len(goods_name) < 40 else goods_name[:40] + "..."
    # Plot original series
    fig.add_trace(go.Scatter(
        x=df_true.index,
        y=df_true,
        mode='lines+markers',
        name=f"original {goods_name_short}",
        line=dict(color='black', width=2),
        marker=dict(size=6),
        hovertemplate='<b>Original</b><br>' +
                     'Date: %{x}<br>' +
                     'Price: %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Create combined series for predictions
    if is_backtest:
        combined_dates = list(df_preds.index) + [last_true_date]
        combined_values = list(df_preds.values)  + [last_true_value]  # Assuming single column
    else:
        combined_dates = [last_true_date] + list(df_preds.index)
        combined_values = [last_true_value] + list(df_preds.values)   # Assuming single column

    mape_val_short = round(mape_val*100, 2) if mape_val > 0 else None
    # Plot predictions
    fig.add_trace(go.Scatter(
        x=combined_dates,
        y=combined_values,
        mode='lines+markers',
        name=f"preds {goods_name_short}; MAPE: {mape_val_short}%",
        line=dict(color='maroon', width=2),
        marker=dict(size=6),
        hovertemplate='<b>Prediction</b><br>' +
                     'Date: %{x}<br>' +
                     'Price: %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Configure layout
    title_text = f"TS for {goods_name} predicted by {predictor_name} with forecasting horizon of {forecasting_horizon} months" if is_backtest else f"TS for {goods_name} predicted by {predictor_name} up to date {up_to_date}"
    
    fig.update_layout(
        title=dict(text=title_text),
        xaxis_title="Date with month frequent",
        yaxis_title="Price",
        plot_bgcolor='lightgray',
        paper_bgcolor='lightgray',
        hovermode='x unified',  # Unified hover mode for better comparison[citation:1]
        showlegend=True
    )
    
    # Save if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_graphics_for_each_ts_plt(    df_true,
                                        df_preds,
                                        df_preds_naive,
                                        goods_name,
                                        mape_val,
                                        mape_val_naive,
                                        predictor_name,
                                        save_dir,
                                        forecasting_horizon,
                                        is_backtest,
                                        up_to_date)
        # fig.write_html(f"{save_dir}/{goods_name}_{predictor_name}_ts_preds.html")
        # fig.write_image(f"{save_dir}/{goods_name}_{predictor_name}_ts_preds.png")
    # Display the figure
    fig.show()
    return fig



def plot_known_feature_value(knwow_series_vals: list[Union[int, float]],
                                    known_series_name: str,
                                    index_values: list[Any]):
    """
    Plot known feature values using Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=index_values,
        y=knwow_series_vals,
        mode='lines+markers',
        name=known_series_name,
        line=dict(color='maroon', width=2),
        marker=dict(size=6),
        hovertemplate=f'<b>{known_series_name}</b><br>' +
                     'Date: %{x}<br>' +
                     'Value: %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f"Future values for column: {known_series_name}"),
        xaxis_title="Date",
        yaxis_title=known_series_name,
        plot_bgcolor='lightgray',
        paper_bgcolor='lightgray',
        hovermode='closest'  # Default hover mode[citation:1]
    )
    
    # Display the figure
    fig.show()
    
    return fig


def get_month_beginnings(date_str):
    """
    Convert a date string into beginning of month strings.
    
    Args:
        date_str (str): Date in "YYYY-MM-DD" format
        
    Returns:
        list: List containing beginning of month string(s)
    """
    # Parse the input date
    input_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Get the beginning of the current month
    current_month_start = input_date.replace(day=1)
    
    # If input is already the first day of month, return just that date
    if input_date.day == 1:
        return [current_month_start.strftime("%Y-%m-%d")]
    
    # Otherwise, return current month start and next month start
    next_month_start = (current_month_start + timedelta(days=32)).replace(day=1)
    
    return [
        current_month_start.strftime("%Y-%m-%d"),
        next_month_start.strftime("%Y-%m-%d")
    ]
