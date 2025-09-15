from collections import defaultdict
from typing import Any, Dict, List, Type

import matplotlib.dates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns

from utils.interface import _calculate_mape



def visualize_ts(
    df,
    columns,
    end=-1,
    test_split_timestamp=None,
    figsize=(12, 8),
    color="black",
    title="Commodity Price Time Series",
):
    """
    Visualize multiple time series in vertically stacked subplots.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index containing the time series data
    columns : list
        List of column names to plot
    end : int, optional
        Index position to end the plot (default: -1, meaning all data)
    test_split_timestamp : datetime or str, optional
        Timestamp to mark the beginning of test period (visualized with different color)
    figsize : tuple, optional
        Figure size (width, height) in inches
    color : color of line
    title : str, optional
        Main title for the figure
    """
    if end == -1:
        plot_df = df.copy()
    else:
        plot_df = df.iloc[:end].copy()

    n_plots = len(columns)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.plot(plot_df.index, plot_df[col], color=color, linewidth=1.5, label=col)

        if test_split_timestamp is not None:
            if isinstance(test_split_timestamp, str):
                test_split_timestamp = pd.to_datetime(test_split_timestamp)

            train_data = plot_df[plot_df.index < test_split_timestamp]
            test_data = plot_df[plot_df.index >= test_split_timestamp]

            ax.plot(
                train_data.index,
                train_data[col],
                color=color,
                linewidth=1.5,
                label=f"{col} (Train)",
            )

            if not test_data.empty:
                ax.plot(
                    test_data.index,
                    test_data[col],
                    color=color,
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.8,
                    label=f"{col} (Test)",
                )

            ax.axvline(
                x=test_split_timestamp,
                color="red",
                linestyle=":",
                alpha=0.7,
                linewidth=1,
            )

        ax.set_ylabel("Price", fontsize=10)
        ax.set_title(f"{col}", fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())

        if i == n_plots - 1:
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("Date", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.95)
    plt.subplots_adjust(top=0.9)

    return fig, axes




def plot_error_matrix(
    error_matrix,
    cmap="RdYlGn_r",
    figsize=(10, 8),
    decimals=0,
    vmin=0,
    vmax=100,
    clip=True,
    title=f"Error Matrix (MAPE %)",
):

    fig, ax = plt.subplots(figsize=figsize)
    data = error_matrix.to_numpy(dtype=float)
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")

    norm = Normalize(vmin=vmin, vmax=vmax, clip=clip)  # normalize cmap

    im = ax.imshow(data, cmap=cmap_obj, aspect="auto", norm=norm)

    xticklabels = [str(c)[:20] for c in error_matrix.columns]
    yticklabels = [str(r)[:20] for r in error_matrix.index]

    ax.set_xticks(np.arange(len(error_matrix.columns)))
    ax.set_yticks(np.arange(len(error_matrix.index)))
    # ax.set_xticklabels(xticklabels, rotation=90)
    # ax.set_yticklabels(yticklabels)

    # for i in range(data.shape[0]):

    #     for j in range(data.shape[1]):
    #         val = data[i, j]
    #         if not np.isnan(val):
    #             ax.text(
    #                 j,
    #                 i,
    #                 f"{val:.{decimals}f}%",
    #                 ha="center",
    #                 va="center",
    #                 color="black",
    #                 fontsize=8,
    #             )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("MAPE (%)")
    ax.set_xticks(np.arange(-0.5, len(error_matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(error_matrix.index), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_model_pred(
    model_class: Type,
    df: pd.DataFrame,
    model_params: Dict[str, Any],
    conditioning_col: str,
    horizon: int,
    cutoff_date: str,
):
    """
    Trains a model, makes a conditional prediction, and plots it against actuals.

    Args:
        model_class: The class of the model to be used.
        df: The full pandas DataFrame with a datetime index.
        model_params: A dictionary of parameters for the model's constructor.
        conditioning_col: The column name to use for the future condition.
        horizon: The number of future time steps to predict.
        cutoff_date: A string representing the date for the train/test split
                     (e.g., '2025-01-31').
    """
    try:
        cutoff_idx = df.index.get_loc(cutoff_date)
    except KeyError:
        print("Error")
        return

    df_train = df.iloc[: cutoff_idx + 1]

    if cutoff_idx + horizon >= len(df):
        print("Error")

        return

    model = model_class(df_train, **model_params)

    conditioning_val_idx = cutoff_idx + horizon
    conditioning_val = df.loc[df.index[conditioning_val_idx], conditioning_col]
    conditioning_date = df.index[conditioning_val_idx]

    predicted_cols, predicted_series = model.make_prediction(
        horizon=horizon, columns=[conditioning_col], values=[conditioning_val]
    )

    for i, pred_col in enumerate(predicted_cols):
        plt.figure(figsize=(14, 7))

        plt.plot(
            df.index,
            df[pred_col],
            label=f"Actual {pred_col}",
            color="royalblue",
            linewidth=2,
        )

        forecast_dates = df.index[cutoff_idx + 1 : cutoff_idx + 1 + horizon]

        plt.plot(
            forecast_dates,
            predicted_series[i],
            label=f"Predicted {pred_col}",
            color="darkorange",
            linestyle="--",
            marker="o",
        )

        plt.axvline(
            x=df.index[cutoff_idx],
            color="crimson",
            linestyle="-.",
            linewidth=2,
            label="Forecast Start",
        )

        actual_vals_horizon = df.loc[forecast_dates, pred_col].values
        predicted_vals_horizon = predicted_series[i]
        mape = _calculate_mape(actual_vals_horizon, predicted_vals_horizon)

        title = (
            f"Conditional Forecast for {pred_col}"
        )
        plt.title(title, fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()


def calc_average_error(matrix):
    print(f"{np.mean(matrix):.2f}%")


def plot_avg_errors(row):
    sorted = np.sort(row)
    plt.plot(np.arange(len(sorted)), sorted)


def plot_delta_matrix_anchored(
    error_matrix1,
    error_matrix2,
    cmap="RdBu_r",
    figsize=(12, 8),
    decimals=1,
    vmin=-50,
    vmax=50,
    center=0,
    clip=True,
    title="Delta Error Matrix",
):
    """
    Plot the difference between two error matrices (Matrix1 - Matrix2).
    Colormap anchored to [-50, 50] with center at 0.
    """

    delta_matrix = error_matrix1 - error_matrix2

    fig, ax = plt.subplots(figsize=figsize)
    data = delta_matrix.to_numpy(dtype=float)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")
    im = ax.imshow(data, cmap=cmap_obj, aspect="auto", norm=norm)

    xticklabels = [str(c)[:20] for c in delta_matrix.columns]
    yticklabels = [str(r)[:20] for r in delta_matrix.index]

    ax.set_xticks(np.arange(len(delta_matrix.columns)))
    ax.set_yticks(np.arange(len(delta_matrix.index)))
    # ax.set_xticklabels(xticklabels, rotation=90)
    # ax.set_yticklabels(yticklabels)

    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         val = data[i, j]
    #         if not np.isnan(val):
    #             if abs(val) > 25:
    #                 text_color = "white"
    #             else:
    #                 text_color = "black"

    #             ax.text(
    #                 j,
    #                 i,
    #                 f"{val:+.{decimals}f}",
    #                 ha="center",
    #                 va="center",
    #                 color=text_color,
    #                 fontsize=8,
    #                 fontweight="bold",
    #             )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ Error")

    ax.set_xticks(np.arange(-0.5, len(delta_matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(delta_matrix.index), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.title(f"{title}")

    plt.tight_layout()
    plt.show()

    return
