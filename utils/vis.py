from collections import defaultdict
from typing import Any, Dict, List, Type

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

        title = f"Conditional Forecast for {pred_col}"
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


def plot_tsne_similarity(
    df: pd.DataFrame,
    metric,
    color_dict: dict = None,
    perplexity: int = 5,
    random_state: int = 42,
    title: str = "t-SNE Visualization of Time Series Similarity",
    annotate=True,
    s=200,
    legend=True,
    figsize=(8, 5),
    n_components=2,
    scaler=MinMaxScaler,
):
    """
    Normalizes time series, applies t-SNE, and plots the 2D result.

    This function helps visualize which time series from a DataFrame are
    similar to each other based on a given distance metric.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex where each column
                           is a non-negative time series (e.g., a commodity price).
        metric (callable or str): The metric for calculating distance between
                                  time series. Can be a string like 'euclidean'
                                  or a callable function.
        color_dict (dict): Dictionary mapping column names to colors.
                          If None, all points will be the same color.
        perplexity (int): A t-SNE parameter related to the number of nearest
                          neighbors considered for each point. Should be less than
                          the number of time series.
        random_state (int): Seed for the random number generator to ensure
                            reproducible results.
        title (str): The title for the plot.
    """
    # 1. Data Preparation
    # The TSNE algorithm expects samples as rows, so we transpose the DataFrame.
    # Now, each row is a time series, and each column is a time step.
    series_data = df.T
    series_labels = series_data.index

    if perplexity >= len(series_labels):
        print(
            f"Perplexity ({perplexity}) must be less than the number of series ({len(series_labels)}). "
            f"Adjusting to {len(series_labels) - 1}."
        )
        perplexity = len(series_labels) - 1

    # 2. Normalization
    # Scale each time series (row) independently to a [0, 1] range.
    # This focuses the comparison on the shape of the series, not the magnitude.
    scaler = scaler()
    normalized_data = np.vstack(
        [
            scaler.fit_transform(series_data.iloc[i, :].values.reshape(-1, 1)).ravel()
            for i in range(series_data.shape[0])
        ]
    )

    # 3. t-SNE Execution
    # Initialize and run the t-SNE algorithm to reduce dimensionality to 2D.
    tsne = TSNE(
        n_components=n_components,
        metric=metric,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    tsne_results = tsne.fit_transform(normalized_data)

    # 4. Plotting
    # Create a scatter plot of the 2D t-SNE results.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=figsize)

    # Handle coloring
    if color_dict is not None:
        # Create color list for each series
        colors = [color_dict.get(label, "gray") for label in series_labels]

        # Create scatter plot with individual colors
        scatter = plt.scatter(
            tsne_results[:, 0], tsne_results[:, 1], c=colors, alpha=0.7, s=s
        )

        # Create legend for the colors
        if legend:
            unique_colors = set(colors)
            legend_elements = []
            for color in unique_colors:
                # Find one label that has this color
                label_example = next(
                    label for label, c in zip(series_labels, colors) if c == color
                )
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=8,
                        label=f"{color} series",
                    )
                )

            plt.legend(handles=legend_elements, loc="best")
    else:
        # Original behavior - all points same color
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7, s=s)

    # Add labels for each point to identify the original time series.
    if annotate:
        for i, label in enumerate(series_labels):
            plt.annotate(
                label,
                (tsne_results[i, 0], tsne_results[i, 1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()


import plotly.express as px
import plotly.graph_objects as go


def plot_tsne_similarity_plotly(
    df: pd.DataFrame,
    metric,
    color_dict: dict = None,
    group_dict: dict = None,  # New argument for group mapping
    perplexity: int = 5,
    random_state: int = 42,
    title: str = "t-SNE Visualization of Time Series Similarity",
    annotate=True,
    s=8,
    legend=True,
    figsize=(800, 500),
    scaler=MinMaxScaler,
):
    """
    Interactive t-SNE visualization using Plotly.

    This function provides the same functionality as the matplotlib version
    but with interactive Plotly visualization.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex where each column
                           is a non-negative time series (e.g., a commodity price).
        metric (callable or str): The metric for calculating distance between
                                  time series. Can be a string like 'euclidean'
                                  or a callable function.
        color_dict (dict): Dictionary mapping column names to colors.
        group_dict (dict): Dictionary mapping column names to group names.
                          Used for hover information and legend.
        perplexity (int): A t-SNE parameter related to the number of nearest
                          neighbors considered for each point. Should be less than
                          the number of time series.
        random_state (int): Seed for the random number generator to ensure
                            reproducible results.
        title (str): The title for the plot.
        annotate (bool): For API consistency with matplotlib version. In Plotly,
                        annotations are always shown on hover.
        s (int): Marker size.
        legend (bool): Whether to show the legend.
        figsize (tuple): Figure size as (width, height) in pixels.
        scaler: Scaler to use for normalization.
    """
    # 1. Data Preparation
    series_data = df.T
    series_labels = series_data.index.tolist()

    if perplexity >= len(series_labels):
        print(f"Adjusting perplexity from {perplexity} to {len(series_labels) - 1}")
        perplexity = len(series_labels) - 1

    # 2. Normalization
    scaler = scaler()
    normalized_data = np.vstack(
        [
            scaler.fit_transform(series_data.iloc[i, :].values.reshape(-1, 1)).ravel()
            for i in range(series_data.shape[0])
        ]
    )

    # 3. t-SNE Execution
    tsne = TSNE(
        n_components=2,
        metric=metric,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    tsne_results = tsne.fit_transform(normalized_data)

    # 4. Prepare data for Plotly
    results_df = pd.DataFrame(
        {"x": tsne_results[:, 0], "y": tsne_results[:, 1], "series": series_labels}
    )

    # Add group information if group_dict is provided
    if group_dict is not None:
        results_df["group"] = [
            group_dict.get(label, "Unknown") for label in series_labels
        ]
    else:
        results_df["group"] = "No group"

    # 5. Create interactive plot
    if color_dict is not None:
        # Add colors and create color groups for proper legend
        results_df["color"] = [color_dict.get(label, "gray") for label in series_labels]
        results_df["color_group"] = results_df["color"]

        # Create custom hover text with series name and group
        results_df["hover_text"] = results_df.apply(
            lambda row: f"Series: {row['series']}<br>Group: {row['group']}", axis=1
        )

        fig = px.scatter(
            results_df,
            x="x",
            y="y",
            color="color_group",
            hover_name="hover_text",  # Use custom hover text
            title=title,
            size_max=s,
            color_discrete_map={color: color for color in results_df["color"].unique()},
        )

        # Update marker colors to use the actual colors from color_dict
        fig.update_traces(
            marker=dict(size=s, opacity=0.7, line=dict(width=1, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )

        # Customize legend to show group names instead of colors
        if legend:
            fig.update_layout(showlegend=True)
            # Rename legend entries to show group names
            for trace in fig.data:
                color = trace.name
                # Find the group name for this color
                matching_rows = results_df[results_df["color"] == color]
                if not matching_rows.empty:
                    group_name = matching_rows["group"].iloc[0]
                    trace.name = f"{group_name}"
                else:
                    trace.name = f"{color} series"
        else:
            fig.update_layout(showlegend=False)

    else:
        # All points same color - still include group info in hover
        if group_dict is not None:
            results_df["hover_text"] = results_df.apply(
                lambda row: f"Series: {row['series']}<br>Group: {row['group']}", axis=1
            )
        else:
            results_df["hover_text"] = results_df["series"]

        fig = px.scatter(
            results_df,
            x="x",
            y="y",
            hover_name="hover_text",
            title=title,
            size_max=s,
        )

        fig.update_traces(
            marker=dict(
                size=s,
                opacity=0.7,
                color="blue",  # Default color
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            selector=dict(mode="markers"),
        )
        fig.update_layout(showlegend=False)

    # 6. Update layout
    fig.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        width=figsize[0],
        height=figsize[1],
        title_x=0.5,  # Center the title
    )

    fig.show()
