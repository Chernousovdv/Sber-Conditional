import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns


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


from matplotlib.colors import Normalize


def plot_error_matrix(
    error_matrix,
    cmap="RdYlGn_r",
    figsize=(10, 8),
    decimals=0,
    vmin=0,
    vmax=100,
    clip=True,
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
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_yticklabels(yticklabels)

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

    plt.title(f"Error Matrix (MAPE %) - Colormap anchored to [{vmin}, {vmax}]")
    plt.tight_layout()
    plt.show()
