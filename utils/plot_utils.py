import pandas as pd
from typing import Dict, List, Optional, Tuple


def plot_horizons_same_lags(
    preds_df: pd.DataFrame,
    series_full: pd.DataFrame,
    model: str,
    lags: int,
    target_level: str,
    predicted_by: str = "",
    horizons: Optional[List[int]] = None,
    fold_selector: str = 'last',   # 'last' or int fold index or 'all' to plot all folds per horizon
    show_actual_full: bool = True,
    figsize: Tuple[int,int] = (12,6)
):
    """
    Plot actual series and predicted series for multiple horizons (same lags) on one plot.
    - preds_df: DataFrame returned by evaluator (contains preds_series and actual_series)
    - model: name of model to plot (string key from regressors)
    - lags: lags value to filter
    - horizons: list of horizons to include (default = all horizons available in preds_df for that model/lags)
    - fold_selector:
        - 'last' -> use last (max) fold_index for each horizon
        - int -> use that fold_index for ALL horizons (if not present for a horizon it is skipped)
        - 'all' -> plot all folds (may clutter)
    """
    # filter preds_df
    df = preds_df.copy()
    df_filtered = df[(df['model'] == model) & (df['lags'] == lags)]
    if df_filtered.empty:
        raise ValueError("No prediction rows found for given model/lags combination.")

    available_horizons = sorted(df_filtered['horizon_months'].unique().tolist())
    if horizons is None:
        horizons = available_horizons
    else:
        horizons = [h for h in horizons if h in available_horizons]
        if not horizons:
            raise ValueError("None of the requested horizons are available for this model/lags.")

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # optionally plot the full actual series
    if show_actual_full:
        series_full = _to_monthly_index(series_full)
        if target_level not in series_full.columns:
            raise ValueError(f"target_level '{target_level}' not found in series_full columns.")
        ax.plot(series_full.index, series_full[target_level], linewidth=2, label='Actual (full)')

    legend_handles = []

    for horizon in horizons:
        subset = df_filtered[df_filtered['horizon_months'] == horizon]
        if subset.empty:
            continue

        if fold_selector == 'last':
            fold_idx = int(subset['fold_index'].max())
            row = subset[subset['fold_index'] == fold_idx].iloc[0]
            preds_series = row['preds_series']
            cutoff = row['cutoff']
            label = f"Pred (h={horizon}m, fold={fold_idx})"
            ax.plot(preds_series.index, preds_series.values, linestyle='--', marker='o', label=label)
        elif fold_selector == 'all':
            # plot all folds for this horizon (semi-transparent)
            for _, row in subset.iterrows():
                preds_series = row['preds_series']
                fold_idx = row['fold_index']
                label = f"Pred (h={horizon}m, fold={fold_idx})"
                ax.plot(preds_series.index, preds_series.values, linestyle='--', alpha=0.5, label=label)
        else:
            # assume integer fold index
            fold_idx = int(fold_selector)
            subset_row = subset[subset['fold_index'] == fold_idx]
            if subset_row.empty:
                warnings.warn(f"No fold={fold_idx} for horizon={horizon}. Skipping.")
                continue
            row = subset_row.iloc[0]
            preds_series = row['preds_series']
            label = f"Pred (h={horizon}m, fold={fold_idx})"
            ax.plot(preds_series.index, preds_series.values, linestyle='--', marker='o', label=label)

    ax.set_title(f"Actual vs predicted {target_level} (model={model}, lags={lags}, predict_base={predicted_by})")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_level)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    os.makedirs(f"results/{target_level}_{predicted_by}", exist_ok=True)
    plt.savefig(f"results/{target_level}_{predicted_by}/{target_level}_{predicted_by}_{lags}.png")
    