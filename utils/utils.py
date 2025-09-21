import math
import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .constants import CATEGORY_MAP
from pathlib import Path
import seaborn as sns
from typing import Optional
from copy import deepcopy


def compute_mode_and_stats(df_subset: pd.DataFrame):
    """
    For a dataframe filtered to a particular (model_family, horizon, lags, step==horizon)
    compute:
      - top_feature_per_fold: Series indexed by fold_index containing the feature_name that had max importance_norm in that fold
      - mode_feature: the mode of top_feature_per_fold (if multiple modes, return first)
      - mode_count: how many folds selected that feature as top
      - stats_df: DataFrame indexed by feature_name with columns ['mean_imp', 'std_imp', 'n_folds', 'top_count']
    """
    if df_subset.empty:
        return None, None, None, pd.DataFrame(columns=['mean_imp','std_imp','n_folds','top_count'])

    # For each fold, find the feature with max importance_norm
    def top_feature_for_fold(g):
        idx = g['importance_norm'].idxmax()
        return g.loc[idx, 'feature_name']

    top_per_fold = df_subset.groupby('fold_index').apply(top_feature_for_fold)
    if len(top_per_fold) == 0:
        return None, None, None, pd.DataFrame(columns=['mean_imp','std_imp','n_folds','top_count'])

    # mode of top features
    mode_series = top_per_fold.mode()
    mode_feature = None
    if isinstance(mode_series, pd.Series) and len(mode_series) > 0:
        mode_feature = mode_series.iloc[0]
    else:
        # fallback: most common
        mode_feature = Counter(top_per_fold).most_common(1)[0][0]

    mode_count = (top_per_fold == mode_feature).sum()

    # per-feature mean/std/count of importance_norm across folds
    grouped = df_subset.groupby('feature_name').importance_norm.agg(['mean','std','count']).rename(
        columns={'mean':'mean_imp','std':'std_imp','count':'n_folds'}
    ).sort_values('mean_imp', ascending=False)

    # how many times each feature was top in folds
    top_counts = top_per_fold.value_counts().rename('top_count')
    grouped = grouped.join(top_counts, how='left').fillna({'top_count': 0})
    grouped['top_count'] = grouped['top_count'].astype(int)

    return top_per_fold, mode_feature, mode_count, grouped


def _read_feat_importance(predictor_name: str, target: str):
    try:
        return pd.read_csv(f"results/{target}_{predictor_name}_tables/feature_imp_df_results.csv")
    except Exception:
        if len(predictor_name) + len(target) >= 120:
            #print(f"len exceed fpr: predictor:{normed_predict_col_name}\tand target: {normed_target_col_name}")
            predictor_name_cp = predictor_name[:50]
            target_cp = target[:50]
            #print(f"cut: predictor:{normed_predict_col_name}\tand target: {normed_target_col_name}")
        try:
            return pd.read_csv(f"results/{target_cp}_{predictor_name_cp}_tables/feature_imp_df_results.csv")
        except Exception:
            if len(predictor_name) + len(target) >= 90:
                #print(f"len exceed fpr: predictor:{normed_predict_col_name}\tand target: {normed_target_col_name}")
                predictor_name_1 = predictor_name[:50]
                target_1 = target[:50]
                #print(f"cut: predictor:{normed_predict_col_name}\tand target: {normed_target_col_name}")
            try:
                return pd.read_csv(f"results/{target_1}_{predictor_name_1}_tables/feature_imp_df_results.csv")
            except Exception:
                raise ValueError(f"There is no feature importance for {target}_{predictor_name}_tables")


def plot_mode_top_features_grid(
    target_series: str,
    predictor_series: str,
    models: list | None = None,
    horizons: list | None = None,
    lags_list: list | None = None,
    top_k: int = 10,
    figsize_per_ax: tuple = (5, 4),
    save_dir: str | None = None,
    show: bool = True
):
    """
    Create a grid of subplots (rows = horizons, cols = lags) showing top_k features for each (horizon,lags)
    using the mode of top-feature across folds (for step == horizon, i.e. last step).

    Parameters
    ----------
    feature_imp_df : pd.DataFrame
        Dataframe with importances (must include columns used in compute_mode_and_stats).
    target_series : str
        The target_series value (filter).
    predictor_series : str
        The predictor_series value (filter), may be "" if not used.
    models : list or None
        List of model_family values to produce figures for. If None -> use all present.
    horizons : list or None
        horizons to include (if None use unique sorted feature_imp_df.horizon)
    lags_list : list or None
        lags values to include (if None use unique sorted feature_imp_df.lags)
    top_k : int
        How many features to show per subplot.
    figsize_per_ax : (w,h)
        Size per axis; final fig size = (w * ncols, h * nrows)
    save_dir : str or None
        If provided, save each model's figure as PNG into this directory.
    show : bool
        Whether to call plt.show() at the end of each figure.
    """
    feature_imp_df = _read_feat_importance(predictor_series, target_series)
    required_cols = {'target_series','predictor_series','model_family','horizon','step','lags','fold_index','feature_name','importance_norm'}
    if not required_cols.issubset(set(feature_imp_df.columns)):
        missing = required_cols.difference(set(feature_imp_df.columns))
        raise ValueError(f"feature_imp_df missing required columns: {missing}")

    df = feature_imp_df.copy()
    # Filter by target & predictor
    df = df[df['target_series'] == target_series]
    if predictor_series != "":
        df = df[df['predictor_series'] == predictor_series]

    if horizons is None:
        horizons = sorted(df['horizon'].dropna().unique().tolist())
    if lags_list is None:
        lags_list = sorted(df['lags'].dropna().unique().tolist())

    if models is None:
        models = sorted(df['model_family'].dropna().unique().tolist())

    if len(horizons) == 0 or len(lags_list) == 0 or len(models) == 0:
        raise ValueError("No horizons/lags/models to plot (after filtering).")

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    figs = {}
    for model in models:
        fname = os.path.join(save_dir, f"feature_mode_top_{target_series}_{predictor_series}_{model}.png")
        if Path(fname).is_file():
            print(f"Skipping {fname}\texperiment was made.")
            return
        df_model = df[df['model_family'] == model]
        nrows = len(horizons)
        ncols = len(lags_list)
        fig_w = figsize_per_ax[0] * ncols
        fig_h = figsize_per_ax[1] * nrows
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
        fig.patch.set_facecolor('lightgray')
        plt.subplots_adjust(hspace=0.5, wspace=0.35)

        for i_h, horizon in enumerate(horizons):
            for j_l, lags in enumerate(lags_list):
                ax = axes[i_h][j_l]
                sub = df_model[
                    (df_model['horizon'] == horizon) &
                    (df_model['lags'] == lags) &
                    (df_model['step'] == horizon)   # last step
                ]

                if sub.empty:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                    ax.set_title(f"h={horizon} lags={lags}\n(empty)")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                top_per_fold, mode_feature, mode_count, stats_df = compute_mode_and_stats(sub)

                if stats_df.empty:
                    ax.text(0.5, 0.5, "No features", ha='center', va='center')
                    ax.set_title(f"h={horizon} lags={lags}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # choose top_k features by mean importance
                top_feats = stats_df.sort_values('mean_imp', ascending=False).head(top_k).index.tolist()
                plot_df = stats_df.loc[top_feats].copy()
                # order them
                plot_df = plot_df.sort_values('mean_imp', ascending=True)  # for horizontal bar

                y = np.arange(len(plot_df))
                means = plot_df['mean_imp'].values
                stds = plot_df['std_imp'].fillna(0).values
                labels = plot_df.index.tolist()

                # horizontal bar chart
                ax.barh(y,
                        means,
                        xerr=stds,
                        align='center',
                        ecolor='maroon',
                        color='forestgreen',
                        capsize=3)
                ax.set_facecolor('lightgray')
                ax.set_yticks(y)
                ax.set_yticklabels(labels, fontsize=9)
                ax.invert_yaxis()  # largest on top
                ax.set_xlabel('mean normalized importance')
                ax.set_title(f"{target_series} predicted by {predictor_series}\nmodel={model} | h={horizon} | lags={lags}")

                ax.grid(axis='x', linestyle=':', alpha=0.6)
                # ax.legend()

        fig.suptitle(f"Top-{top_k} features by mean importance (model={model})\nTarget={target_series} Predictor={predictor_series}", fontsize=14)
        figs[model] = fig

        if save_dir:
            fname = os.path.join(save_dir, f"feature_mode_top_{target_series}_{predictor_series}_{model}.png")
            try:
                fig.savefig(fname, bbox_inches='tight', dpi=150)
                print(f"Saved figure: {fname}")
            except Exception as e:
                print(f"Could not save fig {fname}: {e}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return figs


def create_category(df_in: pd.DataFrame, 
                   predictor_col_name: str,
                   target_col_name: str) -> pd.DataFrame:
    df = df_in.copy()
    df[predictor_col_name] = df[predictor_col_name].map(CATEGORY_MAP)
    df[target_col_name] = df[target_col_name].map(CATEGORY_MAP)

    return df


### Plot graphics aggregated by categories

# fallback names list (all product names)
ALL_PRODUCTS = sorted(list(CATEGORY_MAP.keys()))

# helper: find owner series name from a feature_name (heuristic)
def _get_feature_owner(feature_name: str, product_names: list[str]) -> Optional[str]:
    """
    If feature_name starts with '<product>_lag_' or contains that product name as prefix,
    return product name. Otherwise return None.
    """
    if not isinstance(feature_name, str):
        return None
    # exact prefix match first (most common from _build_feature_names)
    for p in product_names:
        if feature_name.startswith(f"{p}_lag_") or feature_name.startswith(f"{p}__lag_"):
            return p
    # some feature names might be like "<product>_lag_1" without trailing underscore differences
    for p in product_names:
        if feature_name.startswith(p):
            # ensure next char is non-letter (underscore, space, etc.) or length equal
            if len(feature_name) == len(p) or not feature_name[len(p)].isalpha():
                return p
    return None

def _map_owner_to_category(owner: Optional[str]) -> str:
    if owner is None:
        return "USDRUB"
    return CATEGORY_MAP.get(owner, "USDRUB")

# main aggregator + plotting function
def aggregate_feature_importances_by_category(
    products: list[str] = ALL_PRODUCTS,
    models: list[str] | None = None,
    horizons: list[int] | None = None,
    lags_list: list[int] | None = None,
    only_last_step: bool = True,
    drop_missing: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Read all feature importance CSVs for pairs (predictor, target) from CATEGORY_MAP keys,
    aggregate per fold/horizon/lags the importance mass per owner-category and return a
    long DataFrame with columns:
      ['predictor_series','predictor_category','target_series','target_category',
       'model_family','horizon','lags','fold_index','owner_category','importance_sum']
    """
    records = []
    missing = []
    for target in products:
        normed_target_col_name = target.replace("/", " ")
        for predictor in products:
            normed_predict_col_name = predictor.replace("/", " ")
                
            if normed_predict_col_name == normed_target_col_name:
                continue
            try:
                # read using provided helper
                df = _read_feat_importance(normed_predict_col_name, normed_target_col_name)
  
            except Exception as e:
                missing.append((normed_predict_col_name, normed_target_col_name))
                if verbose:
                    print(f"[WARN] missing file for {normed_predict_col_name} -> {normed_target_col_name}: {e}")
                continue

            # basic validation
            if 'importance_norm' not in df.columns or 'feature_name' not in df.columns:
                if verbose:
                    print(f"[WARN] file for {normed_predict_col_name}->{normed_target_col_name} missing required cols; skipping.")
                continue

            # optionally filter to last step rows: step == horizon
            if only_last_step:
                df = df[df['step'] == df['horizon']]

            # filter by model_family/horizon/lags if provided
            if models is not None:
                df = df[df['model_family'].isin(models)]
            if horizons is not None:
                df = df[df['horizon'].isin(horizons)]
            if lags_list is not None:
                df = df[df['lags'].isin(lags_list)]

            if df.empty:
                continue

            # compute owner category for each feature row (fold-level rows)
            # We'll treat EXOG_ prefix as Macros
            def owner_cat_row(row):
                fname = row['feature_name']
                if isinstance(fname, str) and fname.startswith("EXOG_"):
                    return "Macros"
                owner = _get_feature_owner(fname, products)
                return _map_owner_to_category(owner)

            df = df.copy()
            df['owner_category'] = df.apply(owner_cat_row, axis=1)

            # Now group: per fold/model/horizon/lags -> sum importance_norm per owner_category
            group_cols = ['predictor_series','target_series','model_family','horizon','lags','fold_index','owner_category']
            # ensure predictor/target columns exist, else set from args
            if 'predictor_series' not in df.columns:
                df['predictor_series'] = normed_predict_col_name
            if 'target_series' not in df.columns:
                df['target_series'] = normed_target_col_name

            # sum importance_norm within each row-group (fold-level)
            agg = (df
                   .groupby(group_cols, dropna=False, as_index=False)
                   .importance_norm
                   .sum()
                   .rename(columns={'importance_norm':'importance_sum'}))

            # add category labels for predictor & target
            agg['predictor_category'] = agg['predictor_series'].map(CATEGORY_MAP).fillna('Other')
            agg['target_category'] = agg['target_series'].map(CATEGORY_MAP).fillna('Other')

            records.append(agg)

    if not records:
        return pd.DataFrame()  # nothing read

    full = pd.concat(records, ignore_index=True)

    # Optionally drop rows with zero importance_mass
    full = full[full['importance_sum'] > 0.0]

    return full

def plot_category_heatmap_and_stacked(
    aggregated_df: pd.DataFrame,
    horizon: int | None = None,
    lag: int | None = None,
    owner_category_focus: str | None = None,
    agg_method: str = 'mean',   # 'mean' or 'median'
    normalize_rows: bool = False,
    save_dir: Optional[str] = None,
    show: bool = True,
    figsize=(12,8)
):
    """
    Produce:
      1) Heatmap: rows = predictor_category, cols = target_category,
         cell = aggregated importance of owner_category == predictor_category when predicting target_category.
         (i.e. how much a predictor-category's own features contribute to predicting targets).
      2) Stacked bar: for each target_category show contributions from owner categories (mean importance)
    `aggregated_df` columns must include:
      ['predictor_category','target_category','owner_category','importance_sum']
    """

    if aggregated_df.empty:
        raise ValueError("aggregated_df is empty")

    df = aggregated_df.copy()

    # Aggregate across folds/horizons/lags/models using chosen agg_method
    agg_funcs = {'mean': 'mean', 'median': 'median'}
    if agg_method not in agg_funcs:
        raise ValueError("agg_method must be 'mean' or 'median'")

    agg_by_cat = (df
                  .groupby(['predictor_category','target_category','owner_category'], as_index=False)
                  .importance_sum
                  .agg(agg_funcs[agg_method])
                  .rename(columns={'importance_sum': 'importance_agg'}))

    # Heatmap value: for each predictor_category P and target_category T,
    # take importance_agg where owner_category == P (i.e., predictor-category's own features).
    heat = (agg_by_cat[agg_by_cat['owner_category'] == agg_by_cat['predictor_category']]
            .pivot_table(index='predictor_category', columns='target_category', values='importance_agg', fill_value=0.0))

    # ensure all categories present in axis
    cats = sorted(set(agg_by_cat['predictor_category']).union(set(agg_by_cat['target_category'])))
    heat = heat.reindex(index=cats, columns=cats, fill_value=0.0)

    # Optional row-normalize for comparability
    heat_plot = heat.copy()
    if normalize_rows:
        row_sums = heat_plot.sum(axis=1).replace(0, np.nan)
        heat_plot = heat_plot.div(row_sums, axis=0).fillna(0.0)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(heat_plot, annot=True, fmt=".3f", cmap='viridis')
    if horizon is not None:
        heatmap_title = f"Predictor-category -> Target-category (mean importance of predictor's own features on targets) on horizon {horizon} with lag {lag}"
    else:
        heatmap_title = "Predictor-category -> Target-category (mean importance of predictor's own features on targets)"

    plt.title(heatmap_title)
    plt.xlabel("Target category")
    plt.ylabel("Predictor category")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, "category_pair_heatmap.png")
        plt.savefig(fname, bbox_inches='tight', dpi=150)
        print(f"Saved heatmap to {fname}")
    if show:
        plt.show()
    else:
        plt.close()

    # Stacked bar: for each target_category show owner_category contributions
    stacked = (agg_by_cat
               .groupby(['target_category','owner_category'], as_index=False)
               .importance_agg
               .sum()
               .pivot(index='target_category', columns='owner_category', values='importance_agg')
               .fillna(0.0))

    # sort columns by total contribution
    col_order = stacked.sum(axis=0).sort_values(ascending=False).index.tolist()
    stacked = stacked[col_order]

    ax = stacked.plot(kind='bar', stacked=True, figsize=(max(10, len(stacked)*0.6), 6))
    ax.set_ylabel("Aggregated importance (sum of fold-level means)")
    if horizon is not None:
        bars_title = f"Stacked contributions by owner-category for each target-category on horizon {horizon} with lag {lag}"
    else:
        bars_title = "Stacked contributions by owner-category for each target-category"
    ax.set_title(bars_title)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Owner category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_dir:
        fname2 = os.path.join(save_dir, "category_stacked_by_target.png")
        plt.savefig(fname2, bbox_inches='tight', dpi=150)
        print(f"Saved stacked bar to {fname2}")
    if show:
        plt.show()
    else:
        plt.close()