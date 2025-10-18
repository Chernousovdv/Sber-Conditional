import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from utils.data_transform import take_diff


def cluster_from_distance_matrix(distance_matrix, labels, linkage_method="average"):
    """
    Performs hierarchical clustering on a pairwise distance matrix and plots the dendrogram.
    IMPORTANT MATRIX SHOULD BE SYMMETRIC, WONT WORK WITH ENGLE-GRANGER TEST MATRIX
    """

    condensed_dist_matrix = squareform(distance_matrix)
    linked = linkage(condensed_dist_matrix, method=linkage_method)

    plt.figure(figsize=(12, 8))
    dendrogram(
        linked,
        orientation="top",
        labels=labels,
        distance_sort="descending",
        show_leaf_counts=True,
    )

    plt.title(f"Dendrogram (Linkage: {linkage_method})", fontsize=16)
    plt.ylabel("Distance", fontsize=12)
    plt.xlabel("Series", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import warnings

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm


def take_diff(df, diff_order=0):
    if diff_order > 0:
        return df.diff(diff_order).dropna()
    return df


def _pairwise_johansen_distance(x: pd.Series, y: pd.Series, det_order=0, k_ar_diff=1):
    """
    Run Johansen on a 2-var system [x, y] and convert trace statistic -> distance in [0,100].
    Heuristic mapping: distance = 100 * (1 / (1 + trace_stat_r0)),
    so larger trace_stat => smaller distance (stronger cointegration -> "closer").
    """
    data = np.column_stack([x.values, y.values])
    res = coint_johansen(data, det_order, k_ar_diff)
    trace_stat = res.lr1[0]
    dist = 100.0 * (1.0 / (1.0 + float(trace_stat)))
    return float(np.clip(dist, 0.0, 100.0))


def build_coint_mat(df: pd.DataFrame, test="Engle-Granger", **kwargs):
    """
    Build a cointegration matrix for all pairs of series in the DataFrame.

    Supported tests:
      - "Engle-Granger" : pairwise statsmodels.tsa.stattools.coint -> p-value * 100
      - "Johansen"      : pairwise Johansen (2-var system) -> heuristic distance in [0,100]
    Returns a DataFrame (n_vars x n_vars) with values in range ~[0,100] (lower = "closer"/more evidence of cointegration).
    """
    n_vars = len(df.columns)
    coint_mat = np.full((n_vars, n_vars), np.nan)
    cols = list(df.columns)

    if test == "Engle-Granger":
        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if i == j:
                    coint_mat[i, j] = 0.0
                    continue
                try:
                    _, p_value, _ = coint(
                        df[col_i].dropna(), df[col_j].dropna(), **kwargs
                    )
                    coint_mat[i, j] = float(p_value) * 100.0
                except Exception:
                    coint_mat[i, j] = 100.0

        return pd.DataFrame(coint_mat, index=cols, columns=cols)

    elif test == "Johansen":
        # params det_order, k_ar_diff can be passed via kwargs
        det_order = kwargs.pop("det_order", 0)
        k_ar_diff = kwargs.pop("k_ar_diff", 1)

        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if i == j:
                    coint_mat[i, j] = 0.0
                    continue
                try:
                    dist = _pairwise_johansen_distance(
                        df[col_i].dropna(),
                        df[col_j].dropna(),
                        det_order=det_order,
                        k_ar_diff=k_ar_diff,
                    )
                    coint_mat[i, j] = float(dist)
                except Exception:
                    coint_mat[i, j] = 100.0

        return pd.DataFrame(coint_mat, index=cols, columns=cols)

    else:
        raise ValueError(f"TEST '{test}' NOT FOUND")


def get_linear_clusters(
    df, dist_method, linkage_method, threshold, visualize=False, diff_order=0
):
    df = take_diff(df, diff_order)
    # build distance matrix
    if dist_method == "correlation":  #     TODO other dist
        corr_matrix = df.corr()
        distance_matrix = 1 - corr_matrix  # corr is not dist

    elif dist_method == "Engle-Granger":
        coint_matrix = build_coint_mat(df, test="Engle-Granger")
        symmetric_pvalue_matrix = 0.5 * (coint_matrix + coint_matrix.T)
        distance_matrix = symmetric_pvalue_matrix  # TODO distance scaling
    else:
        raise ValueError("Not Implemented")

    # Hierarchical clustering
    condensed_dist_matrix = squareform(distance_matrix)
    linked = linkage(condensed_dist_matrix, method=linkage_method)

    # Assign clusters based on threshold
    labels = fcluster(linked, t=threshold, criterion="distance")

    # output mapping
    clusters = dict(zip(df.columns, labels))

    # optional visualization
    if visualize:  # TODO add horizontal line at threshold
        plt.figure(figsize=(12, 8))
        dendrogram(
            linked,
            orientation="top",
            labels=df.labels,
            distance_sort="descending",
            show_leaf_counts=True,
        )

        plt.title(f"Dendrogram (Linkage: {linkage_method})", fontsize=16)
        plt.ylabel("Distance", fontsize=12)
        plt.xlabel("Series", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    return clusters


def _scale_distance_matrix(mat, method="none"):
    mat = np.array(mat, dtype=float, copy=True)
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 0.0)

    if method == "none":
        return mat

    if method == "minmax":
        vmin, vmax = np.nanmin(mat), np.nanmax(mat)
        if np.isclose(vmax, vmin):
            return np.zeros_like(mat)
        return (mat - vmin) / (vmax - vmin)

    if method == "sqrt":
        mat = np.sqrt(np.clip(mat, 0.0, None))
        return mat

    if method == "log1p":
        return np.log1p(np.clip(mat, 0.0, None))

    raise ValueError(f"Unknown distance scaling method: {method}")


def get_linear_clusters(
    df,
    dist_method,
    linkage_method,
    threshold,
    visualize=False,
    diff_order=0,
    distance_scaling="none",
):
    """
    Hierarchical clustering for multivariate time series.

    Parameters
    df : pd.DataFrame
        Datetime-indexed multivariate time series (columns = series).
    dist_method : str
        One of: "correlation", "Engle-Granger", "Johansen"
        For cointegration-based methods, this calls build_coint_mat(df, test=dist_method).
    linkage_method : str
        Linkage method for scipy.cluster.hierarchy.linkage (e.g., "ward","single","average","complete").
    threshold : float
        Distance threshold used in fcluster(..., criterion="distance").
    visualize : bool, default False
        If True, plot dendrogram and draw a horizontal line at `threshold`.
    diff_order : int
        Optional differencing order applied to df via take_diff(df, diff_order).
    distance_scaling : str
        One of {"none","minmax","minmax_0_100","sqrt","log1p"}.
        - "none": use raw distances as produced
        - "minmax": rescale to [0,1]
        - "sqrt": sqrt transform (preserves ordering)
        - "log1p": log(1 + x) transform (compress large values)
    Returns
    clusters : dict
        Mapping {series_name: cluster_label}, where cluster_label is an integer from fcluster.
    """
    # optional differencing
    df = take_diff(df, diff_order)

    # build raw distance-like matrix
    if dist_method == "correlation":
        corr_matrix = df.corr()
        distance_matrix = 1.0 - corr_matrix.values  # shape (n, n)

    else:
        # support both "Engle-Granger", "Johansen"
        key = dist_method.replace("–", "-")  # tolerate en-dash vs hyphen
        if key not in ("Engle-Granger", "Johansen"):
            raise ValueError(f"Not Implemented distance method: {dist_method}")

        # build_coint_mat is expected to return a DataFrame with numeric entries.
        coint_df = build_coint_mat(df, test=key)
        # ensure numeric ndarray
        distance_matrix = coint_df.values.astype(float)

    # symmetrize (just in case) and set diag to 0
    distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
    np.fill_diagonal(distance_matrix, 0.0)

    if np.isnan(distance_matrix).any():
        finite_max = np.nanmax(distance_matrix[np.isfinite(distance_matrix)])
        fill_val = (
            finite_max * 10.0 if np.isfinite(finite_max) and finite_max > 0 else 1.0
        )
        distance_matrix = np.nan_to_num(
            distance_matrix, nan=fill_val, posinf=fill_val, neginf=fill_val
        )

    # apply scaling
    distance_matrix = _scale_distance_matrix(distance_matrix, method=distance_scaling)

    # condensed form & linkage
    condensed = squareform(distance_matrix, checks=False)
    linked = linkage(condensed, method=linkage_method)

    # get flat clusters using distance threshold
    labels = fcluster(linked, t=threshold, criterion="distance")
    clusters = dict(zip(df.columns, labels))

    # optional visualization — draw dendrogram and horizontal line at threshold
    if visualize:
        plt.figure(figsize=(12, 8))
        dendrogram(
            linked,
            orientation="top",
            labels=list(df.columns),
            distance_sort="descending",
            show_leaf_counts=True,
        )
        plt.axhline(y=threshold, color="red", linestyle="--", linewidth=1)
        plt.title(f"Dendrogram (Linkage: {linkage_method})", fontsize=16)
        plt.ylabel("Distance", fontsize=12)
        plt.xlabel("Series", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return clusters
