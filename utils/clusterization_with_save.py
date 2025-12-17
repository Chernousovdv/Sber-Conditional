import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def take_diff(df, diff_order=0):
    if diff_order > 0:
        return df.diff(diff_order).dropna()
    return df


def _pairwise_johansen_distance(x: pd.Series, y: pd.Series, det_order=0, k_ar_diff=1):
    data = np.column_stack([x.values, y.values])
    res = coint_johansen(data, det_order, k_ar_diff)
    trace_stat = res.lr1[0]
    dist = 100.0 * (1.0 / (1.0 + float(trace_stat)))
    return float(np.clip(dist, 0.0, 100.0))


def build_coint_mat(df: pd.DataFrame, test="Engle-Granger", **kwargs):
    """
    Supported tests:
      - "Engle-Granger" : pairwise statsmodels.tsa.stattools.coint -> p-value * 100
      - "Johansen"      : pairwise Johansen (2-var system) -> heuristic distance in [0,100]
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
    Perform hierarchical clustering of multiple time series

    Parameters
    ----------
    df : pandas.DataFrame
        Time-indexed data where each column is a separate series.
    dist_method : str
        Distance construction method:
        - "correlation": distance = 1 - df.corr()
        - "Engle-Granger": uses build_coint_mat(df, test="Engle-Granger")
        - "Johansen": uses build_coint_mat(df, test="Johansen")
    linkage_method : str
        Linkage method passed to scipy.cluster.hierarchy.linkage
        (e.g., "single", "complete", "average", "ward").
        Note: some linkage methods (e.g., "ward") are intended for Euclidean distances.
    threshold : float
        Distance threshold for scipy.cluster.hierarchy.fcluster with
        criterion="distance". The appropriate scale depends on the chosen
        dist_method and distance_scaling.
    diff_order : int, default 0
        Differencing order applied via df.diff(diff_order). If 0, no differencing.
    distance_scaling : str, default "none"
        Optional transform applied to the distance matrix before clustering:
        - "none": no transform
        - "minmax": rescale all distances to [0, 1]
        - "sqrt": apply sqrt(max(distance, 0))
        - "log1p": apply log1p(max(distance, 0))

    Returns
    -------
    clusters : dict
        Mapping {column_name: cluster_label}, where cluster_label is an integer
        assigned by fcluster().
    """

    df = take_diff(df, diff_order)

    if dist_method == "correlation":
        corr_matrix = df.corr()
        distance_matrix = 1.0 - corr_matrix.values
        distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)

    else:
        key = dist_method.replace("–", "-")
        if key not in ("Engle-Granger", "Johansen"):
            raise ValueError(f"Not Implemented distance method: {dist_method}")

        coint_df = build_coint_mat(df, test=key)
        distance_matrix = coint_df.values.astype(float)

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

    distance_matrix = _scale_distance_matrix(distance_matrix, method=distance_scaling)

    condensed = squareform(distance_matrix, checks=False)
    linked = linkage(condensed, method=linkage_method)
    linked[:, 2] = np.clip(linked[:, 2], 0.0, None)

    labels = fcluster(linked, t=threshold, criterion="distance")
    labels_str: list[str] = [f"Кластер {v}" for v in list(labels)]
    df_columns = [v.replace("/", " ") for v in list(df.columns)]
    clusters = dict(zip(df_columns, labels_str))
    return clusters


def _safe_float_for_filename(x: float) -> str:
    s = f"{x}".strip()
    return s.replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster time series and save clusters to JSON."
    )
    parser.add_argument(
        "--df-path",
        type=str,
        default="/home/danilach/mipt-stats/SberTs/Data/comb_df_v1.csv",
        help="Path to input CSV (index in first column).",
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="correlation",
        help='Distance method: "correlation", "Engle-Granger", or "Johansen".',
    )
    parser.add_argument(
        "--linkage",
        type=str,
        default="ward",
        help='Linkage method: "ward", "single", "average", "complete".',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Distance threshold for fcluster (criterion='distance').",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to write the output JSON.",
    )

    args = parser.parse_args()

    df_path = Path(args.df_path)
    df_name = df_path.stem

    df = pd.read_csv(df_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.drop(
        columns=[
            "Огурцы тепличные, руб./кг",
            "Томаты тепличные, руб./кг",
        ],
        errors="ignore",
    )

    linkage_list = []
    distance_list = []

    clusters = get_linear_clusters(
        df,
        args.dist,
        args.linkage,
        args.threshold,
        visualize=args.visualize,
    )

    thr_tag = _safe_float_for_filename(args.threshold)
    out_name = f"{df_name}__dist-{args.dist}__link-{args.linkage}__thr-{thr_tag}__clusters.json"
    out_path = Path(args.out_dir) / out_name

    payload = {
        "df_name": df_name,
        "df_path": str(df_path),
        "dist_method": args.dist,
        "linkage_method": args.linkage,
        "threshold": args.threshold,
        "clusters": clusters,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
