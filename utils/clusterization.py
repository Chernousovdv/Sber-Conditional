import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def build_coint_mat(df, test="Engle-Granger", **kwargs):
    """
    Build a cointegration matrix for all pairs of series in the DataFrame.
    """

    n_vars = len(df.columns)
    coint_mat = np.full((n_vars, n_vars), np.nan)

    if test == "Engle-Granger":
        for i, col_i in enumerate(df.columns):
            for j, col_j in enumerate(df.columns):
                if i != j:
                    test_stat, p_value, _ = coint(df[col_i], df[col_j], **kwargs)
                    coint_mat[i, j] = p_value * 100
                    # print(p_value)
                else:
                    coint_mat[i, j] = 0

        result_df = pd.DataFrame(coint_mat, index=df.columns, columns=df.columns)
        return result_df

    elif test == "Johansen":  # TODO
        raise ValueError("Not Implemented")

    else:
        raise ValueError("TEST NOT FOUND")


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


def transform_pvalue_todist(pmatrix):
    pass
    return None


def spectral_clustering(distance_matrix, labels, linkage_method="average"):
    pass
    return None


# TODO kmeans, dbscan although probably inferior to t-sne
