# std lib imports

# 3-party imports
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score

# project imports
from sciflow.data.labled_dense_matrix import LabeledDenseMatrix



def _align_clusterings_by_id(
    clustering_a: pd.DataFrame,
    clustering_b: pd.DataFrame,
    id_col: str,
    cluster_col: str = "cluster",
):
    """
        Check that both clustering DataFrames contain the same observations
        and align clustering_b to the order of clustering_a.
    """

    for name, clustering in [("clustering_a", clustering_a), ("clustering_b", clustering_b)]:
        if id_col not in clustering.columns:
            raise ValueError(f"{name} must contain id column '{id_col}'.")

        if cluster_col not in clustering.columns:
            raise ValueError(f"{name} must contain cluster column '{cluster_col}'.")

        if clustering[id_col].duplicated().any():
            duplicated = clustering.loc[clustering[id_col].duplicated(), id_col].tolist()
            raise ValueError(f"{name} contains duplicated ids in '{id_col}': {duplicated[:10]}")

    ids_a = set(clustering_a[id_col])
    ids_b = set(clustering_b[id_col])

    only_in_a = ids_a - ids_b
    only_in_b = ids_b - ids_a

    if only_in_a or only_in_b:
        raise ValueError(
            "Both clusterings must contain exactly the same observations.\n"
            f"Only in clustering_a: {list(only_in_a)[:10]}\n"
            f"Only in clustering_b: {list(only_in_b)[:10]}"
        )

    a = clustering_a[[id_col, cluster_col]].copy()
    b = clustering_b[[id_col, cluster_col]].copy()

    b_aligned = (
        b
        .set_index(id_col)
        .loc[a[id_col]]
        .reset_index()
    )

    labels_a = a[cluster_col].to_numpy().ravel()
    labels_b = b_aligned[cluster_col].to_numpy().ravel()

    return labels_a, labels_b


def clustering_overlap_matrix(
    clustering_a: pd.DataFrame,
    clustering_b: pd.DataFrame,
    id_col: str,
    name_a: str = "clustering_a",
    name_b: str = "clustering_b",
    normalize: str = None,
    as_labeled_matrix: bool = True,
):
    """
    Return the overlap between two clusterings.

    normalize:
        None       -> raw counts
        "rows"     -> each row sums to 1
        "columns"  -> each column sums to 1
        "all"      -> whole table sums to 1
    """

    

    labels_a, labels_b = _align_clusterings_by_id(
        clustering_a=clustering_a,
        clustering_b=clustering_b,
        id_col=id_col,
    )

    table = pd.crosstab(
        pd.Series(labels_a, name=name_a),
        pd.Series(labels_b, name=name_b),
    )

    if normalize is None:
        matrix = table.astype(float)

    elif normalize == "rows":
        matrix = table.div(table.sum(axis=1), axis=0).fillna(0)

    elif normalize == "columns":
        matrix = table.div(table.sum(axis=0), axis=1).fillna(0)

    elif normalize == "all":
        matrix = table / table.to_numpy().sum()

    else:
        raise ValueError("normalize must be None, 'rows', 'columns', or 'all'.")

    if not as_labeled_matrix:
        return matrix

    row_info = pd.DataFrame({
        "cluster": matrix.index.to_list(),
        "source": name_a,
        "cluster_label": [f"{name_a}_{c}" for c in matrix.index],
    })

    col_info = pd.DataFrame({
        "cluster": matrix.columns.to_list(),
        "source": name_b,
        "cluster_label": [f"{name_b}_{c}" for c in matrix.columns],
    })

    return LabeledDenseMatrix(
        matrix=matrix.to_numpy(dtype=np.float32),
        row_info=row_info,
        row_label="cluster_label",
        col_info=col_info,
        name=f"{name_a}_vs_{name_b}_overlap_matrix",
    )


def adjusted_rand_index(
    clustering_a: pd.DataFrame,
    clustering_b: pd.DataFrame,
    id_col: str,
    cluster_col: str = "cluster",
):
    """
        Return the Adjusted Rand Index between two clusterings.

        The two clustering DataFrames are first checked and aligned by id_col,
        so that the same observations are compared in the same order.
    """

    labels_a, labels_b = _align_clusterings_by_id(
        clustering_a=clustering_a,
        clustering_b=clustering_b,
        id_col=id_col,
        cluster_col=cluster_col,
    )

    return adjusted_rand_score(labels_a, labels_b)


def normalized_mutual_information(
    clustering_a: pd.DataFrame,
    clustering_b: pd.DataFrame,
    id_col: str,
    cluster_col: str = "cluster",
):
    """
        Return the Normalized Mutual Information between two clusterings.

        The two clustering DataFrames are first checked and aligned by id_col,
        so that the same observations are compared in the same order.
    """

    labels_a, labels_b = _align_clusterings_by_id(
        clustering_a=clustering_a,
        clustering_b=clustering_b,
        id_col=id_col,
        cluster_col=cluster_col,
    )

    return normalized_mutual_info_score(labels_a, labels_b)

