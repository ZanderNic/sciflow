# std lib imports

# 3-party imports
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import pairwise_distances


def sparse_column_variance(X):
    """
        Compute column-wise variance without densifying the full sparse matrix.
    """
    if scipy.sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()

        return mean_sq - mean ** 2

    return np.var(X, axis=0)


def test_distance_concentration(
    X,
    dimensions,
    n_samples: int = 500,
    metric: str = "cosine",
    feature_strategy: str = "hvg",
    random_state: int = 42,
):
    """
        Test distance concentration on a real data matrix.

        Rows are interpreted as observations.
        Columns are interpreted as features.

        Parameters
        ----------
        X:
            Input matrix. Can be dense or sparse.

        dimensions:
            List of feature dimensions to test.

        n_samples:
            Number of observations sampled from X.

        metric:
            Distance metric used by sklearn.metrics.pairwise_distances.

        feature_strategy:
            Strategy for selecting features.
            Options:
            - "hvg": use features ordered by variance
            - "random": use one fixed random feature order

        random_state:
            Random seed for reproducibility.

        Returns
        -------
        pandas.DataFrame
            Table containing distance concentration statistics.
    """
    rng = np.random.default_rng(random_state)

    if scipy.sparse.issparse(X):
        X = X.tocsr()

    n_rows, n_cols = X.shape

    sample_idx = rng.choice(
        n_rows,
        size=min(n_samples, n_rows),
        replace=False,
    )

    if feature_strategy == "hvg":
        variances = sparse_column_variance(X)
        feature_order = np.argsort(variances)[::-1]

    elif feature_strategy == "random":
        feature_order = rng.permutation(n_cols)

    else:
        raise ValueError("feature_strategy must be 'hvg' or 'random'.")

    records = []

    for dim in dimensions:

        if dim > n_cols:
            continue

        feature_idx = feature_order[:dim]

        X_subset = X[sample_idx, :][:, feature_idx]

        distances = pairwise_distances(
            X_subset,
            metric=metric,
        )

        np.fill_diagonal(distances, np.nan)

        nearest = np.nanmin(distances, axis=1)
        farthest = np.nanmax(distances, axis=1)

        upper = distances[np.triu_indices_from(distances, k=1)]
        upper = upper[~np.isnan(upper)]

        median_nearest = np.median(nearest)
        median_farthest = np.median(farthest)

        records.append({
            "dimension": dim,
            "metric": metric,
            "feature_strategy": feature_strategy,
            "median_nearest_distance": median_nearest,
            "median_farthest_distance": median_farthest,
            "distance_spread": median_farthest - median_nearest,
            "nearest_farthest_ratio": median_nearest / (median_farthest + 1e-12),
            "pairwise_distance_mean": np.mean(upper),
            "pairwise_distance_std": np.std(upper),
            "pairwise_distance_cv": np.std(upper) / (np.mean(upper) + 1e-12),
            "zero_nearest_fraction": np.mean(nearest == 0),
        })

    return pd.DataFrame(records)