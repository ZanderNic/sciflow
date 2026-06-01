# std lib imports

# 3 party import
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import scipy


# projekt imports
from sciflow.reduction import BaseReduction 



class RandomProjektions(torch.nn.Module, BaseReduction):
    """
        A non trainable Module that is based on the paper "Near-Optimal Signal Recovery From Random Projections: Universal Encoding Strategies?" from Emmanuel J. Candes
        that shows that it is possible to recover a d dimensional signal x that is k sparse meeaning that out of the d dimensions there are only k with k << d non zero elements
        with high precission from only k random linear measurements. Therefore we can use a Matrix  M that has the dim d x k with random elements and y = M * x  
        with high precission with y having a dim of k so is mutch smaller than x.
    """
    name = "RandomProjection"
    
    
    def __init__(
        self,
        data_dim: int, 
        feature_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
            Here we init our sensing matrix M with gaussian random variables and the dim of  data_dim x feature_dim so we can multiply our data vector to it and get the 
            not exact but goof enoth feature dim representation of our input dim. 
        """
        super().__init__()
        self.dim_ = feature_dim
        self.device = device 
        
        self.sensing_matrix = torch.randn(
            (data_dim, feature_dim),
            device=device, dtype=torch.float32
        )  / np.sqrt(feature_dim)

        self.is_fitted = True
        

    def forward(
        self,
        x: torch.Tensor
    ):
        """
            Applies the random projection to a batch of input vectors.
            Input: x of shape (batch_size, data_dim)
            Output: y of shape (batch_size, feature_dim)
        """

        x = x.reshape(x.size(0), -1)  # ensure (batch, data_dim)

        with torch.no_grad():
            y = x @ self.sensing_matrix

        return y

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)

        elif scipy.sparse.issparse(X):
            X = X.toarray().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

        return self.forward(X)

    @property
    def feature_names(self):
        return [
            f"RP{i+1}"
            for i in range(self.feature_dim)
        ]


### helper functions

def jl_dimension(n, eps=0.2):
    return int(np.ceil(4 * np.log(n) / (eps**2 / 2 - eps**3 / 3)))


def test_random_projection_matrix(
    matrix,
    dimensions=(50, 100, 200, 300, 500),
    n_samples=500,
    metric="euclidean",
    n_repeats=5,
    random_state=42,
    device="cuda" if torch.cuda.is_available() else "cpu"
):

    rng = np.random.default_rng(random_state)

    n_total = matrix.shape[0]
    n_features = matrix.shape[1]

    sample_idx = rng.choice(
        n_total,
        size=min(n_samples, n_total),
        replace=False
    )

    if isinstance(matrix, pd.DataFrame):
        X_sample = matrix.iloc[sample_idx, :].to_numpy(dtype=np.float32)

    elif scipy.sparse.issparse(matrix):
        X_sample = matrix[sample_idx, :].toarray().astype(np.float32)

    else:
        X_sample = np.asarray(matrix[sample_idx, :], dtype=np.float32)

    D_original = pairwise_distances(X_sample, metric=metric)

    tri_idx = np.triu_indices_from(D_original, k=1)
    original_values = D_original[tri_idx]

    valid = original_values > 1e-12
    original_values = original_values[valid]

    X_tensor = torch.tensor(X_sample, dtype=torch.float32, device=device)

    results = []

    for dim in dimensions:

        repeat_results = []

        for repeat in range(n_repeats):

            torch.manual_seed(random_state + repeat)

            projector = RandomProjektions(
                data_dim=n_features,
                feature_dim=dim,
                device=device
            )

            X_projected = projector(X_tensor)
            X_projected = X_projected.cpu().numpy()

            D_projected = pairwise_distances(
                X_projected,
                metric=metric
            )

            projected_values = D_projected[tri_idx][valid]

            absolute_error = np.abs(
                projected_values - original_values
            )

            relative_error = absolute_error / original_values
            relative_error_percent = relative_error * 100

            repeat_results.append({
                "mae": np.mean(absolute_error),
                "mse": np.mean(absolute_error ** 2),

                "mean_percent_error": np.mean(relative_error_percent),
                "mse_percent_error": np.mean(relative_error_percent ** 2),
                "min_percent_error": np.min(relative_error_percent),
                "max_percent_error": np.max(relative_error_percent)
            })

        repeat_df = pd.DataFrame(repeat_results)

        results.append({
            "n_samples": X_sample.shape[0],
            "original_dimension": X_sample.shape[1],
            "projection_dimension": dim,
            "metric": metric,

            "mae": repeat_df["mae"].mean(),
            "mse": repeat_df["mse"].mean(),
            "min_mae": repeat_df["mae"].min(),
            "max_mae": repeat_df["mae"].max(),
            "min_mse": repeat_df["mse"].min(),
            "max_mse": repeat_df["mse"].max(),

            "mean_percent_error": repeat_df["mean_percent_error"].mean(),
            "mse_percent_error": repeat_df["mse_percent_error"].mean(),
            "min_percent_error": repeat_df["min_percent_error"].min(),
            "max_percent_error": repeat_df["max_percent_error"].max()
        })

    return pd.DataFrame(results)
