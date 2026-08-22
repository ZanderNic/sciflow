# std lib imports

# 3-party import
import numpy as np

# projekt imports
from sciflow.data.labled_sparse_matrix import LabeledSparseMatrix

    
def add_sparse_noise(
    matrix,
    scale=0.1,
    fraction=1.0,
    distribution="normal",
    df=3,
    random_state=42,
):
    X = matrix.copy().tocsr().astype(float)

    rng = np.random.default_rng(random_state)

    n_noisy = int(X.nnz * fraction)
    indices = rng.choice(X.nnz, size=n_noisy, replace=False)

    if distribution == "normal":
        noise = rng.normal(0, scale, n_noisy)

    elif distribution == "t":
        noise = rng.standard_t(df, n_noisy) * scale

    elif distribution == "laplace":
        noise = rng.laplace(0, scale, n_noisy)

    elif distribution == "cauchy":
        noise = rng.standard_cauchy(n_noisy) * scale

    elif distribution == "uniform":
        noise = rng.uniform(-scale, scale, n_noisy)

    else:
        raise ValueError(
            "distribution must be 'normal', 't', 'laplace', "
            "'cauchy', or 'uniform'."
        )

    X.data[indices] += noise

    return X