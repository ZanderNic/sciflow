# std lib imports
import warnings

# 3-party imports
import numpy as np
import scipy
from scipy.stats import wasserstein_distance, spearmanr
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

class BaseDistance:
    name = "BaseDistance"
    metric = None
    blockwise = False
    supports_sparse = False

    
    def __call__(self, x, y):
        return self.distance(x, y)

    def distance(self, x, y):
        x = np.asarray(x).reshape(1, -1)
        y = np.asarray(y).reshape(1, -1)

        return float(pairwise_distances(x, y, metric=self.metric)[0, 0])

    def pairwise(self, X):
    
        if self.blockwise:
            dist = self.blockwise_pairwise(X)
        else:
            if scipy.sparse.issparse(X) and not self.supports_sparse:
                X = X.toarray()
            dist = pairwise_distances(X, metric=self.metric)

            dist = np.asarray(dist, dtype=np.float32)
            dist = (dist + dist.T) / 2
            np.fill_diagonal(dist, 0.0)

        return dist
   
    
    # blockwise parwise distance calculation 
    
    def blockwise_pairwise(self, X, block_size: int = 1000):

        n_samples = X.shape[0]
        dist = np.zeros((n_samples, n_samples), dtype=np.float32)

        for i_start in tqdm(range(0, n_samples, block_size), desc=self.name):
            i_end = min(i_start + block_size, n_samples)

            for j_start in range(i_start, n_samples, block_size):
                j_end = min(j_start + block_size, n_samples)

                if scipy.sparse.issparse(X) and not self.supports_sparse:
                    X_i = X[i_start:i_end, :].toarray()     
                    X_j = X[j_start:j_end, :].toarray()
                else:
                    X_i = X[i_start:i_end, :]  
                    X_j = X[j_start:j_end, :]
                    
                block_dist = pairwise_distances(
                    X_i,
                    X_j,
                    metric=self.metric
                )

                dist[i_start:i_end, j_start:j_end] = block_dist

                if i_start != j_start:
                    dist[j_start:j_end, i_start:i_end] = block_dist.T

        np.fill_diagonal(dist, 0.0)

        return dist
    
class EuclideanDistance(BaseDistance):
    name = "EuclideanDistance"
    metric = "euclidean"
    blockwise = False
    supports_sparse = True


class ManhattanDistance(BaseDistance):
    name = "ManhattanDistance"
    metric = "manhattan"
    blockwise = False
    supports_sparse = True



class CosineDistance(BaseDistance):
    name = "CosineDistance"
    metric = "cosine"
    blockwise = False
    supports_sparse = True


class CorrelationDistance(BaseDistance):
    name = "CorrelationDistance"
    metric = "correlation"
    blockwise = True
    supports_sparse = False
