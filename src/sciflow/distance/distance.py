# std lib imports
import warnings


# 3-party import
import numpy as np
import scipy
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances


# projekt imports


class BaseDistance:
    """
        Base distance class that will be parent class of all other distances 
    """
    name = "BaseDistance"
    supports_sparse = False
    
    
    def __call__(self, x, y):
        return self.distance(x, y)


    def distance(self, x, y):
        raise NotImplementedError


    def pairwise(self, X):
        if scipy.sparse.issparse(X):
            if self.supports_sparse:
                return self.pairwise_sparse(X)

            X = X.toarray()

        return self.pairwise_dense(X)


    def pairwise_dense(self, X):
        """
            Normale pariwise distance calculation that will call self.distance(x,y) for every paars of two ponts to build a distance matrix 
            that will be retunred as np.array. The rows will be interpretet as data points and the columns as features 
            
            
        """
        
        X = np.asarray(X, dtype=np.float32)
        
        dist_array = np.zeros(shape=[X.shape[0], X.shape[0]], dtype=np.float32)
        
        for i in tqdm(range(X.shape[0]), desc=self.name):
            for j in range(i + 1, X.shape[0]):
                dist = self.distance(X[i, :], X[j, :])
                
                dist_array[i][j] = dist
                dist_array[j][i] = dist

        return dist_array


    def pairwise_sparse(self, X):
        raise NotImplementedError(
            f"{self.name} does not implement pairwise_sparse()."
        )
    


class EuclideanDistance(BaseDistance):
    name = "EuclideanDistance"
    supports_sparse = True
    
    def distance(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        x = np.asarray(x)
        y = np.asarray(y)

        return float(np.linalg.norm(x - y))


    def pairwise_dense(self, X: np.typing.ArrayLike) -> np.array:
        """
            Because we can calculate Euclidean Vectorized we will override base method 
        
        """
        X = np.asarray(X, dtype=np.float32)
        
        squared_norms = np.sum(X**2, axis=1, keepdims=True)

        squared_distances = squared_norms + squared_norms.T - 2 * X @ X.T                

        return np.sqrt(np.clip(squared_distances, 0.0, None))
        
        
    def pairwise_sparse(self, X):
        X = X.tocsr().astype(np.float32)

        squared_norms = np.asarray(X.multiply(X).sum(axis=1))

        gram = X @ X.T

        squared_distances = (
            squared_norms
            + squared_norms.T
            - 2 * gram.toarray()
        )

        return np.sqrt(np.clip(squared_distances, 0.0, None))
        

class ManhattanDistance(BaseDistance):
    name = "ManhattanDistance"
    supports_sparse = False
    
    def distance(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        x = np.asarray(x)
        y = np.asarray(y)

        return float(np.sum(np.abs(x - y)))
    


class CosineDistance(BaseDistance):
    name = "CosineDistance"
    supports_sparse = True
    
    def distance(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        x = np.asarray(x)
        y = np.asarray(y)

        numerator = np.dot(x, y)
        denominator = np.linalg.norm(x) * np.linalg.norm(y)

        if denominator == 0:
            return 1.0

        cosine_similarity = numerator / denominator

        return float(1.0 - cosine_similarity)


    def pairwise_dense(self, X):
        X = np.asarray(X, dtype=np.float32)

        norms = np.linalg.norm(X, axis=1)
        norms[norms == 0] = 1.0
        X_norm = X / norms[:, None]

        similarity = X_norm @ X_norm.T

        return 1.0 - similarity


    def pairwise_sparse(self, X):
        return cosine_distances(X)


class CorrelationDistance(BaseDistance):
    name = "CorrelationDistance"
    
    def distance(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        x = np.asarray(x)
        y = np.asarray(y)

        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        numerator = np.dot(x_centered, y_centered)
        denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)

        if denominator == 0:
            return 1.0

        correlation = numerator / denominator

        return float(1.0 - correlation)