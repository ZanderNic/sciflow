# std lib imports

# 3-party import
import numpy as np
import scipy

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
        if scipy.sparse.issparse(X) and self.supports_sparse:
            return self.pairwise_sparse(X)

        return self.pairwise_dense(X)


    def pairwise_dense(self, X):
        """
            Normale pariwise distance calculation that will call self.distance(x,y) for every paars of two ponts to build a distance matrix 
            that will be retunred as np.array. The rows will be interpretet as data points and the columns as features 
            
            
        """
        
        X = np.asarray(X, dtype=np.float32)
        
        dist_array = np.zeros(shape=[X.shape[0], X.shape[0]], dtype=np.float32)
        
        for i in range(X.shape[0]):
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
        X = X.tocsr().astype(np.float32)

        norms = np.sqrt(
            np.asarray(
                X.multiply(X).sum(axis=1)
            )
        ).ravel()

        norms[norms == 0] = 1.0

        X_norm = X.multiply(
            1.0 / norms[:, None]
        )

        similarity = X_norm @ X_norm.T

        return 1.0 - similarity.toarray()



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


    def pairwise_dense(self, X: np.typing.ArrayLike) -> np.array:
        X = np.asarray(X, dtype=float)

        X_centered = X - X.mean(axis=1, keepdims=True)

        norms = np.linalg.norm(X_centered, axis=1)
        norms[norms == 0] = 1.0

        X_norm = X_centered / norms[:, None]

        correlation = X_norm @ X_norm.T

        return 1.0 - correlation


class KendallTauDistance(BaseDistance):
    """
        Distance based on Kendall Tau correlation with measures the ranking diverence betwen x and y

        distance = 1 - tau

        Range:
            0   -> identical ranking
            2   -> completely reversed ranking
    """
    name = "KendallTauDistance"
    def distance(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        x = np.asarray(x)
        y = np.asarray(y)

        tau, _ = scipy.stats.kendalltau(x, y)

        if np.isnan(tau):
            return 1.0

        return float(1.0 - tau)


class DistortionDistance(BaseDistance):
    """
        Weighted combination of:
            - Manhattan (L1) distance
            - Kendall Tau distance (ranking based)
    """
    name = "DistortionDistance"

    def __init__(
        self,
        weights: tuple[float, float] = (0.5, 0.5),
        normalize: bool = True
    ):
        if len(weights) != 2 or not np.isclose(sum(weights), 1.0):
            raise ValueError("weights must sum to 1.")

        self.weights = weights
        self.normalize = normalize

        self.l1_distance = ManhattanDistance()
        self.kendall_distance = KendallTauDistance()

    def _l1_normalize(self, x: np.ndarray) -> np.ndarray:
        norm = np.sum(np.abs(x))

        if norm == 0:
            return x

        return x / norm

    def distance(self, x: np.typing.ArrayLike, y: np.typing.ArrayLike) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")

        if self.normalize:
            x = self._l1_normalize(x)
            y = self._l1_normalize(y)

        l1 = self.l1_distance(x, y)
        kendall = self.kendall_distance(x, y)

        return float(
            self.weights[0] * l1 +
            self.weights[1] * kendall
        )