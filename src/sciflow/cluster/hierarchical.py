# std lib imports


# 3-party imports
import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt

# projekt imports
from sciflow.cluster import BaseCluster
from sciflow.distance import BaseDistance
from sciflow.distance import EuclideanDistance



class HierarchicalClustering(BaseCluster):

    name = "HierarchicalClustering"

    def __init__(
        self,
        k: int,
        method: str = "average",          # "single", "complete", "average", "ward"
        metric: BaseDistance = EuclideanDistance(),
    ):
        self.k = k
        self.method = method
        self.metric = metric.name if isinstance(metric, BaseDistance) else str(metric)

        self.linkage_matrix = None
        self.assign = None
        self.is_fitted = False


    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)

        condensed_dist = pdist(X, metric=self.metric)

        self.linkage_matrix = linkage(condensed_dist, method=self.method)

        self.assign = fcluster(
            self.linkage_matrix,
            t=self.k,
            criterion="maxclust"
        ) - 1

        self.is_fitted = True

        return self.assign


    def fit_predict(self, X: np.ndarray):
        return self.fit(X)


    def predict(self, X):
        raise NotImplementedError(
            "Hierarchical clustering does not naturally support predicting new samples."
        )
        
        
    def plot_dendrogram(
        self,
        truncate_mode: str = "level",   # None, "level", "lastp"
        p: int = 5,
        figsize=(12, 6),
    ):
        if self.linkage_matrix is None:
            raise ValueError("Model must be fitted before plotting dendrogram.")

        fig, ax = plt.subplots(figsize=figsize)

        dendrogram(
            self.linkage_matrix,
            truncate_mode=truncate_mode,
            p=p,
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=8,
        )

        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Samples / clusters")
        ax.set_ylabel("Distance")

        fig.tight_layout()

        return fig