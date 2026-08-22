# std lib imports
import warnings

# 3 party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP as UMAPModel

# project imports
from sciflow.reduction import BaseReduction


class UMAP(BaseReduction):

    name = "UMAP"

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",              #  this can be  "euclidean" or "cosine" or "correlation" or "manhattan" or "canberra" or "braycurtis" 
        random_state: int = 42,
        spread: float = 1.0,
        learning_rate: float = 1.0,
        n_epochs: int = None,
        precomputed_knn: tuple | None = None,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.spread = spread
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.precomputed_knn = precomputed_knn

        if precomputed_knn is None:
            umap_knn = (None, None, None)
        else:
            indices, distances = precomputed_knn
            umap_knn = (indices, distances, None)

        self.model = UMAPModel(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            spread=self.spread,
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            precomputed_knn=umap_knn,
            n_jobs=1
        )

        self.is_fitted = False


    def fit(self, X):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"precomputed_knn\[2\].*is not an NNDescent object"
            )
            self.model.fit(X)

        self.is_fitted = True
        return self


    def fit_transform(self, X):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"precomputed_knn\[2\].*is not an NNDescent object"
            )
            X_reduced = self.model.fit_transform(X)

        self.is_fitted = True
        return X_reduced


    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("UMAP must be fitted before transform.")

        return self.model.transform(X)


    @property
    def feature_names(self):
        return {
            "feature": [
                f"UMAP{i + 1}"
                for i in range(self.n_components)
            ]
        }


    def get_embedding(self):
        if not self.is_fitted:
            raise ValueError("UMAP must be fitted before accessing embedding.")

        return pd.DataFrame(
            self.model.embedding_,
            columns=[
                f"UMAP{i + 1}"
                for i in range(self.n_components)
            ]
        )