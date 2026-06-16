# std lib imports

# 3 party import
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# projekt imports
from sciflow.reduction import BaseReduction 


class PCA(BaseReduction):

    name = "PCA"

    def __init__(
        self,
        n_components: int = 2,
        calculated_components: int = None,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.calculated_components = calculated_components if calculated_components is not None else n_components
        self.random_state = random_state

        if self.calculated_components < self.n_components:
            raise ValueError("calculated_components must be >= n_components.")

        self.model = TruncatedSVD(
            n_components=self.calculated_components,
            random_state=random_state,
        )

        self.is_fitted = False


    def fit(self, X):
        self.model.fit(X)
        self.is_fitted = True
        return self


    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform.")

        X_reduced = self.model.transform(X)

        return X_reduced[:, :self.n_components]


    def fit_transform(self, X):
        X_reduced = self.model.fit_transform(X)

        self.is_fitted = True

        return X_reduced[:, :self.n_components]


    @property
    def feature_names(self):
        return {
            "feature": [
                f"PC{i + 1}"
                for i in range(self.n_components)
            ]
        }


    def get_explained_variance(self):
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before accessing explained variance.")

        return pd.DataFrame({
            "component": np.arange(1, self.calculated_components + 1),
            "explained_variance_ratio": self.model.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                self.model.explained_variance_ratio_
            ),
            "used_for_output": [
                i < self.n_components
                for i in range(self.calculated_components)
            ],
        })


    def plot_explained_variance(self, figsize = (8, 5)):
        df = self.get_explained_variance()

        fig, ax = plt.subplots(figsize = figsize)

        ax.plot(
            df["component"],
            df["cumulative_explained_variance_ratio"],
            marker="o",
            label="Cumulative explained variance",
        )

        ax.axvline(
            self.n_components,
            linestyle="--",
            label=f"Used components: {self.n_components}",
        )

        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("Explained Variance by PCA Components")
        ax.legend()

        fig.tight_layout()

        return fig