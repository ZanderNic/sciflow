# std lib imports

# 3-party import
import numpy as np

# projekt imports




class BaseReduction:
    """
        Base class for all dimensionality reduction methods.
    """

    name: str = "BaseReduction"
    is_fitted: bool = False
    
    def fit(self, X):
        return self

    def transform(self, X):
        raise NotImplementedError()

    def fit_transform(self, X) -> np.array:

        if self.is_fitted:
            return self.transform(X)

        self.fit(X)
        return self.transform(X)
    
    @property
    def feature_names(self):
        return None