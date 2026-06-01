# std lib imports

# 3 party import


# projekt imports
from sciflow.reduction import BaseReduction 



class PCA(BaseReduction):

    name = "PCA"

    def __init__(self, feature_dim):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass



    @property
    def feature_names(self):
        return [
            f"PC{i+1}"
            for i in range(self.n_components)
        ]
