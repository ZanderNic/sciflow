# std lib imports

# 3-party import

# projekt imports
from sciflow.cluster import BaseCluster





class DBSCAN(BaseCluster):

    name = "DBSCAN"

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        pass

    def predict(self, X):
        pass

    def fit_predict(self, X):
        pass