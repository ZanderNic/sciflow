# std lib imports

# 3-party import

# projekt imports



class BaseCluster:
    """
        Base class for all clustering algorithms.
    """

    name = "BaseCluster"

    def __call__(self, X):
        return self.fit_predict(X)

    def fit(self, X):
        """
        Fit the clustering model.
        """
        raise NotImplementedError()

    def predict(self, X):
        """
        Predict cluster labels for new data.
        """
        raise NotImplementedError()

    def fit_predict(self, X):
        """
        Fit and return cluster labels.
        """
        self.fit(X)
        return self.predict(X)