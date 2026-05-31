



class BaseReduction:
    """
        Base class for all dimensionality reduction methods.
    """

    name = "BaseReduction"

    def fit(self, X):
        return self

    def transform(self, X):
        raise NotImplementedError()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)