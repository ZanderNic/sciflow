# std lib imports

# 3-party import

# projekt imports


class Dataset:
    def __init__(self, data, sample_info=None, feature_info=None, name=None):
        self.data = data                        

        self.sample_info = sample_info        
        self.feature_info = feature_info   

        self.representations = {}               # To save representations of the data (scaled data, ...)
        self.projections = {}                   # To save projections from dim reduction
        self.annotations = {}                   # To save labels from clustering 
        
        self.meta = {}        

        self.name = name
        
    
    
    def summary(self):
        return {
            "name": self.name,
            "shape": self.shape,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "matrices": list(self.extra_matrices.keys()),
            "representations": list(self.representations.keys()),
            "results": list(self.results.keys()),
            "meta": list(self.meta.keys()),
        }
        
    def __str__(self):
        pass
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def n_samples(self):
        return self.data.shape[0]

    @property
    def n_features(self):
        return self.data.shape[1]