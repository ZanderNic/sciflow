from sciflow.reduction.base_reduction import BaseReduction
from sciflow.reduction.random_projection import RandomProjektions
from sciflow.reduction.pca import PCA
from sciflow.reduction.autoencoder import Autoencoder
from sciflow.reduction.umap import UMAP


available_reduction_alg = [
    RandomProjektions.name,
    PCA.name,
    Autoencoder.name,
    UMAP.name,
]