# std lib imports
from typing import List


# 3 party import
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import scipy


# projekt imports
from sciflow.reduction import BaseReduction 



class Autoencoder(torch.nn.Module, BaseReduction):

    name = "RandomProjection"

    def __init__(
        self,
        input_dim: int, 
        feature_dim: int = 64,
        
        attention_on_layers: List[bool] = [0]
        
    ):
        self.feature_dim = feature_dim
        
        
        
        
        self.encoder = ...
        
        self.decoder = ...
        
        
        

    def fit(self, X):
        pass

    def transform(self, X):
        pass
