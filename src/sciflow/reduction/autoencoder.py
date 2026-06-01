# std lib imports
from typing import List


# 3 party import
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy


# projekt imports
from sciflow.reduction import BaseReduction 


class Autoencoder(torch.nn.Module, BaseReduction):

    name = "Autoencoder"

    def __init__(
        self,
        input_dim: int, 
        layer_dims: List[int] = [256, 128, 64, 32],
        attention_on_layers: List[bool] = [False, False, False, True],
        
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        sparsity_loss: bool = True,
        sparsity_weight: float = 1e-4,
        
        learning_rate: float = 1e-3,
        loss: str = "mse",
        optimizer: str = "adamw",
        batch_size: int = 256,
        epochs: int = 200,

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.attention_on_layers = attention_on_layers

        self.latent_dim = layer_dims[-1]                        # latent dimension = last encoder layer
        self.n_layers = len(layer_dims)                         # save num layers

        
        # check attention and layer dims match 
        if len(layer_dims) != len(attention_on_layers):
            raise ValueError("layer_dims and attention_on_layers must have the same length.")

        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

        # build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # save training stuff
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.optimizer = optimizer
        self.loss = loss
        self.sparsity_loss = sparsity_loss
        self.sparsity_weight = sparsity_weight

        self.device = device

        self.to(self.device)


    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)

        return x


    def decode(self, x):
        for layer in self.decoder:
            x = layer(x)

        return x


    def fit(self, X):
        """
            The rows of X will be intrepretet as diverent data points and the cloumns as features 
        
        """
    
        dataset = MatrixDataset(X) if not scipy.sparse.issparse(X) else SparseMatrixDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = self._get_optimizer()
        loss_function = self._get_loss()
    
        self.train()
    
        for epoch in tqdm(range(self.epochs), desc="Training Autoencoder"):

            epoch_loss = 0.0

            for batch in dataloader:
                batch = batch.to(self.device)

                reconstructed = self.forward(batch)
                loss = loss_function(batch, reconstructed)

                if self.sparsity_loss:
                    loss += self.sparsity_weight * torch.mean(torch.abs(reconstructed))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)

            tqdm.write(f"Epoch {epoch + 1}/{self.epochs} - loss: {epoch_loss:.6f}")


        self.is_fitted = True
        return self
              
                
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


    def forward(self, X):
        return self.decode(self.encode(X))
    
    
    # helper
    
    def transform(self, X):
        self.eval()
        
        # make sure X is torch tensor on right device 
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)

        elif scipy.sparse.issparse(X):
            X = X.toarray().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            encode = self.encode(X)
        
        return encode
    
    
    def _get_activation(self):
        if self.activation == "relu":
            return torch.nn.ReLU()
        elif self.activation == "gelu":
            return torch.nn.GELU()
        elif self.activation == "tanh":
            return torch.nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")   
    
    def _get_loss(self):
        if self.loss == "mse":
            return torch.nn.MSELoss()

        elif self.loss == "l1":
            return torch.nn.L1Loss()

        else:
            raise ValueError("i dont know that loss")
    
    
    def _get_optimizer(self):

        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        elif self.optimizer == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        else:
            raise ValueError("dont know that optimizer")
    
    
    def _build_encoder(self):
        layers = torch.nn.ModuleList()
        
        previous_dim = self.input_dim

        for layer_dim, use_attention in zip(self.layer_dims, self.attention_on_layers):
            layers.append(torch.nn.Linear(previous_dim, layer_dim))

            if self.batch_norm:
                layers.append(torch.nn.BatchNorm1d(layer_dim))

            layers.append(self._get_activation())

            if self.dropout > 0:
                layers.append(torch.nn.Dropout(self.dropout))

            if use_attention:
                layers.append(SelfAttentionBlock(layer_dim))

            previous_dim = layer_dim

        return layers


    def _build_decoder(self):
        layers = torch.nn.ModuleList()
        
        previous_dim = self.latent_dim
        decoder_dims = list(reversed(self.layer_dims[:-1]))
        decoder_attention = list(reversed(self.attention_on_layers[:-1]))

        for layer_dim, use_attention in zip(decoder_dims, decoder_attention):
            layers.append(
                torch.nn.Linear(previous_dim, layer_dim)
            )

            if self.batch_norm:
                layers.append(
                    torch.nn.BatchNorm1d(layer_dim)
                )

            layers.append(self._get_activation())

            if self.dropout > 0:
                layers.append(
                    torch.nn.Dropout(self.dropout)
                )
                
            if use_attention:
                layers.append(
                    SelfAttentionBlock(layer_dim)
                )
            previous_dim = layer_dim

        layers.append(
            torch.nn.Linear(previous_dim, self.input_dim)
        )

        return layers 

    @property
    def feature_names(self):
        return [f"Latent{i+1}" for i in range(self.latent_dim)]


    def save(self, path: str):
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "layer_dims": self.layer_dims,
                "attention_on_layers": self.attention_on_layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "batch_norm": self.batch_norm,
                "learning_rate": self.learning_rate,
                "loss": self.loss,
                "optimizer": self.optimizer,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "device": self.device,
                "is_fitted": self.is_fitted
            }
        }

        torch.save(checkpoint, path)


    @classmethod
    def load(cls, path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(
            path,
            map_location=device
        )

        config = checkpoint["config"]
        config["device"] = device

        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        return model


class SelfAttentionBlock(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.attention = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        return x * weights
    
    
    
class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = torch.as_tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
    

class SparseMatrixDataset(torch.utils.data.Dataset):

    def __init__(self, X):

        if not scipy.sparse.issparse(X):
            raise ValueError("SparseMatrixDataset expects a scipy sparse matrix.")
        
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        return torch.as_tensor(row.toarray().ravel(),dtype=torch.float32)