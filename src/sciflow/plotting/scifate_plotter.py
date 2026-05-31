# std lib imports


# 3-party import
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage


# projekt imports
from sciflow.data import LabeledDenseMatrix, LabeledDistanceMatrix
from sciflow.data.scifate_dataset import ScifateDataset
from sciflow.distance import BaseDistance




class ScifatePlotter:
    """
    
    
    """
    
    
    def __init__(
        self,
        dataset: ScifateDataset = None
    ):
        self.dataset = dataset
    
    
    
    
    def plot_matrix(
        self, 
        matrix: LabeledDenseMatrix | LabeledDistanceMatrix
    ):
        """
            Plots a Matrix as Heatmap
        
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            matrix.matrix,
            cmap="viridis",
        )

        ax.set_title(matrix.name)

        return fig
    
    
    
    def plot_distance_matrix(
        self,
        dist_matrix: LabeledDistanceMatrix = None,
        entity: str = None, 
        data: str = None, 
        distance: BaseDistance = None,
        cluster_rows: bool = True,
        cluster_cols: bool = False,
        method: str = "average",
        cmap: str = "viridis",
        n_samples: int = 100,
        random_state: int = 42,
        figsize=(10, 10)
    ):
        """
            Plot a clustered heatmap of a distance matrix.

            Parameters
            ----------
            dist_matrix : LabeledDistanceMatrix
                Symmetric NxN distance matrix.

            cluster_rows : bool
                If True, cluster rows.

            cluster_cols : bool
                If True, cluster columns.

            method : str
                Hierarchical clustering linkage method.

            cmap : str
                Matplotlib colormap.

            figsize : tuple
                Figure size.
        """
        if self.dataset is not None and entity is not None and data is not None and distance is not None:
            dist_matrix = self.dataset.get_distance_matrix(entity=entity, data=data, distance=distance)

        if dist_matrix is None:
            raise ValueError("please provide eather a matrix or a Scifatedataset at initialization and the keys to the distance matrix here")

        if n_samples is not None and n_samples < dist_matrix.matrix.shape[0]:
            rng = np.random.default_rng(random_state)
            indices = rng.choice(dist_matrix.matrix.shape[0], size=n_samples, replace=False)
            dist_matrix = dist_matrix[indices]

        matrix = dist_matrix.matrix

        row_linkage = None
        col_linkage = None

        if cluster_rows:
            condensed_rows = squareform(matrix)
            row_linkage = linkage(condensed_rows, method=method)

        if cluster_cols:
            condensed_cols = squareform(matrix)
            col_linkage = linkage(condensed_cols, method=method)

        sns.clustermap(
            matrix,
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            cmap=cmap,
            figsize=figsize,
            xticklabels=True,
            yticklabels=True
        )

        plt.show()