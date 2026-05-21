# std lib imports

# 3-party import
import seaborn as sns
import matplotlib.pyplot as plt

# projekt imports
from sciflow.data import LabeledDenseMatrix, LabeledDistanceMatrix
from sciflow.data.scifate_dataset import ScifateDataset





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
            ax=ax,
            cmap="viridis",
        )

        ax.set_title(matrix.name)

        return fig
    
    
    
    def plot_distance_matrix(
        self, 
    ):
        """
            This function is just a raper around plot_matrix that gets the right distance matrix from the Scifate dataset
        
        """
        pass