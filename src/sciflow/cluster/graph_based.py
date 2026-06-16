# std lib imports

# 3-party imports
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt

# project imports
from sciflow.cluster import BaseCluster
from sciflow.distance import BaseDistance
from sciflow.distance import CosineDistance


class GraphBasedClustering(BaseCluster):

    name = "GraphBasedClustering"

    def __init__(
        self,
        k_neighbors: int = 15,
        distance: BaseDistance = None,
        resolution: float = 1.0,
        seed: int = 42
    ):
        self.k_neighbors = k_neighbors
        self.distance = distance if distance is not None else CosineDistance()
        self.resolution = resolution
        self.seed = seed

        self.graph = None
        self.assign = None
        self.is_fitted = False


    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)

        D = self.distance.pairwise(X)

        self.graph = self.build_knn_graph(D)

        partition = community_louvain.best_partition(
            self.graph,
            weight="weight",
            resolution=self.resolution,
            random_state=self.seed,
        )

        assign = np.empty(D.shape[0], dtype=np.int32)

        for node, cluster in partition.items():
            assign[node] = cluster

        self.assign = assign
        self.is_fitted = True

        return assign


    def fit_predict(self, X: np.ndarray):
        return self.fit(X)


    def predict(self, X):
        raise NotImplementedError(
            "Graph-based clustering does not naturally support predicting new samples."
        )


    def build_knn_graph(self, D: np.ndarray):
        n = D.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))

        for i in range(n):
            distances = D[i].copy()
            distances[i] = np.inf
            neighbors = np.argsort(distances)[:self.k_neighbors]

            for j in neighbors:
                d = D[i, j]
                similarity = 1.0 / (1.0 + d)

                graph.add_edge(
                    i,
                    int(j),
                    weight=float(similarity),
                    distance=float(d),
                )

        return graph
    
    
    def plot_graph(
        self,
        show_edge_labels: bool = False,
        show_node_label: bool = False,
        seed: int = 42,
        layout_k: float = 3.0,
        iterations: int = 500,
        node_size: int = 120,
        edge_alpha: float = 0.25,
        figsize=(16, 12),
    ):
        if self.graph is None:
            raise ValueError("Please fit the model before plotting the graph.")

        G = self.graph

        if self.assign is None:
            raise ValueError("Please fit the model before plotting cluster colors.")

        unique_clusters = np.unique(self.assign)
        cmap = plt.get_cmap("tab20", len(unique_clusters))

        pos = nx.spring_layout(
            G,
            seed=seed,
            weight="weight",
            k=layout_k,
            iterations=iterations,
        )

        fig, ax = plt.subplots(figsize=figsize)

        nx.draw_networkx_edges(
            G,
            pos,
            alpha=edge_alpha,
            ax=ax,
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_size,
            node_color=self.assign,
            cmap=cmap,
            ax=ax,
        )

        if show_node_label:
            nx.draw_networkx_labels(
                G,
                pos,
                font_size=6,
                ax=ax,
            )

        if show_edge_labels:
            edge_labels = {
                (u, v): f"{d['weight']:.2f}"
                for u, v, d in G.edges(data=True)
            }

            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_size=5,
                ax=ax,
            )

        ax.set_title("kNN Graph Colored by Louvain Community")
        ax.axis("off")

        fig.tight_layout()

        return fig