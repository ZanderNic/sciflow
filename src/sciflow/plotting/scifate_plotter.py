# std lib imports


# 3-party import
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import numpy as np
import pandas as pd

import scipy
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

import networkx as nx

# projekt imports
from sciflow.data import LabeledDenseMatrix, LabeledDistanceMatrix
from sciflow.data.scifate_dataset import ScifateDataset
from sciflow.distance import BaseDistance
from sciflow.cluster import BaseCluster
from sciflow.reduction import BaseReduction



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
            Plot a matrix as a heatmap.

            Parameters
            ----------
            matrix : LabeledDenseMatrix | LabeledDistanceMatrix
                Matrix to visualize.

            Returns
            -------
            matplotlib.figure.Figure
                The created figure.
        """

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            matrix.matrix,
            cmap="viridis",
            ax=ax
        )

        ax.set_title(matrix.name)

        fig.tight_layout()

        return fig
    
    
    
    def plot_distance_matrix(
        self,
        dist_matrix: LabeledDistanceMatrix = None,
        entity: str = None,                                           # "cell" | "gene" | "trajectory"
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

        fig = sns.clustermap(
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

        return fig
    
    
    def plot_reduced_matrix(
        self,
        reduced_matrix: LabeledDenseMatrix = None,
        entity: str = None,                                     #        
        data: str = None,                                       # "expression" | "ntr" | "new_rna" | "old_rna"                 
        reduction: BaseReduction = None, 
        color_by = None,                                        # None, np.array-like labels, or row_info column name
        dimensions = (0, 1),
        title: str = None,
        opacity: float = 0.8,
        size: int = 5,
        max_categories: int = 30
    ):

        if self.dataset is not None and entity is not None and data is not None and reduction is not None:
            reduced_matrix = self.dataset.get_reduced_matrix(entity=entity, data=data, reduction=reduction)

        if reduced_matrix is None:
            raise ValueError("please provide either a matrix or a ScifateDataset at initialization and the keys to the reduced matrix here")

        X = reduced_matrix.matrix

        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"reduced_matrix.matrix must be 2D, got shape {X.shape}.")

        if len(dimensions) not in [2, 3]:
            raise ValueError("dimensions must contain either 2 or 3 dimensions.")

        if max(dimensions) >= X.shape[1]:
            raise ValueError(f"Requested dimension {max(dimensions)} but matrix only has {X.shape[1]} columns.")

        df = pd.DataFrame()

        x_dim = dimensions[0]
        y_dim = dimensions[1]

        df["x"] = X[:, x_dim]
        df["y"] = X[:, y_dim]

        if len(dimensions) == 3:
            z_dim = dimensions[2]
            df["z"] = X[:, z_dim]

        hover_cols = []
        row_info = None

        if reduced_matrix.row_info is not None:
            row_info = reduced_matrix.row_info.reset_index(drop=True)

            for col in row_info.columns:
                df[col] = row_info[col]

            hover_cols = list(row_info.columns)

        color = None

        if color_by is None:
            color = None

        elif isinstance(color_by, str):
            if row_info is None:
                raise ValueError("color_by as string requires reduced_matrix.row_info.")

            if color_by == "trajectory":
                mapping = self.dataset.get_cell_trajectory_mapping()
                values = row_info["barcode"].map(mapping)
                df["trajectory"] = values.astype(str)
                color = "trajectory"


            if color_by not in row_info.columns:
                raise ValueError(f"Column '{color_by}' not found in row_info. Available columns: {list(row_info.columns)}")

            values = row_info[color_by].reset_index(drop=True)
            n_unique = values.nunique(dropna=False)

            if n_unique > max_categories:
                df[f"{color_by}_code"] = pd.Categorical(values).codes
                df[color_by] = values.astype(str)
                color = f"{color_by}_code"

                if color_by not in hover_cols:
                    hover_cols.append(color_by)

            else:
                df[color_by] = values.astype(str)
                color = color_by

        else:
            labels = color_by

            if isinstance(labels, LabeledDenseMatrix):
                labels = labels.matrix

            if hasattr(labels, "to_numpy"):
                labels = labels.to_numpy()

            labels = np.asarray(labels).reshape(-1)

            if len(labels) != X.shape[0]:
                raise ValueError(f"Number of labels ({len(labels)}) does not match number of points ({X.shape[0]}).")

            n_unique = len(np.unique(labels))

            if n_unique > max_categories:
                df["label_code"] = pd.Categorical(labels).codes
                df["label"] = labels.astype(str)
                color = "label_code"

                if "label" not in hover_cols:
                    hover_cols.append("label")

            else:
                df["label"] = labels.astype(str)
                color = "label"

        plot_title = title if title is not None else reduced_matrix.name

        if len(dimensions) == 2:
            fig = px.scatter(df, x="x", y="y", color=color, hover_data=hover_cols, title=plot_title)
        else:
            fig = px.scatter_3d(df, x="x", y="y", z="z", color=color, hover_data=hover_cols, title=plot_title)

        fig.update_traces(marker=dict(size=size, opacity=opacity))
        fig.update_layout(template="plotly_white")

        return fig
    
    
    def plot_cluster_trajectories_over_time(
        self,
        cluster_assignment: pd.DataFrame = None,
        data: str = "expression",
        genes: list = None,
        aggregation: str = "mean",
        clustering: BaseCluster = None,
        reduction: BaseReduction = None,
        distance: BaseDistance = None,
        figsize=(10, 6),
    ):

        if aggregation not in ["mean", "sum", "median", "medoid"]:
            raise ValueError("aggregation must be 'mean', 'sum', 'median', or 'medoid'.")

        if cluster_assignment is None:
            if clustering is None:
                raise ValueError("Provide either cluster_assignment or clustering.")

            cluster_assignment = self.dataset.get_cluster_assignment(
                entity="trajectory",
                data=data,
                clustering=clustering,
                reduction=reduction,
            )


        if aggregation == "medoid":
            cluster_assignment = self.dataset.get_cluster_medoids(
                entity="trajectory",
                data=data,
                clustering=clustering,
                reduction=reduction,
                cluster_assignment=cluster_assignment,
                distance=distance,
            )

            id_col = "medoid_id"
        else:
            id_col = "trajectory_id"

        rows = []

        for _, row in cluster_assignment.iterrows():
            trajectory_id = row[id_col]
            cluster = row["cluster"]

            if data == "expression":
                matrix = self.dataset.get_trajectory_expression(trajectory_id)
            elif data == "ntr":
                matrix = self.dataset.get_trajectory_ntr(trajectory_id)
            else:
                raise ValueError("For this compact version use only 'expression' or 'ntr'.")

            if genes is not None:
                matrix = matrix.select_cols(gene_name=genes)

            X = matrix.to_np()

            if aggregation in ["mean", "medoid"]:
                values = X.mean(axis=1)
            elif aggregation == "sum":
                values = X.sum(axis=1)
            else:
                values = np.median(X, axis=1)

            trajectory = self.dataset.get_trajectory(trajectory_id)

            for time, value in zip(trajectory.index, values):
                rows.append({
                    "cluster": cluster,
                    "trajectory_id": trajectory_id,
                    "time": time,
                    "time_num": int(str(time).replace("h", "")),
                    "value": value,
                })

        df = pd.DataFrame(rows)

        if aggregation == "medoid":
            cluster_time = df.sort_values(["cluster", "time_num"])
        else:
            cluster_time = (
                df.groupby(["cluster", "time", "time_num"])["value"]
                .mean()
                .reset_index()
                .sort_values(["cluster", "time_num"])
            )

        fig = px.line(
            cluster_time,
            x="time_num",
            y="value",
            color="cluster",
            markers=True,
            title=f"{aggregation.capitalize()} {data} trajectory per cluster",
            hover_data=["time"]
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=f"{aggregation.capitalize()} {data}",
            template="plotly_white",
            legend_title="Cluster",
        )

        return fig, cluster_time
    
    
    def plot_cluster_value_distribution(
        self,
        cluster_assignment: pd.DataFrame,
        entity: str,
        data: str = "expression",
        mode: str = "value",
        aggregation: str = "mean",
        log1p: bool = True,
        bins: int = 50,
        n_cols: int = 4,
        figsize_per_plot=(4, 3),
    ):
        if self.dataset is None:
            raise ValueError("This plot requires a ScifateDataset.")

        X, info, label = self.dataset._get_entity_vectors(
            entity=entity,
            data=data,
        )

        cluster_assignment = cluster_assignment.copy().reset_index(drop=True)
        clusters = sorted(cluster_assignment["cluster"].unique())

        n_rows = int(np.ceil(len(clusters) / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
            sharex=True,
            sharey=True,
        )

        axes = np.asarray(axes).reshape(-1)

        plot_rows = []

        for ax, cluster in zip(axes, clusters):
            idx = np.where(cluster_assignment["cluster"].values == cluster)[0]

            if mode == "value":
                if scipy.sparse.issparse(X):
                    values = X[idx].data
                else:
                    values = np.asarray(X[idx]).ravel()

                x_label = f"{data} value"

            elif mode == "entity":
                X_cluster = X[idx]

                if scipy.sparse.issparse(X_cluster):
                    if aggregation == "mean":
                        values = np.asarray(X_cluster.mean(axis=1)).ravel()
                    elif aggregation == "sum":
                        values = np.asarray(X_cluster.sum(axis=1)).ravel()
                    else:
                        raise ValueError("median is not supported for sparse matrices.")
                else:
                    if aggregation == "mean":
                        values = X_cluster.mean(axis=1)
                    elif aggregation == "sum":
                        values = X_cluster.sum(axis=1)
                    else:
                        values = np.median(X_cluster, axis=1)

                x_label = f"{aggregation} {data}"

            else:
                raise ValueError("mode must be 'value' or 'entity'.")

            if log1p:
                values = np.log1p(values)
                x_label = f"log1p({x_label})"

            ax.hist(values, bins=bins)
            ax.set_title(f"Cluster {cluster}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", labelbottom=True)
            ax.tick_params(axis="y", labelleft=True)

            for value in values:
                plot_rows.append({
                    "cluster": cluster,
                    "value": value,
                })

        for ax in axes[len(clusters):]:
            ax.axis("off")

        fig.suptitle(
            f"{data.capitalize()} distribution per {entity} cluster",
            y=1.02
        )

        fig.tight_layout()

        return fig, pd.DataFrame(plot_rows)
    
    
    
    def plot_cluster_network(
        self,
        cluster_distance_matrix,
        k_neighbors: int = 2,
        show_edge_labels: bool = False,
        seed: int = 42,
        figsize=(12, 8),
    ):
        import networkx as nx

        D = np.asarray(cluster_distance_matrix.matrix, dtype=np.float32)
        labels = cluster_distance_matrix.info["cluster_label"].values

        G = nx.Graph()

        for label in labels:
            G.add_node(label)

        edges = []

        for i in range(D.shape[0]):
            distances = D[i].copy()
            distances[i] = np.inf

            neighbors = np.argsort(distances)[:k_neighbors]

            for j in neighbors:
                d = D[i, j]
                if np.isfinite(d):
                    edges.append((labels[i], labels[j], d))

        max_d = max(d for _, _, d in edges)
        min_d = min(d for _, _, d in edges)

        for u, v, d in edges:
            similarity = 1.0 - ((d - min_d) / (max_d - min_d + 1e-8))

            G.add_edge(
                u,
                v,
                distance=d,
                similarity=similarity,
                width=1.0 + 5.0 * similarity
            )

        pos = nx.spring_layout(
            G,
            seed=seed,
            weight="similarity",
            k=1.2,
            iterations=200
        )

        fig, ax = plt.subplots(figsize=figsize)

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=900,
            ax=ax
        )

        nx.draw_networkx_labels(
            G,
            pos,
            font_size=10,
            ax=ax
        )

        nx.draw_networkx_edges(
            G,
            pos,
            width=[G[u][v]["width"] for u, v in G.edges()],
            alpha=0.6,
            ax=ax
        )

        if show_edge_labels:
            edge_labels = {
                (u, v): f"{G[u][v]['distance']:.1f}"
                for u, v in G.edges()
            }

            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_size=7,
                ax=ax
            )

        ax.set_title("Cluster Network Based on Cluster Distances")
        ax.axis("off")

        fig.tight_layout()

        return fig, G