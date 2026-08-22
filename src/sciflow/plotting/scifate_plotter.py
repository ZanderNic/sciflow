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
from plotly.subplots import make_subplots
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
        matrix: LabeledDenseMatrix | LabeledDistanceMatrix,
        figsize=(10, 8),
        cmap="viridis",
        annot=True,
        fmt=None,
        linewidths=0.5,
        linecolor="white",
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

        M = matrix.matrix

        if hasattr(M, "toarray"):
            M = M.toarray()

        M = np.asarray(M)

        if fmt is None:
            if np.allclose(M, M.astype(int)):
                fmt = ".0f"
            else:
                fmt = ".2f"

        x_labels = True
        y_labels = True

        if hasattr(matrix, "col_info") and matrix.col_info is not None:
            if hasattr(matrix, "col_label") and matrix.col_label in matrix.col_info.columns:
                x_labels = matrix.col_info[matrix.col_label].astype(str).tolist()

        if hasattr(matrix, "row_info") and matrix.row_info is not None:
            if hasattr(matrix, "row_label") and matrix.row_label in matrix.row_info.columns:
                y_labels = matrix.row_info[matrix.row_label].astype(str).tolist()

        if isinstance(matrix, LabeledDistanceMatrix):
            if hasattr(matrix, "info") and matrix.info is not None:
                if hasattr(matrix, "label") and matrix.label in matrix.info.columns:
                    labels = matrix.info[matrix.label].astype(str).tolist()
                    x_labels = labels
                    y_labels = labels

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            M,
            cmap=cmap,
            ax=ax,
            annot=annot,
            fmt=fmt,
            linewidths=linewidths,
            linecolor=linecolor,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cbar=True,
        )

        ax.set_title(matrix.name)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
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
        feature: str = None,
        feature_aggregation: str = "mean",                      # "mean"
        dimensions = (0, 1),
        title: str = None,
        opacity: float = 0.8,
        size: int = 5,
        continuous: bool = False,
    ):

        if self.dataset is not None and entity is not None and data is not None and reduction is not None:
            reduced_matrix = self.dataset.get_reduced_matrix(entity=entity, data=data, reduction=reduction)

        if reduced_matrix is None:
            raise ValueError("please provide either a matrix or a ScifateDataset at initialization and the keys to the reduced matrix here")
        
        X = reduced_matrix.matrix.to_numpy() if hasattr(reduced_matrix.matrix, "to_numpy") else np.asarray(reduced_matrix.matrix)       # we need this here to handel the costum classes prvided by the package

        if len(dimensions) not in [2, 3]:
            raise ValueError("dimensions must contain either 2 or 3 dimensions.")

        if max(dimensions) >= X.shape[1]:
            raise ValueError(f"Requested dimension {max(dimensions)} but matrix only has {X.shape[1]} columns.")

        df = pd.DataFrame(X[:, dimensions], columns=dimensions)

        hover_cols = []
        row_info = None

        if reduced_matrix.row_info is not None:
            row_info = reduced_matrix.row_info.reset_index(drop=True)

            for col in row_info.columns:
                df[col] = row_info[col]

            hover_cols = list(row_info.columns)

        color = None

        if isinstance(color_by, str):                                                                       # here we handel if we color with information that is given in the row_info df 
            if row_info is None:
                raise ValueError("color_by as string requires reduced_matrix.row_info.")

            if color_by not in row_info.columns:
                raise ValueError(f"Column '{color_by}' not found in row_info. Available columns: {list(row_info.columns)}")

            values = row_info[color_by].reset_index(drop=True)
            values = values.astype(float if continuous else str)
            df[color_by] = values
            color = color_by

            if not continuous and color_by not in hover_cols:
                hover_cols.append(color_by)
        
        elif color_by is not None:                                                                          # here we have a direct mapping of colors to points in the form of for example a np array 
            labels = color_by.matrix if isinstance(color_by, LabeledDenseMatrix) else color_by
            labels = labels.to_numpy() if hasattr(labels, "to_numpy") else labels
            labels = np.asarray(labels).reshape(-1)

            if len(labels) != len(X):
                raise ValueError("Number of labels must match number of points.")

            labels = labels.astype(float if continuous else str)

            df["label"] = labels
            color = "label"
        
        elif isinstance(feature, str):                                                                      # here we handle if the user wants to plot by a feature in the original matrix
            values = self._get_feature_values(
                feature=feature,
                entity=entity,
                data=data,
                aggregation= feature_aggregation,
            )

            df[feature] = values
            color = feature
            continuous = True

        plot_title = title if title is not None else reduced_matrix.name

        if len(dimensions) == 2:
            fig = px.scatter(df, x = dimensions[0], y=dimensions[1], color=color, hover_data=hover_cols, title=plot_title)
        else:
            fig = px.scatter_3d(df, x=dimensions[0], y=dimensions[1], z=dimensions[2], color=color, hover_data=hover_cols, title=plot_title)

        fig.update_traces(marker=dict(size=size, opacity=opacity))
        fig.update_layout(template="plotly_white")

        return fig
    

    def plot_reduced_comparison(
        self,
        reduced_matrix,
        compare_matrix=None,
        color_by=None,
        compare_color_by=None,
        feature=None,
        compare_feature=None,
        entity=None,
        data=None,
        feature_aggregation="mean",
        dimensions=(0, 1),
        titles=None,
        opacity=0.8,
        size=5,
        continuous=False,
    ):
        """
            Plot two reduced representations side by side.

            The second plot can use:
            - another reduced matrix,
            - another coloring,
            - another feature,
            - or any combination of these.

            If no comparison coloring/feature is given, the coloring of the
            first plot is reused.
        """

        # use the same matrix if no second matrix is given
        if compare_matrix is None:
            compare_matrix = reduced_matrix

        # reuse first coloring if no comparison coloring is specified
        if compare_color_by is None and compare_feature is None:
            compare_color_by = color_by
            compare_feature = feature

        # create both plots using the existing plotting function
        fig_left = self.plot_reduced_matrix(
            reduced_matrix=reduced_matrix,
            entity=entity,
            data=data,
            color_by=color_by,
            feature=feature,
            feature_aggregation=feature_aggregation,
            dimensions=dimensions,
            opacity=opacity,
            size=size,
            continuous=continuous,
        )

        fig_right = self.plot_reduced_matrix(
            reduced_matrix=compare_matrix,
            entity=entity,
            data=data,
            color_by=compare_color_by,
            feature=compare_feature,
            feature_aggregation=feature_aggregation,
            dimensions=dimensions,
            opacity=opacity,
            size=size,
            continuous=continuous,
        )

        if titles is None:
            titles = (
                reduced_matrix.name,
                compare_matrix.name,
            )

        # 2D or 3D subplot type
        subplot_type = "scene" if len(dimensions) == 3 else "xy"

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": subplot_type}, {"type": subplot_type}]],
            subplot_titles=titles,
            horizontal_spacing=0.08,
        )

        # add traces from both figures
        for trace in fig_left.data:
            fig.add_trace(trace, row=1, col=1)

        for trace in fig_right.data:
            fig.add_trace(trace, row=1, col=2)

        fig.update_layout(
            template="plotly_white",
            showlegend=True,
        )

        return fig
    
    
    def plot_cluster_trajectories_over_time(
        self,
        cluster_assignment: pd.DataFrame = None,
        data: str = "expression",
        representative_aggregation: str = "mean",
        feature_aggregation: str = "mean",
        genes: list = None,
        clustering: BaseCluster = None,
        reduction: BaseReduction = None,
        distance: BaseDistance = None,
        figsize=(10, 6),
    ):
        """
            Plot one representative trajectory per cluster over time.

            representative_aggregation:
                How trajectories inside one cluster are summarized.

            feature_aggregation:
                How genes/features are summarized at each time point for plotting.
        """

        representatives = self.dataset.create_cluster_representatives(
            cluster_assignment=cluster_assignment,
            data=data,
            entity="trajectory",
            clustering=clustering,
            reduction=reduction,
            aggregation=representative_aggregation,
            distance=distance,
        )

        X = np.asarray(representatives.matrix)
        col_info = representatives.col_info.copy().reset_index(drop=True)

        if "time" not in col_info.columns:
            raise ValueError("representatives.col_info must contain a 'time' column. For trajectory representatives, col_info should describe time x gene features.")

        if genes is not None:
            if "gene_name" not in col_info.columns:
                raise ValueError("Filtering by genes requires representatives.col_info to contain a 'gene_name' column.")

            gene_mask = col_info["gene_name"].isin(genes).to_numpy()

            if not gene_mask.any():
                raise ValueError(f"None of the requested genes were found: {genes}")

            X = X[:, gene_mask]
            col_info = col_info.loc[gene_mask].reset_index(drop=True)

        time_labels = sorted(
            col_info["time"].unique(),
            key=lambda t: int(str(t).replace("h", ""))
        )

        rows = []

        for i, row in representatives.row_info.iterrows():
            cluster = row["cluster"]

            for time in time_labels:
                time_mask = (col_info["time"] == time).to_numpy()
                values = X[i, time_mask]

                value = self._aggregate_feature_values(
                    values=values,
                    method=feature_aggregation,
                )

                rows.append({
                    "cluster": str(cluster),
                    "time": time,
                    "time_num": int(str(time).replace("h", "")),
                    "value": value,
                    "representative_aggregation": representative_aggregation,
                    "feature_aggregation": feature_aggregation,
                    "genes": "all" if genes is None else ", ".join(genes),
                })

        cluster_time = (
            pd.DataFrame(rows)
            .sort_values(["cluster", "time_num"])
            .reset_index(drop=True)
        )

        fig = px.line(
            cluster_time,
            x="time_num",
            y="value",
            color="cluster",
            markers=True,
            title=(
                f"{representative_aggregation.capitalize()} cluster representatives "
                f"with {feature_aggregation} feature aggregation"
            ),
            hover_data=[
                "time",
                "genes",
                "representative_aggregation",
                "feature_aggregation",
            ],
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=f"{feature_aggregation.capitalize()} {data}",
            template="plotly_white",
            legend_title="Cluster",
            width=figsize[0] * 100,
            height=figsize[1] * 100,
        )

        return fig, cluster_time, representatives
    
    
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
        figsize_per_plot=(4, 3)
    ):
        if self.dataset is None:
            raise ValueError("This plot requires a ScifateDataset.")

        X, _, _, _, _ = self.dataset._get_entity_vectors(
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
    
    
    
    
    ### Helper
    
    def _get_feature_values(
        self,
        feature,
        entity,
        data,
        aggregation=None,
    ):
        X, _, _, col_info, col_label = self.dataset._get_entity_vectors(
            entity=entity,
            data=data,
        )

        if entity == "trajectory" and aggregation is not None:
            mask = col_info["gene_name"].eq(feature)
        else:
            mask = col_info[col_label].eq(feature)

        if not mask.any():
            raise ValueError(f"Feature '{feature}' not found.")

        values = X[:, mask.to_numpy()]

        if hasattr(values, "toarray"):
            values = values.toarray()

        values = np.asarray(values)

        if aggregation is not None:
            return getattr(np, aggregation)(values, axis=1)

        return values[:, 0] if values.shape[1] == 1 else values
    
    
    @staticmethod
    def _aggregate_feature_values(
        values,
        method: str = "mean",
    ):
        values = np.asarray(values)

        if method == "mean":
            return values.mean()

        if method == "median":
            return np.median(values)

        if method == "sum":
            return values.sum()

        if method == "max":
            return values.max()

        raise ValueError(
            "method must be 'mean', 'median', 'sum', or 'max'."
        )