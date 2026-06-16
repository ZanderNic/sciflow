# std lib imports
from pathlib import Path
from unittest.mock import Base
import warnings
from typing import List

# 3-party import
import pandas as pd
from scipy.io import mmread

# projekt imports
from sciflow.cluster import BaseCluster
from sciflow.data import LabeledSparseMatrix, LabeledDistanceMatrix, Dataset, LabeledDenseMatrix
from sciflow.distance import *
from sciflow.reduction.base_reduction import BaseReduction



class ScifateDataset(Dataset):
    """
    
    """
    
    
    def __init__(
        self, 
        data: LabeledSparseMatrix, 
        ntr: LabeledSparseMatrix  = None, 
        cell_info=None, 
        gene_info=None, 
        name=None,
        trajectories = None
    ):
        super().__init__(
            data=data,
            sample_info=cell_info,
            feature_info=gene_info,
            name=name
        )

        self.trajectories = trajectories
        self.ntr = ntr
   
        self.distance_matrices = {}
        self.distance_scores = {}
        
        self.cluster_assignments = {}
        
        self.reduced_matrices = {}
        self.reductions = {}
        

    #***# @propertys #***#***#***#***#***#***#***#***#***#***#***#***#
    @property
    def cell_info(self):
        return self.sample_info


    @property
    def gene_info(self):
        return self.feature_info


    @property
    def expression_matrix(self) -> LabeledSparseMatrix:
        return self.data
    
    
    #***# trajectory handling #***#***#***#***#***#***#***#***#***#***#***#***#
    
    def get_trajectory_id(self, barcode):
        if self.trajectories is None:
            raise ValueError("No trajectories available.")

        mask = self.trajectories.eq(barcode)

        if not mask.any().any():
            raise ValueError(f"No trajectory found for barcode: {barcode}")

        return mask.any(axis=1).idxmax()
    
    
    def get_trajectory(self, trajectory_id):
        if self.trajectories is None:
            raise ValueError("No trajectories available.")

        trajectory = self.trajectories.loc[trajectory_id]
        sorted_columns = sorted(trajectory.index,key=lambda x: int(x.replace("h", "")))

        return trajectory[sorted_columns]
    
    
    def get_trajectory_cells(self, trajectory_id):
        trajectory = self.get_trajectory(trajectory_id)
        return trajectory.dropna().tolist()
    
    
    def get_trajectory_expression(self, trajectory_id: int = None, barcode: str = None) -> LabeledSparseMatrix:
        if barcode and trajectory_id is None:
            trajectory_id = self.get_trajectory_id(barcode=barcode)
            
        cells = self.get_trajectory_cells(trajectory_id)
        return self.expression_matrix.select_rows(barcode=cells)
    
     
    def get_trajectory_ntr(self, trajectory_id: int = None, barcode: str = None) -> LabeledSparseMatrix:
        if self.ntr is None: raise ValueError("No NTR matrix available.")
        
        if barcode and trajectory_id is None:
            trajectory_id = self.get_trajectory_id(barcode=barcode)
            
            
        cells = self.get_trajectory_cells(trajectory_id)
        return self.ntr.select_rows(barcode=cells)
    
    
    def _get_trajectory_vectors(
        self,
        data: str       #  "expression" | "ntr"
    ):
        vectors = []
        for trajectory_id in self.trajectories.index: 
            trajectory_matrix = None
            if data == "expression":
                trajectory_matrix = self.get_trajectory_expression(trajectory_id)
            elif data == "ntr":
                trajectory_matrix = self.get_trajectory_ntr(trajectory_id)

            vector = trajectory_matrix.to_np().reshape(-1)

            vectors.append(vector)

        return np.stack(vectors)
    
    def get_cell_trajectory_mapping(self):
        if self.trajectories is None:
            raise ValueError("No trajectories available.")

        mapping = {}

        for trajectory_id in self.trajectories.index:
            cells = self.get_trajectory_cells(trajectory_id)

            for barcode in cells:
                mapping[barcode] = trajectory_id

        return mapping
    
    def add_trajectory_info(self):
        mapping = self.get_cell_trajectory_mapping()

        self.cell_info["trajectory"] = (
            self.cell_info["barcode"]
            .map(mapping)
        )
    
    
    #***# distance matrix create #***#***#***#***#***#***#***#***#***#***#***#***#
    
    
    def create_distance_matrix(
        self,
        matrix: LabeledSparseMatrix | LabeledDistanceMatrix = None,
        entity: str = "cell",                                                # "cell" | "gene" | "trajectory" 
        data: str= "expression",                                             # "expression" | "ntr" | "new_rna" | "old_rna"
        distance: BaseDistance = None,
        reduction: BaseReduction = None,
        save: bool = True
    ) -> LabeledDistanceMatrix:
        
        if distance is None:
            distance = CosineDistance()

        if entity not in ["cell", "gene", "trajectory"]:
                raise ValueError("entity must be 'cell', 'gene', or 'trajectory'.")

        if data not in ["expression", "ntr", "new_rna", "old_rna"]:
            raise ValueError("data must be 'expression', 'ntr', 'new_rna' or 'old_rna'.")

        if matrix is not None:
        #     if not self._is_dataset_matrix_view(matrix):
        #         raise ValueError("The provided matrix is not a view of the orignal matrix")
            X, info, label =  matrix.to_np(), matrix.row_info, matrix.row_label
        else:
            X, info, label = self._get_entity_vectors(entity=entity,data=data,)

        if reduction is not None:
            X = reduction.fit_transform(X)

        dist_m = distance.pairwise(X)

        dist_matrix = LabeledDistanceMatrix(
            matrix=dist_m,
            info=info,
            name=f"{entity}_{data}_{distance.name}_matrix",
            label=label,
            metadata={
                "entity": entity,
                "data": data,
                "distance": distance.name,
            },
            validate_diagonal=True,
            validate_square=True,
            validate_symmetric=True
        )

        if save:
            key = (entity, data, distance.name)
            self.distance_matrices[key] = dist_matrix

        return dist_matrix
        

    def get_distance_matrix(
        self,
        entity: str,                    #
        data: str,                      #
        distance: BaseDistance,         #
    ) -> LabeledDistanceMatrix:
        return self.distance_matrices[(entity, data, distance.name if isinstance(distance, BaseDistance) else distance)]
    
    
    def create_distance_scores(
        self,
        entity: str,
        data: str,
        distance: BaseDistance,
        sort_by: str = "mean_distance",
        ascending: bool = False,
        save: bool = True,
    ) -> pd.DataFrame:


        try:
            dist_matrix = self.get_distance_matrix(
                entity=entity,
                data=data,
                distance=distance,
            )
        except KeyError:
            dist_matrix = self.create_distance_matrix(
                entity=entity,
                data=data,
                distance=distance,
                save=True,
            )
        
        D = np.asarray(dist_matrix.matrix)

        D_no_diag_nan = np.where(np.eye(D.shape[0], dtype=bool), np.nan, D)
        D_no_diag_inf = np.where(np.eye(D.shape[0], dtype=bool), np.inf, D)

        mean_distance = np.nanmean(D_no_diag_nan, axis=1)
        median_distance = np.nanmedian(D_no_diag_nan, axis=1)
        nearest_neighbor_distance = D_no_diag_inf.min(axis=1)

        scores = dist_matrix.info.copy()
        scores["mean_distance"] = mean_distance
        scores["median_distance"] = median_distance
        scores["nearest_neighbor_distance"] = nearest_neighbor_distance
        scores = scores.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

        if save:
            key = (entity, data, distance.name)
            self.distance_scores[key] = scores

        return scores
    
    
    def create_gene_variability_scores(
        self,
        data: str = "expression",
        sort_by: str = "variance",
        ascending: bool = False,
        save: bool = True,
    ) -> pd.DataFrame:

        X, info, label = self._get_entity_vectors(
            entity="gene",
            data=data,
        )

        if scipy.sparse.issparse(X):
            variance = np.asarray(X.power(2).mean(axis=1) - np.square(X.mean(axis=1))).ravel()
            mean = np.asarray(X.mean(axis=1)).ravel()
        else:
            variance = np.var(X, axis=1)
            mean = np.mean(X, axis=1)

        scores = info.copy()
        scores["mean_expression"] = mean
        scores["variance"] = variance
        scores = scores.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

        if save:
            self.gene_scores = scores

        return scores
    
    
    #***# data reduction functions #***#***#***#***#***#***#***#***#***#***#***#***#
    
    def create_reduced_matrix(
        self,
        reduction: BaseReduction,
        entity: str,                                                # "cell" | "gene" | "trajectory" 
        data: str,                                                  # "expression" | "ntr" | "new_rna" | "old_rna"
        save: bool = True,
        save_model: bool = True
    ) -> LabeledDenseMatrix:
        
        if entity not in ["cell", "gene", "trajectory"]:
                raise ValueError("entity must be 'cell', 'gene', or 'trajectory'.")

        if data not in ["expression", "ntr", "new_rna", "old_rna"]:
            raise ValueError("data must be 'expression', 'ntr', 'new_rna' or 'old_rna'.")

        X, info, label = self._get_entity_vectors(entity=entity, data=data,)
        X = reduction.fit_transform(X)
        
        reduced_matrix = LabeledDenseMatrix(
            matrix = X,
            row_info = info,
            row_label = label,
            name = f"{entity}_{data}_{reduction.name}",
            col_info = pd.DataFrame(reduction.feature_names)
        )
        
        if save:
            key = (entity, data, reduction.name)
            self.reduced_matrices[key] = reduced_matrix
        
        if save_model:
            key = (entity, data, reduction.name)
            self.reductions[key] = reduced_matrix
        
        return reduced_matrix
    
    
    def get_reduced_matrix(
        self,
        entity: str,                    
        data: str,                      
        reduction: BaseReduction,         
    ) -> LabeledDenseMatrix:
        return self.reduced_matrices[(entity, data, reduction.name if isinstance(reduction, BaseReduction) else distance)]
    
    
    #***# data clustering functions #***#***#***#***#***#***#***#***#***#***#***#***#
    
    def create_clustering(
        self,
        clustering: BaseCluster,
        entity: str,
        data: str,
        save: bool = True,
        reduction: BaseReduction = None,
    ) -> pd.DataFrame:

        if entity not in ["cell", "gene", "trajectory"]:
            raise ValueError("entity must be 'cell', 'gene', or 'trajectory'.")

        if data not in ["expression", "ntr", "new_rna", "old_rna"]:
            raise ValueError("data must be 'expression', 'ntr', 'new_rna' or 'old_rna'.")

        X, info, label = self._get_entity_vectors(entity=entity, data=data)

        reduction_name = "no_reduction"

        if reduction is not None:
            X = reduction.fit_transform(X)
            reduction_name = reduction.name

        if isinstance(X, (LabeledDenseMatrix, LabeledSparseMatrix)):
            X = X.matrix

        if scipy.sparse.issparse(X):
            X = X.toarray()

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X = np.asarray(X, dtype=np.float32)

        assignment = clustering.fit_predict(X)
        assignment = np.asarray(assignment).reshape(-1, 1)

        cluster_df = info.copy()
        cluster_df["cluster"] = assignment

        if save:
            key = (entity, data, clustering.name, reduction_name)
            self.cluster_assignments[key] = cluster_df

        return cluster_df


    def get_cluster_assignment(
        self,
        entity: str,                    
        data: str,
        clustering: BaseCluster,                      
        reduction: BaseReduction = None,         
    ) -> LabeledDenseMatrix:
        reduction_name =  reduction.name if isinstance(reduction, BaseReduction) else (reduction if isinstance(reduction, str) else "no_reduction")
        cluster_name = clustering.name if isinstance(clustering, BaseCluster) else clustering
        return self.cluster_assignments[(entity, data, cluster_name,  reduction_name)]    
        
    
    def get_cluster_medoids(
        self,
        entity: str,
        data: str,
        clustering: BaseCluster = None, 
        reduction: BaseReduction = None,
        cluster_assignment: pd.DataFrame = None,
        distance: BaseDistance = None,
        save_distance_matrix: bool = True,
    ) -> pd.DataFrame:

        if distance is None:
            distance = CosineDistance()

        if entity not in ["cell", "gene", "trajectory"]:
            raise ValueError("entity must be 'cell', 'gene', or 'trajectory'.")

        if data not in ["expression", "ntr", "new_rna", "old_rna"]:
            raise ValueError("data must be 'expression', 'ntr', 'new_rna' or 'old_rna'.")

        if cluster_assignment is None:
            cluster_assignment = self.get_cluster_assignment(entity, data, clustering, reduction)

        if "cluster" not in cluster_assignment.columns:
            raise ValueError("cluster_assignment must contain a 'cluster' column.")

        try:
            dist_matrix = self.get_distance_matrix(
                entity=entity,
                data=data,
                distance=distance,
            )
        except KeyError:
            dist_matrix = self.create_distance_matrix(
                entity=entity,
                data=data,
                distance=distance,
                save=save_distance_matrix,
            )

        D = np.asarray(dist_matrix.matrix)
        info = dist_matrix.info.copy()

        id_column = {"cell": "barcode", "gene": "gene_name", "trajectory": "trajectory_id"}[entity]

        if id_column not in cluster_assignment.columns:
            raise ValueError(
                f"cluster_assignment must contain '{id_column}' for entity='{entity}'."
            )

        if id_column not in info.columns:
            raise ValueError(
                f"distance matrix info must contain '{id_column}' for entity='{entity}'."
            )

        id_to_pos = {value: idx for idx, value in enumerate(info[id_column].values)}

        rows = []

        for cluster in sorted(cluster_assignment["cluster"].unique()):
            cluster_df = cluster_assignment[cluster_assignment["cluster"] == cluster]

            cluster_ids = cluster_df[id_column].values

            positions = [
                id_to_pos[x]
                for x in cluster_ids
                if x in id_to_pos
            ]

            if len(positions) == 0:
                continue

            if len(positions) == 1:
                medoid_pos = positions[0]
                mean_distance = 0.0

            else:
                D_cluster = D[np.ix_(positions, positions)]

                D_no_diag = np.where(
                    np.eye(D_cluster.shape[0], dtype=bool),
                    np.nan,
                    D_cluster
                )

                mean_distances = np.nanmean(D_no_diag, axis=1)

                local_medoid_idx = int(np.nanargmin(mean_distances))
                medoid_pos = positions[local_medoid_idx]
                mean_distance = mean_distances[local_medoid_idx]

            medoid_info = info.iloc[medoid_pos].to_dict()

            row = {
                "cluster": cluster,
                "cluster_size": len(positions),
                "medoid_index": medoid_pos,
                "medoid_id": info.iloc[medoid_pos][id_column],
                "mean_distance_to_cluster": mean_distance,
            }

            row.update(medoid_info)
            rows.append(row)

        return pd.DataFrame(rows)
        
    
    def cluster_distance_matrix(
        self,
        clustering: BaseCluster,
        entity: str,
        data: str,
        reduction: BaseReduction = None,
        cluster_assignment: pd.DataFrame = None,
        distance: BaseDistance = None
    ):
        
        if distance is None:
            distance = CosineDistance()

        if entity not in ["cell", "gene", "trajectory"]:
            raise ValueError("entity must be 'cell', 'gene', or 'trajectory'.")

        if data not in ["expression", "ntr", "new_rna", "old_rna"]:
            raise ValueError("data must be 'expression', 'ntr', 'new_rna' or 'old_rna'.")

        if cluster_assignment is None:
            cluster_assignment = self.get_cluster_assignment(entity, data, clustering, reduction)

        if "cluster" not in cluster_assignment.columns:
            raise ValueError("cluster_assignment must contain a 'cluster' column.")

        try:
            dist_matrix = self.get_distance_matrix(
                entity=entity,
                data=data,
                distance=distance,
            )
        except KeyError:
            dist_matrix = self.create_distance_matrix(
                entity=entity,
                data=data,
                distance=distance,
                save=True,
            )
        
        D = np.asarray(dist_matrix.matrix)

        clusters = cluster_assignment["cluster"].values
        unique_clusters = sorted(np.unique(clusters))

        cluster_D = np.zeros((len(unique_clusters), len(unique_clusters)), dtype=np.float32)

        for i, c1 in enumerate(unique_clusters):
            idx1 = np.where(clusters == c1)[0]
            for j, c2 in enumerate(unique_clusters):
                idx2 = np.where(clusters == c2)[0]
                cluster_D[i, j] = D[np.ix_(idx1, idx2)].mean()


        cluster_info = pd.DataFrame({
            "cluster": unique_clusters,
            "cluster_label": [f"C{c}" for c in unique_clusters],
        })


        clustering_name = clustering.name if isinstance(clustering, BaseCluster) else clustering
        reduction_name = reduction.name if isinstance(reduction, BaseReduction) else (reduction if reduction is not None else "no_reduction")

        cluster_matrix = LabeledDistanceMatrix(
            matrix=cluster_D,
            info=cluster_info,
            label="cluster_label",
            name=f"{entity}_{data}_{clustering_name}_cluster_distance_matrix",
            metadata={
                "entity": entity,
                "data": data,
                "clustering": clustering_name,
                "distance": distance.name,
                "reduction": reduction_name,
            },
            validate_diagonal=False,
        )

        return cluster_matrix
        
        
    def intra_cluster_distance_matrix(
        self,
        clustering: BaseCluster,
        entity: str,
        data: str,
        distance: BaseDistance = None,
        reduction: BaseReduction = None,
        cluster_assignment: pd.DataFrame = None,
    ) -> LabeledDistanceMatrix:
        
        if distance is None:
            distance = CosineDistance()

        if entity not in ["cell", "gene", "trajectory"]:
            raise ValueError("entity must be 'cell', 'gene', or 'trajectory'.")

        if data not in ["expression", "ntr", "new_rna", "old_rna"]:
            raise ValueError("data must be 'expression', 'ntr', 'new_rna' or 'old_rna'.")

        if cluster_assignment is None:
            cluster_assignment = self.get_cluster_assignment(
                entity=entity,
                data=data,
                clustering=clustering,
                reduction=reduction,
            )

        if "cluster" not in cluster_assignment.columns:
            raise ValueError("cluster_assignment must contain a 'cluster' column.")

        try:
            dist_matrix = self.get_distance_matrix(
                entity=entity,
                data=data,
                distance=distance,
            )
        except KeyError:
            dist_matrix = self.create_distance_matrix(
                entity=entity,
                data=data,
                distance=distance,
                save=True,
            )
        
        D = np.asarray(dist_matrix.matrix)

        clusters = cluster_assignment["cluster"].values
        unique_clusters = sorted(np.unique(clusters))

        intra_D = np.zeros(
            (len(unique_clusters), len(unique_clusters)),
            dtype=np.float32
        )

        for i, cluster in enumerate(unique_clusters):
            idx = np.where(clusters == cluster)[0]

            if len(idx) <= 1:
                intra_D[i, i] = np.nan
                continue

            D_cluster = D[np.ix_(idx, idx)]
            tri = np.triu_indices(len(idx), k=1)

            intra_D[i, i] = D_cluster[tri].mean()

        cluster_info = pd.DataFrame({
            "cluster": unique_clusters,
            "cluster_label": [f"C{c}" for c in unique_clusters],
            "cluster_size": [
                int(np.sum(clusters == c))
                for c in unique_clusters
            ],
        })

        clustering_name = clustering.name if isinstance(clustering, BaseCluster) else clustering
        reduction_name = reduction.name if isinstance(reduction, BaseReduction) else (reduction if reduction is not None else "no_reduction")

        cluster_matrix = LabeledDistanceMatrix(
            matrix=intra_D,
            info=cluster_info,
            label="cluster_label",
            name=f"{entity}_{data}_{clustering_name}_intra_cluster_distance_matrix",
            metadata={
                "entity": entity,
                "data": data,
                "clustering": clustering_name,
                "distance": distance.name,
                "reduction": reduction_name,
            },
            validate_diagonal=False,
        )

        return cluster_matrix
    
    
    #***# info functions #***#***#***#***#***#***#***#***#***#***#***#***#
        
    def summary(self):
        n_cells, n_genes = self.data.shape if self.data is not None else (None, None)

        return str(
            f"ScifateDataset(\n"
            f"  name = {self.name},\n"
            f"  shape = {n_cells} cells x {n_genes} genes,\n"
            f"  ntr = {'yes' if self.ntr is not None else 'no'},\n"
            f"  cell_info = {'yes' if self.cell_info is not None else 'no'},\n"
            f"  gene_info = {'yes' if self.gene_info is not None else 'no'}\n"
            f"  trajectory = {'yes' if self.trajectories is not None else 'no'}\n"
            f")"
        )

    def __str__(self):
        return self.summary()


    def valid_dist_matrix(self) -> List[str]:
        return list(self.distance_matrices.keys())



    def _is_dataset_matrix_view(self, matrix: LabeledSparseMatrix) -> bool:
        if matrix is None:
            return False

        valid_matrices = [
            self.expression_matrix._matrix,
            self.expression_matrix.T._matrix
        ]

        if self.ntr is not None:
            valid_matrices.append(self.ntr._matrix)
            valid_matrices.append(self.ntr.T._matrix)

        return any(matrix._matrix is valid_matrix for valid_matrix in valid_matrices)


    def _get_entity_vectors(
        self, 
        entity: str,                            # "cell" | "gene" | "trajectory" 
        data: str                               # "expression" | "ntr" | "new_rna" | "old_rna"
    ):
        
        if data not in ["expression", "ntr", "new_rna", "old_rna"]:
            raise ValueError("data must be 'expression', 'ntr', 'new_rna', or 'old_rna'.")

        if data in ["ntr", "new_rna", "old_rna"] and self.ntr is None:
            raise ValueError("No NTR matrix available.")
        
        if data == "expression":
            matrix = self.expression_matrix
        elif data == "ntr":
            matrix = self.ntr
        elif data == "new_rna":
            matrix = self.expression_matrix * self.ntr
        elif data == "old_rna":
            matrix = self.expression_matrix - (self.expression_matrix * self.ntr)
            
        if entity == "cell":
            info = self.cell_info.copy()
            if (self.trajectories is not None and "trajectory" not in info.columns):
                info["trajectory"] = info["barcode"].map(self.get_cell_trajectory_mapping())

            return matrix.matrix_sparse, info, "barcode"
        elif entity == "gene":
            return matrix.T.matrix_sparse, self.gene_info, "gene_name"
        elif entity == "trajectory":
            if self.trajectories is None:
                raise ValueError("No trajectories available.")

            X = self._get_trajectory_vectors(data=data)

            trajectory_info = pd.DataFrame(
                {"trajectory_id": self.trajectories.index},
                index=self.trajectories.index,
            )

            return X, trajectory_info, "trajectory_id"

        raise ValueError(f"Unknown entity: {entity}")






class ScifateDataLoader:
    """
    
    
    
    """
    
    def load(
        self, 
        name: str = None,
        folder_path: str = None, 
        expression_matrix_path: str = None,
        ntr_matrix_path: str = None,
        barcodes_path: str = None,
        features_path: str = None,
        trajectories_path: str = None,
        
    ) -> ScifateDataset:
        """
        
        
        """
        
        if folder_path is not None:
            (
                expression_matrix_path,
                ntr_matrix_path,
                barcodes_path,
                features_path,
                trajectories_path,
            ) = self._detect_files_in_folder(
                folder_path=folder_path,
                expression_matrix_path=expression_matrix_path,
                ntr_matrix_path=ntr_matrix_path,
                barcodes_path=barcodes_path,
                features_path=features_path,
                trajectories_path=trajectories_path,
            )
            
        expression_matrix = self.load_expression_matrix(expression_matrix_path)
        ntr_matrix = self.load_ntr_matrix(ntr_matrix_path) if ntr_matrix_path else None
        barcodes = self.load_barcodes(barcodes_path)
        features = self.load_features(features_path)

        trajectories = None
        if trajectories_path is not None:
            trajectories = self.load_trajectories(trajectories_path)
      
        n_cells = len(barcodes) 
        n_genes = len(features)
      
        # we want to make sure that both matrix are n_cells x n_genes
        if expression_matrix is not None:
            if expression_matrix.shape == (n_genes, n_cells):
                expression_matrix = expression_matrix.T.tocsr()
            elif expression_matrix.shape == (n_cells, n_genes):
                pass
            else:
                raise ValueError(f"the Dimensions of the expression_matrix with shape {expression_matrix.shape} doesn't match n_cells: {n_cells} and n_genes: {n_genes}")
        
        if ntr_matrix is not None:
            if ntr_matrix.shape == (n_genes, n_cells):
                ntr_matrix = ntr_matrix.T.tocsr()
            elif ntr_matrix.shape == (n_cells, n_genes):
                pass
            else:
                raise ValueError(f"the Dimensions of the ntr_matrix with shape {ntr_matrix.shape} doesn't match n_cells: {n_cells} and n_genes: {n_genes}")

        cell_info = self.prepare_cell_info(barcodes)
        gene_info = self.prepare_gene_info(features)

        expression = LabeledSparseMatrix(
            matrix=expression_matrix,
            row_info=cell_info,
            col_info=gene_info,
            name="expression_matrix",
            col_label="gene_name",
            row_label="barcode",
        )

        ntr = None
        if ntr_matrix is not None:
            ntr = LabeledSparseMatrix(
                matrix=ntr_matrix,
                row_info=cell_info,
                col_info=gene_info,
                name="ntr",
                col_label="gene_name",
                row_label="barcode",
            )

        dataset = ScifateDataset(
            data=expression,
            ntr=ntr,
            cell_info=cell_info,
            gene_info=gene_info,
            name=name if name is not None else "SciFateDataset",
            trajectories=trajectories,
        )

        return dataset


    def prepare_cell_info(
        self,
        barcodes
    ):
        cell_info = pd.DataFrame({"barcode": barcodes})

        split = cell_info["barcode"].str.split(".", n=2, expand=True)

        cell_info["cell_line"] = split[0]
        cell_info["time"] = split[1]
        cell_info["cell_barcode"] = split[2]
                  
        return cell_info          
            
            
    def prepare_gene_info(
        self,
        features
    ):
        
       gene_info = features
       
       gene_info.columns = [
            "gene_id",
            "gene_name",
            "feature_type",
            "region",
            "length",
        ]
       
       return gene_info
       
        
    def _detect_files_in_folder(
        self,
        folder_path,
        expression_matrix_path=None,
        ntr_matrix_path=None,
        barcodes_path=None,
        features_path=None,
        trajectories_path=None,
    ):
        folder = Path(folder_path)

        files = list(folder.iterdir())

        for file in files:
            if not file.is_file():
                continue

            name = file.name.lower()

            if name == "matrix.mtx":
                if expression_matrix_path is not None:
                    warnings.warn("expression_matrix_path overwritten by file found in folder")
                expression_matrix_path = file

            elif name == "barcodes.tsv":
                if barcodes_path is not None:
                    warnings.warn("barcodes_path overwritten by file found in folder")
                barcodes_path = file

            elif name in ["features.tsv", "genes.tsv"]:
                if features_path is not None:
                    warnings.warn("features_path overwritten by file found in folder")
                features_path = file

            elif name.endswith(".mtx") and "ntr" in name:
                if ntr_matrix_path is not None:
                    warnings.warn("ntr_matrix_path overwritten by file found in folder")
                ntr_matrix_path = file

            elif name.endswith(".csv") and ("traj" in name or "trajectory" in name):
                if trajectories_path is not None:
                    warnings.warn("trajectories_path overwritten by file found in folder")
                trajectories_path = file

        return (
            expression_matrix_path,
            ntr_matrix_path,
            barcodes_path,
            features_path,
            trajectories_path,
        )


    def load_expression_matrix(self, expression_matrix_path):
        expression_matrix_path = Path(expression_matrix_path)

        if not expression_matrix_path.exists():
            raise FileNotFoundError(f"Expression matrix not found: {expression_matrix_path}")

        return mmread(expression_matrix_path).tocsr()


    def load_ntr_matrix(self, ntr_matrix_path):
        ntr_matrix_path = Path(ntr_matrix_path)

        if not ntr_matrix_path.exists():
            raise FileNotFoundError(f"NTR matrix not found: {ntr_matrix_path}")

        return mmread(ntr_matrix_path).tocsr()


    def load_barcodes(self, barcodes_path):
        barcodes_path = Path(barcodes_path)

        if not barcodes_path.exists():
            raise FileNotFoundError(f"Barcodes file not found: {barcodes_path}")

        barcodes = pd.read_csv(
            barcodes_path,
            sep="\t",
            header=None,
        )

        return barcodes.iloc[:, 0].astype(str).tolist()


    def load_features(self, features_path):
        features_path = Path(features_path)

        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        features = pd.read_csv(
            features_path,
            sep="\t",
            header=None,
        )

        return features


    def load_trajectories(self, trajectories_path):
        trajectories_path = Path(trajectories_path)

        if not trajectories_path.exists():
            raise FileNotFoundError(f"Trajectories file not found: {trajectories_path}")

        return pd.read_csv(trajectories_path, index_col=0)
    
    
    
    # helper 
    
    
