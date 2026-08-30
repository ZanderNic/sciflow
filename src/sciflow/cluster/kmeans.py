# std lib imports

# 3-party import
import numpy as np
import pandas as pd
from tqdm import tqdm

# projekt imports
from sciflow.cluster import BaseCluster
from sciflow.distance import BaseDistance
from sciflow.distance.distance import CosineDistance, EuclideanDistance
from sciflow.reduction import BaseReduction



class KMeans(BaseCluster):

    name = "KMeans"

    def __init__(
        self, 
        k: int,
        optimizer: str = "hartigan_wong",                   # "lloyd", "hartigan_wong"
        init: str = "forgy",                                # "forgy", "random_partition", "kmeans++"
        distance: BaseDistance = EuclideanDistance(),
        max_steps: int = 100,
        tol: float = 1e-5,
    ):
        
        if init not in ["forgy", "random_partition", "kmeans++"]:
            raise ValueError()
        
        if optimizer not in ["hartigan_wong", "lloyd"]:
            raise ValueError()
        
        
        self.k = k 
        self.init = init
        self.optimizer = optimizer
        self.distance = distance
        self.max_steps = max_steps
        self.means = None
        self.tol = tol



    def fit(self, X: np.array):
        """
        
        """
        
        if self.optimizer == "lloyd":
            means = self.init_means(X)
            
            for _ in tqdm(range(self.max_steps), desc="lloyd iterations"):
                
                assign = self.lloyd_assignment_step(X, means)
                means_new = self.lloyd_update_step(X, assign)
                
                if np.allclose(means, means_new, atol=self.tol):
                    break
                
                means = means_new
            else:
                print(f"terminated after {self.max_steps} steps becasue of max steps")  
            
            self.means = means
            self.assign = assign
            self.is_fitted = True
           
            return assign
            
        elif self.optimizer == "hartigan_wong":
            means, assign = self.hartigan_wong(X)

            self.means = means
            self.assign = assign
            self.is_fitted = True

            return assign

        else:
            raise ValueError()
    
    
    def hartigan_wong(self, X):
        means = self.init_means(X)
        assign = self.lloyd_assignment_step(X, means)
        means = self.lloyd_update_step(X, assign)


        for _ in tqdm(range(self.max_steps), desc="hartigan_wong iterations"):
            changed = False

            for idx, point in enumerate(X):
                old_cluster = assign[idx]
                best_cluster = old_cluster
                best_cost = self.distance(point, means[old_cluster])

                for new_cluster in range(self.k):
                    if new_cluster == old_cluster:
                        continue

                    cost = self.distance(point, means[new_cluster])

                    if cost < best_cost:
                        best_cost = cost
                        best_cluster = new_cluster

                if best_cluster != old_cluster:
                    assign[idx] = best_cluster
                    changed = True

                    means = self.lloyd_update_step(X, assign)

            if not changed:
                break

        return means, assign


    def predict(self, X):
        if self.is_fitted is False:
            raise ValueError("please fit bevor pred")
        
        return self.lloyd_assignment_step(X, means=self.means)


    def fit_predict(self, X):
        return self.fit(X)
    
    
    def lloyd_assignment_step(self, X, means) -> np.array:
        assign = np.empty(X.shape[0], dtype=np.int32)

        for idx, point in enumerate(X):
            distances = [self.distance(point, mean) for mean in means]
            assign[idx] = np.argmin(distances)

        return assign

    
    def lloyd_update_step(self, X, assign) -> np.array:
        means = np.empty((self.k, X.shape[1]), dtype=np.float32)
        
        for i in range(self.k):
            mask = assign == i 
            cluster_points = X[mask]
            
            if len(cluster_points) == 0:
                raise ValueError(f"Cluster {i} is empty.")

            means[i] = cluster_points.mean(axis=0)
            
        return means

    
    def init_means(self, X) -> np.array:
        means = None
        
        if self.init == "forgy":
            indices = np.random.choice(X.shape[0], size=self.k, replace=False)
            means = X[indices]
            
        elif self.init == "random_partition":
            assign = np.random.randint(0, self.k, size=X.shape[0])
            means = self.lloyd_update_step(X, assign)
            
        elif self.init == "kmeans++":
            means = np.empty(shape=(self.k, X.shape[1]), dtype=np.float32)
            
            idx = np.random.choice(X.shape[0])
            means[0] = X[idx]
            
            for i in range(1, self.k):
                
                proba = np.empty(shape=(X.shape[0]), dtype=np.float32)
                
                for idx, point in enumerate(X):
                    distances = [self.distance(point, mean) for mean in means[:i]]
                    proba[idx] = np.min(distances) ** 2
                
                proba = proba / proba.sum()
                idx = np.random.choice(X.shape[0], p=proba)
                means[i] = X[idx]
        
        
        return means
    
    
    def score(self, X, assign=None, means=None, save_in_class: bool = True):
        if assign is None:
            assign = self.assign

        if means is None:
            means = self.means

        if assign is None or means is None:
            raise ValueError("KMeans must be fitted before scoring.")

        wcss = 0.0

        for i in range(self.k):
            mask = assign == i
            cluster_points = X[mask]

            for point in cluster_points:
                dist = self.distance(point, means[i])
                wcss += dist ** 2

        return wcss
    
    
    def test_ks(
        self,
        X: np.ndarray,
        k_values,
        reduction: BaseReduction = None,
    ):
        
        if reduction != None:
            X = reduction.fit_transform(X)
        
        scores = []

        for k in k_values:
            print(f"Start fitt with k={k}")
            
            model = KMeans(
                k=k,
                optimizer=self.optimizer,
                init=self.init,
                distance=self.distance,
                max_steps=self.max_steps,
                tol=self.tol,
            )

            model.fit(X)

            scores.append(model.score(X))

        return pd.DataFrame({
            "k": list(k_values),
            "score": scores,
        })