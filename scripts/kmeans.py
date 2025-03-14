import numpy as np
import os
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, jensenshannon, cosine
import scripts.utils as utils


class KMeans:
    def __init__(self, k, config:dict, instance_id:int, logging):
        self.k         = k 
        self.distance  = config['distance']
        self.init      = config['kmeans_init']
        self.max_iter  = config['iters']
        self.threads   = config['threads']
        self.seed      = config['init_seed'] + instance_id * 100
        self.nan_value = config['nan_value']
        self.algo      = config['algo']
        self.rng       = np.random.default_rng(self.seed)
        self.logging   = logging

        # set OMP_NUM_THREADS=1 in env
        os.environ["OMP_NUM_THREADS"] = "1"

    def get_centroid_target(self, C):
        return np.mean(C, axis=0).reshape(1, -1)

    def get_centroid_idx(self, i):
        C = self.X[self.labels == i]
        C_target = self.get_centroid_target(C)
        C_dist = self.compute_distance(C, C_target)
        idx = np.arange(len(self.X))[self.labels == i]
        return idx[np.argmin(C_dist)]

    def compute_distance(self, X, centroid):
        dist = cdist(X, centroid, metric=self.distance)
        dist = np.nan_to_num(dist, nan=self.nan_value)
        return dist

    def init_center_idxs(self):
        if self.init == 'kmeans++':
            return self.init_center_idxs_kmeanspp()
        elif self.init == 'random':
            return self.init_center_idxs_random()

    def init_center_idxs_random(self):
        idxs = self.rng.choice(len(self.X), self.k, replace=False)
        return idxs

    def random_choice_prob(self, prob):
        prob = prob / np.sum(prob)
        pos = self.rng.random()

        prob_cumsum = np.cumsum(prob)
        return np.argmin(prob_cumsum < pos)

    def init_center_idxs_kmeanspp(self):
        idxs = [self.rng.integers(len(self.X))]
        cents = self.X[idxs]
        self.logging.info(f"initializing centroids")
        for i in range(1, self.k):
            # self.logging.info(f"initializing centroid {i}")
            # pairwise dist of X and cents
            # dist = self.compute_distance(self.X, cents)
            dist = Parallel(n_jobs=self.threads, prefer="threads")(
                delayed(self.compute_distance)(self.X, np.array([cent])) for cent in cents
            )
            dist = np.array(dist).reshape(len(cents), -1).T
            # foreach x, select min(x, cents)
            dist = np.min(dist, axis=-1)
            # select idx with dist prob
            new_idx = self.random_choice_prob(dist)
            idxs.append(new_idx)
            cents = self.X[idxs]
        return np.array(idxs)
    
    def fit(self, X):
        self.X = X

        if self.algo == 'kmeans':
            return self.fit_kmeans()
        elif self.algo == 'spectralc':
            return self.fit_SpectralClustering()
        elif self.algo == 'aggc':
            return self.fit_AgglomerativeClustering()
        elif self.algo == 'bisecting':
            return self.fit_BisectingKMeans()
        elif self.algo == 'gaussian':
            return self.fit_GaussianMixture()
        elif self.algo == 'birch':
            return self.fit_Birch()
        else:
            raise NotImplementedError
    
    def fit_Birch(self):
        from sklearn.cluster import Birch

        self.logging.info("Birch fitting...")
        clustering = Birch(n_clusters=self.k, threshold=0.005).fit(self.X)
        self.labels = clustering.labels_
        self.centroid_idxs = Parallel(n_jobs=self.threads, prefer="threads")(
            delayed(self.get_centroid_idx)(i) for i in range(self.k)
        )

        # Post processing
        self.post_processing()

        return self
    
    def fit_GaussianMixture(self):
        from sklearn.mixture import GaussianMixture

        self.logging.info("Gaussian Mixture fitting...")
        gmm = GaussianMixture(n_components=self.k, random_state=self.seed, covariance_type='diag').fit(self.X)
        self.labels = gmm.predict(self.X)
        self.centroid_idxs = Parallel(n_jobs=self.threads, prefer="threads")(
            delayed(self.get_centroid_idx)(i) for i in range(self.k)
        )

        # Post processing
        self.post_processing()

        return self
    
    def fit_SpectralClustering(self):
        from sklearn.cluster import SpectralClustering

        self.logging.info("Spectral Clustering fitting...")
        
        clustering = SpectralClustering(n_clusters=self.k, random_state=self.seed, n_jobs=self.threads).fit(self.X)
        self.labels = clustering.labels_
        self.centroid_idxs = Parallel(n_jobs=self.threads, prefer="threads")(
            delayed(self.get_centroid_idx)(i) for i in range(self.k)
        )

        # Post processing
        self.post_processing()

        return self

    def fit_AgglomerativeClustering(self):
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(n_clusters=self.k).fit(self.X)
        self.labels = clustering.labels_
        self.centroid_idxs = Parallel(n_jobs=self.threads, prefer="threads")(
            delayed(self.get_centroid_idx)(i) for i in range(self.k)
        )

        # Post processing
        self.post_processing()

        return self
    
    def fit_BisectingKMeans(self):
        from sklearn.cluster import BisectingKMeans

        clustering = BisectingKMeans(n_clusters=self.k, random_state=self.seed, bisecting_strategy="largest_cluster").fit(self.X)
        self.labels =  clustering.labels_
        self.centroid_idxs = Parallel(n_jobs=self.threads, prefer="threads")(
            delayed(self.get_centroid_idx)(i) for i in range(self.k)
        )

        # Post processing
        self.post_processing()

        return self

    def post_processing(self):
        self.centroids = self.X[self.centroid_idxs]
        self.centroid_slices = self.centroid_idxs
        self.centroid_weights = [np.sum(self.labels == i) / len(self.X) for i in range(self.k)]
        self.slice_labels = self.labels
        self.slice_dists = np.linalg.norm(self.X - self.centroids[self.labels], axis=-1)
        self.score = self.cal_bic_score()

        return

    def fit_kmeans(self):

        # prepare labels
        self.slice_labels = np.zeros(self.X.shape[0], dtype=np.int32)
        self.slice_dists = np.zeros(self.X.shape[0], dtype=np.float32)

        # gen random centroids
        self.centroid_idxs = self.init_center_idxs()

        self.logging.info("kmeans fitting...")
        for i in range(self.max_iter):
            # self.logging.info(f"dealing with iter {i}")
            # get vector of centroids
            self.centroids = self.X[self.centroid_idxs]
            # Compute distances between each sample and all centroids.
            # dist = self.compute_distance(self.X, self.centroids)
            dist = Parallel(n_jobs=self.threads, prefer="threads")(
                delayed(self.compute_distance)(self.X, np.array([centroid])) for centroid in self.centroids
            )
            dist = np.array(dist).reshape(self.k, -1).T
            # Compute new cluster assignments.
            self.labels = np.argmin(dist, axis=-1)
            # special handle
            for i in range(self.k):
                if np.sum(self.labels == i) == 0:
                    k_range = np.arange(self.k)
                    self.rng.shuffle(k_range)
                    self.labels[self.centroid_idxs] = k_range
                    break
            # Compute new cluster means.
            # self.centroid_idxs = np.array([self.get_centroid_idx(i) for i in range(self.k)])
            self.centroid_idxs = Parallel(n_jobs=self.threads, prefer="threads")(
                delayed(self.get_centroid_idx)(i) for i in range(self.k)
            )

        # Post processing
        self.post_processing()

        self.logging.info(f"Clustering score: {self.score}")
        return self
    
    def safe_index(self, array, index, default=0):
        try:
            return array[index]
        except IndexError:
            return default
    
    def distortion(self):
        Xcenter = self.X[np.array(self.centroid_idxs)[self.labels]]
        if self.distance == 'jensenshannon':
            dist = jensenshannon(self.X, Xcenter, axis=-1)
            dist = np.nan_to_num(dist, nan=self.nan_value)
            dist = np.sum(np.square(dist))
        elif self.distance == 'cosine':
            dist = sum([cosine(x, center) for x, center in zip(self.X, Xcenter)])
        elif self.distance == 'cityblock':
            dist = sum([np.sum(np.abs(x - center)) for x, center in zip(self.X, Xcenter)])
        else:
            dist = np.sum(np.square(self.X - Xcenter))
        
        return dist
    
    def cal_bic_score(self):
        n, d = self.X.shape
        k = self.k

        dist = self.distortion()
        sigma2 = dist / (n * d)

        n_param = (k - 1) + (k * d) + 1 # cluster probabilities + cluster means + variances
        penalty = n_param / 2 * np.log(n)

        st = self.stat()
        likelihood = - 0.5 * n * d * (np.log(2.0 * np.pi * sigma2) + 1)
        likelihood += np.sum(st * np.log(st)) - n * np.log(n)

        score = likelihood - penalty

        return score

    def stat(self):
        _, stat = np.unique(self.labels, return_counts=True)
        stat = np.array(stat)
        return stat

    def save(self):
        return {
            'centroids' : self.centroid_slices,
            'weights'   : self.centroid_weights,
            'score'     : self.score,
        }

