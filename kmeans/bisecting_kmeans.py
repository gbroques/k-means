from typing import List

from .kmeans import KMeans
from .kmeans import get_centroids
from .kmeans import get_inertia_per_cluster
from .kmeans import partition_by_cluster


class BisectingKMeans:
    def __init__(self, num_clusters: int, num_trials: int = 10):
        self._num_clusters = num_clusters
        self._num_trials = num_trials
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, data: List[List]) -> 'BisectingKMeans':
        self.labels_ = [0 for _ in range(len(data))]
        self.centroids_ = get_centroids(data, self.labels_)
        while len(set(self.labels_)) != self._num_clusters:
            partitioned_data = partition_by_cluster(data, self.labels_)
            cluster = select_cluster(data, self.centroids_, self.labels_)
            cluster_to_bisect = partitioned_data[cluster]
            trial_results = {}
            for i in range(self._num_trials):
                k_means = KMeans(num_clusters=2)
                k_means.fit(cluster_to_bisect)
                trial_results[k_means.inertia_] = k_means
            best_trial = min(trial_results)
            best_result = trial_results[best_trial]
            self.centroids_ = best_result.centroids_
            self.labels_ = best_result.labels_
            self.inertia_ = best_result.inertia_
        return self


def select_cluster(data: List[List], centroids: List[List], labels: List) -> int:
    """Select a cluster to bisect based upon minimizing the sum of squared errors (SSE).

    Args:
        data: The dataset.
        centroids: The center points of each cluster.
        labels: The cluster each point in the dataset belongs to.

    Returns:
        The cluster to bisect.
    """
    inertia_per_cluster = get_inertia_per_cluster(data, centroids, labels)
    max_inertia = max(inertia_per_cluster)
    return inertia_per_cluster.index(max_inertia)
