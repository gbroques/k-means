from typing import List

from scipy.spatial.distance import euclidean

from .kmeans import KMeans
from .kmeans import get_centroids
from .kmeans import partition_by_cluster


class BisectingKMeans:

    def __init__(self, num_clusters: int, num_trials: int = 10):
        self._num_clusters = num_clusters
        self._num_trials = num_trials
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_per_cluster_ = None

    def fit(self, data: List[List]) -> 'BisectingKMeans':
        clusters = [data]
        self.centroids_ = get_centroids(clusters)
        while len(clusters) != self._num_clusters:
            cluster = select_cluster(clusters, self.centroids_)
            clusters.remove(cluster)

            trial_results = {}
            for i in range(self._num_trials):
                k_means = KMeans(num_clusters=2)
                k_means.fit(cluster)
                trial_results[k_means.inertia_] = k_means
            best_trial = min(trial_results)
            best_result = trial_results[best_trial]
            best_two_clusters = partition_by_cluster(cluster, best_result.labels_)
            clusters.extend(best_two_clusters)
        self.centroids_ = get_centroids(clusters)
        self.labels_ = get_labels(clusters)
        self.inertia_ = get_total_inertia(clusters, self.centroids_)
        self.inertia_per_cluster_ = get_inertia_per_cluster(clusters, self.centroids_)
        return self


def select_cluster(clusters: List[List[List]], centroids: List[List]) -> List[List]:
    """Select a cluster to bisect based upon minimizing the sum of squared errors (SSE).

    Args:
        clusters: A list of clusters.
        centroids: The center points of each cluster.

    Returns:
        The cluster to bisect.
    """
    inertia_per_cluster = get_inertia_per_cluster(clusters, centroids)
    max_inertia = max(inertia_per_cluster)
    index = inertia_per_cluster.index(max_inertia)
    return clusters[index]


def get_total_inertia(clusters: List[List[List]], centroids: List[List]) -> float:
    """Get the total sum of squared errors for each cluster.

    Args:
        clusters: A list of clusters.
        centroids: The center point of each cluster.

    Returns:
        The total sum of squared errors.
    """
    inertia_per_cluster = get_inertia_per_cluster(clusters, centroids)
    return sum(inertia_per_cluster)


def get_inertia_per_cluster(clusters: List[List[List]], centroids: List[List]) -> List[float]:
    """Get the sum of squared errors for each cluster.

    Args:
        clusters: A list of clusters.
        centroids: The center point of each cluster.

    Returns:
        The sum of squared errors for each cluster.
    """
    inertia_per_cluster = []
    for cluster, centroid in zip(clusters, centroids):
        cluster_inertia = get_inertia_for_one_cluster(cluster, centroid)
        inertia_per_cluster.append(cluster_inertia)
    return inertia_per_cluster


def get_inertia_for_one_cluster(cluster: List[List], centroid: List[float]):
    """Get the sum of squared error for one cluster.

    Args:
        cluster: A cluster of points.
        centroid: The center point of the cluster.

    Returns:
        The sum squared error for the cluster.
    """
    squared_errors = []
    for point in cluster:
        squared_error = euclidean(point, centroid) ** 2
        squared_errors.append(squared_error)
    return sum(squared_errors)


def get_labels(clusters: List[List[List]]) -> List[int]:
    labels = []
    for i, cluster in enumerate(clusters):
        labels.extend([i for _ in range(len(cluster))])
    return labels
