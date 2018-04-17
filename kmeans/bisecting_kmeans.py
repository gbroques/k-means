from random import Random
from typing import List

from .kmeans import KMeans
from .kmeans import distance
from .kmeans import get_centroids
from .kmeans import partition_by_cluster


class BisectingKMeans:

    def __init__(self, num_clusters: int, num_trials: int = 10, distance_function='euclidean', seed=None):
        self._num_clusters = num_clusters
        self._num_trials = num_trials
        self._distance_function = distance_function
        self._random = Random(seed)
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_per_cluster_ = None

    def fit(self, data: List[List]) -> 'BisectingKMeans':
        clusters = [data]
        self.centroids_ = get_centroids(clusters)
        while len(clusters) != self._num_clusters:
            cluster = select_cluster(clusters, self.centroids_, self._distance_function)
            clusters.remove(cluster)

            trial_results = {}
            for i in range(self._num_trials):
                random_state = self._random.random()
                k_means = KMeans(num_clusters=2, seed=random_state)
                k_means.fit(cluster)
                trial_results[k_means.inertia_] = k_means
            best_trial = min(trial_results)
            best_result = trial_results[best_trial]
            best_two_clusters = partition_by_cluster(cluster, best_result.labels_)
            clusters.extend(best_two_clusters)
        self.centroids_ = get_centroids(clusters)
        self.labels_ = get_labels(clusters)
        self.inertia_ = get_total_inertia(clusters, self.centroids_, self._distance_function)
        self.inertia_per_cluster_ = get_inertia_per_cluster(clusters, self.centroids_, self._distance_function)
        return self


def select_cluster(clusters: List[List[List]], centroids: List[List], distance_function: str = None) -> List[List]:
    """Select a cluster to bisect based upon minimizing the sum of squared errors (SSE).

    Args:
        clusters: A list of clusters.
        centroids: The center points of each cluster.
        distance_function: Whether to use euclidean or manhattan distance.
                           Defaults to euclidean distance.

    Returns:
        The cluster to bisect.
    """
    inertia_per_cluster = get_inertia_per_cluster(clusters, centroids, distance_function)
    max_inertia = max(inertia_per_cluster)
    index = inertia_per_cluster.index(max_inertia)
    return clusters[index]


def get_total_inertia(clusters: List[List[List]], centroids: List[List], distance_function: str = None) -> float:
    """Get the total sum of squared errors for each cluster.

    Args:
        clusters: A list of clusters.
        centroids: The center point of each cluster.
        distance_function: Whether to use euclidean or manhattan distance.
                           Defaults to euclidean distance.
    Returns:
        The total sum of squared errors.
    """
    inertia_per_cluster = get_inertia_per_cluster(clusters, centroids)
    return sum(inertia_per_cluster)


def get_inertia_per_cluster(clusters: List[List[List]],
                            centroids: List[List],
                            distance_function: str = None) -> List[float]:
    """Get the sum of squared errors for each cluster.

    Args:
        clusters: A list of clusters.
        centroids: The center point of each cluster.
        distance_function: Whether to use euclidean or manhattan distance.
                           Defaults to euclidean distance.

    Returns:
        The sum of squared errors for each cluster.
    """
    inertia_per_cluster = []
    for cluster, centroid in zip(clusters, centroids):
        cluster_inertia = get_inertia_for_one_cluster(cluster, centroid, distance_function)
        inertia_per_cluster.append(cluster_inertia)
    return inertia_per_cluster


def get_inertia_for_one_cluster(cluster: List[List], centroid: List[float], distance_function: str = None):
    """Get the sum of the distances from each point in the cluster to the centroid.

    Args:
        cluster: A cluster of points.
        centroid: The center point of the cluster.
        distance_function: Whether to use euclidean or manhattan distance.
                           Defaults to euclidean distance.

    Returns:
        The sum of the distances from each point in the cluster to the centroid.
    """
    errors = []
    for point in cluster:
        error = distance(point, centroid, func=distance_function)
        errors.append(error)
    return sum(errors)


def get_labels(clusters: List[List[List]]) -> List[int]:
    """Get the label of each cluster.

    Args:
        clusters: A list of clusters.

    Returns:
        A list of labels.
    """
    labels = []
    for i, cluster in enumerate(clusters):
        labels.extend([i for _ in range(len(cluster))])
    return labels
