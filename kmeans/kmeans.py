from random import sample
from typing import List

from scipy.spatial.distance import euclidean


class KMeans:
    def __init__(self, num_clusters=8):
        self._num_clusters = num_clusters
        self.centroids_ = None
        self.labels_ = None

    def fit(self, data: List[List]) -> 'KMeans':
        """Cluster the data into k sets.

        Args:
            data: Data to cluster into k sets.

        Returns:
            self
        """
        self.labels_ = [None for _ in range(len(data))]
        self.centroids_ = self._select_initial_centroids(data)
        # repeat
        #   Form K clusters by assigning each point to its closest centroid
        #   Recompute the centroid of each cluster
        # until centroids do not change
        return self

    def _select_initial_centroids(self, data: List[List]) -> List:
        """Select K points as initial centroids.

        Uses random initialization.

        Args:
            data: The dataset.

        Returns:
            K initial centroids.
        """
        return sample(data, self._num_clusters)


def get_percentage_of_points_changed(previous_labels: List, labels: List) -> float:
    """Get the percentage of points that changed clusters.

    Args:
        previous_labels: The labels of the points before the update.
        labels: The labels of the points after the update.

    Returns:
        The percentage of points that changed clusters.
    """
    num_points = len(labels)
    num_changed = 0
    for i in range(num_points):
        if previous_labels[i] != labels[i]:
            num_changed += 1
    return num_changed / num_points * 100.0


def get_cluster_labels(data: List[List], centroids: List[List]) -> List[int]:
    """Gets the cluster label for a set of points.

    Args:
        data: A set of points. The dataset.
        centroids: The center of each cluster.

    Returns:
        A list of cluster labels. Corresponds to the index of the closest centroid.
    """
    cluster_labels = []
    for point in data:
        cluster_label = get_cluster_label(point, centroids)
        cluster_labels.append(cluster_label)
    return cluster_labels


def get_cluster_label(point: List, centroids: List[List]) -> int:
    """Gets the cluster label of a point.

    Args:
        point: The point to calculate which cluster it belongs to.
        centroids: The center of each cluster.

    Returns:
        The cluster label. This corresponds to the index of the closest centroid.
    """
    distances_from_centroids = [euclidean(point, centroid) for centroid in centroids]
    min_distance = min(distances_from_centroids)
    return distances_from_centroids.index(min_distance)
