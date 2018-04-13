from random import sample
from typing import List


class KMeans:
    def __init__(self, num_clusters=8):
        self._num_clusters = num_clusters
        self.centroids_ = None
        self.labels_ = None

    pass

    def fit(self, data: List[List]) -> 'KMeans':
        """Cluster the data into k sets.

        Args:
            data: Data to cluster into k sets.

        Returns:
            self
        """
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

    @staticmethod
    def _get_percentage_of_points_changed(previous_labels: List, labels: List) -> float:
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
