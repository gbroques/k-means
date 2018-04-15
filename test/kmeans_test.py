import unittest
from random import seed
from typing import List

import numpy as np

from kmeans import KMeans
from kmeans.kmeans import get_centroid
from kmeans.kmeans import get_centroids
from kmeans.kmeans import get_closest_centroids
from kmeans.kmeans import get_closest_centroids_from_labels
from kmeans.kmeans import get_cluster_label
from kmeans.kmeans import get_cluster_labels
from kmeans.kmeans import get_inertia
from kmeans.kmeans import get_inertia_per_cluster
from kmeans.kmeans import get_percentage_of_points_changed
from kmeans.kmeans import partition_by_cluster


class KMeansTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = cls.get_two_clusters()
        cls.num_clusters = 2

    @staticmethod
    def get_two_clusters() -> List[List]:
        """Two clusters in a 2-dimensional plane.
              ^
              | x x
              |   x
        <-----+----->
          o   |
          o o |
              v
        x - Denotes cluster 1
        o - Denotes cluster 2
        """
        return [[-2, -1], [-1, -2], [-2, -2],
                [1, 2], [2, 1], [2, 2]]

    def test_fit(self):
        seed(1)
        expected_labels = [0, 0, 0, 1, 1, 1]
        expected_centroids = [[-1.6666667, -1.6666667], [1.6666667, 1.6666667]]
        expected_inertia = 2.6666667
        k_means = KMeans(num_clusters=self.num_clusters)
        k_means.fit(self.data)
        self.assertEqual(expected_labels, k_means.labels_)
        np.testing.assert_almost_equal(expected_centroids, k_means.centroids_)
        self.assertAlmostEqual(expected_inertia, k_means.inertia_)

    def test_predict(self):
        seed(1)
        test_samples = [[-3, -3], [3, 3], [-1, -1], [1, 1]]
        expected_predictions = [0, 1, 0, 1]
        k_means = KMeans(num_clusters=self.num_clusters)
        k_means.fit(self.data)
        predictions = k_means.predict(test_samples)
        self.assertEqual(expected_predictions, predictions)

    def test_fit_with_different_initial_centroids(self):
        seed(0)
        expected_labels = [0, 0, 0, 1, 1, 1]
        expected_centroids = [[-1.6666667, -1.6666667], [1.6666667, 1.6666667]]
        k_means = KMeans(num_clusters=self.num_clusters)
        k_means.fit(self.data)
        self.assertEqual(expected_labels, k_means.labels_)
        np.testing.assert_almost_equal(expected_centroids, k_means.centroids_)

    def test_select_initial_centroids(self):
        seed(1)
        expected_initial_centroids = [[-1, -2], [2, 1]]
        k_means = KMeans(num_clusters=self.num_clusters)
        initial_centroids = k_means._select_initial_centroids(self.data)
        self.assertEqual(expected_initial_centroids, initial_centroids)
        self.assertEqual(self.num_clusters, len(initial_centroids))

    def test_get_percentage_of_point_changed_with_no_change(self):
        expected_percentage_of_points_changed = 0.0
        previous_labels = [1, 1, 1, 0, 0, 0]
        labels = [1, 1, 1, 0, 0, 0]
        percentage_of_points_changed = get_percentage_of_points_changed(previous_labels, labels)
        self.assertEqual(expected_percentage_of_points_changed, percentage_of_points_changed)

    def test_get_percentage_of_point_changed_with_change(self):
        expected_percentage_of_points_changed = 16.666666667
        previous_labels = [1, 1, 1, 0, 0, 1]
        labels = [1, 1, 1, 0, 0, 0]
        percentage_of_points_changed = get_percentage_of_points_changed(previous_labels, labels)
        self.assertAlmostEqual(expected_percentage_of_points_changed, percentage_of_points_changed)

    def test_get_cluster_label(self):
        expected_cluster = 0
        centroids = [[-1, -2], [2, 1]]
        cluster = get_cluster_label([-2, -1], centroids)
        self.assertEqual(expected_cluster, cluster)

    def test_get_cluster_labels(self):
        expected_clusters = [0, 1]
        centroids = [[-1, -2], [2, 1]]
        clusters = get_cluster_labels([[-2, -1], [1, 1]], centroids)
        self.assertEqual(expected_clusters, clusters)

    def test_get_centroids(self):
        labels = [0, 0, 0, 1, 1, 1]
        expected_centroids = [[-1.6666667, -1.6666667], [1.6666667, 1.6666667]]
        centroids = get_centroids(self.data, labels)
        np.testing.assert_almost_equal(expected_centroids, centroids)

    def test_get_centroid(self):
        cluster_of_points = [[1, 1], [2, 3], [6, 2]]
        expected_centroid = [3, 2]
        centroid = get_centroid(cluster_of_points)
        self.assertEqual(expected_centroid, centroid)

    def test_partition_by_cluster(self):
        labels = [0, 0, 0, 1, 1, 1]
        expected_partition = [[[-2, -1], [-1, -2], [-2, -2]],
                              [[1, 2], [2, 1], [2, 2]]]
        partition = partition_by_cluster(self.data, labels)
        self.assertEqual(expected_partition, partition)

    def test_get_inertia(self):
        expected_inertia = 2.6666667
        closest_centroids = [
            [-1.6666667, -1.6666667], [-1.6666667, -1.6666667], [-1.6666667, -1.6666667],
            [1.6666667, 1.6666667], [1.6666667, 1.6666667], [1.6666667, 1.6666667]
        ]
        inertia = get_inertia(self.data, closest_centroids)
        self.assertAlmostEqual(expected_inertia, inertia)

    def test_get_closest_centroids_from_labels(self):
        expected_closest_centroids = [
            [-1.6666667, -1.6666667], [-1.6666667, -1.6666667], [-1.6666667, -1.6666667],
            [1.6666667, 1.6666667], [1.6666667, 1.6666667], [1.6666667, 1.6666667]
        ]
        centroids = [[-1.6666667, -1.6666667], [1.6666667, 1.6666667]]
        labels = [0, 0, 0, 1, 1, 1]
        closest_centroids = get_closest_centroids_from_labels(self.data, centroids, labels)
        self.assertAlmostEqual(expected_closest_centroids, closest_centroids)

    def test_get_closest_centroids(self):
        expected_closest_centroids = [
            [-1.6666667, -1.6666667], [-1.6666667, -1.6666667], [-1.6666667, -1.6666667],
            [1.6666667, 1.6666667], [1.6666667, 1.6666667], [1.6666667, 1.6666667]
        ]
        centroids = [[-1.6666667, -1.6666667], [1.6666667, 1.6666667]]
        closest_centroids = get_closest_centroids(self.data, centroids)
        self.assertAlmostEqual(expected_closest_centroids, closest_centroids)

    def test_get_inertia_per_cluster(self):
        centroids = [[-1.6666667, -1.6666667], [1.6666667, 1.6666667]]
        labels = [0, 0, 0, 1, 1, 1]
        expected_inertia_per_cluster = [1.3333333, 1.3333333]
        inertia_per_cluster = get_inertia_per_cluster(self.data, centroids, labels)
        np.testing.assert_almost_equal(expected_inertia_per_cluster, inertia_per_cluster)


if __name__ == '__main__':
    unittest.main()
