import unittest
from typing import List

import numpy as np

from kmeans import BisectingKMeans
from kmeans.bisecting_kmeans import select_cluster


class BisectingKMeansTest(unittest.TestCase):

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

    @staticmethod
    def get_two_clusters_with_even_centers() -> List[List]:
        """Two clusters with even centers in a 2-dimensional plane.
        Useful for nice centroid points.
              ^
              | x x
              | x x
        <-----+----->
          o o |
          o o |
              v
        x - Denotes cluster 1
        o - Denotes cluster 2

        Optimal centroids are located at (2, 2) and (-2, -2).
        """
        return [[-3, -1], [-1, -1], [-3, -3], [-1, -3],
                [1, 1], [1, 3], [3, 1], [3, 3]]

    @staticmethod
    def get_four_clusters() -> List[List]:
        """Four clusters in a 2-dimensional plane.
        ^
        | + +       * *
        | +           *
        |
        | x           o
        | x x       o o
        +-------------->
        x - Denotes cluster 1
        o - Denotes cluster 2
        + - Denotes cluster 3
        * - Denotes cluster 4
        """
        return [[1, 1], [1, 2], [2, 1],
                [6, 1], [7, 1], [7, 2],
                [1, 4], [1, 5], [2, 5],
                [6, 5], [7, 4], [7, 5]]

    def test_fit_with_two_cluster(self):
        expected_labels = [0, 0, 0, 1, 1, 1]
        expected_centroids = [[1.6666667, 1.6666667], [-1.6666667, -1.6666667]]
        expected_inertia = 2.6666667
        expected_inertia_per_cluster = [1.3333333, 1.3333333]
        k_means = BisectingKMeans(num_clusters=2, seed=1)
        k_means.fit(self.get_two_clusters())
        self.assertEqual(expected_labels, k_means.labels_)
        np.testing.assert_almost_equal(expected_centroids, k_means.centroids_)
        self.assertAlmostEqual(expected_inertia, k_means.inertia_)
        np.testing.assert_almost_equal(expected_inertia_per_cluster, k_means.inertia_per_cluster_)
        self.assertAlmostEqual(sum(k_means.inertia_per_cluster_), k_means.inertia_)

    def test_fit_with_four_cluster(self):
        expected_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        expected_centroids = [[6.66666667, 4.66666667],
                              [6.66666667, 1.33333333],
                              [1.33333333, 1.33333333],
                              [1.33333333, 4.66666667]]
        expected_inertia = 5.333333333333334
        expected_inertia_per_cluster = [1.3333333, 1.3333333, 1.3333333, 1.3333333]
        k_means = BisectingKMeans(num_clusters=4, seed=1)
        data = self.get_four_clusters()
        k_means.fit(data)
        self.assertEqual(expected_labels, k_means.labels_)
        np.testing.assert_almost_equal(expected_centroids, k_means.centroids_)
        self.assertAlmostEqual(expected_inertia, k_means.inertia_)
        np.testing.assert_almost_equal(expected_inertia_per_cluster, k_means.inertia_per_cluster_)
        self.assertAlmostEqual(sum(k_means.inertia_per_cluster_), k_means.inertia_)

    def test_manhattan_distance(self):
        expected_labels = [0, 0, 0, 0, 1, 1, 1, 1]
        expected_centroids = [[2, 2], [-2, -2]]
        expected_inertia = 16
        expected_inertia_per_cluster = [8, 8]
        k_means = BisectingKMeans(num_clusters=2, distance_function='manhattan', seed=1)
        k_means.fit(self.get_two_clusters_with_even_centers())
        self.assertEqual(expected_labels, k_means.labels_)
        np.testing.assert_almost_equal(expected_centroids, k_means.centroids_)
        self.assertAlmostEqual(expected_inertia, k_means.inertia_)
        np.testing.assert_almost_equal(expected_inertia_per_cluster, k_means.inertia_per_cluster_)
        self.assertAlmostEqual(sum(k_means.inertia_per_cluster_), k_means.inertia_)

    def test_select_cluster(self):
        centroids = [[-1.5, -1.5], [0.75, 0.75]]
        expected_cluster = [[1, 2], [2, 1], [2, 2]]
        clusters = [[[-2, -1], [-1, -2], [-2, -2]],
                    [[1, 2], [2, 1], [2, 2]]]
        cluster = select_cluster(clusters, centroids)
        self.assertEqual(expected_cluster, cluster)


if __name__ == '__main__':
    unittest.main()
