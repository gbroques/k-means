import unittest
from random import seed
from typing import List

import numpy as np

from kmeans import BisectingKMeans
from kmeans.bisecting_kmeans import select_cluster


class BisectingKMeansTest(unittest.TestCase):

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
        expected_labels = [1, 1, 1, 0, 0, 0]
        expected_centroids = [[1.6666667, 1.6666667], [-1.6666667, -1.6666667]]
        expected_inertia = 2.6666667
        k_means = BisectingKMeans(num_clusters=self.num_clusters)
        k_means.fit(self.data)
        self.assertEqual(expected_labels, k_means.labels_)
        np.testing.assert_almost_equal(expected_centroids, k_means.centroids_)
        self.assertAlmostEqual(expected_inertia, k_means.inertia_)

    def test_select_cluster(self):
        labels = [0, 0, 1, 1, 1, 1]
        centroids = [[-1.5, -1.5], [0.75, 0.75]]
        expected_cluster = 1
        cluster = select_cluster(self.data, centroids, labels)
        self.assertEqual(expected_cluster, cluster)


if __name__ == '__main__':
    unittest.main()
