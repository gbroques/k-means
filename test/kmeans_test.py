import unittest
from random import seed
from typing import List

from kmeans import KMeans


class KMeansTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = cls.get_two_clusters()

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
        return [[-2, -1],
                [-1, -2],
                [-2, -2],
                [1, 2],
                [2, 1],
                [2, 2]]

    def test_fit(self):
        seed(1)
        expected_labels = [1, 1, 1, 0, 0, 0]
        k_means = KMeans(num_clusters=2)
        k_means.fit(self.data)
        self.assertEqual(expected_labels, k_means.labels_)

    def test_select_initial_centroids(self):
        seed(1)
        num_clusters = 2
        expected_initial_centroids = [[-1, -2], [2, 1]]
        k_means = KMeans(num_clusters=num_clusters)
        initial_centroids = k_means._select_initial_centroids(self.data)
        self.assertEqual(expected_initial_centroids, initial_centroids)
        self.assertEqual(num_clusters, len(initial_centroids))


if __name__ == '__main__':
    unittest.main()
