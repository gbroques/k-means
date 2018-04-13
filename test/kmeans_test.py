import unittest
from random import seed
from typing import List

from kmeans import KMeans


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
        return [[-2, -1],
                [-1, -2],
                [-2, -2],
                [1, 2],
                [2, 1],
                [2, 2]]

    def test_fit(self):
        seed(1)
        expected_labels = [1, 1, 1, 0, 0, 0]
        k_means = KMeans(num_clusters=self.num_clusters)
        k_means.fit(self.data)
        self.assertEqual(expected_labels, k_means.labels_)

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
        k_means = KMeans(num_clusters=self.num_clusters)
        percentage_of_points_changed = k_means._get_percentage_of_points_changed(previous_labels, labels)
        self.assertEqual(expected_percentage_of_points_changed, percentage_of_points_changed)

    def test_get_percentage_of_point_changed_with_change(self):
        expected_percentage_of_points_changed = 16.666666667
        previous_labels = [1, 1, 1, 0, 0, 1]
        labels = [1, 1, 1, 0, 0, 0]
        k_means = KMeans(num_clusters=self.num_clusters)
        percentage_of_points_changed = k_means._get_percentage_of_points_changed(previous_labels, labels)
        self.assertAlmostEqual(expected_percentage_of_points_changed, percentage_of_points_changed)

    def test_assign_point_to_cluster(self):
        seed(1)
        expected_labels = [0]
        k_means = KMeans(num_clusters=self.num_clusters)
        k_means._assign_point_to_cluster(self.data[0])
        self.assertEqual(expected_labels, k_means.labels_)


if __name__ == '__main__':
    unittest.main()
