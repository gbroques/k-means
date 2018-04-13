import unittest
from typing import List


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


if __name__ == '__main__':
    unittest.main()
