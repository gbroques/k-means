from typing import List

from dataviz import generate_clusters
from dataviz import plot_clusters
from kmeans import BisectingKMeans


def main():
    num_clusters = 4
    clusters = generate_data(num_clusters, seed=1)
    k_means = BisectingKMeans(num_clusters=num_clusters, seed=4235)
    k_means.fit(clusters)
    plot_clusters(clusters, k_means.labels_, k_means.centroids_)


def generate_data(num_clusters: int, seed=None) -> List[List]:
    num_points = 20
    spread = 7
    bounds = (1, 100)
    return generate_clusters(num_clusters, num_points, spread, bounds, bounds, seed)


if __name__ == '__main__':
    main()
