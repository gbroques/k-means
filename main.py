from typing import List

from dataviz import generate_clusters
from dataviz import plot_clusters
from kmeans import KMeans


def main():
    num_clusters = 4
    clusters = generate_data(num_clusters)
    k_means = KMeans(num_clusters=num_clusters)
    k_means.fit(clusters)
    plot_clusters(clusters, k_means.labels_)


def generate_data(num_clusters: int) -> List[List]:
    num_points = 20
    spread = 10
    bounds = (1, 100)
    return generate_clusters(num_clusters, num_points, spread, bounds, bounds)


if __name__ == '__main__':
    main()
