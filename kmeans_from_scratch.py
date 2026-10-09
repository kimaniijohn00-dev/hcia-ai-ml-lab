# Answer to ML Lab question 1.3: "How to implement K-means from scratch using Python?"
# Uses only numpy (matplotlib just for the graph).
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs   # only used to make the same dataset as the lab


def kmeans(X, k, max_iters=100, seed=0):
    rng = np.random.default_rng(seed)
    # 1. Pick k random points from the data as the starting centroids.
    centroids = X[rng.choice(len(X), size=k, replace=False)]

    for _ in range(max_iters):
        # 2. Assign every point to its nearest centroid (Euclidean distance).
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # 3. Move each centroid to the mean of the points assigned to it.
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])

        # 4. Stop when the centroids no longer move.
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels, centroids


# Same dataset as Experiment 5.
X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

for k in (3, 4):
    labels, centroids = kmeans(X, k)
    print("k =", k, "centroids:\n", centroids)

    color = ["red", "pink", "orange", "green"]
    fig, ax = plt.subplots(1)
    for i in range(k):
        ax.scatter(X[labels == i, 0], X[labels == i, 1], marker='o', s=8, c=color[i])
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=15, c="black")
    ax.set_title("K-means from scratch, k = {}".format(k))
    plt.show()
