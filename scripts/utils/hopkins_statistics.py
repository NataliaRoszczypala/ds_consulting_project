import numpy as np
from sklearn.neighbors import NearestNeighbors

def hopkins_statistic(X, sample_size=0.05):
    """
    Compute the Hopkins statistic for the dataset X
    :param X: Array-like, shape (n_samples, n_features)
              The data to compute the Hopkins statistic for.
    :param sample_size: Float, optional, default: 0.05
                        The proportion of the dataset to use for the test.
    :return: Float, the Hopkins statistic
    """
    if isinstance(sample_size, float):
        n_samples = int(sample_size * X.shape[0])
    else:
        n_samples = sample_size

    # Randomly sample n_samples points from X
    np.random.seed(42)  # For reproducibility
    random_indices = np.random.choice(X.shape[0], n_samples, replace=False)
    random_points = X[random_indices]

    # Generate n_samples random points in the same space as X
    min_vals, max_vals = np.min(X, axis=0), np.max(X, axis=0)
    synthetic_points = np.random.uniform(low=min_vals, high=max_vals, size=(n_samples, X.shape[1]))

    # Calculate distances to the nearest neighbor in the original data
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    w_distances, _ = nbrs.kneighbors(random_points)
    u_distances, _ = nbrs.kneighbors(synthetic_points)

    # Calculate the Hopkins statistic
    H = np.sum(w_distances) / (np.sum(w_distances) + np.sum(u_distances))
    return H