import numpy as np
from sklearn.cluster import KMeans


def kmeans_clustering(X, n_clusters=3, random_state=42):
    """
    Perform K-means clustering on the input data.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data for clustering
    n_clusters : int, default=3
        Number of clusters to form
    random_state : int, default=42
        Random state for reproducibility

    Returns:
    --------
    dict
        Dictionary containing:
        - 'model': Trained KMeans model
        - 'labels': Cluster labels for each sample (0-based)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans = kmeans.fit(X)
    labels = kmeans.predict(X)
    labels = labels + 1
    return {"model": kmeans, "labels": labels}
