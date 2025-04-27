import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


def hierarchical_clustering(
    X, n_clusters=3, method="ward", metric="euclidean", criterion="maxclust"
):
    """
    Perform hierarchical clustering on the input data.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data for clustering
    n_clusters : int, default=3
        Number of clusters to form
    method : str, default='ward'
        Linkage method to use
    metric : str, default='euclidean'
        Distance metric to use
    criterion : str, default='maxclust'
        Criterion to form flat clusters

    Returns:
    --------
    dict
        Dictionary containing:
        - 'model': Dictionary with linkage matrix and clustering parameters
        - 'labels': Cluster labels for each sample (1-based)
    """
    Z = linkage(X, method=method, metric=metric)
    labels = fcluster(Z, n_clusters, criterion=criterion)

    model = {
        "linkage_matrix": Z,
        "method": method,
        "metric": metric,
        "criterion": criterion,
        "n_clusters": n_clusters,
    }
    return {"model": model, "labels": labels}
