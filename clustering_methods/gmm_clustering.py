import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


def gmm_clustering(X, n_components=3, covariance_type="diag", random_state=0):
    """
    Perform Gaussian Mixture Model clustering on the input data.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data for clustering
    n_components : int, default=3
        Number of mixture components
    covariance_type : str, default='diag'
        Type of covariance parameters to use
    random_state : int, default=0
        Random state for reproducibility

    Returns:
    --------
    dict
        Dictionary containing:
        - 'model': Trained GMM model
        - 'labels': Cluster labels for each sample (0-based)
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    labels = gmm.fit_predict(X)
    labels = labels + 1
    return {"model": gmm, "labels": labels}


def select_gmm_components(X, max_components=10, covariance_type="diag", random_state=0):
    """
    Select optimal number of components using AIC and BIC.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    max_components : int, default=10
        Maximum number of components to try
    covariance_type : str, default='diag'
        Type of covariance parameters to use
    random_state : int, default=0
        Random state for reproducibility

    Returns:
    --------
    dict
        Dictionary containing:
        - 'aic_scores': AIC scores for each number of components
        - 'bic_scores': BIC scores for each number of components
        - 'optimal_n_components': Optimal number of components based on BIC
    """
    aic_scores = np.zeros(max_components)
    bic_scores = np.zeros(max_components)

    for k in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type=covariance_type, random_state=random_state
        ).fit(X)
        aic_scores[k - 1] = gmm.aic(X)
        bic_scores[k - 1] = gmm.bic(X)

    optimal_n_components = np.argmin(bic_scores) + 1

    return {
        "aic_scores": aic_scores,
        "bic_scores": bic_scores,
        "optimal_n_components": optimal_n_components,
    }
