import numpy as np
from minisom import MiniSom


def som_clustering(
    X, som_shape=(150, 150), sigma=10, learning_rate=0.5, num_iterations=5000
):
    """
    Perform Self-Organizing Map clustering on the input data.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data for clustering
    som_shape : tuple, default=(150, 150)
        Shape of the SOM grid
    sigma : float, default=10
        Spread of the neighborhood function
    learning_rate : float, default=0.5
        Learning rate
    num_iterations : int, default=5000
        Number of training iterations

    Returns:
    --------
    dict
        Dictionary containing:
        - 'model': Trained SOM model
        - 'labels': Cluster labels for each sample (based on SOM grid position)
    """
    som = MiniSom(
        som_shape[0], som_shape[1], X.shape[1], sigma=sigma, learning_rate=learning_rate
    )
    som.pca_weights_init(X)
    som.train_random(X, num_iterations, verbose=True)

    # Get cluster labels based on SOM grid position
    labels = np.array([som.winner(x)[0] * som_shape[1] + som.winner(x)[1] for x in X])
    return {"model": som, "labels": labels}
