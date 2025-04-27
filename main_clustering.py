import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clustering_methods.kmeans_clustering import kmeans_clustering
from clustering_methods.gmm_clustering import gmm_clustering, select_gmm_components
from clustering_methods.som_clustering import som_clustering
from clustering_methods.hierarchical_clustering import hierarchical_clustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


def load_and_prepare_data(
    file_path, target_cols=None, normalize=True, test_size=0.2, random_state=42
):
    # Load data
    df = pd.read_csv(file_path, index_col=0)
    df = df.dropna(axis=1, how="any")

    # Shuffle data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df = pd.get_dummies(df, columns=["Round"], drop_first=True)
    df = pd.get_dummies(df, columns=["Phase"], drop_first=True)
    df["Cohort"] = df["Cohort"].replace("D1_4", "D1_3")
    df["Cohort"] = df["Cohort"].replace("D1_5", "D1_3")
    df["Cohort"] = df["Cohort"].replace("D1_6", "D1_3")
    df["Cohort"] = df["Cohort"].str.split("_").str[1].astype(int)

    if target_cols is None:
        target_cols = ["Cohort"]

    X_cols = list(set(df.columns) - set(target_cols))
    X = df[X_cols].astype(float)
    y = df[target_cols]

    # Normalize features
    if normalize:
        for col in X_cols:
            if X[col].dtype in ["float64", "int64"]:
                X[col] = (X[col] - X[col].min()) / (X[col].max() - X[col].min())

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def calculate_within_class_dissimilarity(X, labels, centers):
    W = 0
    for cluster in range(len(centers)):
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) > 0:
            dists = np.sum((X[cluster_indices] - centers[cluster]) ** 2, axis=1)
            W += np.sum(dists)
    return W


def calculate_gap_statistic(X, n_clusters_range, n_simulations=20, random_state=42):
    N, p = X.shape
    minX = np.min(X, axis=0)
    maxX = np.max(X, axis=0)

    # Initialize arrays
    W = np.zeros(len(n_clusters_range))  # For actual data
    Wu = np.zeros((len(n_clusters_range), n_simulations))  # For simulated data

    for i, k in enumerate(n_clusters_range):
        # === Actual data clustering ===
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(X)
        W[i] = calculate_within_class_dissimilarity(
            X, kmeans.labels_, kmeans.cluster_centers_
        )

        # === Simulated data clustering ===
        for j in range(n_simulations):
            # Generate uniform reference distribution
            X_uniform = np.ones((N, 1)) * minX + np.random.rand(N, p) * (
                np.ones((N, 1)) * maxX - np.ones((N, 1)) * minX
            )

            kmeans_u = KMeans(n_clusters=k, random_state=random_state).fit(X_uniform)
            Wu[i, j] = calculate_within_class_dissimilarity(
                X_uniform, kmeans_u.labels_, kmeans_u.cluster_centers_
            )

    # Calculate Gap statistic
    log_W = np.log(W)
    log_Wu = np.log(Wu)
    Elog_Wu = np.mean(log_Wu, axis=1)
    sk = np.std(log_Wu, axis=1) * np.sqrt(1 + 1 / n_simulations)
    gap = Elog_Wu - log_W

    return gap, sk, W, Wu


def evaluate_clustering(X_train, X_test, y_train, y_test, model, method):

    results = {}

    # For hierarchical clustering, we evaluate on the entire dataset
    if method == "hierarchical":
        # Calculate accuracy (need to align cluster labels with true labels)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train.ravel())
        labels_encoded = le.fit_transform(model["labels"])
        accuracy = accuracy_score(y_encoded, labels_encoded)
        results["accuracy"] = accuracy

        # Calculate within-class dissimilarity
        W = calculate_within_class_dissimilarity(
            X_train,
            model["labels"] - 1,  # Convert to 0-based indexing
            np.array([X_train[model["labels"] == i].mean(axis=0) for i in range(1, 4)]),
        )
        results["within_class_dissimilarity"] = np.log(W)

        return results

    # For other methods, predict on test data
    if method == "kmeans":
        test_labels = model.predict(X_test)
        train_labels = model.predict(X_train)
        centers = model.cluster_centers_
        results["model"] = model

    elif method == "gmm":
        test_labels = model.predict(X_test)
        train_labels = model.predict(X_train)
        centers = model.means_
    # elif method == "som":
    #     test_labels = np.array(
    #         [
    #             model.winner(x)[0] * model._weights.shape[0] + model.winner(x)[1]
    #             for x in X_test
    #         ]
    #     )
    #     train_labels = np.array(
    #         [
    #             model.winner(x)[0] * model._weights.shape[0] + model.winner(x)[1]
    #             for x in X_train
    #         ]
    #     )
    #     centers = model.get_weights().reshape(-1, X_train.shape[1])

    # Calculate accuracy (need to align cluster labels with true labels)
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test.ravel())
    test_labels_encoded = le.fit_transform(test_labels)
    accuracy = accuracy_score(y_test_encoded, test_labels_encoded)

    # Calculate within-class dissimilarity
    W = calculate_within_class_dissimilarity(X_train, train_labels, centers)

    # Calculate Gap statistic (only for K-means)
    if method == "kmeans":
        gap, sk, W_all, Wu = calculate_gap_statistic(X_train, range(2, 11))
        # Find optimal number of clusters using Gap statistic
        optimal_k = np.where(gap[:-1] >= gap[1:] - sk[1:])[0]
        if len(optimal_k) == 0:
            optimal_k = len(gap)
        else:
            optimal_k = optimal_k[0] + 2  # +2 because we start from k=2
        results["optimal_k"] = optimal_k
        results["gap_statistic"] = gap[optimal_k - 2]  # Store gap value for optimal k
        results["gap_standard_error"] = sk[
            optimal_k - 2
        ]  # Store standard error for optimal k

    # Calculate AIC and BIC (only for GMM)
    if method == "gmm":
        aic = model.aic(X_test)
        bic = model.bic(X_test)
        results["aic"] = aic
        results["bic"] = bic

    results["accuracy"] = accuracy
    results["within_class_dissimilarity"] = np.log(W)

    return results


def compare_clustering_methods(
    X_train, X_test, y_train, y_test, methods=None, **kwargs
):

    if methods is None:
        methods = ["kmeans", "gmm", "som", "hierarchical"]

    results = {}
    evaluations = {}

    # Handle hierarchical clustering separately since it doesn't use train/test split
    if "hierarchical" in methods:
        hierarchical_kwargs = kwargs.get("hierarchical", {})
        if "n_clusters" not in hierarchical_kwargs:
            hierarchical_kwargs["n_clusters"] = 3
        # For hierarchical, we use the entire dataset
        X_full = np.vstack((X_train, X_test))
        y_full = np.vstack((y_train, y_test))
        results["hierarchical"] = hierarchical_clustering(X_full, **hierarchical_kwargs)
        evaluations["hierarchical"] = evaluate_clustering(
            X_full, None, y_full, None, results["hierarchical"], "hierarchical"
        )
        methods.remove(
            "hierarchical"
        )  # Remove from methods to handle others with train/test

    # K-means clustering
    if "kmeans" in methods:
        kmeans_kwargs = kwargs.get("kmeans", {})
        results["kmeans"] = kmeans_clustering(X_train, **kmeans_kwargs)
        evaluations["kmeans"] = evaluate_clustering(
            X_train, X_test, y_train, y_test, results["kmeans"]["model"], "kmeans"
        )

    # GMM clustering
    if "gmm" in methods:
        gmm_kwargs = kwargs.get("gmm", {})
        gmm_kwargs["n_components"] = 3
        results["gmm"] = gmm_clustering(X_train, **gmm_kwargs)
        evaluations["gmm"] = evaluate_clustering(
            X_train, X_test, y_train, y_test, results["gmm"]["model"], "gmm"
        )

    # SOM clustering
    if "som" in methods:
        som_kwargs = kwargs.get("som", {})
        results["som"] = som_clustering(X_train, **som_kwargs)
        evaluations["som"] = evaluate_clustering(
            X_train, X_test, y_train, y_test, results["som"]["model"], "som"
        )

    return results, evaluations


def plot_comparison(results, X, y=None, figsize=(10, 8)):

    figures = {}

    for method, result in results.items():
        if method == "gmm_component_selection":
            continue

        title = f"PCA with {method.capitalize()} Clusters"
        fig = plot_clustering_results(
            X, result["model"], cohort_labels=y, title=title, figsize=figsize
        )
        figures[method] = fig

    return figures


def plot_clustering_results(
    X,
    cohort_labels=None,
    predicted_labels=None,
    title="Clustering Results",
    figsize=(10, 8),
    ax=None,
):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Define vibrant colors for cohorts
    cmap = plt.cm.tab10  # Standard colormap supporting up to 10 distinct colors

    # Perform PCA for visualization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # Plot cohort points
    # Plot cohort points with individual labels
    unique_cohorts = np.unique(cohort_labels)
    for i in unique_cohorts:
        mask = cohort_labels == i
        X_pca_mask = X_pca[mask.squeeze()]
        ax.scatter(
            X_pca_mask[:, 0],
            X_pca_mask[:, 1],
            X_pca_mask[:, 2],
            c=[cmap(i)],
            s=100,
            alpha=0.6,
            label=f"Cohort {i}",
        )

    # Plot predicted points with individual labels
    if predicted_labels is not None:
        unique_clusters = np.unique(predicted_labels)
        for i in unique_clusters:
            mask = predicted_labels == i
            X_pca_mask = X_pca[mask.squeeze()]
            ax.scatter(
                X_pca_mask[:, 0],
                X_pca_mask[:, 1],
                X_pca_mask[:, 2],
                c=[cmap(i)],
                alpha=1,
                linewidths=2,
                marker="x",
                s=80,
                label=f"Cluster {i}",
            )

    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")

    # Create custom legend

    ax.legend()
    fig.tight_layout()
    return fig


def plot_evaluation_metrics(evaluations):
    """
    Plot evaluation metrics for all clustering methods.

    Parameters:
    -----------
    evaluations : dict
        Dictionary containing evaluation metrics for each method
    """
    methods = list(evaluations.keys())
    metrics = ["accuracy", "within_class_dissimilarity"]

    # Plot accuracy and within-class dissimilarity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    accuracies = [evaluations[method]["accuracy"] for method in methods]
    ax1.bar(methods, accuracies)
    ax1.set_title("Accuracy")
    ax1.set_ylim(0, 1)

    # Within-class dissimilarity plot
    dissimilarities = [
        evaluations[method]["within_class_dissimilarity"] for method in methods
    ]
    ax2.bar(methods, dissimilarities)
    ax2.set_title("Within-class Dissimilarity")

    plt.tight_layout()
    plt.show()

    # Plot Gap statistic if available
    if "kmeans" in evaluations and "gap_statistic" in evaluations["kmeans"]:
        plt.figure(figsize=(8, 5))
        k_range = range(2, 11)
        gap, sk, W, Wu = calculate_gap_statistic(
            X_train, k_range
        )


        plt.errorbar(k_range, gap, yerr=sk, marker="o", color="blue", capsize=5)
        plt.xlabel("Number of clusters - k")
        plt.ylabel("G(K) ± s_k")
        plt.title("Gap Statistic for K-means")
        plt.axvline(
            x=evaluations["kmeans"]["optimal_k"],
            color="red",
            linestyle="--",
            label=f'Optimal k = {evaluations["kmeans"]["optimal_k"]}',
        )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test = load_and_prepare_data("data/HR_data.csv")

    # Compare all methods
    results, evaluations = compare_clustering_methods(
        X_train,
        X_test,
        y_train,
        y_test,
        methods=["kmeans", "gmm", "hierarchical"],  # Excluding SOM for now
        kmeans={"n_clusters": 3},
        gmm={"n_components": 3},
    )

    # Print evaluation results
    print("\nEvaluation Results:")
    print("-" * 50)
    # evaluations["kmeans"].pop("model")
    for method, metrics in evaluations.items():
        print(f"\n{method.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, np.ndarray):
                print(f"{metric}: {list(value)}")  # Print first value if array
            if metric == "model":
                continue
            else:
                print(f"{metric}: {value:.4f}")

    # Plot evaluation metrics
    plot_evaluation_metrics(evaluations)

    # Plot clustering results
    for method in results.keys():
        # For hierarchical, plot the entire dataset
        if method == "hierarchical":
            X_full = np.vstack((X_train, X_test))
            y_full = np.vstack((y_train, y_test))
            plot_clustering_results(
                X_full,
                cohort_labels=y_full,
                predicted_labels=results[method]["labels"],
                title=method,
            )
        else:
            plot_clustering_results(
                X_train,
                cohort_labels=y_train,
                predicted_labels=results[method]["labels"],
                title=method,
            )

    plt.show()
