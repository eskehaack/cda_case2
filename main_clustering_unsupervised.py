import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from clustering_methods.kmeans_clustering import kmeans_clustering
from clustering_methods.gmm_clustering import gmm_clustering
from clustering_methods.hierarchical_clustering import hierarchical_clustering


# -------------------------------------------------
# 1. DATA HANDLING
# -------------------------------------------------


def load_and_prepare_data(
    file_path: str, target_cols=None, normalize=True, random_state=42
):
    """Read the CSV, minimal cleaning, return X (features) and y (cohort) arrays."""
    df = pd.read_csv(file_path, index_col=0)
    df = df.dropna(axis=0, how="any")
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Basic categorical handling (specific to HR_data.csv)
    df = pd.get_dummies(df, columns=["Round"])
    df = pd.get_dummies(df, columns=["Phase"])
    df["Cohort"] = (
        df["Cohort"]
        .replace({"D1_4": "D1_3", "D1_5": "D1_3", "D1_6": "D1_3"})
        .str.split("_")
        .str[1]
        .astype(int)
    )
    df = pd.get_dummies(df, columns=["Cohort"])

    if target_cols is not None:
        X_cols = list(set(df.columns) - set(target_cols))
        y = df[target_cols].values.squeeze()
    else:
        X_cols = list(df.columns)
        y = None

    X = df[X_cols].astype(float)

    if normalize:
        X = (X - X.min()) / (X.max() - X.min())

    return X.values, y, df


# -------------------------------------------------
# 2. METRIC HELPERS
# -------------------------------------------------


def within_cluster_dispersion(
    X: np.ndarray, labels0: np.ndarray, centers: np.ndarray
) -> float:
    """Total within-cluster squared dispersion (W). labels must be 0-based."""
    W = 0.0
    for k in np.unique(labels0):
        idx = np.where(labels0 == k)[0]
        if idx.size:
            dists = np.sum((X[idx] - centers[k]) ** 2, axis=1)
            W += np.sum(dists)
    return W


def gap_statistic_kmeans(
    X: np.ndarray, k_range=range(2, 11), B: int = 20, random_state: int = 42
):
    """Compute Tibshirani et al. Gap statistic for K‑means over *k_range*."""
    N, p = X.shape
    xmin, xmax = X.min(axis=0), X.max(axis=0)

    def ref_sample():
        return xmin + np.random.rand(N, p) * (xmax - xmin)

    W = np.zeros(len(k_range))
    W_ref = np.zeros((len(k_range), B))

    for i, k in enumerate(k_range):
        km = KMeans(n_clusters=k, random_state=random_state).fit(X)
        W[i] = within_cluster_dispersion(X, km.labels_, km.cluster_centers_)
        for b in range(B):
            Xb = ref_sample()
            km_b = KMeans(n_clusters=k, random_state=random_state).fit(Xb)
            W_ref[i, b] = within_cluster_dispersion(
                Xb, km_b.labels_, km_b.cluster_centers_
            )

    gap = np.mean(np.log(W_ref), axis=1) - np.log(W)
    sk = np.std(np.log(W_ref), axis=1) * np.sqrt(1 + 1.0 / B)
    return list(k_range), gap, sk


def evaluate_generic(
    X: np.ndarray, labels_any: np.ndarray, centers: np.ndarray, include_gap=False
):
    """Silhouette + Within‑cluster dispersion; optionally Gap (K‑means‑only)."""
    labels0 = labels_any - 1 if labels_any.min() == 1 else labels_any  # ensure 0‑based

    res = {}
    if 1 < np.unique(labels0).size < len(X):
        res["silhouette"] = silhouette_score(X, labels0)
    else:
        res["silhouette"] = np.nan

    res["within_dispersion"] = within_cluster_dispersion(X, labels0, centers)

    if include_gap:
        k_range, gap, sk = gap_statistic_kmeans(X)
        res.update({"gap_k": k_range, "gap": gap, "gap_sk": sk})

    return res


# -------------------------------------------------
# 3. RUN ALL ALGORITHMS
# -------------------------------------------------


def run_all(X: np.ndarray, y: np.ndarray, n_clusters: int = 3):
    metrics = {}
    models = {}

    # --- K‑MEANS ---
    km = kmeans_clustering(X, n_clusters=n_clusters)
    km_cent = km["model"].cluster_centers_
    metrics["kmeans"] = evaluate_generic(X, km["labels"], km_cent, include_gap=True)
    models["kmeans"] = km

    # --- GMM ---
    gmm = gmm_clustering(X, n_components=n_clusters)
    gmm_cent = gmm["model"].means_
    gmm_mets = evaluate_generic(X, gmm["labels"], gmm_cent)
    # add AIC/BIC (likelihood‑based; only meaningful for GMM)
    gmm_mets["aic"] = gmm["model"].aic(X)
    gmm_mets["bic"] = gmm["model"].bic(X)
    metrics["gmm"] = gmm_mets
    models["gmm"] = gmm

    # --- HIERARCHICAL ---
    hier = hierarchical_clustering(X, n_clusters=n_clusters)
    hier_cent = np.vstack(
        [X[hier["labels"] == (k + 1)].mean(axis=0) for k in range(n_clusters)]
    )
    metrics["hierarchical"] = evaluate_generic(X, hier["labels"], hier_cent)
    models["hierarchical"] = hier

    return models, metrics


# -------------------------------------------------
# 4. VISUALISATION
# -------------------------------------------------


def plot_3d_clusters(X: np.ndarray, labels: np.ndarray, title: str, cohort_labels=None):
    pca = PCA(n_components=3)
    X3 = pca.fit_transform(X)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.tab10

    if cohort_labels is not None:
        for coh in np.unique(cohort_labels):
            mask = cohort_labels == coh
            ax.scatter(
                X3[mask, 0],
                X3[mask, 1],
                X3[mask, 2],
                s=60,
                alpha=0.8,
                c=[cmap(int(coh) % 10)],
                label=f"Cohort {coh}",
            )

    for k in np.unique(labels):
        mask = labels == k
        ax.scatter(
            X3[mask, 0],
            X3[mask, 1],
            X3[mask, 2],
            marker="x",
            s=90,
            label=f"Cluster {k}",
            c=[cmap(int(k) % 10)],
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    fig.tight_layout()
    return fig


def print_cluster_labels(labels: np.ndarray, name, cohort_labels=None):
    """Prints the cluster labels, their corresponding cohort labels, and proportions."""
    print(f"\n--- {name.upper()} CLUSTER LABELS ---")
    total_samples = len(labels)
    if cohort_labels is not None:
        print("\n--- CLUSTER LABELS ---")
        for k in np.unique(labels):
            mask = labels == k
            cluster_size = mask.sum()
            print(f"Cluster {k}:")
            for coh in np.unique(cohort_labels):
                coh_mask = cohort_labels == coh
                overlap = (mask & coh_mask).sum()
                proportion = (overlap / cluster_size) * 100 if cluster_size > 0 else 0
                print(f"  Cohort {coh}: {overlap} samples ({proportion:.2f}%)")
    else:
        print("\n--- CLUSTER LABELS ---")
        for k in np.unique(labels):
            mask = labels == k
            proportion = mask.sum() / total_samples * 100
            print(f"Cluster {k}: {mask.sum()} samples ({proportion:.2f}%)")

# -------------------------------------------------
# 5. MAIN
# -------------------------------------------------

def main():
    X, y, df = load_and_prepare_data("HR_data.csv", target_cols=None)

    # Reconstruct cohort labels from one-hot columns
    cohort_cols = [col for col in df.columns if col.startswith("Cohort_")]
    plot_labels = (
        df[cohort_cols].idxmax(axis=1).str.extract(r"Cohort_(\d+)").astype(int).squeeze()
        if cohort_cols else None
    )

    models, metrics = run_all(X, y=None, n_clusters=3)

    print("\n--- METRIC SUMMARY ---")
    for name, m in metrics.items():
        print(f"\n{name.upper()}:")
        print(f"Silhouette: {m['silhouette']:.4f}" if not np.isnan(m["silhouette"]) else "Silhouette: n/a")
        print(f"Within‑cluster dispersion (W): {m['within_dispersion']:.2f}")
        if name == "kmeans":
            opt_k = m["gap_k"][np.argmax(m["gap"])]
            print(f"Gap statistic optimal k: {opt_k}")
        if name == "gmm":
            print(f"AIC: {m['aic']:.2f}  |  BIC: {m['bic']:.2f}")

    for name, mdl in models.items():
        plot_3d_clusters(X, mdl["labels"], name, cohort_labels=plot_labels)
        print_cluster_labels(mdl["labels"], name, cohort_labels=plot_labels)

    plt.show()



if __name__ == "__main__":
    main()
