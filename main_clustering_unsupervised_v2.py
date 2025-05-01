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


def load_and_prepare_data(file_path: str, target_col=None, normalize=True, random_state=42):
    """Load CSV, clean, extract target, encode features, return X, y, and full DataFrame (df)."""
    df = pd.read_csv(file_path, index_col=0)
    df = df.dropna(axis=0, how="any")
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    df.columns = df.columns.str.strip()  # Clean column names

    
    df["Cohort"] = (
        df["Cohort"]
        .replace({"D1_4": "D1_3", "D1_5": "D1_3", "D1_6": "D1_3"})
        .str.split("_")
        .str[1]
        .astype(int)
    )
    df["Round"] = df["Round"].str.split("_").str[1].astype(int)  
    df["Phase"] = df["Phase"].str.split("e").str[1].astype(int)

    # Basic categorical handling (specific to HR_data.csv)
    categorical_cols = ["Cohort", "Round", "Phase"]
    categorical_cols = [col for col in categorical_cols if col in df.columns and col != target_col]
    df = pd.get_dummies(df, columns=categorical_cols)

    # Extract target column if specified
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")
        y = df[target_col].values.squeeze()
        X_cols = df.drop(columns = target_col).columns.tolist()
    else:
        y = None
        X_cols = list(df.columns)

    

    
    
    
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


def plot_3d_clusters(X: np.ndarray, labels: np.ndarray, title: str, target_labels=None,  target_name=None):
    pca = PCA(n_components=3)
    X3 = pca.fit_transform(X)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.tab10

    
    if target_labels is not None:
        for val in np.unique(target_labels):
            mask = target_labels == val
            ax.scatter(
                X3[mask, 0],
                X3[mask, 1],
                X3[mask, 2],
                s=60,
                alpha=0.8,
                c=[cmap(int(val) % 10)],
                label=f"{target_name} {val}",
            )

    for k in np.unique(labels):
        mask = labels == k
        ax.scatter(
            X3[mask, 0],
            X3[mask, 1],
            X3[mask, 2],
            marker="x",
            s=90,
            c=[cmap(int(k) % 10)],
            label=f"Cluster {k}",
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    fig.tight_layout()
    return fig


def print_cluster_labels(labels: np.ndarray, name, target_labels=None, target_name= "Target"):
    """Prints the cluster labels, their corresponding cohort labels, and proportions."""
    print(f"\n--- {name.upper()} CLUSTER LABELS ---")
    total_samples = len(labels)
    if target_labels is not None:
        print("\n--- CLUSTER LABELS ---")
        for k in np.unique(labels):
            mask = labels == k
            cluster_size = mask.sum()
            print(f"Cluster {k}:")
            for coh in np.unique(target_labels):
                coh_mask = target_labels == coh
                overlap = (mask & coh_mask).sum()
                proportion = (overlap / cluster_size) * 100 if cluster_size > 0 else 0
                print(f"  {target_name} {coh}: {overlap} samples ({proportion:.2f}%)")
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
    target = "Cohort" #choose target column for supervised or None for unsupervised
    X, y, df = load_and_prepare_data("HR_data.csv", target_col=None) # chose target or none for unsupervised

    # If no explicit y was extracted, reconstruct it from df for plotting only
    if y is None:
        target_cols = [col for col in df.columns if col.startswith(f"{target}_")]
        if target_cols:
            plot_labels = (
                df[target_cols].idxmax(axis=1)
                .str.extract(r"_(\d+)")
                .squeeze()
                .astype(float)
                .dropna()
                .astype(int)
                .values
            )
        else:
            plot_labels = None
    else:
        # Convert categorical y (if necessary) for plotting
        if not np.issubdtype(y.dtype, np.integer):
            try:
                y = pd.Series(y).astype(int).values
            except ValueError:
                y = pd.Series(y).astype("category").cat.codes.values
        plot_labels = y

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
        plot_3d_clusters(X, mdl["labels"], name, target_labels=plot_labels, target_name=target)
        print_cluster_labels(mdl["labels"], name, target_labels=plot_labels)

    plt.show()


if __name__ == "__main__":
    main()
