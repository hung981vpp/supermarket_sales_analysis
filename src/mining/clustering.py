import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Tìm số cụm tối ưu bằng Elbow Method, trả về dict {k: inertia}.
def elbow_method(X: np.ndarray, k_range: range = range(2, 11), random_state: int = 42) -> dict:
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertias[k] = km.inertia_
    return inertias

# Tính Silhouette Score và Davies-Bouldin Index cho từng k.
def evaluate_k(X: np.ndarray, k_range: range = range(2, 11), random_state: int = 42) -> pd.DataFrame:
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        records.append({
            "k": k,
            "inertia":         km.inertia_,
            "silhouette":      silhouette_score(X, labels),
            "davies_bouldin":  davies_bouldin_score(X, labels),
        })
    return pd.DataFrame(records)

# Chạy KMeans với số cụm lấy từ params, trả về labels và model.
def run_kmeans(X: np.ndarray, params: dict) -> tuple:
    n_clusters   = params["mining"]["clustering"]["n_clusters"]
    random_state = params["mining"]["clustering"]["random_state"]
    n_init       = params["mining"]["clustering"]["n_init"]

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    print(f"[kmeans] n_clusters={n_clusters} | "
          f"silhouette={silhouette_score(X, labels):.4f} | "
          f"davies_bouldin={davies_bouldin_score(X, labels):.4f}")
    return labels, km

# Chạy Agglomerative Clustering (HAC) với linkage ward.
def run_hac(X: np.ndarray, params: dict) -> tuple:
    n_clusters = params["mining"]["clustering"]["n_clusters"]
    hac = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = hac.fit_predict(X)
    print(f"[hac] n_clusters={n_clusters} | "
          f"silhouette={silhouette_score(X, labels):.4f} | "
          f"davies_bouldin={davies_bouldin_score(X, labels):.4f}")
    return labels, hac

# Gắn nhãn cụm vào DataFrame khách hàng.
def assign_clusters(df: pd.DataFrame, labels: np.ndarray, col: str = "Cluster") -> pd.DataFrame:
    df = df.copy()
    df[col] = labels
    return df

# Profiling cụm: tính mean các đặc trưng theo từng cụm + số lượng thành viên.
def profile_clusters(df: pd.DataFrame, feature_cols: list, cluster_col: str = "Cluster") -> pd.DataFrame:
    profile = df.groupby(cluster_col)[feature_cols].mean()
    profile["Count"] = df.groupby(cluster_col).size()
    return profile.reset_index()

# Giảm chiều xuống 2D bằng PCA để visualize các cụm.
def reduce_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)

# Chạy toàn bộ pipeline phân cụm, trả về df có cột Cluster và bảng profile.
def run_clustering_pipeline(df_features: pd.DataFrame, params: dict) -> tuple:
    exclude_cols = ["Customer ID"]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]
    X = df_features[feature_cols].values

    labels, model = run_kmeans(X, params)
    df_clustered  = assign_clusters(df_features, labels)
    profile       = profile_clusters(df_clustered, feature_cols)

    print(f"[clustering] Cluster distribution:\n{df_clustered['Cluster'].value_counts().sort_index()}")
    return df_clustered, profile, model
