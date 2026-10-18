import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(
    n_samples=300,
    centers=4,
    n_features=2,
    cluster_std=1.2,
    random_state=42
)

df = pd.DataFrame(X, columns=["Annual Income", "Spending Score"])

print("CUSTOMER CLUSTERING DATASET")
print(df.head())
print("\nDataset Shape:", df.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

hierarchical = AgglomerativeClustering(n_clusters=4)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

results = []

for name, labels in [
    ("K-Means", kmeans_labels),
    ("Hierarchical Clustering", hierarchical_labels)
]:
    silhouette = silhouette_score(X_scaled, labels)
    db_index = davies_bouldin_score(X_scaled, labels)

    results.append([
        name,
        len(np.unique(labels)),
        silhouette,
        db_index
    ])

results_df = pd.DataFrame(
    results,
    columns=["Algorithm", "Number of Clusters", "Silhouette Score", "Davies-Bouldin Index"]
)

print("\nCLUSTERING PERFORMANCE")
print(results_df.round(4).to_string(index=False))

plt.figure(figsize=(8, 5))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    s=40
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids"
)
plt.title("K-Means Clustering")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=hierarchical_labels,
    cmap="viridis",
    s=40
)
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.grid()
plt.show()

linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode="lastp", p=20)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points or Clusters")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

best_algorithm = results_df.loc[
    results_df["Silhouette Score"].idxmax(),
    "Algorithm"
]

print("\nBEST ALGORITHM BASED ON SILHOUETTE SCORE:", best_algorithm)

df["KMeans Cluster"] = kmeans_labels
df["Hierarchical Cluster"] = hierarchical_labels

print("\nSAMPLE CLUSTER ASSIGNMENTS")
print(df.head(10).to_string(index=False))

print("\nClustering Evaluation Completed")
