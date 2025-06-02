import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster

# Load the dataset (Assumes space-separated values, skipping index column)
data = np.loadtxt("C1.txt", usecols=(1, 2))

# Choose the linkage method: 'single' for Single-Link, 'complete' for Complete-Link
linkage_method = 'complete'  # Change to 'complete' for Complete-Link

# Perform hierarchical clustering
Z = linkage(data, method=linkage_method)

# Choose the number of clusters (k = 4)
num_clusters = 4
labels = fcluster(Z, num_clusters, criterion='maxclust')

# Group points into clusters
clusters = {i: [] for i in range(1, num_clusters + 1)}
for point, label in zip(data, labels):
    clusters[label].append(point)

# Convert lists to numpy arrays
clusters = {k: np.array(v) for k, v in clusters.items()}

# Print clustered sets
print(f"Results for {linkage_method.capitalize()}-Link Clustering (k={num_clusters}):\n")
for cluster_id, points in clusters.items():
    print(f"Cluster {cluster_id}:")
    print(points)
    print()

# Plot the clusters
plt.figure(figsize=(6, 6))
for cluster_id, points in clusters.items():
    plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {cluster_id}', s=100, edgecolors='k')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Hierarchical Clustering ({linkage_method.capitalize()}-Link, k={num_clusters})")
plt.legend()
plt.show()
