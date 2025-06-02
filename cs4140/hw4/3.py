import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset C1.txt (assuming the file has space-separated values)
X_c2 = np.loadtxt('C2.txt', usecols=(1,2))

# Run k-means++ with the first centroid as point c1_point
def run_kmeans_plus_plus(X, k):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42)
    kmeans.fit(X)
    return kmeans.cluster_centers_, kmeans.labels_

# Run multiple trials (at least 20)
num_trials = 20
costs = []
labels_list = []
centroids_list = []

for trial in range(num_trials):
    centroids, labels = run_kmeans_plus_plus(X_c2, 3)
    
    # Compute the 3-means cost
    cost = np.sum([np.linalg.norm(X_c2[i] - centroids[labels[i]])**2 for i in range(len(X_c2))]) / len(X_c2)
    costs.append(cost)
    labels_list.append(labels)
    centroids_list.append(centroids)
    
    print(f"Trial {trial+1}: 3-means Cost = {cost:.4f}")

# Scatter plot for one of the trials (e.g., trial 0)
trial_idx = 0  # You can change this to visualize different trials
centroids = centroids_list[trial_idx]
labels = labels_list[trial_idx]

plt.figure(figsize=(8, 6))

# Plot data points, color-coded by cluster label
for i in range(6):  # For each cluster
    cluster_points = X_c2[labels == i]  # Select points belonging to cluster i
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')

# Plot centroids as red stars
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='*', label='Centroids')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title(f'K-Means++ Clustering (Trial {trial_idx+1})')
plt.legend()
plt.show()
