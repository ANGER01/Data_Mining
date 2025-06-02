import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset C2.txt (assuming the file has space-separated values, adapt if needed)
X_c2 = np.loadtxt('C3.txt')  # Read only columns for coordinates

c1_point = X_c2[1]  # Point with index 1

# Run k-means++ with the first centroid as point c1_point
def run_kmeans_plus_plus(X, k):
    # Initialize k-means with k-means++ for the remaining centroids
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42)
    
    # Fit the model first (this generates the initial centroids including k-means++)
    kmeans.fit(X)
    
    return kmeans.cluster_centers_, kmeans.labels_

# Run multiple trials (at least 20)
num_trials = 20
costs = []
labels_list = []
center_list = []
for trial in range(num_trials):
    centroids, labels = run_kmeans_plus_plus(X_c2, 3)
    center_list.append(centroids)
    # Compute the 3-means cost
    cost = np.sum([np.linalg.norm(X_c2[i] - centroids[labels[i]])**2 for i in range(len(X_c2))]) / len(X_c2)
    costs.append(cost)
    
    # Store the labels for comparison later
    labels_list.append(labels)

# Plot the cumulative density function (CDF) of the 3-means cost
plt.figure(figsize=(8, 6))
plt.axis("equal")
plt.hist(costs, bins=30, cumulative=True, density=True, histtype='step', label='CDF of 3-means cost')
plt.xlabel('3-Means Cost')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Density Function of 3-Means Cost')
plt.legend()
plt.show()

# Check fraction of times the clustering matches Gonzalez (for simplicity, assume a function to get Gonzalez clusters)
def compare_with_gonzalez(labels_list, gonzalez_labels):
    match_count = sum(np.array_equal(labels, gonzalez_labels) for labels in labels_list)
    return match_count / len(labels_list)

# Assuming you have the labels from Gonzalez from a previous run
gonzalez_labels = np.array([[-38.07487215, 41.74300547], [200, 60], [32.24723563, 75.95401491]])  # Replace this with the actual Gonzalez clustering result
fraction_matching = compare_with_gonzalez(labels_list, gonzalez_labels)
print(f"Fraction of times subsets match Gonzalez: {fraction_matching}")

# Scatter plot for one of the trials (e.g., trial 0)
centroids, labels = run_kmeans_plus_plus(X_c2, 3)

plt.figure(figsize=(8, 6))

# Plot data points, color-coded by cluster label
for i in range(3):  # For each cluster
    cluster_points = X_c2[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')

# Plot centroids as red stars
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='*', label='Centroids')
plt.axis("equal")
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('K-Means++ Clustering (Trial 0)')
plt.legend()
plt.show()
