import numpy as np
import matplotlib.pyplot as plt

# Gonzalez algorithm implementation
def gonzalez(X, k):
    centers = [X[1]]
    for _ in range(1, k):
        distances = np.array([min(np.linalg.norm(x - c) for c in centers) for x in X])
        next_center = X[np.argmax(distances)]  # Pick the farthest point
        centers.append(next_center)
    
    assignments = np.array([np.argmin([np.linalg.norm(x - c) for c in centers]) for x in X])
    return np.array(centers), assignments

# Gonzalez 3-center cost
def gonzalez_3center_cost(X, centers, labels):
    max_cost = 0
    for x, label in zip(X, labels):
        cost = np.linalg.norm(x - centers[label])
        max_cost = max(max_cost, cost)
    return max_cost

# Gonzalez 3-means cost
def gonzalez_3means_cost(X, centers, labels):
    total_cost = 0
    for x, label in zip(X, labels):
        cost = np.linalg.norm(x - centers[label])**2
        total_cost += cost
    return total_cost / len(X)

# Example usage
X = np.loadtxt("C2.txt", usecols=(1, 2))
k = 3  # Number of clusters
centers, labels = gonzalez(X, k)

# Plotting the clusters and centroids
plt.figure(figsize=(8, 6))
plt.axis("equal")
for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')
plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=100, label='Centroids')
plt.title(f'Clusters and Centroids (k={k})')
plt.legend()
plt.show()

# Compute the costs
cost_3center = gonzalez_3center_cost(X, centers, labels)
cost_3means = gonzalez_3means_cost(X, centers, labels)

print(f"Centers:\n {centers}")
print(f"3-Center Cost: {cost_3center}")
print(f"3-Means Cost: {cost_3means}")
