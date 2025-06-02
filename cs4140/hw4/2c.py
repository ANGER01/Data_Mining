import numpy as np
import matplotlib.pyplot as plt

# Example data (replace with actual data)
X_c2 = np.loadtxt('C2.txt', usecols=(1, 2))  # Replace with your actual data

def lloyd_algorithm(X, k, max_iters=100, tol=1e-4):
    # Randomly initialize k centroids
    
    centroids = np.array([[-34.88537864, 86.48009614],[20.09888579, 60.20276647],[-31.88065781, 43.37429316]])
    #centroids = [[-38.07487215, 41.74300547],[200.0, 60.0],[32.24723563, 75.95401491]]
    #centroids = [X[1],X[2],X[3]]
    for iteration in range(max_iters):
        # Step 1: Assign each point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Compute distances to centroids
        labels = np.argmin(distances, axis=1)  # Assign each point to the closest centroid
        
        # Step 2: Update centroids to the mean of the points assigned to each centroid
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence (if centroids do not change much)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        
        centroids = new_centroids
    
    return centroids, labels

# Run Lloyd's Algorithm
k = 3
centroids, labels = lloyd_algorithm(X_c2, k)

# Report the final clusters and 3-means cost
cost = np.sum([np.linalg.norm(X_c2[i] - centroids[labels[i]])**2 for i in range(len(X_c2))]) / len(X_c2)
print(f"Final Clusters (labels): {labels}")
print(f"Centroids: {centroids}")
print(f"3-Means Cost: {cost}")

# Plot the data points and the final centroids
plt.scatter(X_c2[:, 0], X_c2[:, 1], c=labels, cmap='viridis', label="Data Points")
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', label="Centroids")
plt.legend()
plt.show()
