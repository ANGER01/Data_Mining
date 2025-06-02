import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp

def generate_unit_vectors(t, d):
    """Generates t unit vectors in R^d uniformly from the unit sphere S^(d-1)."""
    vectors = np.random.randn(t, d)  # Gaussian vectors
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize to unit length
    return vectors

def sang(a, b):
    """Computes the angular similarity between two vectors a and b."""
    dot_product = np.dot(a, b)
    cosine_similarity = np.clip(dot_product, -1.0, 1.0)
    return 1 - (1 / np.pi) * np.arccos(cosine_similarity)

def compute_pairwise_similarities(data):
    """Computes all pairwise angular similarities for a dataset."""
    n = data.shape[0]
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(sang(data[i], data[j]))
    return np.array(similarities)

# Generate datasets
np.random.seed(42)
d = 100
t = 500  # Number of points

def load_data(file_path):
    """
    Load data from a CSV file into a numpy array, skipping headers and assuming space-separated.
    Args:
        file_path (str): The path to the file to load.
    """
    df = pd.read_csv(file_path, header=0)  # Adjust delimiter if needed
    return df.to_numpy()

# Generate two datasets with 500 points in 100-dimensional space
np.random.seed(42)  # For reproducibility
R1 = load_data("R1.csv")
R2 = load_data("R2.csv")

# Randomly sampled unit vectors (for comparison)
random_vectors = generate_unit_vectors(t, d)

# Compute pairwise angular similarities
sim_R1 = compute_pairwise_similarities(R1)
sim_R2 = compute_pairwise_similarities(R2)
sim_random = compute_pairwise_similarities(random_vectors)

# Compute empirical CDFs
sorted_R1 = np.sort(sim_R1)
sorted_R2 = np.sort(sim_R2)
sorted_random = np.sort(sim_random)

cdf_R1 = np.arange(1, len(sorted_R1) + 1) / len(sorted_R1)
cdf_R2 = np.arange(1, len(sorted_R2) + 1) / len(sorted_R2)
cdf_random = np.arange(1, len(sorted_random) + 1) / len(sorted_random)

# Plot CDFs
plt.figure(figsize=(8, 5))
plt.plot(sorted_R1, cdf_R1, label="R1 CDF", color='blue')
plt.plot(sorted_R2, cdf_R2, label="R2 CDF", color='red')
plt.plot(sorted_random, cdf_random, label="Random CDF", color='green', linestyle="--")
plt.xlabel("Angular Similarity")
plt.ylabel("CDF")
plt.title("CDFs of Angular Similarities")
plt.legend()
plt.grid()
plt.show()

# Compute KS distances
ks_R1 = ks_2samp(sim_R1, sim_random).statistic
ks_R2 = ks_2samp(sim_R2, sim_random).statistic

print(f"Kolmogorov-Smirnov Distance (R1 vs Random): {ks_R1:.4f}")
print(f"Kolmogorov-Smirnov Distance (R2 vs Random): {ks_R2:.4f}")

# Prediction: Smaller KS distance means the dataset is more similar to a uniform random distribution
if ks_R1 < ks_R2:
    print("R1 is more likely to be uniformly random.")
else:
    print("R2 is more likely to be uniformly random.")
