import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_unit_vectors(t, d):
    """
    Generates t unit vectors in R^d uniformly from the unit sphere S^(d-1).
    """
    vectors = np.random.randn(t, d)  # Generate random Gaussian vectors
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize to unit length
    return vectors

def sang(a, b):
    """Computes the angular similarity between two vectors a and b."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0  # Consider zero similarity for degenerate cases

    cosine_similarity = dot_product / (norm_a * norm_b)
    
    # Ensure the value is within [-1, 1] for numerical stability
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

    return 1 - (1 / np.pi) * np.arccos(cosine_similarity)

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

# Compute all pairwise angular similarities
angular_sim1 = []
angular_sim2 = []
for i in range(500):
    for j in range(i + 1, 500):  # Compute only upper triangle (no repeats)
        sim1 = sang(R1[i], R1[j])
        sim2 = sang(R2[i], R2[j])
        angular_sim1.append(sim1)
        angular_sim2.append(sim2)

angular_sim1 = np.array(angular_sim1)
angular_sim2 = np.array(angular_sim2)

# Compute empirical CDF
sorted_similarities1 = np.sort(angular_sim1)
cdf1 = np.arange(1, len(sorted_similarities1) + 1) / len(sorted_similarities1)

sorted_similarities2 = np.sort(angular_sim2)
cdf2 = np.arange(1, len(sorted_similarities2) + 1) / len(sorted_similarities2)

# Plot CDF
plt.figure(figsize=(8, 5))
plt.plot(sorted_similarities1, cdf1, label="Empirical CDF", color='blue')
plt.axvline(x=0.60, color='g', linestyle='--', label="τ = 0.60")
plt.xlabel("Angular Similarity")
plt.ylabel("CDF")
plt.title("R1 CDF of Angular Similarities")
plt.legend()
plt.grid()
plt.show()

count_above_60_1 = np.sum(angular_sim1 > 0.60)

print(f"Number of pairs with S_ANG > 0.60: {count_above_60_1}")

# Plot CDF
plt.figure(figsize=(8, 5))
plt.plot(sorted_similarities2, cdf2, label="Empirical CDF", color='blue')
plt.axvline(x=0.60, color='g', linestyle='--', label="τ = 0.60")
plt.xlabel("Angular Similarity")
plt.ylabel("CDF")
plt.title("R2 CDF of Angular Similarities")
plt.legend()
plt.grid()
plt.show()

count_above_60_2 = np.sum(angular_sim2 > 0.60)

print(f"R1 Number of pairs with S_ANG > 0.60: {count_above_60_1}")
print(f"R2 Number of pairs with S_ANG > 0.60: {count_above_60_2}")

# Parameters
t = 250  # Number of unit vectors
d = 100  # Dimensionality

# Generate unit vectors
unit_vectors = generate_unit_vectors(t, d)

# Compute all pairwise angular similarities
angular_similarities = []
for i in range(t):
    for j in range(i + 1, t):  # Compute only upper triangle (no repeats)
        sim = sang(unit_vectors[i], unit_vectors[j])
        angular_similarities.append(sim)

angular_similarities = np.array(angular_similarities)

# Compute empirical CDF
sorted_similarities = np.sort(angular_similarities)
cdf = np.arange(1, len(sorted_similarities) + 1) / len(sorted_similarities)

# Plot CDF
plt.figure(figsize=(8, 5))
plt.plot(sorted_similarities, cdf, label="Empirical CDF", color='blue')
plt.axvline(x=0.60, color='g', linestyle='--', label="τ = 0.60")
plt.xlabel("Angular Similarity")
plt.ylabel("CDF")
plt.title("CDF of Angular Similarities for 250 Unit Vectors")
plt.legend()
plt.grid()
plt.show()

# Count pairs above threshold
count_above_60 = np.sum(angular_similarities > 0.60)

print(f"Number of pairs with S_ANG > 0.60: {count_above_60}")