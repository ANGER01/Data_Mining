import numpy as np
import pandas as pd
from scipy import linalg as LA
import matplotlib.pyplot as plt

D = pd.read_csv("D.csv")
d_dest_org = D["ORIGIN/DESTINATION"]
d_num = D.drop(columns="ORIGIN/DESTINATION")

d_sqr = d_num ** 2
d_fro = LA.norm(d_sqr, 'fro')
print(d_fro)
print(d_sqr.shape)

n = 100
I = np.eye(n)  # Identity matrix of shape (100, 100)
one_n = np.ones((n, 1))  # Column vector of ones (100, 1)
C_n = I - (1/n) * np.dot(one_n, one_n.T)  # Centering matrix (100, 100)

# Step 2: Apply the double centering formula: M = -1/2 * C_n * D^2 * C_n
M = -0.5 * np.dot(np.dot(C_n, d_sqr), C_n)
print(LA.norm(M, 'fro'))

eigenvalues, eigenvectors = np.linalg.eigh(M)

# Step 4: Sort the eigenvectors by eigenvalues in descending order (get top 2 eigenvectors)
sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices of eigenvalues in descending order
top_2_eigenvectors = eigenvectors[:, sorted_indices[:2]]  # Select top 2 eigenvectors

# Step 5: Project the data onto the top 2 eigenvectors (this gives the 2D representation)
projected_data = np.dot(M, top_2_eigenvectors)

# Step 6: Plot the projected data in 2D
plt.figure(figsize=(8, 6))
plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.6)
plt.title("Projection of Data on Top 2 Eigenvectors of M")
plt.xlabel("Eigenvector 1")
plt.ylabel("Eigenvector 2")
plt.grid(True)
plt.show()