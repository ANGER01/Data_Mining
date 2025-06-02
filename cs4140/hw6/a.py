import numpy as np
import pandas as pd
from scipy import linalg as LA
import matplotlib.pyplot as plt

A = pd.read_csv("A.csv")
A_words = A["word"]
A_number = A.drop(columns="word")

a_prime = A_number.to_numpy() - np.mean(A_number.to_numpy(), axis=0)
U, s, Vt = LA.svd(a_prime, full_matrices=False)
print(LA.norm(a_prime, 2))
tol = LA.norm(a_prime, 2) * .05
for k in range(1,51):
    Uk = U[:,:k]
    Sk = np.diag(s[:k])
    Vtk = Vt[:k,:]
    Ak = Uk @ Sk @ Vtk
    print(f"-----------{k}-----------")
    print((LA.norm(a_prime - Ak, 2)))

V2 = Vt[:2, :]  # Shape (2, 100)

# Project A onto these two vectors
A_proj = (a_prime @ V2.T)  # Shape (1000, 2)

# Plot the projected points
plt.figure(figsize=(8, 6))
plt.scatter(A_proj[:, 0], A_proj[:, 1], alpha=0.5)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("Projection of A onto the top 2 right singular vectors")
plt.grid(True)
plt.show()