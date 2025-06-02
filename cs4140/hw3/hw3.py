import numpy as np
import pandas as pd
import random
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class LSH:
    def __init__(self, num_hashes, bands):
        """
        Initialize LSH with the specified number of hash functions and bands.
        Args:
            num_hashes (int): The total number of hash functions (t).
            bands (int): The number of bands (b).
        """
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows_per_band = num_hashes // bands
        self.hash_functions = self._generate_hash_functions()
        self.super_hash_tables = [defaultdict(list) for _ in range(bands)]  # LSH buckets for each band

    def _generate_hash_functions(self):
        """
        Generate random hash functions of the form: (a * x + b) % p
        """
        # Using a large prime number as mod
        p = 2**31 - 1
        hash_functions = []
        for _ in range(self.num_hashes):
            a = random.randint(1, p - 1)  # Random a
            b = random.randint(0, p - 1)  # Random b
            hash_functions.append((a, b))
        return hash_functions

    def _minhash(self, vector):
        """
        Generate a MinHash signature for a given vector using the precomputed hash functions.
        Args:
            vector (np.array): The input data point (should be binary or thresholded).
        """
        signature = []
        for a, b in self.hash_functions:
            min_hash = np.min([(a * int(x) + b) % (2**31 - 1) for x in vector])
            signature.append(min_hash)
        return signature

    def fit(self, data):
        """
        Fit the LSH model on the provided data.
        Args:
            data (np.ndarray): The dataset of shape (n, d) where n is the number of points and d is the dimension.
        """
        self.data = data
        self.signatures = {}
        
        # Generate MinHash signatures and store them in the super hash tables
        for idx, vector in enumerate(data):
            signature = self._minhash(vector)
            self.signatures[idx] = signature
            for band in range(self.bands):
                band_signature = tuple(signature[band * self.rows_per_band:(band + 1) * self.rows_per_band])
                self.super_hash_tables[band][band_signature].append(idx)

    def query(self, query_vector, threshold=0.7):
        """
        Query the LSH model to find approximate nearest neighbors.
        Args:
            query_vector (np.array): The query vector to search for approximate nearest neighbors.
            threshold (float): The threshold for the similarity (cosine similarity in this case).
        """
        # Generate the MinHash signature for the query vector
        query_signature = self._minhash(query_vector)
        
        candidate_pairs = set()
        for band in range(self.bands):
            band_signature = tuple(query_signature[band * self.rows_per_band:(band + 1) * self.rows_per_band])
            if band_signature in self.super_hash_tables[band]:
                candidate_pairs.update(self.super_hash_tables[band][band_signature])
        
        # Compute cosine similarity for the candidate pairs
        similar_items = []
        for idx in candidate_pairs:
            cosine_sim = cosine_similarity([query_vector], [self.data[idx]])[0][0]
            if cosine_sim >= threshold:
                similar_items.append((idx, cosine_sim))
        
        return similar_items

def load_data(file_path):
    """
    Load data from a CSV file into a numpy array, skipping headers and assuming space-separated.
    Args:
        file_path (str): The path to the file to load.
    """
    df = pd.read_csv(file_path, header=0)  # Adjust delimiter if needed
    return df.to_numpy()

if __name__ == "__main__":
    # Load dataset (example files, R1 and R2)
    R1 = load_data("R1.csv")
    R2 = load_data("R2.csv")
    
    # Create LSH object with desired number of hash functions and bands
    lsh = LSH(num_hashes=160, bands=40)
    
    # Fit the LSH model on R1 dataset
    print("Processing R1...")
    lsh.fit(R1)
    
    # Query the LSH model with a vector from R1 (for demonstration)
    query_vector = R1[0]  # Let's query with the first point
    similar_R1 = lsh.query(query_vector, threshold=0.7)
    print(f"R1 - Found {len(similar_R1)} similar pairs.")
    
    # Fit the LSH model on R2 dataset
    print("Processing R2...")
    lsh.fit(R2)
    
    # Query the LSH model with a vector from R2 (for demonstration)
    query_vector = R2[0]  # Let's query with the first point
    similar_R2 = lsh.query(query_vector, threshold=0.7)
    print(f"R2 - Found {len(similar_R2)} similar pairs.")
