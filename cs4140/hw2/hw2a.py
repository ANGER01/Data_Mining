import hashlib
import random

def hash_function(seed, x, m=10000):
    """Generates a hash function using SHA-1 and a random seed."""
    sha1 = hashlib.sha1((str(seed) + x).encode()).hexdigest()
    return int(sha1, 16) % m  # Ensure value is within [0, m-1]

def minhash_signature(shingle_set, num_hashes=10, m=10000):
    """Computes the MinHash signature for a given set."""
    minhashes = [min(hash_function(seed, shingle, m) for shingle in shingle_set) 
                 for seed in range(num_hashes)]
    return minhashes

def minhash_jaccard(set1, set2, num_hashes=10):
    """Estimates Jaccard Similarity using MinHash signatures."""
    sig1 = minhash_signature(set1, num_hashes)
    sig2 = minhash_signature(set2, num_hashes)
    return sum(1 for i in range(num_hashes) if sig1[i] == sig2[i]) / num_hashes

# Example sets (Shingles)
setA = {"apple", "banana", "cherry"}
setB = {"banana", "cherry", "date", "fig"}

# Compute MinHash-based Jaccard Similarity
estimated_jaccard = minhash_jaccard(setA, setB, num_hashes=100)
print(f"Estimated Jaccard Similarity: {estimated_jaccard:.2f}")

# Compute Exact Jaccard Similarity
exact_jaccard = len(setA & setB) / len(setA | setB)
print(f"Exact Jaccard Similarity: {exact_jaccard:.2f}")
