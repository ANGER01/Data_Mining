import hashlib
import random
import time

def hash_function(seed, x, m=10000):
    sha1 = hashlib.sha1((str(seed) + x).encode()).hexdigest()
    return int(sha1, 16) % m  # Ensure value is within [0, m-1]

def minhash_signature(shingle_set, num_hashes=10, m=10000):
    minhashes = [min(hash_function(seed, shingle, m) for shingle in shingle_set) 
                 for seed in range(num_hashes)]
    return minhashes

def minhash_jaccard(set1, set2, num_hashes=10):
    sig1 = minhash_signature(set1, num_hashes)
    sig2 = minhash_signature(set2, num_hashes)
    return sum(1 for i in range(num_hashes) if sig1[i] == sig2[i]) / num_hashes

def get_k_gram(text, k=2):
    bigrams = [text[i:i+k] for i in range((len(text) + 1) - k)]
    if len(bigrams[0]) > 1:
        bigrams = [" ".join(pair) for pair in bigrams]
    unique_bigrams = set(bigrams)
    return unique_bigrams

def get_docs():
    docs = []
    for i in range(1,5):
        filename = f"d{i}.txt"
        
        with open(filename, 'r', encoding='utf-8') as file:
            docs.append(file.read())
            
    return docs

def compute_jaccard_sim(set1: set, set2: set):
    numerator = len(set1 & set2)
    denominator = len(set1 | set2)
    return numerator/denominator if denominator != 0 else 0

def compare_to_all(list: list):
    results = {}
    
    for i in range(0, len(list)):
        for j in range(i + 1, len(list)):
            set1, set2 = list[i], list[j]
            results[f"D{i+1} and D{j+1}"] = (compute_jaccard_sim(set1, set2))
    return results

if __name__ == "__main__":
    docs = get_docs()
    G1 = []
    G2 = []
    G3 = []

    for doc in docs:
        G1.append(get_k_gram(doc))
        
    for doc in docs:
        G2.append(get_k_gram(doc, k=3))
        
    for doc in docs:
        doc = doc.split()
        G3.append(get_k_gram(doc))
    
    for i in range(1,5):
        temp = G1[i - 1]
        print(f"Document {i} 2-grams: {len(temp)}")
        
    for i in range(1,5):
        temp = G2[i - 1]
        print(f"Document {i} 3-grams: {len(temp)}")
        
    for i in range(1,5):
        temp = G3[i - 1]
        print(f"Document {i} 2-grams-words: {len(temp)}")
        
    res1 = compare_to_all(G1)
    print(f"G1 similarities: {res1}")
    
    res2= compare_to_all(G2)
    print(f"G2 similarities: {res2}")
    
    res3 = compare_to_all(G3)
    print(f"G3 similarities: {res3}")

    D1 = G2[0]
    D2 = G2[1]

    t = [20, 50, 150, 300, 600]
    for num in t:
        start = time.time()
        estimated_jaccard = minhash_jaccard(D1, D2, num_hashes=num)
        end = time.time()
        print(f"Estimated Jaccard Similarity at t={num}: {estimated_jaccard:.3f}\n Time in seconds: {end - start}")
