import mmh3
import numpy as np

class CountMinSketch:
    def __init__(self, width, depth):
        """
        Initializes a Count-Min Sketch with given width (w) and depth (d).

        :param width: Number of hash buckets per row (w)
        :param depth: Number of hash functions (d)
        """
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int32)  # 2D NumPy array for counts

    def _hash(self, item, i):
        """
        Generates a hash index for the item using different hash functions.
        :param item: The item to hash
        :param i: Hash function index
        :return: An index between 0 and width-1
        """
        return mmh3.hash(item, i) % self.width

    def add(self, item):
        """
        Adds an item to the Count-Min Sketch.
        :param item: The item to increment the count for
        """
        for i in range(self.depth):
            index = self._hash(item, i)
            self.table[i, index] += 1

    def estimate(self, item):
        """
        Estimates the count of an item.
        :param item: The item to estimate the count for
        :return: Estimated count (minimum across all hash functions)
        """
        return min(self.table[i, self._hash(item, i)] for i in range(self.depth))

# Example Usage
cms = CountMinSketch(width=10, depth=5)

with open("S1_hw5.csv", "r", encoding="utf-8") as file:
    content = file.read()  # Read the entire file as a string

with open("S2_hw5.csv", "r", encoding="utf-8") as file1:
    content1 = file1.read()
for char in content:
    cms.add(char)

# Estimating counts
s1 = 3_000_000
print("Error bound is 10%")
print("Estimated Count of 'a':", cms.estimate('a'))
print("Percent for a:", cms.estimate('a')/s1)
print("Estimated Count of 'b':", cms.estimate('b'))
print("Percent for b:", cms.estimate('b')/s1)
print("Estimated Count of 'c':", cms.estimate('c'))
print("Percent for c:", cms.estimate('c')/s1)
print("-------------")
cms1 = CountMinSketch(width=10, depth=5)

for char in content1:
    cms1.add(char)

# Estimating counts
s2 = 4_000_000
print("Error bound is 10%")
print("Estimated Count of 'a':", cms1.estimate('a'))
print("Percent for a:", cms1.estimate('a')/s2)
print("Estimated Count of 'b':", cms1.estimate('b'))
print("Percent for b:", cms1.estimate('b')/s2)
print("Estimated Count of 'c':", cms1.estimate('c'))
print("Percent for c:", cms1.estimate('c')/s2)