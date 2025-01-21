import faiss
import numpy as np

# Create float32 vectors (e.g., 1000 vectors of 128 dimensions)
d = 128  # Dimension of vectors
nb = 1000  # Number of database vectors
vectors = np.random.rand(nb, d).astype(np.float32)  # Create random vectors

# Normalize vectors to unit length (recommended before quantization)
faiss.normalize_L2(vectors)

# Convert to int8 (FAISS expects float first, then quantizes internally)
quantizer = faiss.IndexFlatL2(d)  # Base index for L2 search
index = faiss.IndexIVFPQ(quantizer, d, 10, 8, 8)  # IVF-PQ with 8-bit storage

# Train the index
index.train(vectors)
index.add(vectors)  # Add int8-quantized vectors

# Perform search
query_vector = np.random.rand(1, d).astype(np.float32)
faiss.normalize_L2(query_vector)  # Normalize query vector
distances, indices = index.search(query_vector, k=5)  # Search top-5 results

print("Top indices:", indices)
print("Top distances:", distances)
