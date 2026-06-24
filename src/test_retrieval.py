import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import ast
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from utils.train_config import MODEL

df = pd.read_csv("data/arxiv_dataset_cleaned.csv")

model = SentenceTransformer(MODEL)
embeddings = np.load("data/arxiv_embeddings.npy")
X = np.array(embeddings).astype("float32")

#cosine similarity indexing
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
query = "graph neural networks for molecules"
query_vec = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

k = 5  # top 5 results

distances, indices = index.search(query_vec, k)

#indices are the indices for the document (here numpy indices are used which are same to document)
for i, idx in enumerate(indices[0]):
    print(f"\nRank {i+1}")
    print("Paper:", df.iloc[idx]["document"])
    print("Distance:", distances[0][i])