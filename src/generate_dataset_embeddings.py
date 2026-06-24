import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device {device}")

#root dir
project_root = str(Path(__file__).resolve().parent.parent)

if project_root not in sys.path:
    sys.path.append(project_root)
    
    
from utils.train_config import CORPUS_PATH, EMBED_PATH, MODEL

#load corpus
df = pd.read_csv(CORPUS_PATH)

#load pre-trained encoder
model = SentenceTransformer(MODEL)

#document column
texts = df['document'].fillna("").tolist()

print("\n Encoding Documents... \n")

#embed the documents
embeddings = model.encode(
    texts,
    batch_size = 32,
    show_progress_bar = True,
    convert_to_numpy = True,
    normalize_embeddings = True, #helpful during cosine similarity
    device = device 
)


print("\n Embeddings generated successfully...\n")
print(f"Generated {embeddings.shape[0]} embeddings \n")
print(f"Embeddings shape: {embeddings.shape} \n")


np.save(EMBED_PATH, embeddings)

print('Embeddings saved successfully...')