MODEL = "BAAI/bge-large-en-v1.5"

CORPUS_PATH = "./data/arxiv_dataset_cleaned.csv"

EMBED_PATH = "./data/arxiv_embeddings.npy"

QUERY_PATH = "./data/arxiv_dataset_queries.csv"

KS = (1, 5, 10)

MODEL_SAVE_PATH = "./models/fine_tuned_model"

INDEX_SAVE_PATH = "./data/faiss.index"

CORPUS_SAVE_PATH = "./data/arxiv_dataset_with_index.csv"