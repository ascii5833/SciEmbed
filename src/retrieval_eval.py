import numpy as np
import pandas as pd
import faiss
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch



#project root
project_root = str(Path(__file__).resolve().parent.parent)

if project_root not in sys.path:
    sys.path.append(project_root)
    

from utils.eval_config import MODEL, CORPUS_PATH, EMBED_PATH, QUERY_PATH, KS

from utils.train_config import MODEL as TRAIN_MODEL

assert MODEL == TRAIN_MODEL

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device {device}")

#load data
df = pd.read_csv(CORPUS_PATH)
df_q = pd.read_csv(QUERY_PATH)
embeddings = np.load(EMBED_PATH)
embeddings = np.array(embeddings).astype("float32")

#build faiss index and add embeddings
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

#load the model 
model = SentenceTransformer(MODEL)

print("\n Calculating metrics..... \n")

#calculate the metrics
def calc_metrics(queries, model, index, Ks):
    
    #get the queries
    texts = queries["query"].tolist()
    true_ids = queries["doc_id"].astype(int).to_numpy()
    
    #batch predictions
    query_embeddings = model.encode(
        texts,
        convert_to_numpy = True,
        normalize_embeddings = True,
        batch_size = 64,
        show_progress_bar = True
    ).astype("float32")
    
    #faiss call
    top_k  = max(Ks)
    D, I = index.search(query_embeddings, top_k)
    
    #hits for found docs
    hits = {k:0 for k in Ks}
    mrr = 0
    
    #loop through all queries
    for i in range(len(true_ids)):
        #get results for ith query
        retrieved = I[i]
        #get true document for ith query
        true_id = true_ids[i]
        
        #find index where match is
        match  = np.where(retrieved == true_id)[0]
        
        #if we found a match
        if len(match) > 0:
            rank = match[0] + 1 #starts with 0 rank
            mrr += 1 / rank
            
            #loop through each k
            for k in Ks:
                #if its in top k
                if rank <= k:
                    hits[k] += 1
    
    #length of true ids
    n = len(true_ids)
    #return the metrics
    recall = {f"Recall@{k}" : hits[k] / n for k in Ks}
    mrr = mrr / n
    return recall, mrr


# get the metrics
recall, mrr = calc_metrics(df_q, model = model, index = index, Ks = KS)

print(f'\n mrr is: {mrr}\n')
print(f'Recalls: {recall}\n')


            
        
    
    
    
 
    


                   
        
    



