import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import random
import torch
import gc

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#set up imports
parent_dir = str(Path(__file__).resolve().parent.parent)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from utils.fine_tuning_config import EMBED_PATH, CORPUS_PATH, QUERY_PATH, MODEL, KS, MODEL_SAVE_PATH, INDEX_SAVE_PATH, CORPUS_SAVE_PATH

from utils.train_config import MODEL as TRAIN_MODEL

assert MODEL == TRAIN_MODEL
#load data
df = pd.read_csv(CORPUS_PATH)
df_q = pd.read_csv(QUERY_PATH)

#split data into train and test
train_q, test_q = train_test_split(df_q, test_size = 0.2, random_state = 42)

#extract text
texts = df["document"].tolist()
#indices
ids = df.index.tolist()

#baseline model evaluation
print("\n Loading the base model...\n")

baseline_model = SentenceTransformer(MODEL)

print("\n Baseline model loaded... \n")

#embeddings for baseline model
print("\n Generating baseline embeddings... \n")

baseline_emb = np.load(EMBED_PATH)

#float32 check
baseline_emb = baseline_emb.astype("float32")


assert baseline_emb.shape[0] == len(df)

#baseline embeddings index
index_baseline = faiss.IndexFlatIP(baseline_emb.shape[1])
index_baseline.add(baseline_emb)

#function to evaluate
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

#get the baseline results
baseline_rec, baseline_mrr = calc_metrics(test_q, baseline_model, index_baseline, Ks=KS)


#print the baseline results
print(f"\n Baseline MRR: {baseline_mrr} \n")
print(f"\n Baseline Recall: {baseline_rec}")

print("\n Done with Baseline results... \n")

#---------------- Training Module---------------------------

#set up training pairs
train_examples = []
#loop through training queries
for _, row in train_q.iterrows():
    #get the query
    q = row["query"]
    #get the document text
    d = df.iloc[row["doc_id"]]["document"]
    #append to the training examples
    train_examples.append(InputExample(texts = [q, d]))
    
print(f"\n Fine tuning model...\n")
#load model
model = SentenceTransformer(MODEL)
#dataloader
train_loader = DataLoader(train_examples, batch_size = 4, shuffle = True)
#loss function
loss_fn = losses.MultipleNegativesRankingLoss(model)


#train model
print("\n Starting model training \n")

#clear gpu memory
del baseline_model
gc.collect()
torch.cuda.empty_cache()

model.fit(
    train_objectives = [(train_loader, loss_fn)],
    epochs = 3,
    warmup_steps = 100,
    show_progress_bar = True
)

print("\n Model Trained ... \n")

#get document embeddings
doc_embeddings = model.encode(texts,
        convert_to_numpy = True,
        normalize_embeddings = True,
        batch_size = 64,
        show_progress_bar = True).astype("float32")

#check shape for error
assert doc_embeddings.shape[0] == len(df)

#baseline embeddings index
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)

#get the fine-tuned results
ft_rec, ft_mrr = calc_metrics(test_q, model = model, index = index, Ks=KS)


#print the fine-tuned results
print(f"\n Fine-tuned MRR: {ft_mrr} \n")
print(f"\n Fine-tuned Recall: {ft_rec}")

print("\n Done with Fine-tuned results... \n")

#save model
model.save(MODEL_SAVE_PATH)

#save index
faiss.write_index(index, INDEX_SAVE_PATH)

#save dataframe
df.to_csv(CORPUS_SAVE_PATH, index = True)


    
    

