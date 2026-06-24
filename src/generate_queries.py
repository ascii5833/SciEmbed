from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import re


#project root
project_root = str(Path(__file__).resolve().parent.parent)

#check if project root in import path
if project_root not in sys.path:
    sys.path.append(project_root)
    
from utils.train_config import QUERY_MODEL_NAME, CORPUS_PATH, SAMPLE_SIZE, QUERIES_PATH


#device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Using device {device}")
#tokenizer using pre-trained model
tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)

#pre_trained model
model = AutoModelForCausalLM.from_pretrained(QUERY_MODEL_NAME).to(device)

#generate query function
def generate(title, summary, q_size = 3):
    # prompt = f"""
    # Generate exactly 3 search queries.

    # Rules:
    # - Output only the queries
    # - One query per line
    # - No numbering
    # - No explanations
    # - Stop after the third query

    # Title: {title}

    # Summary: {summary}
    # """
    #define the role of the system and the user and the rules
    messages = [
    {
        "role": "system",
        "content": (
            "You are an information retrieval assistant. "
            "You generate search queries that researchers would type into academic search engines."
        )
    },
    {
        "role": "user",
        "content": f"""
    Generate exactly 3 search queries for the given paper.

    IMPORTANT RULES:
    - Output ONLY the 3 queries
    - One query per line
    - No numbering (no 1., 2., 3.)
    - No labels (no Query 1:, etc.)
    - No explanations
    - No extra text

    Paper Title:
    {title}

    Paper Summary:
    {summary}
    """
        }
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = True
    )
    inputs = tokenizer(text, return_tensors = "pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    
    #output 
    #**inputs unpacks, token stream and the attention mask to generator (decode) for the next series of output
    #write response but stop generating after 100 words
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens = 40, 
            do_sample = False,
            temperature = 0.7,
            top_p = 0.9,
            repetition_penalty = 1.1,
            eos_token_id = tokenizer.eos_token_id)

        #decode output to string
        # we have a batch of outputs but want only the first one so out[0]
        #we also want to skip tokens like <cls> etc 
        res = tokenizer.decode(out[0][input_len:], skip_special_tokens = True)
        return res


#load the df
df = pd.read_csv(CORPUS_PATH)


print(f'\n Dataframe length: {len(df)} \n')

#sample dataframe
sample = df.sample(n = SAMPLE_SIZE, random_state = 42).reset_index()

rows = []

print(f'Generating Queries .....')

#iterate through the samples to generate queries
for _ , row in tqdm(sample.iterrows(), total = len(sample)):
    queries_text = generate(row['title'], row['summary'])
    
    #get the indv queries
    queries = queries_text.strip().split("\n")
    
    #loop through each query
    for q in queries:
        q = q.strip()
        #regex starts with one or more digits followed by .) or - and has s* means more than one or more white space tabs etc.
        q = re.sub(r"^\d+[\.\)\-]\s*", "", q)
        #if empty string
        if not q:
            continue
        
        rows.append({
            "query" : q,
            "doc_id" : int(row["index"])
        })
        
    
#create output dataframe
out = pd.DataFrame(rows)
#save the queries
out.to_csv(QUERIES_PATH, index = False)

print(f'\n Queries Generated .....')

print(f"\n Queries generated for {SAMPLE_SIZE} documents \n")
