import arxiv
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import time
import random

#project root
project_root = str(Path(__file__).resolve().parent.parent)

#check if project root in import path
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.scraper_config import MAX_RESULTS, TOPICS, DF_SAVE_PATH

#function to fetch the papers
def fetch_arxiv_papers(query = "machine learning", max_results = 100):
    #api client
    client = arxiv.Client(
        page_size = 50,
        delay_seconds = 7.0,
        num_retries = 5
    )
    #search arxiv using the wrapper
    search = arxiv.Search(
        query = query,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )
    
    results = client.results(search = search)
    
    #paper list
    papers = []
    #loop through the results
    for result in results:
        papers.append({
            "id": result.entry_id,
            "title": result.title,
            "summary": result.summary,
            "categories": result.categories,
            "published": str(result.published),
            "authors": [a.name for a in result.authors]
        })
        
    return pd.DataFrame(papers)
        

#final dataframe
dfs = []


#loop through topics and generate final dataframe
for topic in tqdm(TOPICS, desc = "Scraping arxiv topics"):
    tqdm.write(f"Fetching topic: {topic}")
    #get the result dataframe
    df = fetch_arxiv_papers(query = topic, max_results = MAX_RESULTS)
    
    dfs.append(df)
    
    #jitter for getting through bot like behavior
    time.sleep(10 + random.uniform(2, 5))

final_df = pd.concat(dfs, ignore_index = True)
#for the final dataframe aggregate the duplicates
final_df = final_df.groupby("id").agg({
    "title" : "first",
    "summary": "first",
    "published": "first",
    "authors": "first",
    "categories": lambda x: list(set(cat for sub in x for cat in sub)) #x is series of list
    
    }).reset_index()



final_df.to_csv(DF_SAVE_PATH, index = False)