import re
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
#project root
project_root = str(Path(__file__).resolve().parent.parent)

#sys path append
if project_root not in sys.path:
    sys.path.append(project_root)
    

from utils.df_clean_config import DATASET_PATH, DF_SAVE_PATH

#clean data
def cleaner(text):
    #handle nan
    if pd.isna(text):
        return ""
    
    text = str(text)

    #replace newlines and tabs
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    #replace multiple spaces
    text = re.sub(r"\s+", " ", text)

    #strip trailing and leading spaces
    text = text.strip()

    return text

print('\n Initializing script \n')    


df = pd.read_csv(DATASET_PATH)
print("Dataset Loaded Successfully... \n")
#clean title and summary
df['title'] = df['title'].apply(cleaner)
df['summary'] = df['summary'].apply(cleaner)
#create document column
df['document'] = (df['title'] + " " + df['summary']).str.strip()
#remove duplicate documents (ne not equal return true for documents not having "")
df = df[df['document'].ne("")]
    
print("Dataset Cleaned Successfully... \n")
    
df.to_csv(DF_SAVE_PATH, index = False)


print('Dataframe saved successfully\n')    

    
    
    
    