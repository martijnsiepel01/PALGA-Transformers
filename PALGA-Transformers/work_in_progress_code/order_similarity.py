import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import time

# Specify the path to your CSV file
csv_file = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/work_in_progress_code/similar_terms_tfidf.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Order the column "Similarity" in descending order
df = df.sort_values(by='Count', ascending=False)

df.to_csv(csv_file, index=False)

# def normalize_text(text):
#     if pd.isnull(text):
#         return ""  # Return an empty string for NaN or None values
#     text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = text.strip()
#     text = text.lower()
#     return text



# thesaurus_location = '/home/gburger01/snomed_20230426.txt'
# thesaurus = pd.read_csv(thesaurus_location, sep='|', encoding='latin-1')
# thesaurus['Normalized_DETEROM'] = thesaurus['DETEROM'].apply(normalize_text)
# thesaurus['Normalized_DEADVOM'] = thesaurus['DEADVOM'].apply(normalize_text)

# combined_columns = thesaurus['Normalized_DETEROM'].tolist() + thesaurus['Normalized_DEADVOM'].tolist()

# for not_found_term, found_term in zip(df['Not_Found_Term'], df['Found_Term']):
#     if found_term not in combined_columns: print(f"{not_found_term} was mapped to {found_term} but {found_term} is not in the thesaurus.")

