import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import time

def normalize_text(text):
    if pd.isnull(text):
        return ""  # Return an empty string for NaN or None values
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.lower()
    return text

def encode_text(texts, tokenizer, model):
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model(**inputs)
        # Mean pooling - take attention mask into account for correct averaging
        attention_mask = inputs['attention_mask']
        outputs = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
        sum_embeddings = torch.sum(outputs * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return embeddings

def find_most_similar(term, thesaurus_terms, tokenizer, model):
    term_embedding = encode_text([term], tokenizer, model)
    thesaurus_embeddings = encode_text(thesaurus_terms, tokenizer, model)
    cosine_similarities = cosine_similarity(term_embedding, thesaurus_embeddings).squeeze()
    max_sim_index = cosine_similarities.argmax()
    max_sim_score = cosine_similarities[max_sim_index].item()
    matching_term = thesaurus_terms[max_sim_index]
    return term, matching_term, max_sim_score

tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
# model = AutoModel.from_pretrained("bert-base-multilingual-uncased")
model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")


if torch.cuda.is_available():
    model.cuda()
    print("Using GPU.")
else:
    print("Using CPU.")

csv_location = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/work_in_progress_code/similar_terms2.csv'

similar_terms = pd.read_csv(csv_location)
all_not_found_terms = similar_terms['Not_Found_Term'].tolist()

thesaurus_location = '/home/gburger01/snomed_20230426.txt'
thesaurus = pd.read_csv(thesaurus_location, sep='|', encoding='latin-1')
thesaurus['Normalized_DETEROM'] = thesaurus['DETEROM'].apply(normalize_text)
thesaurus['Normalized_DEADVOM'] = thesaurus['DEADVOM'].apply(normalize_text)

combined_columns = thesaurus['Normalized_DETEROM'].tolist() + thesaurus['Normalized_DEADVOM'].tolist()

similar_terms_list = []

for not_found_term in all_not_found_terms:
    term, found_term, similarity = find_most_similar(not_found_term, combined_columns, tokenizer, model)
    similar_terms_list.append([term, found_term, similarity])

similar_terms_df = pd.DataFrame(similar_terms_list, columns=['Not_Found_Term', 'Found_Term', 'Similarity'])

output_csv_location = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/work_in_progress_code/similar_terms3.csv'
similar_terms_df.to_csv(output_csv_location, index=False)