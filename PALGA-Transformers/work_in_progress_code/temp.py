import pandas as pd
import numpy as np
from datasets import concatenate_datasets


# tsv_file = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/pretrain/pretrain.tsv'
tsv_file = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/combined_gold_standard_with_codes.tsv"
df = pd.read_csv(tsv_file, sep='\t')
df.dropna(inplace=True)

# tsv_file_all = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_train_with_codes.tsv'
# tsv_file_histology = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_train_with_codes_RAG.tsv'
# tsv_file_autopsy = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/autopsies/autopsies_norm_train_with_codes.tsv'

# all = pd.read_csv(tsv_file_all, sep='\t')
# histology = pd.read_csv(tsv_file_histology, sep='\t')
# autopsy = pd.read_csv(tsv_file_autopsy, sep='\t')
# autopsy['Type'] = 'S'

# df = pd.concat([all, histology, autopsy], ignore_index=True)
# df.dropna(inplace=True)

histology = df[df['Type'] == 'T']
cytology = df[df['Type'] == 'C']
autopsy = df[df['Type'] == 'S']

print(len(histology))
print(len(cytology))
print(len(autopsy))

column_name = 'Conclusie'

average_words_histology = histology[column_name].str.split().apply(len).mean()
average_words_cytology = cytology[column_name].str.split().apply(len).mean()
average_words_autopsy = autopsy[column_name].str.split().apply(len).mean()

median_words_histology = histology[column_name].str.split().apply(len).median()
median_words_cytology = cytology[column_name].str.split().apply(len).median()
median_words_autopsy = autopsy[column_name].str.split().apply(len).median()

std_dev_histology = histology[column_name].str.split().apply(len).std()
std_dev_cytology = cytology[column_name].str.split().apply(len).std()
std_dev_autopsy = autopsy[column_name].str.split().apply(len).std()

q1_histology = np.percentile(histology[column_name].str.split().apply(len), 25)
q1_cytology = np.percentile(cytology[column_name].str.split().apply(len), 25)
q1_autopsy = np.percentile(autopsy[column_name].str.split().apply(len), 25)

q3_histology = np.percentile(histology[column_name].str.split().apply(len), 75)
q3_cytology = np.percentile(cytology[column_name].str.split().apply(len), 75)
q3_autopsy = np.percentile(autopsy[column_name].str.split().apply(len), 75)

print(f"Average words histology {average_words_histology}, median {median_words_histology}, standard deviation {std_dev_histology}, q1 {q1_histology}, q3 {q3_histology}")
print(f"Average words cytology {average_words_cytology}, median {median_words_cytology}, standard deviation {std_dev_cytology}, q1 {q1_cytology}, q3 {q3_cytology}")
print(f"Average words autopsy {average_words_autopsy}, median {median_words_autopsy}, standard deviation {std_dev_autopsy}, q1 {q1_autopsy}, q3 {q3_autopsy}")