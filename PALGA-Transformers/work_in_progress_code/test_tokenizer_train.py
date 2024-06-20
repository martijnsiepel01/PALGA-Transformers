from transformers import AutoTokenizer
import pandas as pd
from itertools import islice

tokenizer = AutoTokenizer.from_pretrained("/home/msiepel/PALGA-Transformers/PALGA-Transformers/T5_small_32128_pretrain_with_codes")

# Load the tsv file
df = pd.read_csv('/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_test_with_codes.tsv', delimiter='\t')
filtered_df = df[df['Codes'].str.contains('C-SEP')]


for row in islice(filtered_df.iterrows(), 10):
    text = row[1]['Conclusie']
    code = row[1]['Codes']
    print(tokenizer.tokenize(text.lower()))
    print(tokenizer.tokenize(code))
    print('--')
