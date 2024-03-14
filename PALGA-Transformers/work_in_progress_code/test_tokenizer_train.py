from transformers import AutoTokenizer
import pandas as pd

old_tokenizer = AutoTokenizer.from_pretrained("google/mT5-small")

# Load the tsv file
df = pd.read_csv('/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/pretrain/pretrain.tsv', delimiter='\t')

# Select only the "Conclusie" column
training_corpus = df['Conclusie'].dropna().tolist()

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 32128)

sentence = "Vacuum biopten linker mamma met een fibroadenoom."

old_tokenizer_tokens = old_tokenizer.tokenize(sentence)
new_tokenizer_tokens = tokenizer.tokenize(sentence)

print(old_tokenizer_tokens)
print(new_tokenizer_tokens)