import pandas as pd
import string
from transformers import AutoTokenizer, AutoModel, MT5Tokenizer, MBart50TokenizerFast


# -- load a PALGA thesaurus file (snomed_xxxxxxxx.txt) and extract the terms
# -- split these on space and add the term words to the additional vocabulary list if not therein already
# -- returns this list
def get_unique_thesaurus_term_words(thesaurus_file_name):
    unique_term_words = []

    with open(thesaurus_file_name, "r", encoding='latin-1') as fh: # utf-8 encoding gives an error
        for line in fh:
            term = line.split("|")[0]
            term_words = term.split()
            for w in term_words:
                if not w in unique_term_words: # add if not already in list
                    unique_term_words.append(w)

    return unique_term_words


utw = get_unique_thesaurus_term_words('/home/msiepel/data/snomed_20230426.txt')

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


# -- delete the words from the list of unique PALGA thesaurus term words already in the tokenizer's vocabulary:
utw = set(utw) - set(tokenizer.vocab.keys())

# -- add the words from the PALGA thesaurus:
tokenizer.add_tokens(list(utw))

# -- add special tokens ('[C-SEP]' in the code series)
tokenizer.add_tokens('[C-SEP]', special_tokens=True)

tokenizer.save_pretrained('/home/msiepel/mbart_mmt_tokenizer')