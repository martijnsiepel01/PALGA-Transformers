from transformers import T5Tokenizer


tokenizer = T5Tokenizer('/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_32128/combined_data_unigram_t5_custom_32128_1_lower_case.model')

special_tokens_dict = {
    "unk_token": "<unk>",
    "start_token": "<s>",
    "end_token": "</s>",
    "pad_token": "[PAD]",
    "unk_token_2": "[UNK]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
    "c_sep_token": "[C-SEP]",
}

# Get the token IDs for the special tokens
special_token_ids = {token: tokenizer.convert_tokens_to_ids(special) for token, special in special_tokens_dict.items()}


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


allowed_words = get_unique_thesaurus_term_words('/home/gburger01/snomed_20230426.txt')

# Get the full vocabulary
vocab = tokenizer.get_vocab()

# Convert allowed words to their token IDs
allowed_ids = [vocab[word] for word in allowed_words if word in vocab]

# List of special token strings you want to include
special_tokens_strings = ['<unk>', '<s>', '</s>', '[PAD]', '[UNK]', '[CLS]', '[MASK]', '[C-SEP]']

# Convert special token strings to their token IDs using the vocabulary
special_tokens_ids = [vocab.get(token) for token in special_tokens_strings if token in vocab]

# Combine allowed IDs with special token IDs, ensuring no duplicates and removing None values
allowed_ids_set = set(id for id in (allowed_ids + special_tokens_ids) if id is not None)

print(allowed_ids_set)

# Create a set of all token IDs in the vocabulary
all_ids = set(vocab.values())

# Subtract the set of allowed token IDs from the set of all token IDs to get the bad words IDs
bad_words_ids = list(all_ids - allowed_ids_set)