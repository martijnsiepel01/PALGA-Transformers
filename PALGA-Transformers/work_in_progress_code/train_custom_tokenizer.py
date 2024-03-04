import sentencepiece as spm
import csv
import os

def train_sentencepiece_tokenizer(input_file_path, model_prefix, vocab_size, special_tokens):
    """
    Trains a SentencePiece tokenizer.
    
    Parameters:
    - input_file_path: Path to the text file containing training data.
    - model_prefix: Prefix for the output model files.
    - vocab_size: Target vocabulary size.
    - special_tokens: List of special tokens to include in the tokenizer.
    """
    # Preparing SentencePiece command
    spm_command = f'--input={input_file_path} --model_prefix={model_prefix} ' \
                  f'--vocab_size={vocab_size} --character_coverage=1.0' \
                  f'--model_type=unigram --control_symbols={",".join(special_tokens)}'
    
    # Train the SentencePiece model
    spm.SentencePieceTrainer.Train(spm_command)

# Path to your dataset
# input_files = [
#     "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_train.tsv",
#     "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/autopsies/autopsies_norm_train.tsv",
#     "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_train.tsv"
# ]
    
input_files = ['/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_train_with_codes.tsv']
temp_file_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/temp_train_data.txt"

# Convert your dataset to the required format if it's not already in plain text
# For example, converting from TSV to plain text as in the initial question
with open(temp_file_path, "w") as temp_file:
    for input_file in input_files:
        with open(input_file, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            count = 0
            for row in reader:
                temp_file.write(row[0].lower().replace("[c-sep]", "").replace("[crlf]", "") + "\n")  # Convert input to lowercase and remove [C-SEP]
                temp_file.write(row[1].lower() + "\n")  # Convert input to lowercase

# Training parameters
model_prefix = "histo_unigram_t5_custom_32128_1_lower_case_with_codes"
vocab_size = 16000
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[MASK]", "[C-SEP]"]  # Include your special tokens here

# Train the tokenizer
train_sentencepiece_tokenizer(temp_file_path, model_prefix, vocab_size, special_tokens)

# Clean up the temporary file if necessary
# os.remove(temp_file_path)

# The tokenizer is now trained and saved to "t5_custom.model" and "t5_custom.vocab"
