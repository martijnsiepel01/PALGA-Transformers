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
                  f'--vocab_size={vocab_size} --character_coverage=1.0 ' \
                  f'--model_type=bpe --control_symbols={",".join(special_tokens)}'
    
    # Train the SentencePiece model
    spm.SentencePieceTrainer.Train(spm_command)

# Path to your dataset
input_file = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_train.tsv"
temp_file_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/temp_train_data.txt"

# Convert your dataset to the required format if it's not already in plain text
# For example, converting from TSV to plain text as in the initial question
with open(input_file, "r") as file, open(temp_file_path, "w") as temp_file:
    reader = csv.reader(file, delimiter="\t")
    for row in reader:
        temp_file.write(row[0] + "\n")

# Training parameters
model_prefix = "t5_custom_128000_1"
vocab_size = 128000
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[MASK]", "[C-SEP]"]  # Include your special tokens here

# Train the tokenizer
train_sentencepiece_tokenizer(temp_file_path, model_prefix, vocab_size, special_tokens)

# Clean up the temporary file if necessary
os.remove(temp_file_path)

# The tokenizer is now trained and saved to "t5_custom.model" and "t5_custom.vocab"
