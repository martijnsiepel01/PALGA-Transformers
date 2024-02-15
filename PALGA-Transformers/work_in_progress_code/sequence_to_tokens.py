import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
import wandb
import pandas as pd
from transformers import MT5Tokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AdamW, T5Tokenizer
import evaluate
from accelerate import Accelerator
from datasets import concatenate_datasets

def load_tokenizer(local_tokenizer_path = 'PALGA-Transformers/flan_tokenizer'):
    tokenizer = T5Tokenizer.from_pretrained(local_tokenizer_path)
    # tokenizer = MT5Tokenizer(vocab_file=local_tokenizer_path)
    return tokenizer

tokenizer = load_tokenizer("/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_32128/combined_data_unigram_t5_custom_32128_098.model")

sequence = "poliep colon resectie geen diagnose C-E poliep colon resectie tubulair adenoom laaggradige dysplasie C-E poliep colon resectie tubulair adenoom laaggradige dysplasie C-E poliep colon resectie tubulair adenoom laaggradige dysplasie C-E poliep colon resectie tubulair adenoom laaggradige dysplasie C-E poliep colon resectie tubulovilleus adenoom hooggradige dysplasie snijvlak vrij"

tokens = tokenizer(sequence, return_tensors="pt")

print(tokens)
print(len(tokens['input_ids'][0]))