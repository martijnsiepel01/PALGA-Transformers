import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset, load_metric
import numpy as np
import wandb
import csv
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, MT5Tokenizer, TrainingArguments, Trainer, MT5ForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer, MT5Model
import evaluate
from torch.utils.data import random_split
from transformers import Adafactor
from datasets import load_dataset
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
from wandb.keras import WandbCallback
import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt')  # Download NLTK's punkt tokenizer
stemmer = PorterStemmer()


data = "tsv"
num_train_epochs = 50
max_generate_length = 32
train_batch_size = 8
validation_batch_size = 4
learning_rate = 1e-4
max_length_sentence = 512
update_tokenizer = False
data_set = 'autopsies'

config={'num_train_epochs': num_train_epochs, 'max_length_sentence': max_length_sentence, 'train_batch_size': train_batch_size, 'validation_batch_size': validation_batch_size, 'lr': learning_rate, 'max_generate_length': max_generate_length, 'update_tokenizer': update_tokenizer, 'data_set': data_set}
run_name = f'epochs{num_train_epochs}_maxlengthsentence{max_length_sentence}_trainbatchsize{train_batch_size}_validationbatchsize{validation_batch_size}_lr{learning_rate}_maxgeneratelength{max_generate_length}_updatetokenizer{update_tokenizer}_dataset{data_set}'
print(run_name)

# Initialize WandB for logging
wandb.init(project="Transformers-PALGA", entity="srp-palga", config=config)

local_tokenizer_path = '/home/msiepel/tokenizer'
tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)

def stem_text(text):
    words = nltk.word_tokenize(text)
    
    stemmed_words = [stemmer.stem(word) for word in words]
    
    stemmed_text = ' '.join(stemmed_words)
    
    return stemmed_text

def preprocess_function(examples):
    if update_tokenizer:
        inputs = [stem_text(ex) for ex in examples["Conclusie"]]
    else:
        inputs = [ex for ex in examples["Conclusie"]]
    targets = [ex for ex in examples["Codes"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
    return model_inputs

if data == 'tsv':
    data_files = {"train": f"data/{data_set}/{data_set}_norm_train.tsv", "test": f"data/{data_set}/{data_set}_norm_test.tsv", "validation": f"data/{data_set}/{data_set}_norm_validation.tsv"}
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_dataset_for_tokenizer = tokenized_datasets
    tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes"])
    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets['validation']
elif data == 'csv':
    file_path = 'data/Mixed.csv'
    df = pd.read_csv(file_path, sep=';', encoding='latin-1')  # Change the encoding as necessary
    df.columns = df.columns.str.strip().str.replace(r'\W', '_')
    dataset = df[['PALGA-diagnose', 'Conclusie']].copy()
    dataset = dataset.rename(columns={'PALGA-diagnose': 'Codes'})
    dataset = dataset.dropna(subset=['Codes'])
    dataset = dataset[dataset['Codes'].astype(str).str.strip().str.len() > 0]  # Remove empty values
    dataset = dataset[dataset['Codes'].astype(str).str.contains(r'[a-zA-Z*]')]
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes", "__index_level_0__"])
    tokenized_datasets.set_format("torch")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(tokenized_datasets, [train_size, val_size])


if update_tokenizer:
    combined_text = []

    for example in tokenized_dataset_for_tokenizer['train']:
        stemmed_text = stem_text(example['Conclusie'])
        combined_text.append(stemmed_text)

    for example in tokenized_dataset_for_tokenizer['validation']:
        stemmed_text = stem_text(example['Conclusie'])
        combined_text.append(stemmed_text)
    unique_words = set()

    for text in combined_text:
        words = text.lower().split()  # Split text into words
        unique_words.update(words)  # Add unique words to the set

    # Display the list of unique words
    unique_words_list = list(unique_words)
    unique_words_list = set(unique_words_list) - set(tokenizer.vocab.keys())
    print(len(tokenizer))
    tokenizer.add_tokens(list(unique_words_list))
    print(len(tokenizer))

model = AutoModelForSeq2SeqLM.from_pretrained("/home/msiepel/mT5", local_files_only=True)
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=train_batch_size,
)
eval_dataloader = DataLoader(
    val_dataset, collate_fn=data_collator, batch_size=validation_batch_size
)

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=learning_rate)

from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch


def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels

metric = evaluate.load("sacrebleu")
rouge = evaluate.load('rouge')
progress_bar = tqdm(range(num_training_steps))

best_bleu_rouge_f1 = 0.0

# Loop over training epochs
for epoch in range(num_train_epochs):
    # Training
    model.train()
    total_train_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_train_epochs} - Training"):
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)

    # Evaluation
    model.eval()
    total_eval_loss = 0.0
    all_preds = []
    all_labels = []
    all_metrics = []

    for batch in tqdm(eval_dataloader, desc=f"Epoch {epoch + 1}/{num_train_epochs} - Evaluation"):
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length,
            )

        labels = batch["labels"]
        loss = model(**batch).loss
        total_eval_loss += loss.item()

        # Filter out -100 tokens from generated_tokens and labels before extending
        filtered_generated_tokens = [token[token != -100] for token in generated_tokens]
        filtered_labels = [label[label != -100] for label in labels]

        all_preds.extend(filtered_generated_tokens)
        all_labels.extend(filtered_labels)

    avg_eval_loss = total_eval_loss / len(eval_dataloader)

    decoded_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(all_labels, skip_special_tokens=True)

    # Compute BLEU score using sacrebleu
    metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    bleu_score = metric.compute()['score']

    # Compute ROUGE scores
    rouge.add_batch(predictions=decoded_preds, references=decoded_labels)
    rouge_scores = rouge.compute()

    ROUGE_1 = rouge_scores['rouge1'] 
    ROUGE_2 = rouge_scores['rouge2'] 
    ROUGE_L = rouge_scores['rougeL'] 
    ROUGE_Lsum = rouge_scores['rougeLsum'] 
    bleu_score = bleu_score / 100

    average_rouge = (ROUGE_1 + ROUGE_2 + ROUGE_L + ROUGE_Lsum)/4

    epsilon = 1e-7
    bleu_rouge_f1 = (2 * bleu_score * average_rouge) / (bleu_score + average_rouge + epsilon)

    # Log metrics to WandB
    wandb.log({
        "epoch/epoch": epoch,
        "loss/train_loss": avg_train_loss,
        "loss/eval_loss": avg_eval_loss,
        "eval/BLEU": bleu_score,
        "eval/ROUGE-1": ROUGE_1,
        "eval/ROUGE-2": ROUGE_2,
        "eval/ROUGE-L": ROUGE_L,
        "eval/ROUGE-Lsum": ROUGE_Lsum,
        "eval/F1-Bleu-Rouge": bleu_rouge_f1,
    })


    
    if bleu_rouge_f1 > best_bleu_rouge_f1:  # Update best_bleu_score accordingly
        best_bleu_rouge_f1 = bleu_rouge_f1
        torch.save(model.state_dict(), f'/home/msiepel/models/{run_name}.pth')  # Save the model weights

# Save the best model weights as a W&B artifact
artifact = wandb.Artifact("best_model", type="model")
artifact.add_file(f'/home/msiepel/models/{run_name}.pth')
wandb.log_artifact(artifact)