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
from datasets import load_dataset, concatenate_datasets
from transformers import get_scheduler
from tqdm.auto import tqdm
import nltk
from nltk.stem import PorterStemmer
from transformers import AdamW
from accelerate import Accelerator
import argparse

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from collections import defaultdict
from nltk.util import ngrams





nltk.download('punkt')  # Download NLTK's punkt tokenizer
stemmer = PorterStemmer()


def load_tokenizer(local_tokenizer_path = '/home/msiepel/tokenizer'):
    return AutoTokenizer.from_pretrained(local_tokenizer_path)

def generate_config_and_run_name(num_train_epochs, max_length_sentence, train_batch_size, validation_batch_size, learning_rate, max_generate_length, update_tokenizer, data_set, local_model_path, comment):
    config = {
        'num_train_epochs': num_train_epochs,
        'max_length_sentence': max_length_sentence,
        'train_batch_size': train_batch_size,
        'validation_batch_size': validation_batch_size,
        'lr': learning_rate,
        'max_generate_length': max_generate_length,
        'update_tokenizer': update_tokenizer,
        'data_set': data_set,
        'local_model_path': local_model_path,
        'comment': comment
    }

    run_name = f'epochs{num_train_epochs}_maxlengthsentence{max_length_sentence}_trainbatchsize{train_batch_size}_validationbatchsize{validation_batch_size}_lr{learning_rate}_maxgeneratelength{max_generate_length}_updatetokenizer{update_tokenizer}_dataset{data_set}_model{local_model_path.split("/")[-1]}_comment{comment}'

    return config, run_name

def stem_text(text):
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

def preprocess_function(examples, update_tokenizer, tokenizer, max_length_sentence):
    if update_tokenizer:
        inputs = [stem_text(ex) for ex in examples["Conclusie"]]
    else:
        inputs = [ex for ex in examples["Conclusie"]]
    targets = [ex for ex in examples["Codes"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
    return model_inputs

def prepare_datasets_tsv(data_set, update_tokenizer, tokenizer, max_length_sentence):
    data_files = {"train": f"data/{data_set}/{data_set}_norm_train.tsv", "test": f"data/{data_set}/{data_set}_norm_test.tsv", "validation": f"data/{data_set}/{data_set}_norm_validation.tsv"}
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, update_tokenizer, tokenizer, max_length_sentence),
        batched=True
    )
    tokenized_dataset_for_tokenizer = tokenized_datasets
    tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes"])
    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets['validation']
    return train_dataset, val_dataset, tokenized_dataset_for_tokenizer


def update_tokenizer_if_needed(tokenizer, train_dataset, val_dataset):
    combined_text = []

    for example in train_dataset:
        stemmed_text = stem_text(example['Conclusie'])
        combined_text.append(stemmed_text)

    for example in val_dataset:
        stemmed_text = stem_text(example['Conclusie'])
        combined_text.append(stemmed_text)

    unique_words = set()

    for text in combined_text:
        words = text.lower().split()  
        unique_words.update(words)  

    unique_words_list = list(unique_words)
    unique_words_list = set(unique_words_list) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(unique_words_list))

def setup_model(tokenizer, local_model_path = '/home/msiepel/mT5_small'):
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))
    return model

def prepare_datacollator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return data_collator

def prepare_dataloaders(train_dataset, val_dataset, data_collator, train_batch_size, validation_batch_size):
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size,
    )
    eval_dataloader = DataLoader(
        val_dataset, collate_fn=data_collator, batch_size=validation_batch_size
    )
    return train_dataloader, eval_dataloader

def prepare_training_objects(learning_rate, model, train_dataloader, eval_dataloader):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    return optimizer, accelerator, model, optimizer, train_dataloader, eval_dataloader

def train_model(model, optimizer, accelerator, max_generate_length, train_dataloader, eval_dataloader, num_train_epochs, tokenizer, run_name): 
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    metric = evaluate.load("sacrebleu")
    rouge = evaluate.load('rouge')

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

        for batch in tqdm(eval_dataloader, desc=f"Epoch {epoch + 1}/{num_train_epochs} - Evaluation"):
            with torch.no_grad():
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=max_generate_length,
                    num_beams=4,  # Set the beam width to 4
                    early_stopping=True,  # Stop the beam search when the first beam is finished
                    length_penalty=1,  # Apply a length penalty of 0.6
                    no_repeat_ngram_size=3  # Optional: Prevents the model from repeating n-grams
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

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', punctuation)).lower()
    return text

def preprocess_and_tokenize(text):
    stop_words = set(stopwords.words('dutch'))  
    preprocessed_text = preprocess_text(text)  
    tokens = word_tokenize(preprocessed_text)
    tokens_without_stopwords = [word for word in tokens if word not in stop_words]
    return tokens_without_stopwords

def readAndPreprocessData(dataset_for_graph):
    # nltk.download('punkt')
    # nltk.download('stopwords')
    df = dataset_for_graph
    df = df.iloc[1:]
    df['Conclusie_Tokens'] = df['Conclusie '].apply(preprocess_and_tokenize)
    return df

def createBiGrams(df):
    bigram_counts = defaultdict(int)

    # Iterate through each row of the DataFrame and count bigram occurrences
    for tokens in df['Conclusie_Tokens']:
        # Create bigrams from the tokenized sentences
        token_bigrams = ngrams(tokens, 2)
        for bigram in token_bigrams:
            # Join the tokens in the bigram into a string
            bigram_str = ' '.join(bigram)
            # Increment the count for the bigram
            bigram_counts[bigram_str] += 1

    print(f"{len(bigram_counts)} bigrams read")
    sorted_bigram_counts = dict(sorted(bigram_counts.items(), key=lambda item: item[1], reverse=True))
    return sorted_bigram_counts

def getGraph(dataset_for_graph):
    df = readAndPreprocessData(dataset_for_graph)
    sorted_bigram_counts = createBiGrams(df)
    print(sorted_bigram_counts)
    # g, idx_to_node = createDGLGraph(sorted_bigram_counts)

    # return g, idx_to_node

def main(num_train_epochs, max_generate_length, train_batch_size, validation_batch_size, learning_rate,
         max_length_sentence, update_tokenizer, data_set, local_tokenizer_path, local_model_path, comment):
    
    config, run_name = generate_config_and_run_name(
        num_train_epochs,
        max_length_sentence,
        train_batch_size,
        validation_batch_size,
        learning_rate,
        max_generate_length,
        update_tokenizer,
        data_set,
        local_model_path,
        comment
    )

    # Initialize WandB for logging
    # wandb.init(project="Transformers-PALGA", entity="srp-palga", config=config)

    tokenizer = load_tokenizer(local_tokenizer_path)

    if data_set == "autopsies" or data_set == "histo" or "all" in data_set:
        train_dataset, val_dataset, tokenized_dataset_for_tokenizer = prepare_datasets_tsv(data_set, update_tokenizer, tokenizer, max_length_sentence)
    else:
        train_dataset_histo, val_dataset_histo, tokenizer_dataset_histo = prepare_datasets_tsv("histo", update_tokenizer, tokenizer, max_length_sentence)
        train_dataset_autopsies, val_dataset_autopsies, tokenizer_dataset_autopsies = prepare_datasets_tsv("autopsies", update_tokenizer, tokenizer, max_length_sentence)
        train_dataset = concatenate_datasets([train_dataset_histo, train_dataset_autopsies])
        val_dataset = concatenate_datasets([val_dataset_histo, val_dataset_autopsies])
    
    # existing_graph, idx_to_node = getGraph(tokenized_dataset_for_tokenizer)
    getGraph(tokenized_dataset_for_tokenizer)
    print("done!")



    # # Update tokenizer if needed
    # if update_tokenizer:
    #     tokenize_dataset_train = concatenate_datasets([tokenizer_dataset_histo['train'], tokenizer_dataset_autopsies['train']])
    #     tokenize_dataset_val = concatenate_datasets([tokenizer_dataset_histo['validation'], tokenizer_dataset_autopsies['validation']])
    #     update_tokenizer_if_needed(tokenizer, tokenize_dataset_train, tokenize_dataset_val)

    # # Setup model and tokenizer
    # model = setup_model(tokenizer, local_model_path)
    
    # data_collator = prepare_datacollator(tokenizer, model)

    # train_dataloader, eval_dataloader = prepare_dataloaders(train_dataset, val_dataset, data_collator, train_batch_size, validation_batch_size)
    
    # optimizer, accelerator, model, optimizer, train_dataloader, eval_dataloader = prepare_training_objects(learning_rate, model, train_dataloader, eval_dataloader)

    # # Training and evaluation
    # train_model(model, optimizer, accelerator, max_generate_length, train_dataloader, eval_dataloader, num_train_epochs, tokenizer, run_name)

# Entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for fine-tuning a hugginface transformer model")

    # Define the command-line arguments
    parser.add_argument("--num_train_epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--max_generate_length", type=int, default=32, help="Max length for generation")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--validation_batch_size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length_sentence", type=int, default=512, help="Max length of sentence")
    parser.add_argument("--update_tokenizer", type=bool, default=False, help="Whether to update tokenizer")
    parser.add_argument("--data_set", type=str, default='histo', help="Dataset name")
    parser.add_argument("--local_tokenizer_path", type=str, default='/home/msiepel/tokenizer', help="Local tokenizer path")
    parser.add_argument("--local_model_path", type=str, default='/home/msiepel/mT5_small', help="Local model path")
    parser.add_argument("--comment", type=str, default='', help="Comment for the current run")

    args = parser.parse_args()

    # Call main function with the parsed arguments
    main(args.num_train_epochs, args.max_generate_length, args.train_batch_size, args.validation_batch_size,
         args.learning_rate, args.max_length_sentence, args.update_tokenizer, args.data_set, args.local_tokenizer_path, args.local_model_path, args.comment)