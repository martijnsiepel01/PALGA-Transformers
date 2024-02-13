import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
import wandb
import pandas as pd
from transformers import MT5Tokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AdamW, T5Tokenizer
import evaluate
from accelerate import Accelerator

def load_tokenizer(local_tokenizer_path = 'PALGA-Transformers/flan_tokenizer'):
    # tokenizer = T5Tokenizer.from_pretrained(local_tokenizer_path)
    tokenizer = MT5Tokenizer(vocab_file=local_tokenizer_path)
    return tokenizer




def generate_config_and_run_name(num_train_epochs, max_length_sentence, train_batch_size, validation_batch_size, learning_rate, max_generate_length, data_set, local_model_path, comment, patience, freeze_all_but_x_layers):
    config = {
        'num_train_epochs': num_train_epochs,
        'max_length_sentence': max_length_sentence,
        'train_batch_size': train_batch_size,
        'validation_batch_size': validation_batch_size,
        'lr': learning_rate,
        'max_generate_length': max_generate_length,
        'data_set': data_set,
        'local_model_path': local_model_path,
        'comment': comment,
        'patience': patience,
        'freeze_all_but_x_layers': freeze_all_but_x_layers,
    }

    run_name = f'epochs{num_train_epochs}_maxlengthsentence{max_length_sentence}_trainbatchsize{train_batch_size}_validationbatchsize{validation_batch_size}_lr{learning_rate}_maxgeneratelength{max_generate_length}_dataset{data_set}_model{local_model_path.split("/")[-1]}_comment{comment}_patience{patience}_freezeallbutxlayers{freeze_all_but_x_layers}'

    return config, run_name

def preprocess_function(examples, tokenizer, max_length_sentence):
    inputs = [ex for ex in examples["Conclusie"]]
    targets = [ex for ex in examples["Codes"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
    return model_inputs

def prepare_datasets_tsv(data_set, tokenizer, max_length_sentence):
    data_files = {"train": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_train.tsv", "test": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_test.tsv", "validation": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_validation.tsv"}
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    dataset = dataset.filter(lambda example: example["Conclusie"] is not None and example["Conclusie"] != '')
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
        batched=True
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes"])
    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets['validation']
    return train_dataset, val_dataset


def prepare_test_dataset(tokenizer, max_length_sentence):
    test_data_location = "PALGA-Transformers/data/gold_P1.tsv"
    dataset = load_dataset("csv", data_files=test_data_location, delimiter="\t")
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    dataset = dataset.filter(lambda example: example["Conclusie"] is not None and example["Conclusie"] != '')
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
        batched=True
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes"])
    tokenized_datasets.set_format("torch")
    test_dataset = tokenized_datasets['train']
    return test_dataset

def freeze_layers(model, freeze_all_but_x_layers):
    # Freeze all the parameters in the model first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last x encoder layers
    for param in model.encoder.block[-freeze_all_but_x_layers:].parameters():
        param.requires_grad = True

    # Unfreeze the last x decoder layers
    for param in model.decoder.block[-freeze_all_but_x_layers:].parameters():
        param.requires_grad = True

    model.encoder.final_layer_norm.weight.requires_grad = True
    model.decoder.final_layer_norm.weight.requires_grad = True
    model.lm_head.weight.requires_grad = True


    # Optionally, print out which layers are trainable to verify
    for name, param in model.named_parameters():
        print(f"{name} is {'trainable' if param.requires_grad else 'frozen'}")

    return model


def setup_model(tokenizer, freeze_all_but_x_layers, local_model_path = 'PALGA-Transformers/models/flan-t5-small'):
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))

    if freeze_all_but_x_layers > 0:
        model = freeze_layers(model, freeze_all_but_x_layers)
        
    return model

def prepare_datacollator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return data_collator

def prepare_dataloaders(train_dataset, val_dataset, test_dataset, data_collator, train_batch_size, validation_batch_size):
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size,
    )
    eval_dataloader = DataLoader(
        val_dataset, 
        collate_fn=data_collator, 
        batch_size=validation_batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, 
        collate_fn=data_collator, 
        batch_size=validation_batch_size
    )
    return train_dataloader, eval_dataloader, test_dataloader

def prepare_training_objects(learning_rate, model, train_dataloader, eval_dataloader, test_dataloader):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )
    return optimizer, accelerator, model, optimizer, train_dataloader, eval_dataloader, test_dataloader


def train_step(model, dataloader, optimizer, accelerator):
    model.train()
    total_train_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(dataloader)
    return avg_train_loss

def validation_step(model, dataloader, tokenizer, max_generate_length):
    metric = evaluate.load("sacrebleu")
    rouge = evaluate.load('rouge')

    model.eval()
    total_eval_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluation"):
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length,
            )

        labels = batch["labels"]
        loss = model(**batch).loss
        total_eval_loss += loss.item()

        # Filter and extend tokens and labels
        filtered_generated_tokens = [token[token != -100] for token in generated_tokens]
        filtered_labels = [label[label != -100] for label in labels]

        all_preds.extend(filtered_generated_tokens)
        all_labels.extend(filtered_labels)

    # Compute Metrics
    decoded_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(all_labels, skip_special_tokens=True)
    metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    rouge.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Compute BLEU score
    bleu_score = metric.compute()['score']
    # Compute ROUGE scores
    rouge_score = rouge.compute()
    ROUGE_1 = rouge_score['rouge1'] 
    ROUGE_2 = rouge_score['rouge2'] 
    ROUGE_L = rouge_score['rougeL'] 
    ROUGE_Lsum = rouge_score['rougeLsum'] 
    bleu_score = bleu_score / 100

    epsilon = 1e-7
    average_rouge_test = (ROUGE_1 + ROUGE_2 + ROUGE_L + ROUGE_Lsum)/4

    bleu_rouge_f1 = (2 * bleu_score * average_rouge_test) / (bleu_score + average_rouge_test + epsilon)
    
    eval_metrics = {
        "loss": total_eval_loss / len(dataloader),
        "bleu": bleu_score,
        "ROUGE_1": ROUGE_1,
        "ROUGE_2": ROUGE_2,
        "ROUGE_L": ROUGE_L,
        "ROUGE_Lsum": ROUGE_Lsum,
        "bleu_rouge_f1": bleu_rouge_f1
    }
    
    return eval_metrics

def test_step(model, dataloader, tokenizer, max_generate_length):
    metric = evaluate.load("sacrebleu")
    rouge = evaluate.load('rouge')

    model.eval()
    total_test_loss = 0.0
    test_preds = []
    test_labels = []
    decoded_test_preds = []  
    decoded_test_inputs = []  # Added line

    for batch in tqdm(dataloader, desc="Testing"):
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length,
            )

        labels = batch["labels"]
        loss = model(**batch).loss
        total_test_loss += loss.item()

        filtered_generated_tokens = [token[token != -100] for token in generated_tokens]
        filtered_labels = [label[label != -100] for label in labels]

        test_preds.extend(filtered_generated_tokens)
        test_labels.extend(filtered_labels)

        # Decode original input sequence
        decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        decoded_test_inputs.extend(decoded_inputs)

    decoded_test_preds = tokenizer.batch_decode(test_preds, skip_special_tokens=True)
    decoded_test_labels = tokenizer.batch_decode(test_labels, skip_special_tokens=True)
    metric.add_batch(predictions=decoded_test_preds, references=decoded_test_labels)
    rouge.add_batch(predictions=decoded_test_preds, references=decoded_test_labels)

    # Compute BLEU score
    bleu_score = metric.compute()['score']
    # Compute ROUGE scores
    rouge_score = rouge.compute()
    ROUGE_1 = rouge_score['rouge1'] 
    ROUGE_2 = rouge_score['rouge2'] 
    ROUGE_L = rouge_score['rougeL'] 
    ROUGE_Lsum = rouge_score['rougeLsum'] 
    bleu_score = bleu_score / 100

    epsilon = 1e-7
    average_rouge_test = (ROUGE_1 + ROUGE_2 + ROUGE_L + ROUGE_Lsum)/4

    bleu_rouge_f1 = (2 * bleu_score * average_rouge_test) / (bleu_score + average_rouge_test + epsilon)
    
    test_metrics = {
        "loss": total_test_loss / len(dataloader),
        "bleu": bleu_score,
        "ROUGE_1": ROUGE_1,
        "ROUGE_2": ROUGE_2,
        "ROUGE_L": ROUGE_L,
        "ROUGE_Lsum": ROUGE_Lsum,
        "bleu_rouge_f1": bleu_rouge_f1
    }
    
    return test_metrics, decoded_test_preds, decoded_test_labels, decoded_test_inputs

def print_test_predictions(decoded_test_preds, decoded_test_labels, decoded_test_input):
    print("Predictions on Test Data in the Last Epoch:")
    for input_seq, label, pred in zip(decoded_test_input, decoded_test_labels, decoded_test_preds):
        print("Input Sequence:", input_seq)
        print("Label:", label)
        print("Prediction:", pred)
        print('-'*100)  

def wandb_log_metrics(epoch, avg_train_loss, eval_metrics, test_metrics):
    wandb.log({
                "epoch/epoch": epoch,
                "loss/train_loss": avg_train_loss,
                "loss/eval_loss": eval_metrics["loss"],
                "loss/test_loss": test_metrics["loss"],
                "eval/BLEU": eval_metrics["bleu"],
                "eval/ROUGE-1": eval_metrics["ROUGE_1"],
                "eval/ROUGE-2": eval_metrics["ROUGE_2"],
                "eval/ROUGE-L": eval_metrics["ROUGE_L"],
                "eval/ROUGE-Lsum": eval_metrics["ROUGE_Lsum"],
                "eval/F1-Bleu-Rouge": eval_metrics["bleu_rouge_f1"],
                "test/BLEU": test_metrics["bleu"],
                "test/ROUGE-1": test_metrics["ROUGE_1"],
                "test/ROUGE-2": test_metrics["ROUGE_2"],
                "test/ROUGE-L": test_metrics["ROUGE_L"],
                "test/ROUGE-Lsum": test_metrics["ROUGE_Lsum"],
                "test/F1-Bleu-Rouge": test_metrics["bleu_rouge_f1"],
            })

def train_model(model, optimizer, accelerator, max_generate_length, train_dataloader, eval_dataloader, test_dataloader, num_train_epochs, tokenizer, run_name, patience=3):
    num_training_steps = num_train_epochs * len(train_dataloader)
    print(f"Number of training steps: {num_training_steps}")

    lowest_loss = float("inf")
    early_stopping_counter = 0
    best_model_state_dict = None

    for epoch in range(num_train_epochs):
        avg_train_loss = train_step(model, train_dataloader, optimizer, accelerator)
        eval_metrics = validation_step(model, eval_dataloader, tokenizer, max_generate_length)
        test_metrics, decoded_test_preds, decoded_test_labels, decoded_test_input = test_step(model, test_dataloader, tokenizer, max_generate_length)

        # Log metrics to WandB
        wandb_log_metrics(epoch, avg_train_loss, eval_metrics, test_metrics)

        if eval_metrics["loss"] < lowest_loss:  # Update loss accordingly
            lowest_loss = eval_metrics["loss"]
            best_model_state_dict = model.state_dict()  # Save the state dict of the best model
            torch.save(model.state_dict(), f'PALGA-Transformers/models/trained_models/{run_name}.pth')  # Save the model weights
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience: # Stop training if loss does not improve for patience epochs
                print(f"Early stopping triggered. No improvement in {patience} epochs.") 
                break

    # Load the best model state dict
    model.load_state_dict(best_model_state_dict)

    # Run print_test_predictions with the best model
    test_metrics, decoded_test_preds, decoded_test_labels, decoded_test_input = test_step(model, test_dataloader, tokenizer, max_generate_length)
    print_test_predictions(decoded_test_preds, decoded_test_labels, decoded_test_input)

    # Save the best model weights as a W&B artifact
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(f'PALGA-Transformers/models/trained_models/{run_name}.pth')
    wandb.log_artifact(artifact)