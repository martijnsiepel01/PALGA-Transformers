from transformers import T5Config, T5ForConditionalGeneration, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import torch
import math



def load_model(tokenizer):
    config = T5Config(decoder_start_token_id=tokenizer.pad_token_id) 
    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    return model


def preprocess_function(examples, tokenizer, max_length_sentence, task):
    if task == 'span_corruption':
        inputs = [ex for ex in examples["input_sequence_span_corruption"]]
        targets = [ex for ex in examples["output_sequence_span_corruption"]]
    elif task == 'translation_pair_span_corruption':
        inputs = [ex for ex in examples["input_sequence_translation_pair_span_corruption"]]
        targets = [ex for ex in examples["output_sequence_translation_pair_span_corruption"]]
    elif task == 'span_corruption_with_target_concat':
        inputs = [ex for ex in examples["input_sequence_source_only_span_corruption_with_target_concat"]]
        targets = [ex for ex in examples["output_sequence_source_only_span_corruption_with_target_concat"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
    return model_inputs

def prepare_datasets_tsv(data_set, tokenizer, max_length_sentence, task):
    data_files = {
        "train": f"PALGA-Transformers/data/{data_set}/{data_set}_{task}_train.tsv",
        "validation": f"PALGA-Transformers/data/{data_set}/{data_set}_{task}_validation.tsv",
        "test": f"PALGA-Transformers/data/{data_set}/{data_set}_{task}_test.tsv"
    }

    # Load the dataset
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

    # Filter out examples with missing values in any relevant column
    for split in dataset.keys():
        dataset[split] = dataset[split].filter(lambda example: all(example[col] is not None and example[col] != '' for col in example))

    # Tokenize and preprocess the dataset
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length_sentence, task),
        batched=True,
    )

    # Remove unnecessary columns
    for split in tokenized_datasets.keys():
        tokenized_datasets[split] = tokenized_datasets[split].remove_columns(
            [col for col in tokenized_datasets[split].column_names if col not in ['input_ids', 'attention_mask', 'labels']]
        )
    tokenized_datasets[split].set_format("torch")

    # Optionally select a subset of the data for each split
    # train_dataset = tokenized_datasets["train"].select(range(int(len(tokenized_datasets["train"]) * 0.05)))
    # val_dataset = tokenized_datasets["validation"].select(range(int(len(tokenized_datasets["validation"]) * 0.05)))
    # test_dataset = tokenized_datasets["test"].select(range(int(len(tokenized_datasets["test"]) * 0.05)))

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    return train_dataset, val_dataset, test_dataset


def prepare_training_objects(learning_rate, model, train_dataloader, eval_dataloader, test_dataloader):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )
    return optimizer, accelerator, model, optimizer, train_dataloader, eval_dataloader, test_dataloader

def prepare_dataloaders_pretrain(train_dataset, val_dataset, test_dataset, data_collator, train_batch_size, validation_batch_size):
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size,
    )
    
    eval_dataloader = DataLoader(
        val_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size,
    )
    
    return train_dataloader, eval_dataloader, test_dataloader

# Define the learning rate scheduler
def inverse_square_root_schedule(optimizer, step, warmup_steps=1e4, init_lr=0.01):
    step = max(step, warmup_steps)
    lr = init_lr / math.sqrt(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Modify the train_step function to include the learning rate scheduler
def train_step(model, dataloader, optimizer, accelerator, current_step, warmup_steps=1e4, init_lr=0.01):
    model.train()
    total_train_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        # Update the learning rate
        # inverse_square_root_schedule(optimizer, current_step, warmup_steps, init_lr)
        
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        total_train_loss += loss.item()
        # current_step += 1  # Increment the step count

    avg_train_loss = total_train_loss / len(dataloader)
    train_perplexity = torch.exp(torch.tensor(avg_train_loss))  # Calculate train perplexity

    train_metrics = {
            "loss": avg_train_loss,
            "perplexity": train_perplexity.item()  # Convert to Python float for logging or printing
        }
    
    return train_metrics, current_step  # Return train metrics and the updated step count



def validation_step(model, dataloader):
    model.eval()
    total_eval_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluation"):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_eval_loss))

    eval_metrics = {
        "loss": avg_eval_loss,
        "perplexity": perplexity.item()  # Convert to Python float for logging or printing
    }
    
    return eval_metrics

def wandb_log_metrics(epoch, train_metrics, eval_metrics):
    wandb.log({
        "epoch/epoch": epoch,
        "loss/train_loss": train_metrics['loss'],
        "train/perplexity": train_metrics['perplexity'], 
        "loss/eval_loss": eval_metrics["loss"],
        "eval/perplexity": eval_metrics["perplexity"]
    })


def train_model(model, optimizer, accelerator, train_dataloader, eval_dataloader, test_dataloader, num_train_epochs, run_name, patience):
    num_training_steps = num_train_epochs * len(train_dataloader)
    print(f"Number of training steps: {num_training_steps}")

    lowest_loss = float("inf")
    early_stopping_counter = 0
    best_model_state_dict = None
    current_step = 0
    for epoch in range(num_train_epochs):
        train_metrics, current_step = train_step(model, train_dataloader, optimizer, accelerator, current_step)
        eval_metrics = validation_step(model, eval_dataloader)

        # Log metrics to WandB
        wandb_log_metrics(epoch, train_metrics, eval_metrics)

        if eval_metrics["loss"] < lowest_loss:  # Update loss accordingly
            lowest_loss = eval_metrics["loss"]
            best_model_state_dict = model.state_dict()  # Save the state dict of the best model
            torch.save(model.state_dict(), f'PALGA-Transformers/models/pretrained_models/{run_name}.pth')  # Save the model weights
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience: # Stop training if loss does not improve for patience epochs
                print(f"Early stopping triggered. No improvement in {patience} epochs.") 
                break

    # Load the best model state dict
    model.load_state_dict(best_model_state_dict)

    # Save the best model weights as a W&B artifact
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(f'PALGA-Transformers/models/pretrained_models/{run_name}.pth')
    wandb.log_artifact(artifact)