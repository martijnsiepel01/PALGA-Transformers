import torch
from tqdm import tqdm
from datasets import load_dataset
import wandb
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AdamW, Adafactor, AutoConfig, T5ForConditionalGeneration
import evaluate
from accelerate import Accelerator
from datasets import concatenate_datasets
import random
import re

from shared_utils import *

experiment_id = str(random.randint(1, 1000000))

def preprocess_function(examples, tokenizer, max_length_sentence):
    inputs = examples["Conclusie_Conclusie"]
    targets = examples["Codes"]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length_sentence, truncation=True)
    # Calculate lengths based on the inputs for histogram-based splitting
    lengths = [len(input.split()) for input in inputs]  # Assuming length is based on word count before tokenization
    model_inputs["length"] = lengths
    return model_inputs

def prepare_datasets_tsv(data_set, tokenizer, max_length_sentence):
    data_files = {
        "train": f"/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/train.tsv",
        "test": f"/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/test.tsv",
        "validation": f"/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/val.tsv"
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

    # Function to get 1% of the dataset
    def get_percentage(ds):
        return ds.shuffle(seed=42).select(range(max(1, int(0.05 * len(ds)))))

    # Filter rows where 'Codes' or 'Conclusie' is not null or empty, and preprocess
    for split in dataset.keys():
        dataset[split] = dataset[split].filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
        dataset[split] = dataset[split].filter(lambda example: example["Conclusie_Conclusie"] is not None and example["Conclusie_Conclusie"] != '')
        dataset[split] = dataset[split].map(
            lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
            batched=True
        )

        dataset[split] = dataset[split].remove_columns(["Conclusie_Conclusie", "Codes", "Type", "Jaar", "Epicrise", "Conclusie_Epicrise", "Conclusie"])
        # dataset[split] = get_percentage(dataset[split])

    # Sort the validation dataset by 'length' for histogram-based splitting
    val_dataset = dataset['validation'].sort("length")

    # Split the validation dataset into 5 parts
    total_length = len(val_dataset)
    split_sizes = [total_length // 5] * 4 + [total_length - (total_length // 5 * 4)]  # Handle remainder in the last split
    val_datasets = []
    for i in range(5):
        start_index = sum(split_sizes[:i])  # Calculate the starting index for each split
        end_index = start_index + split_sizes[i]
        val_datasets.append(val_dataset.select(range(start_index, end_index)))

    train_dataset = dataset['train'].remove_columns(["length"])
    val_datasets = [vd.remove_columns(["length"]) for vd in val_datasets]  # Remove the 'length' column from the validation splits

    return train_dataset, val_datasets
    
# def prepare_test_dataset(tokenizer, max_length_sentence):
#     test_data_location = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/gold_resolved_with_codes.tsv"
#     dataset = load_dataset("csv", data_files={"test": test_data_location}, delimiter="\t")['test']
    
#     dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
#     dataset = dataset.filter(lambda example: example["Conclusie"] is not None and example["Conclusie"] != '')
#     tokenized_datasets = dataset.map(
#         lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
#         batched=True
#     )
#     tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes"])

#     # Sort the test dataset by 'length' for histogram-based splitting
#     test_dataset_sorted = tokenized_datasets.sort("length")

#     # Split the test dataset into 5 parts
#     total_length = len(test_dataset_sorted)
#     split_sizes = [total_length // 5] * 4 + [total_length - (total_length // 5 * 4)]  # Handle remainder in the last split
#     test_datasets = []
#     for i in range(5):
#         start_index = sum(split_sizes[:i])  # Calculate the starting index for each split
#         end_index = start_index + split_sizes[i]
#         split_dataset = test_dataset_sorted.select(range(start_index, end_index))
#         split_dataset.set_format("torch")  # Set format to torch if necessary for each split
#         test_datasets.append(split_dataset)

#     return test_datasets

def setup_model(tokenizer, freeze_all_but_x_layers, local_model_path='PALGA-Transformers/models/flan-t5-small', dropout_rate=0.1, rank=16, lora_alpha=32, lora_dropout=0.01):
    config = AutoConfig.from_pretrained(local_model_path)

    if dropout_rate is not None:
        config.dropout_rate = dropout_rate
    
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path, config=config)

    model.resize_token_embeddings(len(tokenizer))
    # checkpoint = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs20_data_setcombined_commentmt5_final.pth"

    # config = AutoConfig.from_pretrained(local_model_path)
    # model = T5ForConditionalGeneration(config)
    # model.resize_token_embeddings(len(tokenizer))
    # model.load_state_dict(torch.load(checkpoint))
    return model



def prepare_training_objects(learning_rate, model, train_dataloader, eval_dataloaders, lr_strategy, total_steps, optimizer_type='adamw'):
    if optimizer_type.lower() == 'adamw':
        # optimizer = AdamW(model.base_model.model.parameters(), lr=learning_rate)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'adafactor':
        optimizer = Adafactor(
            model.parameters(),
            lr=1e-4,  # Learning rate
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,  # Disabling momentum
            weight_decay=0.0,  # No weight decay
            relative_step=False,  # Absolute lr
            scale_parameter=False,  # No scaling
            warmup_init=False  # No warmup
        )
    else:
        raise ValueError("Unsupported optimizer type. Please choose 'adamw' or 'adafactor'.")

    accelerator = Accelerator()

    # Prepare model, optimizer, and train_dataloader as usual
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Prepare each eval_dataloader separately
    eval_dataloaders = [accelerator.prepare(eval_dl) for eval_dl in eval_dataloaders]

    scheduler = None

    return optimizer, accelerator, model, train_dataloader, eval_dataloaders, scheduler


thesaurus_location = '/home/gburger01/snomed_20230426.txt'
thesaurus = pd.read_csv(thesaurus_location, sep='|', encoding='latin-1')

# Function to get word from code
def get_word_from_code(code):
    if code == '[C-SEP]' or code == '[c-sep]':
        return '[C-SEP]'
    else:
        word = thesaurus[(thesaurus['DEPALCE'].str.lower() == code.lower()) & (thesaurus['DESTACE'] == 'V')]['DETEROM'].values
        return word[0] if len(word) > 0 else 'Unknown'
    
def custom_decode(tokenizer, token_ids):
    tokens = [tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]
    decoded_string = ' '.join(tokens).replace(' </s>', '').strip()
    return decoded_string

def train_step(model, dataloader, optimizer, accelerator, scheduler, tokenizer):
    model.train()
    total_train_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        outputs = model(**batch)
        loss = outputs.loss  # This is the total loss for the batch

        # Assuming 'input_ids' and 'labels' in batch, and outputs contain logits
        input_tokens = batch['input_ids']
        output_tokens = outputs.logits.argmax(-1)

        # Calculate 'c-sep' counts for each sequence in input and output
        input_c_sep_counts = count_c_sep(input_tokens, tokenizer)
        output_c_sep_counts = count_c_sep(output_tokens, tokenizer)

        # Calculate reweight factors for each sequence in the batch
        reweight_factors = [calculate_reweight_factor(inp, out) for inp, out in zip(input_c_sep_counts, output_c_sep_counts)]

        # Adjust the loss for each sequence and sum them up
        weighted_losses = loss * torch.tensor(reweight_factors).to(loss.device)  # Ensure reweight factors are on the same device as loss
        weighted_loss = weighted_losses.mean()  # Calculate mean to combine individual losses

        optimizer.zero_grad()
        accelerator.backward(weighted_loss)
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_train_loss += weighted_loss.item()

    avg_train_loss = total_train_loss / len(dataloader)
    # print(f"Average training loss: {avg_train_loss}")
    return avg_train_loss

def count_c_sep(tokens, tokenizer):
    # Adjust to handle a batch of sequences and return a list of counts
    counts = []
    for sequence in tokens:
        # token_list = custom_decode(tokenizer, sequence).split()
        token_list = tokenizer.decode(sequence, skip_special_tokens=True).split()
        counts.append(sum(1 for token in token_list if 'c-sep' in token.lower()))
        # counts.append(sum(1 for token in token_list if 'c-sep' in token))
    return counts

def calculate_reweight_factor(input_count, output_count):
    # No change needed, works on individual counts
    return 1
    diff = abs(input_count - output_count)
    if diff > 0:
        reweight_factor = (1 + diff)  # Adjust formula as needed
        return reweight_factor
    return 1

def create_prefix_allowed_tokens_fn(Palga_trie, tokenizer):
    # c_sep_token_id = tokenizer.encode('[C-SEP]', add_special_tokens=False)  # Get the token ID for '[C-SEP]'
    c_sep_token_id = [491, 424, 264, 155719, 439]
    # print(f"CSEP: {c_sep_token_id}")

    def prefix_allowed_tokens_fn(batch_id, sent):
        # Convert the sentence to a list excluding the first element (usually the BOS token)
        sent = sent.tolist()[1:]
        
        # Initialize the last_split_index to -1 (no split found initially)
        last_split_index = -1
        
        # Check if c_sep_token_id is a list (multiple token IDs) or a single token ID
        if isinstance(c_sep_token_id, list):
            # Iterate through the sentence to find the last occurrence of the c_sep_token_id sequence
            for i in range(len(sent) - len(c_sep_token_id) + 1):
                if sent[i:i+len(c_sep_token_id)] == c_sep_token_id:
                    last_split_index = i + len(c_sep_token_id) - 1
        else:
            # If c_sep_token_id is a single ID, find its last occurrence
            try:
                reversed_index = sent[::-1].index(c_sep_token_id)
                last_split_index = len(sent) - 1 - reversed_index
            except ValueError:
                print("No occurrence of single ID found")

        # If a split was found, update the sentence to only include tokens after the last split
        if last_split_index != -1:
            sent = sent[last_split_index + 1:]

        # Get allowed tokens using Palga_trie
        out = list(Palga_trie.get(sent))
        # If no allowed tokens found, return the EOS token
        if len(out) > 0:
            return out
        else:
            return list(tokenizer.encode(tokenizer.eos_token))
    return prefix_allowed_tokens_fn

def validation_step(model, eval_dataloaders, tokenizer, max_generate_length, constrained_decoding, Palga_trie):
    dataloader_names = ["shortest", "short", "average", "long", "longest"]
    all_validation_metrics = {}
    
    decoded_validation_inputs = []
    decoded_validation_preds = []
    decoded_validation_labels = []

    if len(eval_dataloaders) != 5:
        raise ValueError("There must be exactly 5 validation dataloaders.")

    for dataloader, name in zip(eval_dataloaders, dataloader_names):
        metric = evaluate.load("sacrebleu", experiment_id=experiment_id)
        rouge = evaluate.load('rouge', experiment_id=experiment_id)

        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc=f"Validating {name}"):
            if 'length' in batch:
                del batch['length']

            with torch.no_grad():
                if constrained_decoding:
                    print('constrained decoding')
                    prefix_allowed_tokens_fn = create_prefix_allowed_tokens_fn(Palga_trie, tokenizer)
                    outputs =  model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=max_generate_length,
                        diversity_penalty=0.3,
                        num_beams=6,
                        num_beam_groups=2,
                        no_repeat_ngram_size=3,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    )
                else:
                    outputs =  model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=max_generate_length,
                        diversity_penalty=0.3,
                        num_beams=6,
                        num_beam_groups=2,
                        no_repeat_ngram_size=3,
                    )
                
                loss = model(**batch).loss
                total_loss += loss.item()

            # Filtering predictions and labels before decoding
            filtered_preds = [[token_id for token_id in token if token_id != tokenizer.pad_token_id] for token in outputs]
            filtered_labels = [[token_id for token_id in token if token_id != -100] for token in batch["labels"]]
            
            # Decoding filtered predictions and labels # .replace('[C-SEP]', ' [C-SEP] ')
            # decoded_preds = [custom_decode(tokenizer, pred) for pred in filtered_preds]
            decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in filtered_preds]
            # decoded_labels = [custom_decode(tokenizer, label) for label in filtered_labels]
            decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in filtered_labels]

            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)

            # Decoding input sequences for detailed analysis or printing later
            decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

            
            decoded_validation_inputs.extend(decoded_inputs)

            # Print decoded examples for this batch
            print(f"Validation examples from {name}:")
            for i in range(min(1, len(decoded_preds))):  # Ensure there are enough examples to print
                pred_words = [get_word_from_code(code) for code in decoded_preds[i].split()]
                label_words = [get_word_from_code(code) for code in decoded_labels[i].split()]

                print(f"Input:            {decoded_inputs[i]}")
                print(f"Prediction:       {pred_words}")
                print(f"Prediction codes: {decoded_preds[i]}")
                print(f"Label:            {label_words}")
                print(f"Label codes:      {decoded_labels[i]}")
                print("----------")

        # Compute metrics for each dataloader
        bleu = metric.compute(predictions=all_preds, references=[[label] for label in all_labels])
        rouge_scores = rouge.compute(predictions=all_preds, references=all_labels)

        ROUGE_1 = rouge_scores['rouge1'] 
        ROUGE_2 = rouge_scores['rouge2'] 
        ROUGE_L = rouge_scores['rougeL'] 
        ROUGE_Lsum = rouge_scores['rougeLsum'] 
        bleu_score = bleu["score"] / 100

        epsilon = 1e-7
        average_rouge_validation = (ROUGE_1 + ROUGE_2 + ROUGE_L + ROUGE_Lsum)/4

        bleu_rouge_f1 = (2 * bleu_score * average_rouge_validation) / (bleu_score + average_rouge_validation + epsilon)

        # Storing metrics for each validation dataloader
        all_validation_metrics[f"loss_{name}"] = total_loss / len(dataloader)
        all_validation_metrics[f"bleu_{name}"] = bleu_score
        all_validation_metrics[f"average_rouge_{name}"] = average_rouge_validation
        all_validation_metrics[f"bleu_rouge_f1_{name}"] = bleu_rouge_f1


        decoded_validation_preds.extend(all_preds)
        decoded_validation_labels.extend(all_labels)

    return all_validation_metrics, decoded_validation_preds, decoded_validation_labels, decoded_validation_inputs

def print_test_predictions(decoded_test_preds, decoded_test_labels, decoded_test_input):
    print("Predictions on Validation Data in the Last Epoch:")
    # Load the thesaurus

    random_indices = random.sample(range(len(decoded_test_input)), 100)

    for index in random_indices:
        input_seq = decoded_test_input[index]
        label = decoded_test_labels[index]
        pred = decoded_test_preds[index]

        print("Input Sequence:", input_seq)
        print("Label:           ", label)
        print("Prediction:      ", pred)

        # Convert codes to words for label
        label_words = [get_word_from_code(code) for code in label.split()]
        print("Label Words:     ", ' '.join(label_words))

        # Convert codes to words for prediction
        pred_words = [get_word_from_code(code) for code in pred.split()]
        print("Prediction Words:", ' '.join(pred_words))

        print('-'*100)

# def wandb_log_metrics(epoch, avg_train_loss, eval_metrics, test_metrics):
def wandb_log_metrics(epoch, avg_train_loss, eval_metrics, suffix="", run=None):
    # Compute average evaluation metrics
    avg_eval_loss = (eval_metrics["loss_shortest"] + eval_metrics["loss_short"] + eval_metrics["loss_average"] + eval_metrics["loss_long"] + eval_metrics["loss_longest"]) / 5
    avg_eval_bleu = (eval_metrics["bleu_shortest"] + eval_metrics["bleu_short"] + eval_metrics["bleu_average"] + eval_metrics["bleu_long"] + eval_metrics["bleu_longest"]) / 5
    avg_eval_rouge = (eval_metrics["average_rouge_shortest"] + eval_metrics["average_rouge_short"] + eval_metrics["average_rouge_average"] + eval_metrics["average_rouge_long"] + eval_metrics["average_rouge_longest"]) / 5
    avg_eval_f1_bleu_rouge = (eval_metrics["bleu_rouge_f1_shortest"] + eval_metrics["bleu_rouge_f1_short"] + eval_metrics["bleu_rouge_f1_average"] + eval_metrics["bleu_rouge_f1_long"] + eval_metrics["bleu_rouge_f1_longest"]) / 5
    
    # Prepare metrics with suffix
    metrics = {
        f"epoch/epoch{suffix}": epoch,
        f"training/train_loss{suffix}": avg_train_loss,
        f"shortest/eval_loss_shortest{suffix}": eval_metrics["loss_shortest"],
        f"short/eval_loss_short{suffix}": eval_metrics["loss_short"],
        f"average/eval_loss_average{suffix}": eval_metrics["loss_average"],
        f"long/eval_loss_long{suffix}": eval_metrics["loss_long"],
        f"longest/eval_loss_longest{suffix}": eval_metrics["loss_longest"],
        f"shortest/eval_BLEU_shortest{suffix}": eval_metrics["bleu_shortest"],
        f"short/eval_BLEU_short{suffix}": eval_metrics["bleu_short"],
        f"average/eval_BLEU_average{suffix}": eval_metrics["bleu_average"],
        f"long/eval_BLEU_long{suffix}": eval_metrics["bleu_long"],
        f"longest/eval_BLEU_longest{suffix}": eval_metrics["bleu_longest"],
        f"shortest/eval_average_ROUGE_shortest{suffix}": eval_metrics["average_rouge_shortest"],
        f"short/eval_average_ROUGE_short{suffix}": eval_metrics["average_rouge_short"],
        f"average/eval_average_ROUGE_average{suffix}": eval_metrics["average_rouge_average"],
        f"long/eval_average_ROUGE_long{suffix}": eval_metrics["average_rouge_long"],
        f"longest/eval_average_ROUGE_longest{suffix}": eval_metrics["average_rouge_longest"],
        f"shortest/eval_F1-Bleu-Rouge_shortest{suffix}": eval_metrics["bleu_rouge_f1_shortest"],
        f"short/eval_F1-Bleu-Rouge_short{suffix}": eval_metrics["bleu_rouge_f1_short"],
        f"average/eval_F1-Bleu-Rouge_average{suffix}": eval_metrics["bleu_rouge_f1_average"],
        f"long/eval_F1-Bleu-Rouge_long{suffix}": eval_metrics["bleu_rouge_f1_long"],
        f"longest/eval_F1-Bleu-Rouge_longest{suffix}": eval_metrics["bleu_rouge_f1_longest"],
        f"evaluation/avg_loss{suffix}": avg_eval_loss,
        f"evaluation/avg_bleu{suffix}": avg_eval_bleu,
        f"evaluation/avg_rouge{suffix}": avg_eval_rouge,
        f"evaluation/avg_f1_bleu_rouge{suffix}": avg_eval_f1_bleu_rouge,
    }
    
    # Log metrics to the specified run
    if run:
        run.log(metrics)
    else:
        wandb.log(metrics)

def train_model(model, optimizer, accelerator, max_generate_length, train_dataloader, eval_dataloaders, num_train_epochs, tokenizer, run_name, patience, scheduler, Palga_trie, config, constrained_decoding):
    # Initialize WandB run for logging
    run = wandb.init(project="Transformers-PALGA", entity="srp-palga", config=config, name=run_name, reinit=True)

    lowest_loss = float("inf")
    early_stopping_counter = 0
    best_model_state_dict = None

    for epoch in range(num_train_epochs):
        avg_train_loss = train_step(model, train_dataloader, optimizer, accelerator, scheduler, tokenizer)
        
        all_eval_metrics, decoded_eval_preds, decoded_eval_labels, decoded_eval_input = validation_step(model, eval_dataloaders, tokenizer, max_generate_length, constrained_decoding, Palga_trie)
        
        # Log metrics to WandB run
        wandb_log_metrics(epoch, avg_train_loss, all_eval_metrics, run=run)

        # Calculate average loss for constrained decoding
        total_eval_loss = sum(all_eval_metrics[f"loss_{name}"] for name in ["shortest", "short", "average", "long", "longest"])
        average_eval_loss = total_eval_loss / 5

        # Update model for constrained decoding
        if average_eval_loss < lowest_loss:
            lowest_loss = average_eval_loss
            best_model_state_dict = model.state_dict()  # Save the state dict of the best model
            torch.save(model.state_dict(), f'PALGA-Transformers/models/trained_models/{run_name}.pth')  # Save the model weights
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience: # Stop training if loss does not improve for patience epochs
                print(f"Early stopping triggered. No improvement in {patience} epochs.") 
                break


    # Load the best model state dicts and log final metrics
    model.load_state_dict(best_model_state_dict)
    print("Running final validation")
    eval_metrics, decoded_eval_preds, decoded_eval_labels, decoded_eval_input = validation_step(
        model, eval_dataloaders, tokenizer, max_generate_length, constrained_decoding, Palga_trie)
    print_test_predictions(decoded_eval_preds, decoded_eval_labels, decoded_eval_input)
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(f'PALGA-Transformers/models/trained_models/{run_name}.pth')
    run.log_artifact(artifact)

    # Finish the W&B run
    run.finish()

