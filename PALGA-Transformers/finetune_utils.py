import torch
from tqdm import tqdm
from datasets import load_dataset
import wandb
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AdamW, Adafactor
import evaluate
from accelerate import Accelerator
from datasets import concatenate_datasets
import random

experiment_id = str(random.randint(1, 1000000))

def preprocess_function(examples, tokenizer, max_length_sentence):
    inputs = [ex.lower() for ex in examples["Conclusie"]]
    targets = [ex.lower() for ex in examples["Codes"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length_sentence, truncation=True)
    # Calculate lengths based on the inputs for histogram-based splitting
    lengths = [len(input.split()) for input in inputs]  # Assuming length is based on word count before tokenization
    model_inputs["length"] = lengths
    return model_inputs

def prepare_datasets_tsv(data_set, tokenizer, max_length_sentence):
    data_files = {"train": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_train_with_codes.tsv", "test": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_test_with_codes.tsv", "validation": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_validation_with_codes.tsv"}
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

    if data_set == "all":
        # Define and load the second dataset
        data_set = "histo"
        data_files2 = {"train": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_train_with_codes.tsv", "test": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_test_with_codes.tsv", "validation": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_validation_with_codes.tsv"}
        dataset2 = load_dataset("csv", data_files=data_files2, delimiter="\t")
        
        # Define and load the third dataset
        data_set = "autopsies"
        data_files3 = {"train": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_train_with_codes.tsv", "test": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_test_with_codes.tsv", "validation": f"PALGA-Transformers/data/{data_set}/{data_set}_norm_validation_with_codes.tsv"}
        dataset3 = load_dataset("csv", data_files=data_files3, delimiter="\t")


        # Concatenate the datasets for each split separately
        train_datasets = concatenate_datasets([dataset["train"], dataset2["train"], dataset3["train"]])
        test_datasets = concatenate_datasets([dataset["test"], dataset2["test"], dataset3["test"]])
        validation_datasets = concatenate_datasets([dataset["validation"], dataset2["validation"], dataset3["validation"]])

        # Combine the splits back into a single dataset dictionary
        dataset = {"train": train_datasets, "test": test_datasets, "validation": validation_datasets}

    for split in dataset.keys():
        dataset[split] = dataset[split].filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
        dataset[split] = dataset[split].filter(lambda example: example["Conclusie"] is not None and example["Conclusie"] != '')
        dataset[split] = dataset[split].map(
            lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
            batched=True
        )
        dataset[split] = dataset[split].remove_columns(["Conclusie", "Codes"])

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

def prepare_test_dataset(tokenizer, max_length_sentence):
    test_data_location = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/gold_resolved_with_codes.tsv"
    dataset = load_dataset("csv", data_files={"test": test_data_location}, delimiter="\t")['test']
    
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    dataset = dataset.filter(lambda example: example["Conclusie"] is not None and example["Conclusie"] != '')
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
        batched=True
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes"])

    # Sort the test dataset by 'length' for histogram-based splitting
    test_dataset_sorted = tokenized_datasets.sort("length")

    # Split the test dataset into 5 parts
    total_length = len(test_dataset_sorted)
    split_sizes = [total_length // 5] * 4 + [total_length - (total_length // 5 * 4)]  # Handle remainder in the last split
    test_datasets = []
    for i in range(5):
        start_index = sum(split_sizes[:i])  # Calculate the starting index for each split
        end_index = start_index + split_sizes[i]
        split_dataset = test_dataset_sorted.select(range(start_index, end_index))
        split_dataset.set_format("torch")  # Set format to torch if necessary for each split
        test_datasets.append(split_dataset)

    return test_datasets

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


def slanted_triangular_learning_rate(step, total_steps, lr_start, lr_max, cut_frac, ratio):
    if step < total_steps * cut_frac:
        p = step / (total_steps * cut_frac)
    else:
        p = 1 - (step - total_steps * cut_frac) / (total_steps * (1 - cut_frac))
    lr = lr_start + (lr_max - lr_start) * p
    return lr * (1 / (1 + ratio * step))


def prepare_training_objects(learning_rate, model, train_dataloader, eval_dataloaders, test_dataloaders, lr_strategy, total_steps, optimizer_type='adamw'):
    # Select optimizer based on optimizer_type
    if optimizer_type.lower() == 'adamw':
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

    # Prepare each test_dataloader separately
    test_dataloaders = [accelerator.prepare(test_dl) for test_dl in test_dataloaders]

    scheduler = None
    if lr_strategy == "ST-LR":
        # Adjusted parameters for ST-LR to keep the learning rate close to 1e-4
        lr_start = 5e-5  # Start slightly lower than the target learning rate
        lr_max = 2e-4    # Peak learning rate, slightly above the target
        cut_frac = 0.1   # Proportion of training steps to increase the learning rate
        ratio = 32       # Controls the steepness of the learning rate decrease

        lr_lambda = lambda step: slanted_triangular_learning_rate(step, total_steps, lr_start, lr_max, cut_frac, ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, accelerator, model, train_dataloader, eval_dataloaders, test_dataloaders, scheduler


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

        # Print 'c-sep' counts for verification
        # print(f"Batch {batch_idx}: Input 'c-sep' counts: {input_c_sep_counts}")
        # print(f"Batch {batch_idx}: Output 'c-sep' counts: {output_c_sep_counts}")

        # Calculate reweight factors for each sequence in the batch
        reweight_factors = [calculate_reweight_factor(inp, out) for inp, out in zip(input_c_sep_counts, output_c_sep_counts)]
        # reweight_factors = [1 for _ in range(len(input_c_sep_counts))]  # Set to 1 for each sequence


        # Print reweight factors for verification
        # print(f"Batch {batch_idx}: Reweight factors: {reweight_factors}")

        # Adjust the loss for each sequence and sum them up
        weighted_losses = loss * torch.tensor(reweight_factors).to(loss.device)  # Ensure reweight factors are on the same device as loss
        weighted_loss = weighted_losses.mean()  # Calculate mean to combine individual losses

        # Print weighted loss for verification
        # print(f"Batch {batch_idx}: Weighted loss: {weighted_loss.item()}")

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
        token_list = tokenizer.decode(sequence, skip_special_tokens=True).split()
        counts.append(sum(1 for token in token_list if 'c-sep' in token))
    return counts

def calculate_reweight_factor(input_count, output_count):
    # No change needed, works on individual counts
    diff = abs(input_count - output_count)
    reweight_factor = (1 + diff)  # Adjust formula as needed
    return reweight_factor



def validation_step(model, eval_dataloaders, tokenizer, max_generate_length):
    # Predefined names for each dataloader
    dataloader_names = [
        "shortest", "short", "average", "long", "longest"
    ]

    # Initialize evaluation metrics storage
    all_eval_metrics = {}

    # Ensure there are exactly 5 dataloaders
    if len(eval_dataloaders) != len(dataloader_names):
        raise ValueError("The number of evaluation dataloaders must be exactly 5 to match the predefined names.")

    # Iterate through each eval dataloader
    for dataloader, name in zip(eval_dataloaders, dataloader_names):
        metric = evaluate.load("sacrebleu", experiment_id=experiment_id)
        rouge = evaluate.load('rouge', experiment_id=experiment_id)

        model.eval()
        total_eval_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc=f"Evaluation {name}"):
            with torch.no_grad():
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=max_generate_length,
                    diversity_penalty=0.3,
                    num_beams=6,
                    num_beam_groups=2,
                    # no_repeat_ngram_size = 4
                )

            labels = batch["labels"]
            loss = model(**batch).loss
            total_eval_loss += loss.item()

            filtered_preds = [[token_id for token_id in token if token_id != tokenizer.pad_token_id] for token in generated_tokens]
            filtered_labels = [[token_id for token_id in token if token_id != -100] for token in labels]

            decoded_generated = tokenizer.batch_decode(filtered_preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

            # Extend predictions and labels for evaluation
            all_preds.extend(decoded_generated)
            # all_labels.extend(decoded_labels)
            for label in decoded_labels:
                all_labels.append([label])  # Encapsulate each label in a list

        # Compute Metrics for current dataloader
        bleu_score = metric.compute(predictions=all_preds, references=all_labels)['score']
        rouge_score = rouge.compute(predictions=all_preds, references=all_labels)

        ROUGE_1 = rouge_score['rouge1'] 
        ROUGE_2 = rouge_score['rouge2'] 
        ROUGE_L = rouge_score['rougeL'] 
        ROUGE_Lsum = rouge_score['rougeLsum'] 
        bleu_score = bleu_score / 100

        epsilon = 1e-7
        average_rouge_test = (ROUGE_1 + ROUGE_2 + ROUGE_L + ROUGE_Lsum)/4

        bleu_rouge_f1 = (2 * bleu_score * average_rouge_test) / (bleu_score + average_rouge_test + epsilon)
            
        # Append metrics with dynamic names based on dataloader names
        all_eval_metrics[f"loss_{name}"] = total_eval_loss / len(dataloader)
        all_eval_metrics[f"bleu_{name}"] = bleu_score
        all_eval_metrics[f"average_rouge_{name}"] = average_rouge_test
        all_eval_metrics[f"bleu_rouge_f1_{name}"] = bleu_rouge_f1

    return all_eval_metrics

def test_step(model, test_dataloaders, tokenizer, max_generate_length):
    dataloader_names = ["shortest", "short", "average", "long", "longest"]
    all_test_metrics = {}
    
    decoded_test_inputs = []
    decoded_test_preds = []
    decoded_test_labels = []

    if len(test_dataloaders) != 5:
        raise ValueError("There must be exactly 5 test dataloaders.")

    for dataloader, name in zip(test_dataloaders, dataloader_names):
        metric = evaluate.load("sacrebleu", experiment_id=experiment_id)
        rouge = evaluate.load('rouge', experiment_id=experiment_id)

        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc=f"Testing {name}"):
            if 'length' in batch:
                del batch['length']

            with torch.no_grad():
                outputs =  model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=max_generate_length,
                    diversity_penalty=0.3,
                    num_beams=6,
                    num_beam_groups=2,
                    # no_repeat_ngram_size = 4
                )
                loss = model(**batch).loss
                total_loss += loss.item()

            # Filtering predictions and labels before decoding
            filtered_preds = [[token_id for token_id in token if token_id != tokenizer.pad_token_id] for token in outputs]
            filtered_labels = [[token_id for token_id in token if token_id != -100] for token in batch["labels"]]
            
            # Decoding filtered predictions and labels
            decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in filtered_preds]
            decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in filtered_labels]

            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)

            # Decoding input sequences for detailed analysis or printing later
            decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            decoded_test_inputs.extend(decoded_inputs)

        # Compute metrics for each dataloader
        bleu = metric.compute(predictions=all_preds, references=[[label] for label in all_labels])
        rouge_scores = rouge.compute(predictions=all_preds, references=all_labels)

        ROUGE_1 = rouge_scores['rouge1'] 
        ROUGE_2 = rouge_scores['rouge2'] 
        ROUGE_L = rouge_scores['rougeL'] 
        ROUGE_Lsum = rouge_scores['rougeLsum'] 
        bleu_score = bleu["score"] / 100

        epsilon = 1e-7
        average_rouge_test = (ROUGE_1 + ROUGE_2 + ROUGE_L + ROUGE_Lsum)/4

        bleu_rouge_f1 = (2 * bleu_score * average_rouge_test) / (bleu_score + average_rouge_test + epsilon)

        # Storing metrics for each test dataloader
        all_test_metrics[f"loss_{name}"] = total_loss / len(dataloader)
        all_test_metrics[f"bleu_{name}"] = bleu_score
        all_test_metrics[f"average_rouge_{name}"] = average_rouge_test
        all_test_metrics[f"bleu_rouge_f1_{name}"] = bleu_rouge_f1


        decoded_test_preds.extend(all_preds)
        decoded_test_labels.extend(all_labels)

    return all_test_metrics, decoded_test_preds, decoded_test_labels, decoded_test_inputs


def print_test_predictions(decoded_test_preds, decoded_test_labels, decoded_test_input):
    print("Predictions on Test Data in the Last Epoch:")
    # Load the thesaurus
    thesaurus_location = '/home/msiepel/snomed_20230426.txt'
    thesaurus = pd.read_csv(thesaurus_location, sep='|', encoding='latin-1')

    # Function to get word from code
    def get_word_from_code(code):
        if code == '[C-SEP]' or code == '[c-sep]':
            return '[C-SEP]'
        else:
            word = thesaurus[(thesaurus['DEPALCE'].str.lower() == code.lower()) & (thesaurus['DESTACE'] == 'V')]['DETEROM'].values
            return word[0] if len(word) > 0 else 'Unknown'

    for input_seq, label, pred in zip(decoded_test_input, decoded_test_labels, decoded_test_preds):
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

def wandb_log_metrics(epoch, avg_train_loss, eval_metrics, test_metrics):
    # Compute average evaluation metrics
    avg_eval_loss = (eval_metrics["loss_shortest"] + eval_metrics["loss_short"] + eval_metrics["loss_average"] + eval_metrics["loss_long"] + eval_metrics["loss_longest"]) / 5
    avg_eval_bleu = (eval_metrics["bleu_shortest"] + eval_metrics["bleu_short"] + eval_metrics["bleu_average"] + eval_metrics["bleu_long"] + eval_metrics["bleu_longest"]) / 5
    avg_eval_rouge = (eval_metrics["average_rouge_shortest"] + eval_metrics["average_rouge_short"] + eval_metrics["average_rouge_average"] + eval_metrics["average_rouge_long"] + eval_metrics["average_rouge_longest"]) / 5
    avg_eval_f1_bleu_rouge = (eval_metrics["bleu_rouge_f1_shortest"] + eval_metrics["bleu_rouge_f1_short"] + eval_metrics["bleu_rouge_f1_average"] + eval_metrics["bleu_rouge_f1_long"] + eval_metrics["bleu_rouge_f1_longest"]) / 5
    
    # Compute average test metrics
    avg_test_loss = (test_metrics["loss_shortest"] + test_metrics["loss_short"] + test_metrics["loss_average"] + test_metrics["loss_long"] + test_metrics["loss_longest"]) / 5
    avg_test_bleu = (test_metrics["bleu_shortest"] + test_metrics["bleu_short"] + test_metrics["bleu_average"] + test_metrics["bleu_long"] + test_metrics["bleu_longest"]) / 5
    avg_test_rouge = (test_metrics["average_rouge_shortest"] + test_metrics["average_rouge_short"] + test_metrics["average_rouge_average"] + test_metrics["average_rouge_long"] + test_metrics["average_rouge_longest"]) / 5
    avg_test_f1_bleu_rouge = (test_metrics["bleu_rouge_f1_shortest"] + test_metrics["bleu_rouge_f1_short"] + test_metrics["bleu_rouge_f1_average"] + test_metrics["bleu_rouge_f1_long"] + test_metrics["bleu_rouge_f1_longest"]) / 5

    wandb.log({
                "epoch/epoch": epoch,
                "training/train_loss": avg_train_loss,
                "shortest/eval_loss_shortest": eval_metrics["loss_shortest"],
                "short/eval_loss_short": eval_metrics["loss_short"],
                "average/eval_loss_average": eval_metrics["loss_average"],
                "long/eval_loss_long": eval_metrics["loss_long"],
                "longest/eval_loss_longest": eval_metrics["loss_longest"],
                "shortest/test_loss_shortest": test_metrics["loss_shortest"],
                "short/test_loss_short": test_metrics["loss_short"],
                "average/test_loss_average": test_metrics["loss_average"],
                "long/test_loss_long": test_metrics["loss_long"],
                "longest/test_loss_longest": test_metrics["loss_longest"],
                "shortest/eval_BLEU_shortest": eval_metrics["bleu_shortest"],
                "short/eval_BLEU_short": eval_metrics["bleu_short"],
                "average/eval_BLEU_average": eval_metrics["bleu_average"],
                "long/eval_BLEU_long": eval_metrics["bleu_long"],
                "longest/eval_BLEU_longest": eval_metrics["bleu_longest"],
                "shortest/test_BLEU_shortest": test_metrics["bleu_shortest"],
                "short/test_BLEU_short": test_metrics["bleu_short"],
                "average/test_BLEU_average": test_metrics["bleu_average"],
                "long/test_BLEU_long": test_metrics["bleu_long"],
                "longest/test_BLEU_longest": test_metrics["bleu_longest"],
                "shortest/eval_average_ROUGE_shortest": eval_metrics["average_rouge_shortest"],
                "short/eval_average_ROUGE_short": eval_metrics["average_rouge_short"],
                "average/eval_average_ROUGE_average": eval_metrics["average_rouge_average"],
                "long/eval_average_ROUGE_long": eval_metrics["average_rouge_long"],
                "longest/eval_average_ROUGE_longest": eval_metrics["average_rouge_longest"],
                "shortest/test_average_ROUGE_shortest": test_metrics["average_rouge_shortest"],
                "short/test_average_ROUGE_short": test_metrics["average_rouge_short"],
                "average/test_average_ROUGE_average": test_metrics["average_rouge_average"],
                "long/test_average_ROUGE_long": test_metrics["average_rouge_long"],
                "longest/test_average_ROUGE_longest": test_metrics["average_rouge_longest"],
                "shortest/eval_F1-Bleu-Rouge_shortest": eval_metrics["bleu_rouge_f1_shortest"],
                "short/eval_F1-Bleu-Rouge_short": eval_metrics["bleu_rouge_f1_short"],
                "average/eval_F1-Bleu-Rouge_average": eval_metrics["bleu_rouge_f1_average"],
                "long/eval_F1-Bleu-Rouge_long": eval_metrics["bleu_rouge_f1_long"],
                "longest/eval_F1-Bleu-Rouge_longest": eval_metrics["bleu_rouge_f1_longest"],
                "shortest/test_F1-Bleu-Rouge_shortest": test_metrics["bleu_rouge_f1_shortest"],
                "short/test_F1-Bleu-Rouge_short": test_metrics["bleu_rouge_f1_short"],
                "average/test_F1-Bleu-Rouge_average": test_metrics["bleu_rouge_f1_average"],
                "long/test_F1-Bleu-Rouge_long": test_metrics["bleu_rouge_f1_long"],
                "longest/test_F1-Bleu-Rouge_longest": test_metrics["bleu_rouge_f1_longest"],
                "evaluation/avg_loss": avg_eval_loss,
                "evaluation/avg_bleu": avg_eval_bleu,
                "evaluation/avg_rouge": avg_eval_rouge,
                "evaluation/avg_f1_bleu_rouge": avg_eval_f1_bleu_rouge,
                "test/avg_loss": avg_test_loss,
                "test/avg_bleu": avg_test_bleu,
                "test/avg_rouge": avg_test_rouge,
                "test/avg_f1_bleu_rouge": avg_test_f1_bleu_rouge,
            })

def train_model(model, optimizer, accelerator, max_generate_length, train_dataloader, eval_dataloaders, test_dataloaders, num_train_epochs, tokenizer, run_name, patience, scheduler):
    lowest_loss = float("inf")
    early_stopping_counter = 0
    best_model_state_dict = None

    for epoch in range(num_train_epochs):
        avg_train_loss = train_step(model, train_dataloader, optimizer, accelerator, scheduler, tokenizer)
        all_eval_metrics = validation_step(model, eval_dataloaders, tokenizer, max_generate_length)
        all_test_metrics, decoded_test_preds, decoded_test_labels, decoded_test_input = test_step(model, test_dataloaders, tokenizer, max_generate_length)

        # Log metrics to WandB
        wandb_log_metrics(epoch, avg_train_loss, all_eval_metrics, all_test_metrics)
        total_eval_loss = all_test_metrics["loss_shortest"] + all_test_metrics["loss_short"] + all_test_metrics["loss_average"] + all_test_metrics["loss_long"] + all_test_metrics["loss_longest"]
        average_eval_loss = total_eval_loss / 5
        if average_eval_loss < lowest_loss:  # Update loss accordingly
            lowest_loss = average_eval_loss
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
    test_metrics, decoded_test_preds, decoded_test_labels, decoded_test_input = test_step(model, test_dataloaders, tokenizer, max_generate_length)
    print_test_predictions(decoded_test_preds, decoded_test_labels, decoded_test_input)

    # Save the best model weights as a W&B artifact
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(f'PALGA-Transformers/models/trained_models/{run_name}.pth')
    wandb.log_artifact(artifact)