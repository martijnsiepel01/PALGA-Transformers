from transformers import T5Tokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration, T5Config, MT5Tokenizer, MT5ForConditionalGeneration, MT5Config, AutoTokenizer
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from evaluate import load
import numpy as np
import random


import sacrebleu

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_function(examples, tokenizer, max_length_sentence):
    inputs = [ex.lower() for ex in examples["Conclusie"]]
    targets = [ex.lower() for ex in examples["Codes"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
    return model_inputs



def prepare_test_dataset(test_data_location, tokenizer, max_length_sentence):
    # Load and initially filter the dataset
    dataset = load_dataset("csv", data_files=test_data_location, delimiter="\t")
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    dataset = dataset.filter(lambda example: example["Conclusie"] is not None and example["Conclusie"] != '')

    # data_files = {
    #         "train": f"all_norm_train_with_codes.tsv",
    #         "test": f"all_norm_test_with_codes.tsv",
    #         "validation": f"all_norm_validation_with_codes.tsv"
    #     }
    # dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

    # data_files2 = {
    #         "train": f"histo_norm_train_with_codes.tsv",
    #         "test": f"histo_norm_test_with_codes.tsv",
    #         "validation": f"histo_norm_validation_with_codes.tsv"
    #     }
    # dataset2 = load_dataset("csv", data_files=data_files2, delimiter="\t")
    # dataset2 = dataset2.map(lambda example: {'Type': 'T', **example})

    # data_set = "autopsies"
    # data_files3 = {
    #     "train": f"autopsies_norm_train_with_codes.tsv",
    #     "test": f"autopsies_norm_test_with_codes.tsv",
    #     "validation": f"autopsies_norm_validation_with_codes.tsv"
    # }
    # dataset3 = load_dataset("csv", data_files=data_files3, delimiter="\t")
    # dataset3 = dataset3.map(lambda example: {'Type': 'S', **example})

    # # Concatenate the datasets
    # dataset = {
    #     "train": concatenate_datasets([dataset["train"], dataset2["train"], dataset3["train"]]),
    #     "test": concatenate_datasets([dataset["test"], dataset2["test"], dataset3["test"]]),
    #     "validation": concatenate_datasets([dataset["validation"], dataset2["validation"], dataset3["validation"]])
    # }

    # # Function to sample n_samples for each type from the dataset
    # def sample_by_type(dataset, n_samples):
    #     sampled = []
    #     for type_value in ['C', 'T', 'S']:
    #         if type_value == 'T':
    #             n_samples = n_samples * 2
    #         filtered = dataset.filter(lambda example: example['Type'] == type_value)
    #         sampled.append(filtered.shuffle().select(range(min(n_samples, len(filtered)))))
    #     return concatenate_datasets(sampled)

    # # Sampling datasets to contain specified number of rows for each Type
    # dataset["train"] = sample_by_type(dataset["train"], 40000)
    # dataset["test"] = sample_by_type(dataset["test"], 5000)
    # dataset["validation"] = sample_by_type(dataset["validation"], 5000)

    # dataset = dataset["test"]
    dataset = dataset["train"]
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    dataset = dataset.filter(lambda example: example["Conclusie"] is not None and example["Conclusie"] != '')
    
    # Tokenize and preprocess the dataset
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
        batched=True
    )
    
    # Remove unnecessary columns and set format for PyTorch
    tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes", "Type"])
    # tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes"])
    tokenized_datasets.set_format("torch")
    
    test_dataset = tokenized_datasets
    return test_dataset.select(range(int(len(test_dataset) * 0.1)))
    # return test_dataset




def prepare_datacollator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return data_collator

def prepare_dataloader(dataset, data_collator, batch_size):
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )
    return dataloader


thesaurus_location = '/home/gburger01/snomed_20230426.txt'
thesaurus = pd.read_csv(thesaurus_location, sep='|', encoding='latin-1')

# Function to get word from code
def get_word_from_code(code):
    if code == '[C-SEP]' or code == '[c-sep]':
        return '[C-SEP]'
    else:
        word = thesaurus[(thesaurus['DEPALCE'].str.lower() == code.lower()) & (thesaurus['DESTACE'] == 'V')]['DETEROM'].values
        return word[0] if len(word) > 0 else 'Unknown'
        

def generate_predictions_and_update_csv(model, dataloader, tokenizer, original_csv_path, updated_csv_path):
    df = pd.read_csv(original_csv_path, delimiter='\t')

    # Ensure there's a place to store the predictions
    df['Predictions'] = ""

    model.eval()

    for index, batch in enumerate(tqdm(dataloader, desc="Generating Predictions")):
        # batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length,
                diversity_penalty=0.3,
                num_beams=6,
                num_beam_groups=2,
            )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Assuming your dataloader returns batches in the same order as your DataFrame
        start_index = index * dataloader.batch_size
        end_index = start_index + len(decoded_preds)
        # decoded_preds = [' '.join([get_word_from_code(code) for code in prediction.split()]) for prediction in decoded_preds]
        df.loc[start_index:end_index - 1, 'Predictions'] = decoded_preds

    # Save the updated DataFrame to a new CSV
    df.to_csv(updated_csv_path, sep='\t', index=False)




tokenizer_mt5_default = AutoTokenizer.from_pretrained("google/mt5-small")
tokenizer_pathologyt5_dutch = AutoTokenizer.from_pretrained("/home/gburger01/PALGA-Transformers/PALGA-Transformers/T5_small_32128_with_codes_csep_normal_token")

test_dataset_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/combined_test_with_codes.tsv"
test_dataset_mt5_default = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_mt5_default, max_length_sentence=2048)
test_dataset_pathologyt5_dutch = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pathologyt5_dutch, max_length_sentence=2048)


gold_test_dataset_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/combined_gold_standard_with_codes.tsv"
gold_standard_dataset_mt5_default = prepare_test_dataset(test_data_location=gold_test_dataset_path, tokenizer=tokenizer_mt5_default, max_length_sentence=2048)
gold_standard_dataset_pathologyt5_dutch = prepare_test_dataset(test_data_location=gold_test_dataset_path, tokenizer=tokenizer_pathologyt5_dutch, max_length_sentence=2048)


# checkpoint = "num_train_epochs25_data_setall_commentmT5_small_pretrained_v1_all_custom_loss.pth"
# checkpoint = "num_train_epochs25_data_setall_commentmT5_small_pretrained_v1_all_custom_loss_50k.pth"
checkpoint_mt5_default = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs15_data_setall_commentmT5_small_all_custom_loss_default_from_checkpoint_2.pth"
checkpoint_pathologyt5_dutch = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs25_data_setall_commentmT5_small_pretrained_v1_all_80k_histo.pth"


config_mt5 = MT5Config(decoder_start_token_id=tokenizer_mt5_default.pad_token_id) 
mt5_default = MT5ForConditionalGeneration(config_mt5)
mt5_default.resize_token_embeddings(len(tokenizer_mt5_default))
# mt5_default.to(device)

mt5_default.load_state_dict(torch.load(checkpoint_mt5_default))
data_collator_mt5_default = prepare_datacollator(tokenizer_mt5_default, mt5_default)


config_pathologyt5_dutch = T5Config(decoder_start_token_id=tokenizer_pathologyt5_dutch.pad_token_id) 
pathologyt5_dutch = T5ForConditionalGeneration(config_pathologyt5_dutch)
pathologyt5_dutch.resize_token_embeddings(len(tokenizer_pathologyt5_dutch))
# pathologyt5_dutch.to(device)

pathologyt5_dutch.load_state_dict(torch.load(checkpoint_pathologyt5_dutch))
data_collator_pathologyt5_dutch = prepare_datacollator(tokenizer_pathologyt5_dutch, pathologyt5_dutch)


batch_size = 8
max_generate_length = 128


# mt5 test dataset
test_dataloader_mt5_default = prepare_dataloader(test_dataset_mt5_default, data_collator_mt5_default, batch_size)
updated_csv_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/predictions/mt5_default_test_dataset_predictions.tsv"
generate_predictions_and_update_csv(mt5_default, test_dataloader_mt5_default, tokenizer_mt5_default, test_dataset_path, updated_csv_path)

# mt5 gold standard dataset
gold_standard_dataloader_mt5_default = prepare_dataloader(gold_standard_dataset_mt5_default, data_collator_mt5_default, batch_size)
updated_csv_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/predictions/mt5_default_gold_standard_dataset_predictions.tsv"
generate_predictions_and_update_csv(mt5_default, gold_standard_dataloader_mt5_default, tokenizer_mt5_default, gold_test_dataset_path, updated_csv_path)


# pathologyt5_dutch test dataset
test_dataloader_pathologyt5_dutch = prepare_dataloader(test_dataset_pathologyt5_dutch, data_collator_pathologyt5_dutch, batch_size)
updated_csv_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/predictions/pathologyt5_dutch_default_test_dataset_predictions.tsv"
generate_predictions_and_update_csv(pathologyt5_dutch, test_dataloader_pathologyt5_dutch, tokenizer_pathologyt5_dutch, test_dataset_path, updated_csv_path)

# pathologyt5_dutch gold standard dataset
gold_standard_dataloader_pathologyt5_dutch = prepare_dataloader(gold_standard_dataset_pathologyt5_dutch, data_collator_pathologyt5_dutch, batch_size)
updated_csv_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/predictions/pathologyt5_dutch_default_gold_standard_dataset_predictions.tsv"
generate_predictions_and_update_csv(pathologyt5_dutch, gold_standard_dataloader_pathologyt5_dutch, tokenizer_pathologyt5_dutch, gold_test_dataset_path, updated_csv_path)

