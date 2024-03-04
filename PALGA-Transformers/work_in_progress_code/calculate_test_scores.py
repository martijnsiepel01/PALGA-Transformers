from transformers import T5Tokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration, T5Config
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from evaluate import load
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_function(examples, tokenizer, max_length_sentence):
    inputs = [ex.lower() for ex in examples["Conclusie"]]
    targets = [ex.lower() for ex in examples["Codes"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
    return model_inputs

def prepare_test_dataset(test_data_location, tokenizer, max_length_sentence, report_type=None, min_conclusion_length=None, max_conclusion_length=None, required_codes=None, excluded_codes=None):
    # Load and initially filter the dataset
    dataset = load_dataset("csv", data_files=test_data_location, delimiter="\t")
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')

    # Filter by report type if specified
    if report_type is not None:
        dataset = dataset.filter(lambda example: example["Type"] == report_type)

    
    # Adjusted filter for required and excluded codes to handle special characters accurately
    def codes_filter(example):
        example = example['Codes']

        # Check for required codes, if specified
        required_check = True
        if required_codes:
            required_check = required_codes.lower() in example.lower()

        # Check for excluded codes, if specified
        excluded_check = True
        if excluded_codes:
            required_check = not excluded_codes.lower() in example.lower()

        # if (required_check and excluded_check):
        #     print(example)

        return required_check and excluded_check

    dataset = dataset.filter(codes_filter)
    
    # Filter by conclusion length if specified
    if min_conclusion_length is not None or max_conclusion_length is not None:
        def conclusion_length_filter(example):
            conclusion_length = len(example["Conclusie"])
            if min_conclusion_length is not None and conclusion_length < min_conclusion_length:
                return False
            if max_conclusion_length is not None and conclusion_length > max_conclusion_length:
                return False
            return True
        dataset = dataset.filter(conclusion_length_filter)


    conclusion_lengths = [len(conclusie) for conclusie in dataset['train']['Conclusie']]

    # Tokenize and preprocess the dataset
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
        batched=True
    )
    
    # Remove unnecessary columns and set format for PyTorch
    tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes", "Type"])
    # tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes"])
    tokenized_datasets.set_format("torch")
    
    # Select the subset of the dataset
    test_dataset = tokenized_datasets['train']
    return test_dataset, conclusion_lengths



def test_step(model, dataloader, tokenizer, max_generate_length):
    model.eval()
    total_test_loss = 0.0
    decoded_test_preds = []  
    decoded_test_inputs = []
    individual_bleu_scores = []  # Store individual BLEU scores
    individual_rouge_scores = []  # Store individual average ROUGE scores

    # Initialize metrics
    bleu_metric = load("sacrebleu")
    rouge_metric = load('rouge')

    for batch in tqdm(dataloader, desc="Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length,
                diversity_penalty=0.3,
                num_beams=6,
                num_beam_groups=2,
            )

        labels = batch["labels"]
        loss = model(**batch).loss
        total_test_loss += loss.item()

        # Filter out -100 tokens from generated tokens and labels
        filtered_generated_tokens = [token[token != -100] for token in generated_tokens]
        filtered_labels = [label[label != -100] for label in labels]

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(filtered_generated_tokens, skip_special_tokens=True)
        decoded_labels = [tokenizer.decode(filtered_labels[i], skip_special_tokens=True) for i in range(len(filtered_labels))]

        for pred, label in zip(decoded_preds, decoded_labels):
            # Compute individual BLEU score
            bleu_score = bleu_metric.compute(predictions=[pred], references=[[label]])['score']
            individual_bleu_scores.append(bleu_score/100)

            # Compute individual ROUGE scores
            rouge_result = rouge_metric.compute(predictions=[pred], references=[[label]])
            # Calculate average F1 score across ROUGE metrics directly using the scores
            avg_rouge_score = np.mean([rouge_result[key] for key in rouge_result])
            individual_rouge_scores.append(avg_rouge_score)

        decoded_test_preds.extend(decoded_preds)
        decoded_test_inputs.extend(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True))

    # Compute overall BLEU and ROUGE scores
    overall_bleu_score = np.mean(individual_bleu_scores)
    overall_avg_rouge_score = np.mean(individual_rouge_scores)

    test_metrics = {
        "average_bleu": overall_bleu_score,
        "average_rouge": overall_avg_rouge_score,
    }

    return test_metrics, decoded_test_preds, decoded_test_inputs, individual_bleu_scores, individual_rouge_scores



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

# tokenizer_path = "C:/Users/Martijn/Thesis/flan_tokenizer"
# tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
tokenizer_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_32128/combined_data_unigram_t5_custom_32128_1_lower_case.model"
tokenizer = T5Tokenizer(tokenizer_path)

test_dataset_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_test.tsv"
# test_dataset, conclusion_lengths = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type=None, min_conclusion_length=None, max_conclusion_length=None, required_codes=None, excluded_codes=None)
# test_dataset_C, conclusion_lenghts_C = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type="C", min_conclusion_length=None, max_conclusion_length=None, required_codes=None, excluded_codes=None)
# test_dataset_T, conclusion_lengths_T = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type="T", min_conclusion_length=None, max_conclusion_length=None, required_codes=None, excluded_codes=None)
# test_dataset_S, conclusion_lengths_S = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type="S", min_conclusion_length=None, max_conclusion_length=None, required_codes=None, excluded_codes=None)

# test_dataset_080, conclusion_lengths_080 = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type=None, min_conclusion_length=0, max_conclusion_length=80, required_codes=None, excluded_codes=None)
# test_dataset_81200, conclusion_lengths_81200 = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type=None, min_conclusion_length=81, max_conclusion_length=200, required_codes=None, excluded_codes=None)
# test_dataset_201400, conclusion_lengths_201400 = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type=None, min_conclusion_length=201, max_conclusion_length=400, required_codes=None, excluded_codes=None)
# test_dataset_401800, conclusion_lengths_401800 = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type=None, min_conclusion_length=401, max_conclusion_length=800, required_codes=None, excluded_codes=None)
# test_dataset_801x, conclusion_lengths_801x = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type=None, min_conclusion_length=801, max_conclusion_length=None, required_codes=None, excluded_codes=None)

codes = 'C-SEP'
test_dataset_noCSEP, conclusion_lengths_noCSEP = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type=None, min_conclusion_length=None, max_conclusion_length=None, required_codes=None, excluded_codes=codes)
test_dataset_CSEP, conclusion_lengths_CSEP = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type=None, min_conclusion_length=None, max_conclusion_length=None, required_codes=codes, excluded_codes=None)

test_datasets = {
    # "all": {
    #     "dataset": test_dataset,
    #     "lengths": conclusion_lengths
    # },
    # "C": {
    #     "dataset": test_dataset_C,
    #     "lengths": conclusion_lenghts_C
    # },
    # "T": {
    #     "dataset": test_dataset_T,
    #     "lengths": conclusion_lengths_T
    # },
    # "S": {
    #     "dataset": test_dataset_S,
    #     "lengths": conclusion_lengths_S
    # },
    # "0-80": {
    #     "dataset": test_dataset_080,
    #     "lengths": conclusion_lengths_080
    # },
    # "81-200": {
    #     "dataset": test_dataset_81200,
    #     "lengths": conclusion_lengths_81200
    # },
    # "201-400": {
    #     "dataset": test_dataset_201400,
    #     "lengths": conclusion_lengths_201400
    # },
    # "401-800": {
    #     "dataset": test_dataset_401800,
    #     "lengths": conclusion_lengths_401800
    # },
    # "801x": {
    #     "dataset": test_dataset_801x,
    #     "lengths": conclusion_lengths_801x
    # },
    "noCSEP": {
        "dataset": test_dataset_noCSEP,
        "lengths": conclusion_lengths_noCSEP
    },
    "CSEP": {
        "dataset": test_dataset_CSEP,
        "lengths": conclusion_lengths_CSEP
    }

}
# print(f"Len dataset: {len(test_dataset)}")
# print(f"Len C dataset: {len(test_dataset_C)}")
# print(f"Len T dataset: {len(test_dataset_T)}")
# print(f"Len S dataset: {len(test_dataset_S)}")
# print(f"Len 0-80 dataset: {len(test_dataset_080)}")
# print(f"Len 81-200 dataset: {len(test_dataset_81200)}")
# print(f"Len 201-400 dataset: {len(test_dataset_201400)}")
# print(f"Len 400+ dataset: {len(test_dataset_400x)}")
# print(f"Len noCSEP dataset: {len(test_dataset_noCSEP)}")
# print(f"Len CSEP dataset: {len(test_dataset_CSEP)}")

# print(f"Len noCSEP_400x dataset: {len(test_dataset_noCSEP_400x)}")
# print(f"Len CSEP_400x dataset: {len(test_dataset_CSEP_400x)}")


config = T5Config.from_pretrained('/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/flan-t5-small/config.json')
checkpoint = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs15_data_setall_commentcombined_data_unigram_lower_case_32128_1_combined_data.pth"
model = T5ForConditionalGeneration(config)
model.resize_token_embeddings(len(tokenizer))

model.to(device)

model.load_state_dict(torch.load(checkpoint))
data_collator = prepare_datacollator(tokenizer, model)
batch_size = 8
max_generate_length = 32

for dataset_name, dataset in test_datasets.items():
    print(f"Processing dataset: {dataset_name}")
    test_dataset = dataset['dataset']
    conclusion_lengths = dataset['lengths']

    # num_rows = len(test_dataset)

    # avg_conclusion_length = np.mean([len(data['Conclusie']) for data in test_dataset])

    # median_conclusion_length = np.median([len(data['Conclusie']) for data in test_dataset])
    # q1_conclusion_length = np.percentile([len(data['Conclusie']) for data in test_dataset], 25)

    # q3_conclusion_length = np.percentile([len(data['Conclusie']) for data in test_dataset], 75)
    # print(f"Number of rows: {num_rows}")
    # print(f"Average length of conclusion text: {avg_conclusion_length}")
    # print(f"Median length of conclusion text: {median_conclusion_length}")
    # print(f"Q1 value of conclusion length: {q1_conclusion_length}")
    # print(f"Q3 value of conclusion length: {q3_conclusion_length}")

    test_dataloader = prepare_dataloader(test_dataset, data_collator, batch_size)
    test_metrics, decoded_test_preds, decoded_test_inputs, individual_bleu_scores, individual_rouge_scores = test_step(model, test_dataloader, tokenizer, max_generate_length)
    rouge_scores = individual_bleu_scores
    bleu_scores = individual_rouge_scores

    print(f"Metrics for dataset {dataset_name}: {test_metrics}, len: {test_dataset}")
    plt.figure(figsize=(12, 6))

    # Plotting Rouge Scores
    plt.subplot(1, 2, 1)
    plt.scatter(conclusion_lengths, rouge_scores, color='r')
    plt.title('Input Length vs. Rouge Score')
    plt.xlabel('Input Length')
    plt.ylabel('Rouge Score')

    # Plotting Bleu Scores
    plt.subplot(1, 2, 2)
    plt.scatter(conclusion_lengths, bleu_scores, color='b')
    plt.title('Input Length vs. Bleu Score')
    plt.xlabel('Input Length')
    plt.ylabel('Bleu Score')

    plt.tight_layout()
    
    # Save the plot with a custom name
    custom_name = f"{dataset_name}_bleu_rouge_plot.png"  # Replace with your desired custom name
    plt.savefig(custom_name)
    
    plt.show()
    


# Metrics for dataset C: {'bleu': 0.47934815894602195, 'average rouge': 0.6890611648765586}
# Metrics for dataset T: {'bleu': 0.36288439973790226, 'average rouge': 0.6672269656131028}
# Metrics for dataset S: {'bleu': 0.12969550134666738, 'average rouge': 0.34463529242141766}
    
# Metrics for dataset 0-80: {'bleu': 0.5090214461427426, 'average rouge': 0.7530345081047309}
# Metrics for dataset 81-200: {'bleu': 0.41490041832998614, 'average rouge': 0.6805519146226103}
# Metrics for dataset 201-400: {'bleu': 0.31732368698044516, 'average rouge': 0.5924982698070032}
# Metrics for dataset 400+: {'bleu': 0.2720127060160415, 'average rouge': 0.5227936979302007}
    
# Metrics for dataset noCSEP: {'bleu': 0.3539822935661532, 'average rouge': 0.6698618329712577}
# Metrics for dataset CSEP: {'bleu': 0.4403903210574924, 'average rouge': 0.6476207867135182}
