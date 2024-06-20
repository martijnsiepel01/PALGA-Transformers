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

from annotate import split_and_compile


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
    dataset = dataset.filter(lambda example: example["Conclusie"] is not None and example["Conclusie"] != '')

    # dataset = dataset["test"]
    dataset = dataset["train"]
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    dataset = dataset.filter(lambda example: example["Conclusie"] is not None and example["Conclusie"] != '')
    print(len(dataset))

    # Filter by report type if specified
    if report_type is not None:
        dataset = dataset.filter(lambda example: example["Type"] == report_type)

    if report_type == "S":
        dataset = dataset.map(lambda example: {'Conclusie': split_and_compile(example['Conclusie'])[0]})

    
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
    # test_dataset = tokenized_datasets['train']
    test_dataset = tokenized_datasets
    # return test_dataset.select(range(int(len(test_dataset) * 0.01)))
    return test_dataset

def prepare_datacollator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return data_collator

def prepare_dataloader(dataset, data_collator, batch_size):
    print("start dataloader")
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )
    return dataloader


tokenizer_path = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/T5_small_32128_pretrain_with_codes"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def custom_decode(tokenizer, token_ids):
    tokens = [tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]
    decoded_string = ' '.join(tokens).replace(' </s>', '').replace('  c  s e p ', "[C-SEP]").strip()
    return decoded_string


def test_step2(model, test_dataloader, tokenizer, max_generate_length):
    print("start test step 2")
    dataloader_names = ["shortest", "short", "average", "long", "longest"]
    all_test_metrics = {}


    if len(test_dataloader) == 0:
        all_test_metrics[f"loss"] = 0
        all_test_metrics[f"bleu"] = 0
        all_test_metrics[f"average_rouge"] = 0
        all_test_metrics[f"bleu_rouge_f1"] = 0
        return all_test_metrics, None, None, None
    
    decoded_test_inputs = []
    decoded_test_preds = []
    decoded_test_labels = []

    metric = evaluate.load("sacrebleu")
    rouge = evaluate.load('rouge')
    meteor = load("meteor")

    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(test_dataloader, desc="Testing"):
        # batch = {k: v.to(device) for k, v in batch.items()}
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
                no_repeat_ngram_size=3,
            )
            loss = model(**batch).loss
            total_loss += loss.item()

        # Filtering predictions and labels before decoding
        filtered_preds = [[token_id for token_id in token if token_id != tokenizer.pad_token_id] for token in outputs]
        filtered_labels = [[token_id for token_id in token if token_id != -100] for token in batch["labels"]]
        
        # Decoding filtered predictions and labels
        decoded_preds = [custom_decode(tokenizer, pred) for pred in filtered_preds]
        decoded_labels = [custom_decode(tokenizer, label) for label in filtered_labels]

        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)

        # Decoding input sequences for detailed analysis or printing later
        decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        decoded_test_inputs.extend(decoded_inputs)

    # Compute metrics for each dataloader

    bleu = metric.compute(predictions=all_preds, references=[[label] for label in all_labels])
    # meteor_score = meteor.compute(predictions=all_preds, references=[[label] for label in all_labels])

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
    all_test_metrics[f"loss"] = total_loss / len(test_dataloader)
    all_test_metrics[f"bleu"] = bleu_score
    all_test_metrics[f"average_rouge"] = average_rouge_test
    all_test_metrics[f"bleu_rouge_f1"] = bleu_rouge_f1
    # all_test_metrics[f"meteor"] = meteor_score['meteor']

    decoded_test_preds.extend(all_preds)
    decoded_test_labels.extend(all_labels)

    return all_test_metrics, decoded_test_preds, decoded_test_labels, decoded_test_inputs


thesaurus_location = '/home/msiepel/snomed_20230426.txt'
thesaurus = pd.read_csv(thesaurus_location, sep='|', encoding='latin-1')

# Function to get word from code
def get_word_from_code(code):
    if code == '[C-SEP]' or code == '[c-sep]':
        return '[C-SEP]'
    else:
        word = thesaurus[(thesaurus['DEPALCE'].str.lower() == code.lower()) & (thesaurus['DESTACE'] == 'V')]['DETEROM'].values
        return word[0] if len(word) > 0 else 'Unknown'
        
def write_test_predictions(dataset_name, all_test_metrics, decoded_test_inputs, decoded_test_preds, decoded_test_labels):
    print(f"Metrics for dataset {dataset_name}:")
    print(f"Model {dataset_name}: {all_test_metrics}")
    
    if all_test_metrics['bleu'] == 0:
        return

    random_indices = random.sample(range(len(decoded_test_inputs)), min(len(decoded_test_inputs), 20))

    for index in random_indices:
        input_seq = decoded_test_inputs[index]
        label = decoded_test_labels[index]
        pred = decoded_test_preds[index]

        label_words = ' '.join([get_word_from_code(code) for code in label.split()])
        pred_words = ' '.join([get_word_from_code(code) for code in pred.split()])
 
        print(f"Input Sequence:                               {input_seq}")
        print(f"Label:                                        {label_words}")
        print(f"Prediction model:                             {pred_words}")
        print('-'*100 + '\n')


def create_outcome_matrix(outcomes):
    # Initialize matrix with dynamically constructed keys
    matrix = {
        'T': {'0-80': outcomes['0-80_T'], '81-200': outcomes['81-200_T'], '201-400': outcomes['201-400_T'], '400+': outcomes['400+_T'], 'Average': None},
        'C': {'0-80': outcomes['0-80_C'], '81-200': outcomes['81-200_C'], '201-400': outcomes['201-400_C'], '400+': outcomes['400+_C'], 'Average': None},
        'S': {'0-80': outcomes['0-80_S'], '81-200': outcomes['81-200_S'], '201-400': outcomes['201-400_S'], '400+': outcomes['400+_S'], 'Average': None},
        'Average': {'0-80': None, '81-200': None, '201-400': None, '400+': None, 'Average': None}
    }

    # Calculate subgroup averages
    for subgroup in 'TCS':
        total_bleu = sum(outcomes[f"{length}_{subgroup}"][0] * outcomes[f"{length}_{subgroup}"][1]['bleu'] for length in ['0-80', '81-200', '201-400', '400+'])
        total_count = sum(outcomes[f"{length}_{subgroup}"][0] for length in ['0-80', '81-200', '201-400', '400+'])
        matrix[subgroup]['Average'] = (total_count, {'bleu': total_bleu / total_count})

    # Calculate column averages and overall average
    for length in ['0-80', '81-200', '201-400', '400+']:
        total_bleu = sum(matrix[subgroup][length][0] * matrix[subgroup][length][1]['bleu'] for subgroup in 'TCS')
        total_count = sum(matrix[subgroup][length][0] for subgroup in 'TCS')
        matrix['Average'][length] = (total_count, {'bleu': total_bleu / total_count})

    # Calculate overall average
    total_bleu = sum(matrix['Average'][length][0] * matrix['Average'][length][1]['bleu'] for length in ['0-80', '81-200', '201-400', '400+'])
    total_count = sum(matrix['Average'][length][0] for length in ['0-80', '81-200', '201-400', '400+'])
    matrix['Average']['Average'] = (total_count, {'bleu': total_bleu / total_count})

    data_len = len(dataset)  # Number of items in dataset
    print(f"Data length for {dataset_name}: {data_len}")
    print(f"Matrix {matrix}")



    # Print the matrix
    for subgroup in 'TCS':
        print(f"{subgroup}: " + ', '.join(f"{length}: {matrix[subgroup][length][1]['bleu']:.2f} (N={matrix[subgroup][length][0]})" for length in ['0-80', '81-200', '201-400', '400+']) + f", Average: {matrix[subgroup]['Average'][1]['bleu']:.2f} (N={matrix[subgroup]['Average'][0]})")
    print('Average: ' + ', '.join(f"{length}: {matrix['Average'][length][1]['bleu']:.2f} (N={matrix['Average'][length][0]})" for length in ['0-80', '81-200', '201-400', '400+']) + f", Overall Average: {matrix['Average']['Average'][1]['bleu']:.2f} (N={matrix['Average']['Average'][0]})")
    print('------------------')


# test_dataset_path = "validation_combined_with_codes.tsv"
test_dataset_path = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/combined_gold_standard_with_codes.tsv"
# test_dataset_path = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/combined_test_with_codes.tsv"
test_dataset = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=None, report_type=None, min_conclusion_length=None, max_conclusion_length=None, required_codes=None, excluded_codes=None)
test_dataset_080_C = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='C', min_conclusion_length=0, max_conclusion_length=80, required_codes=None, excluded_codes=None)
test_dataset_81200_C = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='C', min_conclusion_length=81, max_conclusion_length=200, required_codes=None, excluded_codes=None)
test_dataset_201400_C = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='C', min_conclusion_length=201, max_conclusion_length=400, required_codes=None, excluded_codes=None)
test_dataset_400x_C = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='C', min_conclusion_length=401, max_conclusion_length=10000, required_codes=None, excluded_codes=None)

test_dataset_080_T = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='T', min_conclusion_length=0, max_conclusion_length=80, required_codes=None, excluded_codes=None)
test_dataset_81200_T = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='T', min_conclusion_length=81, max_conclusion_length=200, required_codes=None, excluded_codes=None)
test_dataset_201400_T = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='T', min_conclusion_length=201, max_conclusion_length=400, required_codes=None, excluded_codes=None)
test_dataset_400x_T = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='T', min_conclusion_length=401, max_conclusion_length=10000, required_codes=None, excluded_codes=None)

test_dataset_080_S = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='S', min_conclusion_length=0, max_conclusion_length=80, required_codes=None, excluded_codes=None)
test_dataset_81200_S = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='S', min_conclusion_length=81, max_conclusion_length=200, required_codes=None, excluded_codes=None)
test_dataset_201400_S = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='S', min_conclusion_length=201, max_conclusion_length=400, required_codes=None, excluded_codes=None)
test_dataset_400x_S = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=2048, report_type='S', min_conclusion_length=401, max_conclusion_length=10000, required_codes=None, excluded_codes=None)


print(test_dataset_400x_S)
exit()



test_datasets = {
    "0-80_C": test_dataset_080_C,
    "81-200_C": test_dataset_81200_C,
    "201-400_C": test_dataset_201400_C,
    "400+_C": test_dataset_400x_C,
    "0-80_T": test_dataset_080_T,
    "81-200_T": test_dataset_81200_T,
    "201-400_T": test_dataset_201400_T,
    "400+_T": test_dataset_400x_T,
    "0-80_S": test_dataset_080_S,
    "81-200_S": test_dataset_81200_S,
    "201-400_S": test_dataset_201400_S,
    "400+_S": test_dataset_400x_S,
}


print(f"080_C len: {len(test_dataset_080_C)}")
print(f"81200_C len: {len(test_dataset_81200_C)}")
print(f"201400_C len: {len(test_dataset_201400_C)}")
print(f"400x_C len: {len(test_dataset_400x_C)}")
print(f"080_T len: {len(test_dataset_080_T)}")
print(f"81200_T len: {len(test_dataset_81200_T)}")
print(f"201400_T len: {len(test_dataset_201400_T)}")
print(f"400x_T len: {len(test_dataset_400x_T)}")
print(f"080_S len: {len(test_dataset_080_S)}")
print(f"81200_S len: {len(test_dataset_81200_S)}")
print(f"201400_S len: {len(test_dataset_201400_S)}")
print(f"400x_S len: {len(test_dataset_400x_S)}")


checkpoint = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs15_data_setall_commentT5_small_pretrained_v2_all_custom_loss_autopsies_split.pth"


config = T5Config(decoder_start_token_id=tokenizer.pad_token_id) 
model = T5ForConditionalGeneration(config)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load(checkpoint))

data_collator = prepare_datacollator(tokenizer, model)

batch_size = 8
max_generate_length = 32

outcomes = {}

print(f"Test dataset path {test_dataset_path}")
print(f"Tokenizer path {tokenizer_path}")
print(f"Checkpoint path {checkpoint}")


for dataset_name, dataset in test_datasets.items():
    print(f"Processing datasets: {dataset_name}")

    test_dataloader = prepare_dataloader(dataset, data_collator, batch_size)

    data_len = len(dataset)

    all_test_metrics, decoded_test_preds, decoded_test_labels, decoded_test_inputs = test_step2(model, test_dataloader, tokenizer, max_generate_length)
    
    outcomes[dataset_name] = data_len, all_test_metrics

    print(f"Metrics for dataset {dataset_name}: {all_test_metrics}")

    write_test_predictions(dataset_name, all_test_metrics, decoded_test_inputs, decoded_test_preds, decoded_test_labels)

create_outcome_matrix(outcomes)

print('--------------')
test_dataloader = prepare_dataloader(test_dataset, data_collator, batch_size)

data_len = len(test_dataset)

all_test_metrics, decoded_test_preds, decoded_test_labels, decoded_test_inputs = test_step2(model, test_dataloader, tokenizer, max_generate_length)
print(f"Outcomes: {all_test_metrics}")