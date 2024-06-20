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
    return test_dataset.select(range(int(len(test_dataset) * 0.3)))
    # return test_dataset

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


tokenizer_path_pretrain = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/T5_small_32128_with_codes_csep_normal_token"
# tokenizer_path_finetune = "google/mt5-small"
tokenizer_pretrain = AutoTokenizer.from_pretrained(tokenizer_path_pretrain)
# tokenizer_finetune = AutoTokenizer.from_pretrained(tokenizer_path_finetune)

tokenizer = tokenizer_pretrain

df = pd.read_csv('/home/msiepel/snomed_20230426.txt', delimiter='|', encoding='latin')
unique_codes = df[df["DESTACE"] == "V"]["DEPALCE"].str.lower().unique().tolist()
print(f"Unique codes read successfully")

topography = [code for code in unique_codes if code.startswith("t")]
procedure = [code for code in unique_codes if code.startswith("p")]
morphology = [code for code in unique_codes if not code.startswith("t") and not code.startswith("p")]

topography_tokens = set()
procedure_tokens = set()
morphology_tokens = set()

for word in topography:
    # Tokenize the word
    tokens = tokenizer.tokenize(word)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    topography_tokens.update(token_ids)
# # print(f"Topography tokens: {tokenizer.convert_ids_to_tokens(list(topography_tokens))}")

for word in procedure:
    # Tokenize the word
    tokens = tokenizer.tokenize(word)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    procedure_tokens.update(token_ids)
# # print(f"Procedure tokens: {tokenizer.convert_ids_to_tokens(list(procedure_tokens))}")

for word in morphology:
    # Tokenize the word
    tokens = tokenizer.tokenize(word)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    morphology_tokens.update(token_ids)
# # print(f"Morphology tokens: {tokenizer.convert_ids_to_tokens(list(morphology_tokens))}")



def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        # Split by lines first, then split each line by comma
        data = [line.strip().lower().split(',') for line in file]
    return data

file_path = '/home/msiepel/mutually_exclusive_values.txt'
data = load_txt_file(file_path)

# Creating DataFrame directly from the list of lists
mutually_exclusive_terms = pd.DataFrame(data, columns=['Term1', 'Term2'])

exclusive_dict = {}
for index, row in mutually_exclusive_terms.iterrows():
    # Convert terms to integers before adding them to the dictionary
    term1 = int(row['Term1'])
    term2 = int(row['Term2'])

    exclusive_dict[term1] = term2
    exclusive_dict[term2] = term1

def custom_decode(tokenizer, token_ids):
    tokens = [tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]
    decoded_string = ' '.join(tokens).replace(' </s>', '').replace('  c  s e p ', "[C-SEP]").strip()
    return decoded_string


def create_prefix_allowed_tokens_fn(Palga_trie, tokenizer):
    def prefix_allowed_tokens_fn(batch_id, sent):
        sent = sent.tolist()[1:]
        if len(sent) > 0:
            try:
                index = sent.index(462)
                sent = sent[index + 1:]
            except ValueError:
                sent = sent

        out = list(Palga_trie.get(sent))
        if len(out) > 0:
            return out
        else:
            return list(tokenizer.encode(tokenizer.eos_token))
    return prefix_allowed_tokens_fn

def test_step2(model, test_dataloader, tokenizer, max_generate_length, constrained_decoding, Palga_trie):
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
            if constrained_decoding:
                prefix_allowed_tokens_fn = create_prefix_allowed_tokens_fn(Palga_trie, tokenizer)
                outputs =  model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=max_generate_length,
                    diversity_penalty=0.3,
                    num_beams=6,
                    num_beam_groups=2,
                    no_repeat_ngram_size=3,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
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
        
def write_test_predictions(dataset_name_pretrain, dataset_name_finetune, all_test_metrics_pretrain, all_test_metrics_finetune, decoded_test_inputs_pretrain, decoded_test_inputs_finetune, decoded_test_preds_pretrain, decoded_test_preds_finetune, decoded_test_labels_pretrain, decoded_test_labels_finetune):
    # Load BLEU and ROUGE metrics
    # bleu_metric = evaluate.load('sacrebleu')
    # rouge_metric = evaluate.load('rouge')
    # meteor = load("meteor")
    
    # all_bleu_scores = []
    # all_meteor_scores = []

    print(f"Metrics for dataset {dataset_name_pretrain} and {dataset_name_finetune}:")
    print(f"Model {dataset_name_pretrain}: {all_test_metrics_pretrain}")
    print(f"Model {dataset_name_finetune}: {all_test_metrics_finetune}")
    
    if all_test_metrics_pretrain['bleu'] == 0 or all_test_metrics_finetune['bleu'] == 0:
        return

    random_indices = random.sample(range(len(decoded_test_inputs_pretrain)), min(len(decoded_test_inputs_pretrain), 20))

    for index in random_indices:
        input_seq_pretrain = decoded_test_inputs_pretrain[index]
        label_pretrain = decoded_test_labels_pretrain[index]
        pred_pretrain = decoded_test_preds_pretrain[index]

        # input_seq_finetune = decoded_test_inputs_finetune[index]
        # label_finetune = decoded_test_labels_finetune[index]
        pred_finetune = decoded_test_preds_finetune[index]

        label_words_pretrain = ' '.join([get_word_from_code(code) for code in label_pretrain.split()])
        pred_words_pretrain = ' '.join([get_word_from_code(code) for code in pred_pretrain.split()])

        # label_words_finetune = ' '.join([get_word_from_code(code) for code in label_finetune.split()])
        pred_words_finetune = ' '.join([get_word_from_code(code) for code in pred_finetune.split()])
 
        print(f"Input Sequence:                               {input_seq_pretrain}")
        print(f"Label:                                        {label_words_pretrain}")
        print(f"Prediction model pretrain:                    {pred_words_pretrain}")
        # print(f"Prediction model finetune:                    {pred_words_finetune}")
        print('-'*100 + '\n')


def create_outcome_matrix(outcomes, dataset_type):
    print(dataset_type)
    # Determine dataset type suffix based on input
    suffix = '_' + dataset_type

    # Initialize matrix with dynamically constructed keys
    matrix = {
        'T': {'0-80': outcomes['0-80_T' + suffix], '81-200': outcomes['81-200_T' + suffix], '201-400': outcomes['201-400_T' + suffix], '400+': outcomes['400+_T' + suffix], 'Average': None},
        'C': {'0-80': outcomes['0-80_C' + suffix], '81-200': outcomes['81-200_C' + suffix], '201-400': outcomes['201-400_C' + suffix], '400+': outcomes['400+_C' + suffix], 'Average': None},
        'S': {'0-80': outcomes['0-80_S' + suffix], '81-200': outcomes['81-200_S' + suffix], '201-400': outcomes['201-400_S' + suffix], '400+': outcomes['400+_S' + suffix], 'Average': None},
        'Average': {'0-80': None, '81-200': None, '201-400': None, '400+': None, 'Average': None}
    }

    # Calculate subgroup averages
    for subgroup in 'TCS':
        total_bleu = sum(outcomes[f"{length}_{subgroup}{suffix}"][0] * outcomes[f"{length}_{subgroup}{suffix}"][1]['bleu'] for length in ['0-80', '81-200', '201-400', '400+'])
        total_count = sum(outcomes[f"{length}_{subgroup}{suffix}"][0] for length in ['0-80', '81-200', '201-400', '400+'])
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

    data_len_pretrain = len(dataset_pretrain)  # Number of items in dataset_pretrain
    data_len_finetune = len(dataset_finetune)  # Number of items in dataset_finetune
    print(f"Data length for {dataset_name_pretrain}: {data_len_pretrain}")
    print(f"Data length for {dataset_name_finetune}: {data_len_finetune}")
    print(f"Matrix {matrix}")



    # Print the matrix
    for subgroup in 'TCS':
        print(f"{subgroup}: " + ', '.join(f"{length}: {matrix[subgroup][length][1]['bleu']:.2f} (N={matrix[subgroup][length][0]})" for length in ['0-80', '81-200', '201-400', '400+']) + f", Average: {matrix[subgroup]['Average'][1]['bleu']:.2f} (N={matrix[subgroup]['Average'][0]})")
    print('Average: ' + ', '.join(f"{length}: {matrix['Average'][length][1]['bleu']:.2f} (N={matrix['Average'][length][0]})" for length in ['0-80', '81-200', '201-400', '400+']) + f", Overall Average: {matrix['Average']['Average'][1]['bleu']:.2f} (N={matrix['Average']['Average'][0]})")
    print('------------------')


# test_dataset_path = "validation_combined_with_codes.tsv"
# test_dataset_path = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/combined_gold_standard_with_codes.tsv"
test_dataset_path = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/combined_test_with_codes.tsv"
test_dataset_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=None, report_type=None, min_conclusion_length=None, max_conclusion_length=None, required_codes=None, excluded_codes=None)
test_dataset_080_C_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='C', min_conclusion_length=0, max_conclusion_length=80, required_codes=None, excluded_codes=None)
test_dataset_81200_C_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='C', min_conclusion_length=81, max_conclusion_length=200, required_codes=None, excluded_codes=None)
test_dataset_201400_C_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='C', min_conclusion_length=201, max_conclusion_length=400, required_codes=None, excluded_codes=None)
test_dataset_400x_C_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='C', min_conclusion_length=401, max_conclusion_length=10000, required_codes=None, excluded_codes=None)

test_dataset_080_T_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='T', min_conclusion_length=0, max_conclusion_length=80, required_codes=None, excluded_codes=None)
test_dataset_81200_T_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='T', min_conclusion_length=81, max_conclusion_length=200, required_codes=None, excluded_codes=None)
test_dataset_201400_T_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='T', min_conclusion_length=201, max_conclusion_length=400, required_codes=None, excluded_codes=None)
test_dataset_400x_T_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='T', min_conclusion_length=401, max_conclusion_length=10000, required_codes=None, excluded_codes=None)

test_dataset_080_S_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='S', min_conclusion_length=0, max_conclusion_length=80, required_codes=None, excluded_codes=None)
test_dataset_81200_S_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='S', min_conclusion_length=81, max_conclusion_length=200, required_codes=None, excluded_codes=None)
test_dataset_201400_S_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='S', min_conclusion_length=201, max_conclusion_length=400, required_codes=None, excluded_codes=None)
test_dataset_400x_S_pretrain = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_pretrain, max_length_sentence=2048, report_type='S', min_conclusion_length=401, max_conclusion_length=10000, required_codes=None, excluded_codes=None)

# test_dataset_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=None, report_type=None, min_conclusion_length=None, max_conclusion_length=None, required_codes=None, excluded_codes=None)
# test_dataset_080_C_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='C', min_conclusion_length=0, max_conclusion_length=80, required_codes=None, excluded_codes=None)
# test_dataset_81200_C_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='C', min_conclusion_length=81, max_conclusion_length=200, required_codes=None, excluded_codes=None)
# test_dataset_201400_C_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='C', min_conclusion_length=201, max_conclusion_length=400, required_codes=None, excluded_codes=None)
# test_dataset_400x_C_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='C', min_conclusion_length=401, max_conclusion_length=10000, required_codes=None, excluded_codes=None)

# test_dataset_080_T_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='T', min_conclusion_length=0, max_conclusion_length=80, required_codes=None, excluded_codes=None)
# test_dataset_81200_T_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='T', min_conclusion_length=81, max_conclusion_length=200, required_codes=None, excluded_codes=None)
# test_dataset_201400_T_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='T', min_conclusion_length=201, max_conclusion_length=400, required_codes=None, excluded_codes=None)
# test_dataset_400x_T_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='T', min_conclusion_length=401, max_conclusion_length=10000, required_codes=None, excluded_codes=None)

# test_dataset_080_S_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='S', min_conclusion_length=0, max_conclusion_length=80, required_codes=None, excluded_codes=None)
# test_dataset_81200_S_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='S', min_conclusion_length=81, max_conclusion_length=200, required_codes=None, excluded_codes=None)
# test_dataset_201400_S_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='S', min_conclusion_length=201, max_conclusion_length=400, required_codes=None, excluded_codes=None)
# test_dataset_400x_S_finetune = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer_finetune, max_length_sentence=2048, report_type='S', min_conclusion_length=401, max_conclusion_length=10000, required_codes=None, excluded_codes=None)




test_datasets_pretrain = {
    "0-80_C_pretrain": test_dataset_080_C_pretrain,
    "81-200_C_pretrain": test_dataset_81200_C_pretrain,
    "201-400_C_pretrain": test_dataset_201400_C_pretrain,
    "400+_C_pretrain": test_dataset_400x_C_pretrain,
    "0-80_T_pretrain": test_dataset_080_T_pretrain,
    "81-200_T_pretrain": test_dataset_81200_T_pretrain,
    "201-400_T_pretrain": test_dataset_201400_T_pretrain,
    "400+_T_pretrain": test_dataset_400x_T_pretrain,
    "0-80_S_pretrain": test_dataset_080_S_pretrain,
    "81-200_S_pretrain": test_dataset_81200_S_pretrain,
    "201-400_S_pretrain": test_dataset_201400_S_pretrain,
    "400+_S_pretrain": test_dataset_400x_S_pretrain,
}

# test_datasets_finetune = {
#     "0-80_C_finetune": test_dataset_080_C_finetune,
#     "81-200_C_finetune": test_dataset_81200_C_finetune,
#     "201-400_C_finetune": test_dataset_201400_C_finetune,
#     "400+_C_finetune": test_dataset_400x_C_finetune,
#     "0-80_T_finetune": test_dataset_080_T_finetune,
#     "81-200_T_finetune": test_dataset_81200_T_finetune,
#     "201-400_T_finetune": test_dataset_201400_T_finetune,
#     "400+_T_finetune": test_dataset_400x_T_finetune,
#     "0-80_S_finetune": test_dataset_080_S_finetune,
#     "81-200_S_finetune": test_dataset_81200_S_finetune,
#     "201-400_S_finetune": test_dataset_201400_S_finetune,
#     "400+_S_finetune": test_dataset_400x_S_finetune,
# }


print(f"080_C_pretrain len: {len(test_dataset_080_C_pretrain)}")
print(f"81200_C_pretrain len: {len(test_dataset_81200_C_pretrain)}")
print(f"201400_C_pretrain len: {len(test_dataset_201400_C_pretrain)}")
print(f"400x_C_pretrain len: {len(test_dataset_400x_C_pretrain)}")
print(f"080_T_pretrain len: {len(test_dataset_080_T_pretrain)}")
print(f"81200_T_pretrain len: {len(test_dataset_81200_T_pretrain)}")
print(f"201400_T_pretrain len: {len(test_dataset_201400_T_pretrain)}")
print(f"400x_T_pretrain len: {len(test_dataset_400x_T_pretrain)}")
print(f"080_S_pretrain len: {len(test_dataset_080_S_pretrain)}")
print(f"81200_S_pretrain len: {len(test_dataset_81200_S_pretrain)}")
print(f"201400_S_pretrain len: {len(test_dataset_201400_S_pretrain)}")
print(f"400x_S_pretrain len: {len(test_dataset_400x_S_pretrain)}")

# print(f"080_C_finetune len: {len(test_dataset_080_C_finetune)}")
# print(f"81200_C_finetune len: {len(test_dataset_81200_C_finetune)}")
# print(f"201400_C_finetune len: {len(test_dataset_201400_C_finetune)}")
# print(f"400x_C_finetune len: {len(test_dataset_400x_C_finetune)}")
# print(f"080_T_finetune len: {len(test_dataset_080_T_finetune)}")
# print(f"81200_T_finetune len: {len(test_dataset_81200_T_finetune)}")
# print(f"201400_T_finetune len: {len(test_dataset_201400_T_finetune)}")
# print(f"400x_T_finetune len: {len(test_dataset_400x_T_finetune)}")
# print(f"080_S_finetune len: {len(test_dataset_080_S_finetune)}")
# print(f"81200_S_finetune len: {len(test_dataset_81200_S_finetune)}")
# print(f"201400_S_finetune len: {len(test_dataset_201400_S_finetune)}")
# print(f"400x_S_finetune len: {len(test_dataset_400x_S_finetune)}")

checkpoint_pretrain = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs25_data_setall_commentmT5_small_pretrained_v1_all_custom_loss.pth"
# checkpoint = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs25_data_setall_commentmT5_small_pretrained_v1_all_80k_histo.pth"
# checkpoint_finetune = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs15_data_setall_commentmT5_small_all_custom_loss_default_from_checkpoint_2.pth"
# checkpoint = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs25_data_setall_commentmT5_small_pretrained_v1_all.pth"


config_pretrain = T5Config(decoder_start_token_id=tokenizer_pretrain.pad_token_id) 
model_pretrain = T5ForConditionalGeneration(config_pretrain)
model_pretrain.resize_token_embeddings(len(tokenizer_pretrain))
model_pretrain.load_state_dict(torch.load(checkpoint_pretrain))

# config_finetune = MT5Config(decoder_start_token_id=tokenizer_finetune.pad_token_id) 
# model_finetune = MT5ForConditionalGeneration(config_finetune)
# model_finetune.resize_token_embeddings(len(tokenizer_finetune))
# model_finetune.load_state_dict(torch.load(checkpoint_finetune))

data_collator_pretrain = prepare_datacollator(tokenizer_pretrain, model_pretrain)
# data_collator_finetune = prepare_datacollator(tokenizer_pretrain, model_finetune)

batch_size = 8
max_generate_length = 32

outcomes_pretrain = {}
outcomes_finetune = {}

print(f"Test dataset path {test_dataset_path}")
print(f"Tokenizer path pretrain {tokenizer_path_pretrain}")
# print(f"Tokenizer path finetune {tokenizer_path_finetune}")
print(f"Checkpoint path pretrain {checkpoint_pretrain}")
# print(f"Checkpoint path finetune {checkpoint_finetune}")


# for (dataset_name_pretrain, dataset_pretrain), (dataset_name_finetune, dataset_finetune) in zip(test_datasets_pretrain.items(), test_datasets_finetune.items()):
for (dataset_name_pretrain, dataset_pretrain), (dataset_name_finetune, dataset_finetune) in zip(test_datasets_pretrain.items(), test_datasets_pretrain.items()):
    print(f"Processing datasets: {dataset_name_pretrain} + {dataset_name_finetune}")

    test_dataloader_pretrain = prepare_dataloader(dataset_pretrain, data_collator_pretrain, batch_size)
    # test_dataloader_finetune = prepare_dataloader(dataset_finetune, data_collator_finetune, batch_size)

    data_len_pretrain = len(dataset_pretrain)
    # data_len_finetune = len(dataset_finetune)

    all_test_metrics_pretrain, decoded_test_preds_pretrain, decoded_test_labels_pretrain, decoded_test_inputs_pretrain = test_step2(model_pretrain, test_dataloader_pretrain, tokenizer_pretrain, max_generate_length, 'pretrain')
    # all_test_metrics_finetune, decoded_test_preds_finetune, decoded_test_labels_finetune, decoded_test_inputs_finetune = test_step2(model_finetune, test_dataloader_finetune, tokenizer_finetune, max_generate_length, 'finetune')
    
    outcomes_pretrain[dataset_name_pretrain] = data_len_pretrain, all_test_metrics_pretrain
    # outcomes_finetune[dataset_name_finetune] = data_len_finetune, all_test_metrics_finetune

    print(f"Metrics for dataset {dataset_name_pretrain}: {all_test_metrics_pretrain}")
    # print(f"Metrics for dataset {dataset_name_finetune}: {all_test_metrics_finetune}")

    # write_test_predictions(dataset_name_pretrain, dataset_name_finetune, all_test_metrics_pretrain, all_test_metrics_finetune, decoded_test_inputs_pretrain, decoded_test_inputs_finetune, decoded_test_preds_pretrain, decoded_test_preds_finetune, decoded_test_labels_pretrain, decoded_test_labels_finetune)
    write_test_predictions(dataset_name_pretrain, dataset_name_finetune, all_test_metrics_pretrain, all_test_metrics_pretrain, decoded_test_inputs_pretrain, decoded_test_inputs_pretrain, decoded_test_preds_pretrain, decoded_test_preds_pretrain, decoded_test_labels_pretrain, decoded_test_labels_pretrain)

create_outcome_matrix(outcomes_pretrain, "pretrain")
# create_outcome_matrix(outcomes_finetune, "finetune")

print('--------------')
test_dataloader_pretrain = prepare_dataloader(test_dataset_pretrain, data_collator_pretrain, batch_size)
# test_dataloader_finetune = prepare_dataloader(test_dataset_finetune, data_collator_finetune, batch_size)

data_len_pretrain = len(test_dataset_pretrain)
# data_len_finetune = len(test_dataset_finetune)

all_test_metrics_pretrain, decoded_test_preds_pretrain, decoded_test_labels_pretrain, decoded_test_inputs_pretrain = test_step2(model_pretrain, test_dataloader_pretrain, tokenizer_pretrain, max_generate_length, 'pretrain')
# all_test_metrics_finetune, decoded_test_preds_finetune, decoded_test_labels_finetune, decoded_test_inputs_finetune = test_step2(model_finetune, test_dataloader_finetune, tokenizer_finetune, max_generate_length, 'finetune')
print(f"Outcomes pretrain: {all_test_metrics_pretrain}")
# print(f"Outcomes finetune: {all_test_metrics_finetune}")
