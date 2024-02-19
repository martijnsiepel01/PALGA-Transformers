from transformers import MT5ForConditionalGeneration, MT5Config, MT5Tokenizer, AutoTokenizer, PretrainedConfig, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Config, T5Tokenizer, DataCollatorForSeq2Seq
import torch
from datasets import load_dataset
import random
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
import numpy as np
import random


def preprocess_function(examples, tokenizer, max_length_sentence):
    inputs = [ex.lower() for ex in examples["Conclusie"]]
    targets = [ex.lower() for ex in examples["Codes"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
    return model_inputs

def prepare_test_dataset(test_data_location, tokenizer, max_length_sentence):
    # test_data_location = "/home/msiepel/data/all/all_norm_test.tsv"
    dataset = load_dataset("csv", data_files=test_data_location, delimiter="\t")
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
        batched=True
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes"])
    tokenized_datasets.set_format("torch")
    test_dataset = tokenized_datasets['train']
    # return test_dataset
    return test_dataset.select(range(1000))

def print_test_predictions(decoded_input_ids, decoded_labels, decoded_preds):
    # Load BLEU and ROUGE metrics
    bleu_metric = evaluate.load('sacrebleu')
    rouge_metric = evaluate.load('rouge')

    for i, (input_seq, label) in enumerate(zip(decoded_input_ids, decoded_labels)):
        print(f"Input Sequence:                {input_seq}")
        print(f"Label:                         {label}")

        for key, preds in decoded_preds.items():
            prediction = preds[i]
            # Format inputs correctly for the metrics
            # Note: 'sacrebleu' expects a list of references per prediction for BLEU calculation
            bleu_metric.add_batch(predictions=[prediction], references=[[label]])
            bleu_score = bleu_metric.compute()['score']

            rouge_metric.add_batch(predictions=[prediction], references=[[label]])
            rouge_scores = rouge_metric.compute()

            count = 0
            sum_scores = 0

            for _, score in rouge_scores.items():
                count += 1 
                sum_scores += score

            average_rouge_scores = sum_scores / count if count else 0


            print(f"Prediction ({key.capitalize()}):".ljust(31) + f"{prediction}, BLEU: {bleu_score:.2f}, ROUGE: {average_rouge_scores*100:.2f}")
        print('-' * 100)

def validation_step(model, dataloader, tokenizer, max_generate_length, comment,
                    greedy_params={}, contrastive_params={}, sampling_params={}, 
                    beam_params={}, beam_sampling_params={}, diverse_beam_params={}):
    
    params = locals()  # This captures all local variables, including parameters
    print("Validation Step Parameters:")
    for param, value in params.items():
        if "params" in param:  # Check if 'params' is in the variable name
            print(f"{param}: {value}")


    metric_greedy = evaluate.load("sacrebleu")
    metric_greedy_long = evaluate.load("sacrebleu")
    metric_contrastive = evaluate.load("sacrebleu")
    metric_contrastive_long = evaluate.load("sacrebleu")
    metric_sampling = evaluate.load("sacrebleu")
    metric_sampling_long = evaluate.load("sacrebleu")
    # metric_beam = evaluate.load("sacrebleu")
    # metric_beam_sampling = evaluate.load("sacrebleu")
    metric_diverse_beam = evaluate.load("sacrebleu")
    metric_diverse_beam_long = evaluate.load("sacrebleu")

    rouge_greedy = evaluate.load('rouge')
    rouge_greedy_long = evaluate.load('rouge')
    rouge_contrastive = evaluate.load('rouge')
    rouge_contrastive_long = evaluate.load('rouge')
    rouge_sampling = evaluate.load('rouge')
    rouge_sampling_long = evaluate.load('rouge')
    # rouge_beam = evaluate.load('rouge')
    # rouge_beam_sampling = evaluate.load('rouge')
    rouge_diverse_beam = evaluate.load('rouge')
    rouge_diverse_beam_long = evaluate.load('rouge')

    model.eval()
    all_preds = {
        'greedy': [],
        'greedy_long': [],
        'contrastive': [],
        'contrastive_long': [],
        'sampling': [],
        'sampling_long': [],
        # 'beam': [],
        # 'beam_sampling': [],
        'diverse_beam': [],
        'diverse_beam_long': [],
    }
    all_labels = []
    all_input_ids = []

    max_generate_length_long = max_generate_length * 2

    for batch in tqdm(dataloader, desc="Evaluation"):
        with torch.no_grad():
            # Greedy search
            all_preds['greedy'].extend(model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length,
                **greedy_params
            ))

            # Greedy search long
            all_preds['greedy_long'].extend(model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length_long,
                **greedy_params
            ))

            # Contrastive search
            all_preds['contrastive'].extend(model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length,
                **contrastive_params
            ))

            # Contrastive search long
            all_preds['contrastive_long'].extend(model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length_long,
                **contrastive_params
            ))

            # Multinomial Sampling
            all_preds['sampling'].extend(model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length,
                **sampling_params
            ))

            # Multinomial Sampling long
            all_preds['sampling_long'].extend(model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length_long,
                **sampling_params
            ))

            # # Beam search
            # all_preds['beam'].extend(model.generate(
            #     batch["input_ids"],
            #     attention_mask=batch["attention_mask"],
            #     max_length=max_generate_length,
            #     **beam_params
            # ))

            # # Beam search with Multinomial Sampling
            # all_preds['beam_sampling'].extend(model.generate(
            #     batch["input_ids"],
            #     attention_mask=batch["attention_mask"],
            #     max_length=max_generate_length,
            #     **beam_sampling_params
            # ))

            # Diverse Beam Search Decoding
            all_preds['diverse_beam'].extend(model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length,
                **diverse_beam_params
            ))

            # Diverse Beam Search Decoding long
            all_preds['diverse_beam_long'].extend(model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length_long,
                **diverse_beam_params
            ))

        all_input_ids.extend(batch["input_ids"])
        all_labels.extend(batch["labels"])

    # Decode all at once after processing all batches

    filtered_all_input_ids =  [token[token != 16100] for token in all_input_ids]  
    filtered_all_labels = [token[token != -100] for token in all_labels]

    for key in all_preds.keys():
        # Apply filtering to remove -100 from each list in the dictionary's values
        filtered_lists = [[token for token in pred if token != -100] for pred in all_preds[key]]
        # Update the dictionary with the filtered lists
        all_preds[key] = filtered_lists

    decoded_input_ids = tokenizer.batch_decode(filtered_all_input_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(filtered_all_labels, skip_special_tokens=True)

    
    decoded_preds = {}

    # Iterate through each key in the all_preds dictionary
    for key in all_preds.keys():
        # Decode the filtered predictions for the current key, skipping special tokens
        decoded_preds[key] = tokenizer.batch_decode(all_preds[key], skip_special_tokens=True)


        # Assume metric_objects is a dictionary mapping prediction types to their respective metric objects
    metric_objects = {
        'greedy': metric_greedy,
        'greedy_long': metric_greedy_long,
        'contrastive': metric_contrastive,
        'contrastive_long': metric_contrastive_long,
        'sampling': metric_sampling,
        'sampling_long': metric_sampling_long,
        # 'beam': metric_beam,
        # 'beam_sampling': metric_beam_sampling,
        'diverse_beam': metric_diverse_beam,
        'diverse_beam_long': metric_diverse_beam_long,
    }

    # Initialize dictionaries to hold BLEU and ROUGE scores
    bleu_scores = {}
    rouge_scores = {}

    # Iterate through each prediction type and its corresponding decoded predictions
    for key, predictions in decoded_preds.items():
        # Add batch for the current prediction type
        metric_objects[key].add_batch(predictions=predictions, references=decoded_labels)
        # Compute BLEU score for the current prediction type
        bleu_scores[key] = metric_objects[key].compute()['score']

    # Assuming similar setup for ROUGE metrics
    rouge_metric_objects = {
        'greedy': rouge_greedy,
        'greedy_long': rouge_greedy_long,
        'contrastive': rouge_contrastive,
        'contrastive_long': rouge_contrastive_long,
        'sampling': rouge_sampling,
        'sampling_long': rouge_sampling_long,
        # 'beam': rouge_beam,
        # 'beam_sampling': rouge_beam_sampling,
        'diverse_beam': rouge_diverse_beam,
        'diverse_beam_long': rouge_diverse_beam_long,
    }

    # Compute ROUGE scores in a similar fashion
    for key, predictions in decoded_preds.items():
        rouge_metric_objects[key].add_batch(predictions=predictions, references=decoded_labels)
        rouge_scores[key] = rouge_metric_objects[key].compute()

    # Print test predictions using the separate function
    print_test_predictions(decoded_input_ids, decoded_labels, decoded_preds)

    # Dynamically print BLEU scores from the bleu_scores dictionary
    for key, score in bleu_scores.items():
        print(f"BLEU Score ({key.replace('_', ' ').capitalize()}): {score:.2f}")

    # Add a separator for readability, if desired
    print('-' * 50)

    # First, calculate the average ROUGE F1 score for each strategy
    average_rouge_f1_scores = {}

    for key, metrics in rouge_scores.items():
        sum_f1_scores = 0
        count = 0
        for metric, f1_score in metrics.items():  # Assuming 'f1_score' is directly the F1 score
            sum_f1_scores += f1_score
            count += 1
        average_rouge_f1_scores[key] = sum_f1_scores / count if count else 0

    # Then print the averages
    print("Average ROUGE F1 Scores:")
    for key, avg_score in average_rouge_f1_scores.items():
        print(f"ROUGE Score ({key.replace('_', ' ').capitalize()}): {avg_score:.2f}")

    return bleu_scores



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


tokenizer = T5Tokenizer('/home/msiepel/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_32128/combined_data_unigram_t5_custom_32128_098_lower_case.model')
max_length_sentence = 512

# test_dataset_gold = prepare_test_dataset('/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/gold_P1.tsv', tokenizer, max_length_sentence)
test_dataset = prepare_test_dataset('/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_test.tsv', tokenizer, max_length_sentence)

config = T5Config.from_pretrained('/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/flan-t5-small/config.json')

checkpoint_model = '/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/trained_models/epochs15_datasetall_modelflan-t5-small_commentcombined_data_unigram_32128_098_lower_case_on_combined_data_with_flan_small_model_patience3_freezeallbutxlayers0.pth'


# tokenizer = T5Tokenizer('/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_128000/t5_custom_128000_1.model')
# max_length_sentence = 512

# test_dataset_gold = prepare_test_dataset('/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/processed_gold_P1.tsv', tokenizer, max_length_sentence)
# test_dataset = prepare_test_dataset('/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_test.tsv', tokenizer, max_length_sentence)

# config = T5Config.from_pretrained('/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/flan-t5-small/config.json')
# checkpoint_model = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/epochs15_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelflan-t5-small_commentcustom_tokenizer_vocab_128000_patience3_freezeallbutxlayers0.pth'


model = T5ForConditionalGeneration(config)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load(checkpoint_model))

batch_size = 32  

collator = prepare_datacollator(tokenizer, model)

# test_dataloader_gold = prepare_dataloader(test_dataset_gold, collator, batch_size)
test_dataloader = prepare_dataloader(test_dataset, collator, batch_size)


def random_search(model, dataloader, tokenizer, max_generate_length, comment, parameter_space, n_runs=100):
    best_score = float('-inf')  # Initialize the best score as the lowest possible value
    best_params = {}  # Dictionary to store the best parameters

    print(comment)
    for _ in range(n_runs):
        # Generate current parameters for static definitions
        current_params = {
            key: {param: random.choice(values) for param, values in strategy.items()}
            for key, strategy in parameter_space.items()
        }

        # Dynamically generate diverse_beam_params for each iteration
        current_params['diverse_beam_params'] = select_diverse_beam_params()

        # Call the validation step with the current parameters
        scores = validation_step(model=model, dataloader=dataloader, tokenizer=tokenizer,
                                 max_generate_length=max_generate_length, comment=comment,
                                 **current_params)

        # Assuming scores contain BLEU and/or ROUGE scores, determine the comparison metric
        # This example assumes you have a function to calculate or extract a single comparison metric from scores
        print(scores)
        comparison_metric = calculate_comparison_metric(scores)

        # Check if the current comparison metric is higher than the best score observed so far
        if comparison_metric > best_score:
            best_score = comparison_metric  # Update the best score
            best_params = current_params  # Save the parameters that led to the new best score
            print(f"New best score: {best_score}")
            print("With parameters:")
            for param, value in best_params.items():
                print(f"{param}: {value}")

    return best_score, best_params

def calculate_comparison_metric(scores):
    # If scores is a dictionary with strategy names as keys and float scores as values
    return max(scores.values())


def select_diverse_beam_params():
    # Example ranges as previously defined
    num_beam_groups_choices = range(2, 4)  # Assuming you want at least some diversity
    diversity_penalty_choices = np.linspace(0.1, 0.5, 5)
    
    # Select num_beam_groups first
    num_beam_groups = random.choice(num_beam_groups_choices)
    
    # Ensure num_beams is divisible by num_beam_groups by constructing a compatible range
    max_beams = 6  # Set the maximum feasible number of beams for your setup
    num_beams_choices = [n for n in range(num_beam_groups, max_beams + 1) if n % num_beam_groups == 0]
    
    # Now select num_beams from the filtered choices
    num_beams = random.choice(num_beams_choices)
    
    # Select diversity_penalty
    diversity_penalty = random.choice(diversity_penalty_choices)
    
    return {'num_beams': num_beams, 'num_beam_groups': num_beam_groups, 'diversity_penalty': diversity_penalty}


# parameter_space = {
#     'contrastive_params': {
#         'penalty_alpha': np.linspace(0.7, 1.0, 4).tolist(),  # Convert to list
#         'top_k': list(range(1, 6))  # Convert range to list
#     },
#     'sampling_params': {
#         'temperature': np.linspace(0.5, 0.7, 3).tolist(),  # Convert to list
#         'top_k': list(range(20, 51, 10))  # Convert range to list
#     },
#     'beam_params': {
#         'num_beams': list(range(4, 10)),  # Convert range to list
#         'no_repeat_ngram_size': [1]  # Convert range to list
#     },
#     'beam_sampling_params': {
#         'num_beams': list(range(4, 10)),  # Convert range to list
#         'no_repeat_ngram_size': [1],  # Convert range to list
#         'do_sample': [True]
#     },
# }

# best_score, best_params = random_search(
#     model=model,
#     dataloader=test_dataloader_gold,
#     tokenizer=tokenizer,
#     max_generate_length=32,
#     parameter_space=parameter_space,
#     comment="Hyperparameter search with 16000_098 model"
# )

# print(f"Best score: {best_score}")
# print(f"Best params: {best_params}")

settings = {
    'greedy_params': {},
        'contrastive_params': {
            'penalty_alpha': 0.7,
            'top_k': 3
        },
        'sampling_params': {
            'temperature': 0.7,
            'top_k': 20
        },
        # 'beam_params': {
        #     'num_beams': 8,
        #     'no_repeat_ngram_size': 1
        # },
        # 'beam_sampling_params': {
        #     'num_beams': 7,
        #     'no_repeat_ngram_size': 1,
        #     'do_sample': True
        # },
        'diverse_beam_params': {
            'diversity_penalty': 0.3,
            'num_beams': 6,
            'num_beam_groups': 2
            # 'diversity_penalty_scale': 0.8
        }
    }

validation_step(model, test_dataloader, tokenizer, 32, "Validation Step", **settings)