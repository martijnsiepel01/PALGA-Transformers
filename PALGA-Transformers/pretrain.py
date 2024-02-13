from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, MT5Tokenizer
from datasets import load_dataset


def load_tokenizer(local_tokenizer_path = 'PALGA-Transformers/flan_tokenizer'):
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    return tokenizer


def load_model(tokenizer):
    config = T5Config()
    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    return model


# def preprocess_function(examples, tokenizer, max_length_sentence, task):
#     if task == 'span_corruption':
#         inputs = [ex for ex in examples["input_sequence_span_corruption"]]
#         targets = [ex for ex in examples["output_sequence_span_corruption"]]
#     elif task == 'translation_pair_span_corruption':
#         inputs = [ex for ex in examples["input_sequence_translation_pair_span_corruption"]]
#         targets = [ex for ex in examples["output_sequence_translation_pair_span_corruption"]]
#     elif task == 'span_corruption_with_target_concat':
#         inputs = [ex for ex in examples["input_sequence_source_only_span_corruption_with_target_concat"]]
#         targets = [ex for ex in examples["output_sequence_source_only_span_corruption_with_target_concat"]]
#     model_inputs = tokenizer(
#         inputs, text_target=targets, max_length=max_length_sentence, truncation=True
#     )
#     return model_inputs

def prepare_datasets_tsv(tokenizer, max_length_sentence, task):
    data_file = f"PALGA-Transformers/data/all/pretrain_all_train.tsv"
    dataset = load_dataset("csv", data_files=data_file, delimiter="\t")
    dataset = dataset.filter(lambda example: all(example[col] is not None and example[col] != '' for col in example))
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length_sentence, task),
        batched=True
    )
    # tokenized_datasets = tokenized_datasets.remove_columns(
    #     [col for col in tokenized_datasets.column_names if col not in ['input_ids', 'attention_mask', 'labels']]
    # )
    tokenized_datasets.set_format("torch")
    return tokenized_datasets


from transformers import PreTrainedTokenizer
from typing import Dict, List

def preprocess_function(examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizer, max_length_sentence: int, task: str):
    if task == 'span_corruption':
        inputs = examples["input_sequence_span_corruption"]
        targets = examples["output_sequence_span_corruption"]
    elif task == 'translation_pair_span_corruption':
        inputs = examples["input_sequence_translation_pair_span_corruption"]
        targets = examples["output_sequence_translation_pair_span_corruption"]
    elif task == 'span_corruption_with_target_concat':
        inputs = examples["input_sequence_source_only_span_corruption_with_target_concat"]
        targets = examples["output_sequence_source_only_span_corruption_with_target_concat"]
    
    # Check if inputs or targets would be cut off
    for i, (input_text, target_text) in enumerate(zip(inputs, targets)):
        input_len = len(tokenizer.tokenize(input_text))
        target_len = len(tokenizer.tokenize(target_text))
        
        if input_len > max_length_sentence:
            print(f"Input sentence {i} is longer than max length ({input_len} tokens). It will be cut off.")
        if target_len > max_length_sentence:
            print(f"Target sentence {i} is longer than max length ({target_len} tokens). It will be cut off.")

    # Proceed with tokenization and truncation
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
    return model_inputs

tokenizer = load_tokenizer()
max_length_sentence = 1024
task = 'translation_pair_span_corruption'
train_dataset = prepare_datasets_tsv(tokenizer, max_length_sentence, task)