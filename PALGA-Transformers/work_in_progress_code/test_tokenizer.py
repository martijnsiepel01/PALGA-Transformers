from transformers import AutoTokenizer
from datasets import load_dataset


# Initialize your tokenizer (replace with your tokenizer)
tokenizer = AutoTokenizer.from_pretrained('PALGA-Transformers/PALGA-Transformers/flan_tokenizer')


def preprocess_function(examples, tokenizer, max_length_sentence):
    inputs = examples["Conclusie"]
    targets = examples["Palga_codes"]
    print(inputs)
    print(targets)

    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )

    # Decode the token IDs to text for both inputs and targets
    decoded_inputs = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in model_inputs['input_ids']]
    if 'labels' in model_inputs:  # Ensure 'labels' exists before decoding (depends on the tokenizer)
        decoded_targets = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in model_inputs['labels']]
        print("Decoded Targets:", decoded_targets)
    print("Decoded Inputs:", decoded_inputs)
    exit()
    return model_inputs


def prepare_datasets_tsv(data_set, tokenizer, max_length_sentence):
    # Define file paths for the first dataset
    data_files = {"train": f"PALGA-Transformers/PALGA-Transformers/data/{data_set}/{data_set}_norm_train_with_codes.tsv", "test": f"PALGA-Transformers/PALGA-Transformers/data/{data_set}/{data_set}_norm_test_with_codes.tsv", "validation": f"PALGA-Transformers/PALGA-Transformers/data/{data_set}/{data_set}_norm_validation_with_codes.tsv"}
    
    # Load the first dataset
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

   # Further processing (filtering and tokenizing)
    print(dataset)
    print(dataset.keys())
    for split in dataset.keys():
        dataset[split] = dataset[split].filter(lambda example: example["Palga_codes"] is not None and example["Palga_codes"] != '')
        dataset[split] = dataset[split].filter(lambda example: example["Conclusie"] is not None and example["Conclusie"] != '')
        dataset[split] = dataset[split].map(
            lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
            batched=False
        )
        dataset[split] = dataset[split].remove_columns(["Conclusie", "Palga_codes"])
        dataset[split].set_format("torch")

    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    return train_dataset, val_dataset


train_dataset, val_dataset = prepare_datasets_tsv("histo", tokenizer, 512)

print(train_dataset[:5])
