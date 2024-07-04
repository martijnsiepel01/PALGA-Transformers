from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, T5Tokenizer, MT5Tokenizer, AutoTokenizer


# def load_tokenizer(local_tokenizer_path):
#     tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
#     return tokenizer

def load_tokenizer(local_tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
    
    # Check if '[C-SEP]' is in the tokenizer's vocabulary
    if '[C-SEP]' not in tokenizer.get_vocab():
        # Adding '[C-SEP]' to the tokenizer's vocabulary
        tokenizer.add_tokens(['[C-SEP]'])
    
    return tokenizer

def prepare_datacollator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return data_collator

def prepare_dataloaders(train_dataset, val_datasets, data_collator, train_batch_size, validation_batch_size):
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size,
    )
    
    # Create a list of DataLoaders, one for each validation dataset
    eval_dataloaders = [
        DataLoader(
            dataset, 
            collate_fn=data_collator, 
            batch_size=validation_batch_size
        ) for dataset in val_datasets
    ]
    
    
    return train_dataloader, eval_dataloaders

def generate_config_and_run_name(**kwargs):
    # The config dictionary is directly constructed from kwargs
    config = kwargs

    # Construct the run_name string dynamically from the kwargs
    run_name_parts = [f'{key}{value}' for key, value in kwargs.items()]
    run_name = '_'.join(run_name_parts)

    return config, run_name