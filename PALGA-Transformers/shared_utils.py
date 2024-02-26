from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, T5Tokenizer


def load_tokenizer(local_tokenizer_path):
    tokenizer = T5Tokenizer.from_pretrained(local_tokenizer_path)
    return tokenizer

def prepare_datacollator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return data_collator

def prepare_dataloaders(train_dataset, val_dataset, test_dataset, data_collator, train_batch_size, validation_batch_size):
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size,
    )
    eval_dataloader = DataLoader(
        val_dataset, 
        collate_fn=data_collator, 
        batch_size=validation_batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, 
        collate_fn=data_collator, 
        batch_size=validation_batch_size
    )
    return train_dataloader, eval_dataloader, test_dataloader

def generate_config_and_run_name(**kwargs):
    # The config dictionary is directly constructed from kwargs
    config = kwargs

    # Construct the run_name string dynamically from the kwargs
    run_name_parts = [f'{key}{value}' for key, value in kwargs.items()]
    run_name = '_'.join(run_name_parts)

    return config, run_name