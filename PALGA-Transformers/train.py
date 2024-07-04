from finetune_utils import *
from shared_utils import *
import argparse
from TRIE import create_palga_trie

def main(num_train_epochs, max_generate_length, train_batch_size, validation_batch_size, learning_rate,
         max_length_sentence, data_set, local_tokenizer_path, local_model_path, comment, patience, freeze_all_but_x_layers, lr_strategy, optimizer_type
         , dropout_rate, constrained_decoding):
    
    # Generate config and run name
    config, run_name = generate_config_and_run_name(
        num_train_epochs=num_train_epochs,
        data_set=data_set,
        comment=comment,
    )


    # Print run name
    print(f"Run name: {run_name}")

    # # Initialize WandB for logging
    # wandb.init(project="Transformers-PALGA", entity="srp-palga", config=config)

    # Load tokenizer
    tokenizer = load_tokenizer(local_tokenizer_path)

    # Load datasets
    train_dataset, val_datasets = prepare_datasets_tsv(data_set, tokenizer, max_length_sentence)
    # test_datasets = prepare_test_dataset(tokenizer, max_length_sentence)
    
    # Setup model and tokenizer
    model = setup_model(tokenizer, freeze_all_but_x_layers, local_model_path, dropout_rate)
    
    # Prepare datacollator and dataloaders
    data_collator = prepare_datacollator(tokenizer, model)
    train_dataloader, eval_dataloaders = prepare_dataloaders(train_dataset, val_datasets, data_collator, train_batch_size, validation_batch_size)

    num_training_steps = num_train_epochs * len(train_dataloader)
 
    # Prepare training objects
    optimizer, accelerator, model, train_dataloader, eval_dataloaders, scheduler = prepare_training_objects(learning_rate, model, train_dataloader, eval_dataloaders, lr_strategy, num_training_steps, optimizer_type)

    if constrained_decoding:
        thesaurus_location = "/home/gburger01/snomed_20230426.txt"
        tokenizer_location = local_tokenizer_path
        data_location = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/pretrain.tsv"
        exclusive_terms_file_path = "/home/gburger01/mutually_exclusive_values.txt"
        Palga_trie = create_palga_trie(thesaurus_location, tokenizer_location, data_location, exclusive_terms_file_path)
    else:
        Palga_trie = None
    # Training and evaluation
    train_model(model, optimizer, accelerator, max_generate_length, train_dataloader, eval_dataloaders, num_train_epochs, tokenizer, run_name, patience, scheduler, Palga_trie, config, constrained_decoding)

# Entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for fine-tuning a hugginface transformer model")

    # Define the command-line arguments
    parser.add_argument("--num_train_epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--max_generate_length", type=int, default=128, help="Max length for generation")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--validation_batch_size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length_sentence", type=int, default=2048, help="Max length of sentence")
    parser.add_argument("--data_set", type=str, default='all', help="Dataset name")
    parser.add_argument("--local_tokenizer_path", type=str, default='google/mT5-small', help="Local tokenizer path")
    parser.add_argument("--local_model_path", type=str, default='PALGA-Transformers/models/mT5-small', help="Local model path")
    parser.add_argument("--comment", type=str, default='', help="Comment for the current run")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--freeze_all_but_x_layers", type=int, default=0, help="Number of layers NOT to freeze")
    parser.add_argument("--lr_strategy", type=str, default='AdamW', help="AdamW or ST-LR")
    parser.add_argument("--optimizer_type", type=str, default='AdamW', help="Optimizer type")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--constrained_decoding", action='store_true', help="Enable constrained decoding")
    args = parser.parse_args()

    # Call main function with the parsed arguments
    main(args.num_train_epochs, args.max_generate_length, args.train_batch_size, args.validation_batch_size,
         args.learning_rate, args.max_length_sentence, args.data_set, args.local_tokenizer_path, args.local_model_path, 
         args.comment, args.patience, args.freeze_all_but_x_layers, args.lr_strategy, args.optimizer_type, args.dropout_rate, args.constrained_decoding)
