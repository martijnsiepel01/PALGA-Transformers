from utils_mbart import *
import argparse

def main(num_train_epochs, max_generate_length, train_batch_size, validation_batch_size, learning_rate,
         max_length_sentence, data_set, local_tokenizer_path, local_model_path, comment, patience, freeze_all_but_x_layers):
    
    # Generate config and run name
    config, run_name = generate_config_and_run_name(
        num_train_epochs,
        max_length_sentence,
        train_batch_size,
        validation_batch_size,
        learning_rate,
        max_generate_length,
        data_set,
        local_model_path,
        comment,
        patience,
        freeze_all_but_x_layers
    )

    # Print run name
    print(f"Run name: {run_name}")

    # Initialize WandB for logging
    wandb.init(project="Transformers-PALGA", entity="srp-palga", config=config)

    # Load tokenizer
    tokenizer = load_tokenizer(local_tokenizer_path)

    # Load datasets
    if data_set == "autopsies" or data_set == "histo" or "all" in data_set:
        train_dataset, val_dataset = prepare_datasets_tsv(data_set, tokenizer, max_length_sentence)
        test_dataset = prepare_test_dataset(tokenizer, max_length_sentence)
    
    # Setup model and tokenizer
    model = setup_model(tokenizer, freeze_all_but_x_layers, local_model_path)
    
    # Prepare datacollator and dataloaders
    data_collator = prepare_datacollator(tokenizer, model)
    train_dataloader, eval_dataloader, test_dataloader = prepare_dataloaders(train_dataset, val_dataset, test_dataset, data_collator, train_batch_size, validation_batch_size)
    
    # Prepare training objects
    optimizer, accelerator, model, optimizer, train_dataloader, eval_dataloader, test_dataloader = prepare_training_objects(learning_rate, model, train_dataloader, eval_dataloader, test_dataloader)

    # Training and evaluation
    train_model(model, optimizer, accelerator, max_generate_length, train_dataloader, eval_dataloader, test_dataloader, num_train_epochs, tokenizer, run_name, patience)

# Entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for fine-tuning a hugginface transformer model")

    # Define the command-line arguments
    parser.add_argument("--num_train_epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--max_generate_length", type=int, default=32, help="Max length for generation")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--validation_batch_size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length_sentence", type=int, default=512, help="Max length of sentence")
    parser.add_argument("--data_set", type=str, default='all', help="Dataset name")
    parser.add_argument("--local_tokenizer_path", type=str, default='/home/msiepel/mbart_tokenizer', help="Local tokenizer path")
    parser.add_argument("--local_model_path", type=str, default='/home/msiepel/models/mbart-large-50', help="Local model path")
    parser.add_argument("--comment", type=str, default='', help="Comment for the current run")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--freeze_all_but_x_layers", type=int, default=0, help="Number of layers NOT to freeze")

    args = parser.parse_args()

    # Call main function with the parsed arguments
    main(args.num_train_epochs, args.max_generate_length, args.train_batch_size, args.validation_batch_size,
         args.learning_rate, args.max_length_sentence, args.data_set, args.local_tokenizer_path, args.local_model_path, 
         args.comment, args.patience, args.freeze_all_but_x_layers)
    
    # sbatch train.sh --num_train_epochs 50 --data_set all --local_model_path "/home/msiepel/models/mT5_small" --comment "mt5-small_default_settings_final" --patience 10
         