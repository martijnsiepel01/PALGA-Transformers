from shared_utils import *
from pretrain_utils import *
import argparse
import wandb

def main(num_train_epochs, max_length_sentence, train_batch_size, validation_batch_size, lr, data_set, comment, patience, task, local_tokenizer_path):
    
    # Generate config and run name
    config, run_name = generate_config_and_run_name(
        num_train_epochs=num_train_epochs,
        data_set=data_set,
        comment=comment,
        task=task  
    )

    # Initialize WandB for logging
    wandb.init(project="Transformers-PALGA", entity="srp-palga", config=config)

    # Load tokenizer and ensure pad_token_id is set
    tokenizer = load_tokenizer(local_tokenizer_path)
    assert tokenizer.pad_token_id is not None, "Tokenizer's pad_token_id is not set."

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets_tsv(data_set, tokenizer, max_length_sentence, task)
    
    # Load model
    model = load_model(tokenizer)
    
    # Prepare data collator and dataloaders
    data_collator = prepare_datacollator(tokenizer, model)
    train_dataloader, eval_dataloader, test_dataloader = prepare_dataloaders_pretrain(train_dataset, val_dataset, test_dataset, data_collator, train_batch_size, validation_batch_size)
    
    # Prepare training objects
    optimizer, accelerator, model, optimizer, train_dataloader, eval_dataloader, test_dataloader = prepare_training_objects(lr, model, train_dataloader, eval_dataloader, test_dataloader)

    # Output dataloader sizes
    print(f"Train Dataloader Length: {len(train_dataloader)}")
    print(f"Eval Dataloader Length: {len(eval_dataloader)}")
    print(f"Test Dataloader Length: {len(test_dataloader)}")

    # Start training
    train_model(model, optimizer, accelerator, train_dataloader, eval_dataloader, test_dataloader, num_train_epochs, run_name, patience)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for pretraining a transformer model")

    # Define command-line arguments
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_length_sentence", type=int, default=512, help="Max length of sentence")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--validation_batch_size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data_set", type=str, default='pretrain', help="Dataset name")
    parser.add_argument("--comment", type=str, default='test', help="Comment for the current run")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--task", type=str, default='span_corruption', help="Specific task for pretraining")
    parser.add_argument("--local_tokenizer_path", type=str, default='google/mT5-small', help="Local tokenizer path")

    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.num_train_epochs, args.max_length_sentence, args.train_batch_size, args.validation_batch_size,
         args.learning_rate, args.data_set, args.comment, args.patience, args.task, args.local_tokenizer_path)